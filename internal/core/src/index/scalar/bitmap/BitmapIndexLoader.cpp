// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"
#include "storage/LocalFileIOPool.h"
#include "index/scalar/bitmap/BitmapIndexLoader.h"

#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>
#include "folly/ScopeGuard.h"
#include "folly/coro/BlockingWait.h"
#include "folly/coro/CurrentExecutor.h"
#include "folly/coro/WithCancellation.h"
#include "common/OpContext.h"
#include "storage/FileWriter.h"
#include "storage/EntryStreamUtils.h"

#include "index/ParamUtils.h"
#include "index/IndexLoadUtils.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "storage/artifact/FileSourceUtils.h"
#include "storage/artifact/LocalFileUtils.h"
#include "index/Families.h"
#include "index/Meta.h"
#include "index/Utils.h"
#include "index/contracts/Registry.h"
#include "index/scalar/bitmap/BitmapIndexReader.h"
#include "index/scalar/json/JsonProjectedIndexLoad.h"
#include "nlohmann/json.hpp"

namespace milvus::index {
namespace {

using bitmap_params::IsStringType;

constexpr size_t kFrozenAlignment = 32;
constexpr uint64_t kMaxCoordinateCount =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
constexpr std::string_view kLegacyNestedKey = "is_nested_index";
constexpr std::string_view kV3NestedKey = "is_nested";

struct RuntimeParams {
    DataType field_type{DataType::NONE};
    DataType value_type{DataType::NONE};
    bool nullable{false};
    bool nested{false};
    bool value_lookup{true};
    bool offset_cache{false};
};

RuntimeParams
ParseRuntimeParams(const Config& params) {
    RuntimeParams result;
    result.field_type =
        ReadDataTypeParam(params, "field_type").value_or(DataType::NONE);
    const auto array_element_type =
        ReadDataTypeParam(params, "array_element_type")
            .value_or(DataType::NONE);
    const auto configured_value_type =
        ReadDataTypeParam(params, "value_type").value_or(DataType::NONE);
    if (result.field_type == DataType::NONE &&
        array_element_type != DataType::NONE) {
        result.field_type = DataType::ARRAY;
    }
    result.value_type = result.field_type == DataType::ARRAY &&
                                array_element_type != DataType::NONE
                            ? array_element_type
                            : configured_value_type;
    if ((result.value_type == DataType::NONE ||
         result.value_type == DataType::ARRAY) &&
        result.field_type != DataType::ARRAY) {
        result.value_type = result.field_type;
    }
    result.nullable =
        GetValueFromConfigOrFallback<bool>(params, "nullable", false);
    result.nested = ReadRequiredNestedParam(params, "bitmap loader");
    result.value_lookup = result.field_type != DataType::ARRAY || result.nested;
    result.offset_cache =
        GetValueFromConfigOrFallback<bool>(params, ENABLE_OFFSET_CACHE, false);
    return result;
}

struct BitmapMeta {
    size_t index_length{0};
    size_t count{0};
    bool nested{false};
    bool has_nested{false};
};

BitmapMeta
ParseLegacyMeta(const std::vector<uint8_t>& encoded) {
    const std::string text(encoded.begin(), encoded.end());
    try {
        const auto json = nlohmann::json::parse(text);
        BitmapMeta result{
            .index_length = json.at(BITMAP_INDEX_LENGTH).get<size_t>(),
            .count = json.at(BITMAP_INDEX_NUM_ROWS).get<size_t>()};
        if (json.contains(kLegacyNestedKey)) {
            result.nested = json.at(kLegacyNestedKey).get<bool>();
            result.has_nested = true;
        }
        return result;
    } catch (const nlohmann::json::parse_error&) {
        try {
            const auto yaml = YAML::Load(text);
            BitmapMeta result{
                .index_length = yaml[BITMAP_INDEX_LENGTH].as<size_t>(),
                .count = yaml[BITMAP_INDEX_NUM_ROWS].as<size_t>()};
            const auto nested = yaml[std::string(kLegacyNestedKey)];
            if (nested) {
                result.nested = nested.as<bool>();
                result.has_nested = true;
            }
            return result;
        } catch (const YAML::Exception& error) {
            ThrowInfo(DataFormatBroken,
                      "invalid bitmap V1/V2 metadata: {}",
                      error.what());
        }
    } catch (const nlohmann::json::exception& error) {
        ThrowInfo(DataFormatBroken,
                  "invalid bitmap V1/V2 metadata: {}",
                  error.what());
    }
}

BitmapMeta
ReadMeta(storage::FileSource& source) {
    return ParseLegacyMeta(source.ReadEntry(BITMAP_INDEX_META));
}

void
ValidateMeta(const BitmapMeta& meta, const RuntimeParams& params) {
    if (meta.count > kMaxCoordinateCount ||
        (meta.count == 0 && meta.index_length != 0)) {
        ThrowInfo(DataFormatBroken,
                  "bitmap coordinate count {} is invalid for {} postings "
                  "(maximum {})",
                  meta.count,
                  meta.index_length,
                  kMaxCoordinateCount);
    }
    if (params.value_lookup && meta.index_length > meta.count) {
        ThrowInfo(DataFormatBroken,
                  "bitmap scalar posting count {} exceeds coordinate count "
                  "{}",
                  meta.index_length,
                  meta.count);
    }
}

size_t
PackedValidityBytes(size_t count) {
    return count / 8 + static_cast<size_t>(count % 8 != 0);
}

TargetBitmap
DecodeValidity(const std::vector<uint8_t>& encoded, size_t count) {
    const auto expected = PackedValidityBytes(count);
    if (encoded.size() != expected) {
        ThrowInfo(DataFormatBroken,
                  "bitmap validity size mismatch: expected {}, got {}",
                  expected,
                  encoded.size());
    }
    TargetBitmap result(count, false);
    for (size_t i = 0; i < count; ++i) {
        if ((encoded[i / 8] & static_cast<uint8_t>(1U << (i % 8))) != 0) {
            result.set(i);
        }
    }
    return result;
}

TargetBitmap
ToBitset(const roaring::Roaring& posting, size_t count) {
    TargetBitmap result(count, false);
    for (auto coordinate : posting) {
        if (coordinate >= count) {
            ThrowInfo(DataFormatBroken,
                      "bitmap posting coordinate {} exceeds count {}",
                      coordinate,
                      count);
        }
        result.set(coordinate);
    }
    return result;
}

template <typename T>
T
ReadKey(const uint8_t*& cursor, const uint8_t* end) {
    if constexpr (std::is_same_v<T, std::string>) {
        if (static_cast<size_t>(end - cursor) < sizeof(size_t)) {
            ThrowInfo(DataFormatBroken, "truncated bitmap string key length");
        }
        size_t length = 0;
        std::memcpy(&length, cursor, sizeof(length));
        cursor += sizeof(length);
        if (length > static_cast<size_t>(end - cursor)) {
            ThrowInfo(DataFormatBroken,
                      "truncated bitmap string key: expected {} bytes, got {}",
                      length,
                      end - cursor);
        }
        std::string result(reinterpret_cast<const char*>(cursor), length);
        cursor += length;
        return result;
    } else {
        if (static_cast<size_t>(end - cursor) < sizeof(T)) {
            ThrowInfo(DataFormatBroken, "truncated bitmap numeric key");
        }
        T result;
        std::memcpy(&result, cursor, sizeof(T));
        cursor += sizeof(T);
        return result;
    }
}

template <typename T>
roaring::Roaring
ReadPosting(const uint8_t*& cursor,
            const uint8_t* end,
            size_t ordinal,
            size_t count) {
    if (cursor == end) {
        ThrowInfo(DataFormatBroken, "truncated bitmap posting {}", ordinal);
    }
    roaring::Roaring posting;
    try {
        posting = roaring::Roaring::readSafe(
            reinterpret_cast<const char*>(cursor), end - cursor);
    } catch (const std::bad_alloc&) {
        ThrowInfo(MemAllocateFailed,
                  "failed to allocate while decoding bitmap posting");
    } catch (const std::exception& error) {
        ThrowInfo(DataFormatBroken,
                  "invalid bitmap posting {}: {}",
                  ordinal,
                  error.what());
    }
    const auto consumed = posting.getSizeInBytes(true);
    if (consumed > static_cast<size_t>(end - cursor)) {
        ThrowInfo(DataFormatBroken,
                  "bitmap posting {} exceeds serialized entry",
                  ordinal);
    }
    for (auto coordinate : posting) {
        if (coordinate >= count) {
            ThrowInfo(DataFormatBroken,
                      "bitmap posting coordinate {} exceeds count {}",
                      coordinate,
                      count);
        }
    }
    cursor += consumed;
    return posting;
}

template <typename T>
BitmapRoaringPostingMap<T>
DecodePostings(const uint8_t* data,
               size_t size,
               size_t index_length,
               size_t count) {
    if (size == 0) {
        if (index_length != 0) {
            ThrowInfo(DataFormatBroken,
                      "bitmap data is empty for {} postings",
                      index_length);
        }
        return {};
    }
    const auto* cursor = data;
    const auto* end = data + size;
    BitmapRoaringPostingMap<T> postings;
    for (size_t i = 0; i < index_length; ++i) {
        auto key = ReadKey<T>(cursor, end);
        auto posting = ReadPosting<T>(cursor, end, i, count);
        const auto inserted =
            postings.emplace(std::move(key), std::move(posting)).second;
        if (!inserted) {
            ThrowInfo(DataFormatBroken,
                      "bitmap data contains duplicate key at posting {}",
                      i);
        }
    }
    if (cursor != end) {
        ThrowInfo(DataFormatBroken,
                  "bitmap data has {} trailing bytes",
                  end - cursor);
    }
    return postings;
}

template <typename T>
void
ValidateCoordinateOwnership(const BitmapRoaringPostingMap<T>& postings,
                            bool one_value_per_coordinate) {
    if (!one_value_per_coordinate) {
        return;
    }
    roaring::Roaring occupied;
    for (const auto& [_, posting] : postings) {
        if (occupied.intersect(posting)) {
            ThrowInfo(DataFormatBroken,
                      "bitmap scalar postings contain an overlapping "
                      "coordinate");
        }
        occupied |= posting;
    }
}

class TemporaryFileGuard {
 public:
    explicit TemporaryFileGuard(std::string path) : path_(std::move(path)) {
    }

    TemporaryFileGuard(const TemporaryFileGuard&) = delete;
    TemporaryFileGuard&
    operator=(const TemporaryFileGuard&) = delete;

    TemporaryFileGuard(TemporaryFileGuard&& other) noexcept
        : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)) {
    }

    TemporaryFileGuard&
    operator=(TemporaryFileGuard&& other) noexcept {
        if (this != &other) {
            Cleanup();
            path_ = std::move(other.path_);
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    ~TemporaryFileGuard() {
        Cleanup();
    }

    // Replace the mkstemp template with an owned, open file.
    void
    Create() {
        AssertInfo(fd_ == -1, "bitmap temporary file was already created");
        fd_ = mkstemp(path_.MutablePath());
        if (fd_ == -1) {
            const auto error = errno;
            const auto path = path_.Release();
            ThrowInfo(FileCreateFailed,
                      "failed to create bitmap mmap file {}: {}",
                      path,
                      std::strerror(error));
        }
    }

    // Close the descriptor with error reporting, retaining path cleanup.
    void
    Close() {
        AssertInfo(fd_ != -1, "bitmap temporary file is already closed");
        const auto fd = std::exchange(fd_, -1);
        if (close(fd) != 0) {
            ThrowInfo(FileWriteFailed,
                      "failed to close bitmap mmap file {}: {}",
                      path_.Path(),
                      std::strerror(errno));
        }
    }

    const std::string&
    Path() const {
        return path_.Path();
    }

    // Transfer path cleanup only after the descriptor has been closed.
    std::string
    ReleasePath() {
        AssertInfo(fd_ == -1,
                   "bitmap temporary path cannot be released while open");
        return path_.Release();
    }

 private:
    // Best-effort descriptor cleanup; the path member owns unlinking.
    void
    Cleanup() noexcept {
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
    }

    storage::LocalEntryGuard path_;
    int fd_{-1};
};

TemporaryFileGuard
CreateTemporaryFile(const std::string& directory, std::string_view prefix) {
    if (directory.empty()) {
        ThrowInfo(FileCreateFailed, "bitmap mmap directory must not be empty");
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        ThrowInfo(FileCreateFailed,
                  "failed to create bitmap mmap directory {}: {}",
                  directory,
                  error.message());
    }
    TemporaryFileGuard file(
        (std::filesystem::path(directory) / (std::string(prefix) + "_XXXXXX"))
            .string());
    file.Create();
    return file;
}

size_t
AlignFrozenSize(size_t size) {
    if (size > std::numeric_limits<size_t>::max() - (kFrozenAlignment - 1)) {
        ThrowInfo(DataFormatBroken, "bitmap frozen posting size overflows");
    }
    return (size + kFrozenAlignment - 1) & ~(kFrozenAlignment - 1);
}

template <typename T>
struct FrozenPostings {
    // Declared first so the posting views are destroyed before their mapping.
    std::shared_ptr<BitmapMmapOwner> owner;
    BitmapRoaringPostingMap<T> postings;
};

std::shared_ptr<BitmapMmapOwner>
FinishFrozenFile(TemporaryFileGuard file, size_t file_size) {
    AssertInfo(file_size != 0,
               "bitmap mmap requires at least one non-empty posting file");

    storage::MappedRegionGuard mapping;
    {
        const auto map_fd = open(file.Path().c_str(), O_RDONLY);
        if (map_fd == -1) {
            ThrowInfo(FileOpenFailed,
                      "failed to open bitmap mmap file {}: {}",
                      file.Path(),
                      std::strerror(errno));
        }
        storage::FileDescriptorGuard descriptor(map_fd);
        auto* mapped = static_cast<char*>(
            mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, map_fd, 0));
        const auto saved_errno = errno;
        if (mapped == MAP_FAILED) {
            ThrowInfo(MmapError,
                      "failed to map bitmap frozen file {}: {}",
                      file.Path(),
                      std::strerror(saved_errno));
        }
        mapping = storage::MappedRegionGuard(mapped, file_size);
    }

    auto owner = std::make_shared<BitmapMmapOwner>(
        mapping.Data(), file_size, file.Path());
    mapping.Release();
    static_cast<void>(file.ReleasePath());
    return owner;
}

template <typename T>
folly::coro::Task<FrozenPostings<T>>
DecodeFrozenPostings(const uint8_t* data,
                     size_t size,
                     size_t index_length,
                     size_t count,
                     TargetBitmap& validity,
                     bool rebuild_validity,
                     const storage::LoadOptions& opts,
                     const folly::CancellationToken& token,
                     bool use_async) {
    if (size == 0) {
        ThrowInfo(DataFormatBroken,
                  "bitmap mmap data is empty for {} postings",
                  index_length);
    }
    const auto* cursor = data;
    const auto* end = data + size;
    const auto priority =
        opts.op_ctx && opts.op_ctx->runtime_load_priority.value_or(0) != 0
            ? proto::common::LoadPriority::LOW
            : proto::common::LoadPriority::HIGH;
    std::optional<TemporaryFileGuard> file;
    std::unique_ptr<storage::FileWriter> writer;
    FrozenPostings<T> result;
    std::exception_ptr failure;
    try {
        auto local_io = [&] {
            file.emplace(
                CreateTemporaryFile(opts.mmap_dir_path, "bitmap_frozen"));
            file->Close();
            writer = std::make_unique<storage::FileWriter>(
                file->Path(),
                storage::io::GetPriorityFromLoadPriority(priority));
        };
        if (use_async) {
            co_await storage::RunLocalFileIOAsync(local_io, priority);
        } else {
            local_io();
        }
        std::map<T, std::pair<size_t, size_t>> locations;
        std::vector<uint8_t> frozen_buffer;
        size_t file_offset = 0;
        for (size_t i = 0; i < index_length; ++i) {
            storage::ThrowIfCancelled(token, "Bitmap::ConvertFrozen");
            auto key = ReadKey<T>(cursor, end);
            if (locations.find(key) != locations.end()) {
                ThrowInfo(DataFormatBroken,
                          "bitmap data contains duplicate key at posting {}",
                          i);
            }
            {
                // Keep only one decoded portable posting resident at a time.
                // The mapped reader later points at the frozen output file.
                auto posting = ReadPosting<T>(cursor, end, i, count);
                if (rebuild_validity) {
                    for (auto coordinate : posting) {
                        validity.set(coordinate);
                    }
                }
                const auto frozen_size = posting.getFrozenSizeInBytes();
                const auto aligned_size = AlignFrozenSize(frozen_size);
                if (aligned_size >
                    std::numeric_limits<size_t>::max() - file_offset) {
                    ThrowInfo(DataFormatBroken,
                              "bitmap frozen file size overflows");
                }
                const auto begin = frozen_buffer.size();
                if (aligned_size > std::numeric_limits<size_t>::max() - begin) {
                    ThrowInfo(DataFormatBroken,
                              "bitmap frozen batch size overflows");
                }
                if (begin + aligned_size > frozen_buffer.capacity()) {
                    frozen_buffer.reserve(std::max(kBitmapFrozenBatchBytes,
                                                   begin + aligned_size));
                }
                frozen_buffer.resize(begin + aligned_size, 0);
                posting.writeFrozen(
                    reinterpret_cast<char*>(frozen_buffer.data() + begin));
                if (frozen_buffer.size() >= kBitmapFrozenBatchBytes ||
                    i + 1 == index_length) {
                    storage::ThrowIfCancelled(token, "Bitmap::WriteFrozen");
                    auto write_batch = [&] {
                        storage::ThrowIfCancelled(token, "Bitmap::WriteFrozen");
                        writer->Write(frozen_buffer.data(),
                                      frozen_buffer.size());
                    };
                    if (use_async) {
                        co_await storage::RunLocalFileIOAsync(write_batch,
                                                              priority);
                    } else {
                        write_batch();
                    }
                    frozen_buffer.clear();
                }
                locations.emplace(std::move(key),
                                  std::make_pair(file_offset, frozen_size));
                file_offset += aligned_size;
            }
            // Release the worker after each bounded write batch. Sync loading has
            // no executor requirement and stays on its caller thread.
            if (use_async && frozen_buffer.empty() && i + 1 < index_length) {
                co_await folly::coro::co_reschedule_on_current_executor;
            }
        }
        if (cursor != end) {
            ThrowInfo(DataFormatBroken,
                      "bitmap data has {} trailing bytes",
                      end - cursor);
        }
        auto finish_file = [&] {
            writer->Finish();
            writer.reset();
            storage::ThrowIfCancelled(token, "Bitmap::MapFrozen");
            result.owner = FinishFrozenFile(std::move(*file), file_offset);
        };
        if (use_async) {
            co_await storage::RunLocalFileIOAsync(finish_file, priority);
        } else {
            finish_file();
        }
        for (const auto& [key, location] : locations) {
            result.postings.emplace(
                key,
                roaring::Roaring::frozenView(
                    result.owner->Data() + location.first, location.second));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    auto cleanup = [&] {
        if (failure)
            result = {};
        writer.reset();
        file.reset();
    };
    if (use_async) {
        co_await storage::RunLocalFileIOAsync(cleanup, priority);
    } else {
        cleanup();
    }
    if (failure)
        std::rethrow_exception(failure);
    co_return result;
}

template <typename T>
void
RebuildValidity(const BitmapRoaringPostingMap<T>& postings,
                TargetBitmap& validity) {
    for (const auto& [_, posting] : postings) {
        for (auto coordinate : posting) {
            validity.set(coordinate);
        }
    }
}

BitmapReaderOptions
MakeReaderOptions(TargetBitmap validity,
                  const BitmapMeta& meta,
                  const RuntimeParams& params) {
    return BitmapReaderOptions{.valid_bitset = std::move(validity),
                               .total_num_rows = meta.count,
                               .nested = meta.nested,
                               .value_lookup = params.value_lookup,
                               .value_type = params.value_type,
                               .offset_cache = params.offset_cache};
}

template <typename T>
std::unique_ptr<IIndexReaderBase>
OpenDecodedState(const uint8_t* data,
                 size_t size,
                 const BitmapMeta& meta,
                 const RuntimeParams& params,
                 TargetBitmap validity,
                 bool rebuild_validity,
                 BitmapLayout layout) {
    auto postings =
        DecodePostings<T>(data, size, meta.index_length, meta.count);
    ValidateCoordinateOwnership(postings, params.value_lookup);
    if (rebuild_validity) {
        RebuildValidity(postings, validity);
    }
    auto options = MakeReaderOptions(std::move(validity), meta, params);
    if (layout == BitmapLayout::Roaring) {
        return CreateBitmapIndexReader<T>(std::move(postings),
                                          std::move(options));
    }

    BitmapBitsetPostingMap<T> bitset_postings;
    for (const auto& [key, posting] : postings) {
        bitset_postings.emplace(key, ToBitset(posting, meta.count));
    }
    return CreateBitmapIndexReader<T>(std::move(bitset_postings),
                                      std::move(options));
}

template <typename T>
folly::coro::Task<std::unique_ptr<IIndexReaderBase>>
OpenDecodedMmapState(const uint8_t* data,
                     size_t size,
                     const BitmapMeta& meta,
                     const RuntimeParams& params,
                     TargetBitmap validity,
                     bool rebuild_validity,
                     const storage::LoadOptions& opts,
                     const folly::CancellationToken& token,
                     bool use_async) {
    auto frozen = co_await DecodeFrozenPostings<T>(data,
                                                   size,
                                                   meta.index_length,
                                                   meta.count,
                                                   validity,
                                                   rebuild_validity,
                                                   opts,
                                                   token,
                                                   use_async);
    ValidateCoordinateOwnership(frozen.postings, params.value_lookup);
    auto options = MakeReaderOptions(std::move(validity), meta, params);
    co_return CreateBitmapIndexReader<T>(std::move(frozen.postings),
                                         std::move(options),
                                         std::move(frozen.owner));
}

template <typename Result, typename F>
Result
DispatchBitmapType(DataType value_type, F&& fn) {
    switch (value_type) {
        case DataType::BOOL:
            return fn.template operator()<bool>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
        case DataType::INT64:
            return fn.template operator()<int64_t>();
        case DataType::FLOAT:
            return fn.template operator()<float>();
        case DataType::DOUBLE:
            return fn.template operator()<double>();
        case DataType::STRING:
        case DataType::VARCHAR:
            return fn.template operator()<std::string>();
        default:
            ThrowInfo(DataTypeInvalid,
                      "unsupported bitmap value type {}",
                      static_cast<int>(value_type));
    }
}

std::unique_ptr<IIndexReaderBase>
DispatchOpen(DataType value_type,
             const uint8_t* data,
             size_t size,
             const BitmapMeta& meta,
             const RuntimeParams& params,
             TargetBitmap validity,
             bool rebuild_validity,
             BitmapLayout layout) {
    return DispatchBitmapType<std::unique_ptr<IIndexReaderBase>>(
        value_type, [&]<typename T>() {
            return OpenDecodedState<T>(data,
                                       size,
                                       meta,
                                       params,
                                       std::move(validity),
                                       rebuild_validity,
                                       layout);
        });
}

folly::coro::Task<std::unique_ptr<IIndexReaderBase>>
DispatchMmapOpen(DataType value_type,
                 const uint8_t* data,
                 size_t size,
                 const BitmapMeta& meta,
                 const RuntimeParams& params,
                 TargetBitmap validity,
                 bool rebuild_validity,
                 const storage::LoadOptions& opts,
                 const folly::CancellationToken& token,
                 bool use_async) {
    return DispatchBitmapType<
        folly::coro::Task<std::unique_ptr<IIndexReaderBase>>>(
        value_type, [&]<typename T>() {
            return OpenDecodedMmapState<T>(data,
                                           size,
                                           meta,
                                           params,
                                           std::move(validity),
                                           rebuild_validity,
                                           opts,
                                           token,
                                           use_async);
        });
}

std::unique_ptr<IIndexReaderBase>
LoadBitmapPayload(storage::FileSource& source,
                  const storage::LoadOptions& opts,
                  const RuntimeParams& params) {
    auto meta = ReadMeta(source);
    if (meta.has_nested && meta.nested != params.nested) {
        ThrowInfo(DataFormatBroken,
                  "bitmap persisted nested value {} disagrees with runtime "
                  "value {}",
                  meta.nested,
                  params.nested);
    }
    // Old artifacts may lack this metadata. The adapter-supplied runtime
    // value is mandatory, so capability derivation and opening still use the
    // same coordinate domain without eagerly reading artifact metadata.
    meta.nested = params.nested;
    ValidateMeta(meta, params);

    TargetBitmap validity(meta.count, meta.nested || !params.nullable);
    bool rebuild_validity = params.nullable && !meta.nested;
    if (source.HasEntry(BITMAP_INDEX_VALID_BITSET)) {
        auto encoded = source.ReadEntry(BITMAP_INDEX_VALID_BITSET);
        validity = DecodeValidity(encoded, meta.count);
        rebuild_validity = false;
    }

    const auto layout =
        meta.index_length <=
                static_cast<size_t>(DEFAULT_BITMAP_INDEX_BUILD_MODE_BOUND)
            ? BitmapLayout::Bitset
            : BitmapLayout::Roaring;
    if (opts.enable_mmap && layout == BitmapLayout::Roaring) {
        auto portable_file =
            CreateTemporaryFile(opts.mmap_dir_path, "bitmap_portable");
        portable_file.Close();
        source.ReadEntryToLocalFile(BITMAP_INDEX_DATA, portable_file.Path());
        const auto size = storage::LocalFileSize(
            portable_file.Path(), "failed to size bitmap mmap input");
        if (size == 0) {
            ThrowInfo(DataFormatBroken,
                      "bitmap data is empty for non-empty mmap index");
        }
        const auto fd = open(portable_file.Path().c_str(), O_RDONLY);
        if (fd == -1) {
            ThrowInfo(FileOpenFailed,
                      "failed to open bitmap file {}: {}",
                      portable_file.Path(),
                      std::strerror(errno));
        }
        storage::FileDescriptorGuard descriptor(fd);
        // Unmap before evicting the one-pass input; keep fd alive for advice.
        auto evict_input = folly::makeGuard(
            [fd] { (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); });
        auto* mapped = static_cast<char*>(
            mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        const auto saved_errno = errno;
        if (mapped == MAP_FAILED) {
            ThrowInfo(MmapError,
                      "failed to map bitmap file {}: {}",
                      portable_file.Path(),
                      std::strerror(saved_errno));
        }
        storage::MappedRegionGuard mapping(mapped, size);
        // Conversion scans portable postings once, in order. Best-effort only.
        (void)::madvise(mapped, size, MADV_SEQUENTIAL);
        const auto operation_token = opts.op_ctx
                                         ? opts.op_ctx->cancellation_token
                                         : folly::CancellationToken{};
        const auto token = operation_token;
        return folly::coro::blockingWait(
            DispatchMmapOpen(params.value_type,
                             reinterpret_cast<const uint8_t*>(mapping.Data()),
                             size,
                             meta,
                             params,
                             std::move(validity),
                             rebuild_validity,
                             opts,
                             token,
                             false));
    }

    auto data = source.ReadEntry(BITMAP_INDEX_DATA);
    return DispatchOpen(params.value_type,
                        data.data(),
                        data.size(),
                        meta,
                        params,
                        std::move(validity),
                        rebuild_validity,
                        layout);
}

struct PackedBitmapState {
    RuntimeParams params;
    BitmapMeta meta;
    BitmapLayout layout;
    JsonProjectedOpenPlan projection;
    std::shared_ptr<TargetBitmap> validity;
    bool rebuild_validity{false};
    std::shared_ptr<std::vector<uint8_t>> data;
    std::shared_ptr<storage::IndexFileTarget> file;
};

}  // namespace

ReaderCaps
BitmapIndexLoader::DeriveCaps(const Config& index_meta) {
    const auto params = ParseRuntimeParams(index_meta);
    return DeriveJsonProjectedCaps(
        families::kBitmap,
        index_meta,
        ReaderCaps{
            .predicate = true,
            .pattern_match = IsStringType(params.value_type),
            .nested = params.nested,
            .value_lookup = params.value_lookup,
            .cheap_value_lookup = params.value_lookup && params.offset_cache,
            .exact = !params.nested});
}

IIndexReaderBasePtr
BitmapIndexLoader::Open(storage::FileSource& source,
                        const storage::LoadOptions& opts) {
    auto projection = PrepareJsonProjectedOpen(families::kBitmap, source, opts);
    const auto params = ParseRuntimeParams(opts.params);
    if (params.value_type == DataType::NONE ||
        params.value_type == DataType::ARRAY) {
        ThrowInfo(DataTypeInvalid,
                  "bitmap loader requires value_type or array_element_type");
    }
    auto inner = LoadBitmapPayload(source, opts, params);
    return FinishJsonProjectedOpen(
        std::move(projection), source, std::move(inner));
}

IndexLoadPlan
BitmapIndexLoader::PlanPacked(const storage::IndexEntryDirectory& directory,
                              const nlohmann::json& metadata,
                              const storage::LoadOptions& opts) {
    auto state = std::make_shared<PackedBitmapState>();
    state->params = ParseRuntimeParams(opts.params);
    if (state->params.value_type == DataType::NONE ||
        state->params.value_type == DataType::ARRAY) {
        ThrowInfo(DataTypeInvalid,
                  "bitmap loader requires value_type or array_element_type");
    }
    state->meta.index_length =
        ReadRequiredIndexMeta<size_t>(metadata, BITMAP_INDEX_LENGTH);
    state->meta.count =
        ReadRequiredIndexMeta<size_t>(metadata, BITMAP_INDEX_NUM_ROWS);
    state->meta.nested = state->params.nested;
    if (metadata.contains(kV3NestedKey) &&
        ReadRequiredIndexMeta<bool>(metadata, "is_nested") !=
            state->params.nested) {
        ThrowInfo(DataFormatBroken,
                  "bitmap persisted nested flag disagrees with runtime");
    }
    ValidateMeta(state->meta, state->params);
    state->layout =
        state->meta.index_length <=
                static_cast<size_t>(DEFAULT_BITMAP_INDEX_BUILD_MODE_BOUND)
            ? BitmapLayout::Bitset
            : BitmapLayout::Roaring;
    const bool has_validity = directory.HasEntry(BITMAP_INDEX_VALID_BITSET);
    state->rebuild_validity =
        !has_validity && state->params.nullable && !state->meta.nested;
    state->validity = std::make_shared<TargetBitmap>(
        state->meta.count,
        !has_validity && (state->meta.nested || !state->params.nullable));
    IndexLoadPlan plan;
    plan.load_context = state;
    plan.entries.reserve(3);
    state->projection = PreparePackedJsonProjectedOpen(
        families::kBitmap, directory, metadata, opts, plan, state->meta.count);
    if (has_validity) {
        const auto bytes = PackedValidityBytes(state->meta.count);
        if (directory.At(BITMAP_INDEX_VALID_BITSET).plaintext_size != bytes) {
            ThrowInfo(DataFormatBroken,
                      "bitmap validity size disagrees with row count");
        }
        static_assert(std::endian::native == std::endian::little);
        plan.entries.push_back(
            {BITMAP_INDEX_VALID_BITSET,
             storage::MemoryEntryTarget{
                 state->validity,
                 reinterpret_cast<uint8_t*>(state->validity->data()),
                 bytes}});
    }
    const auto bytes = directory.At(BITMAP_INDEX_DATA).plaintext_size;
    if (opts.enable_mmap && state->layout == BitmapLayout::Roaring) {
        if (bytes == 0) {
            ThrowInfo(DataFormatBroken, "bitmap mmap payload is empty");
        }
        auto file = CreateTemporaryFile(opts.mmap_dir_path, "bitmap_portable");
        file.Close();
        state->file = std::make_shared<storage::IndexFileTarget>(
            file.Path(), bytes, false);
        // The plan must own even an unprepared temporary file on failure.
        state->file->Prepare(storage::io::GetPriorityFromLoadPriority(
            opts.op_ctx && opts.op_ctx->runtime_load_priority.value_or(0) != 0
                ? proto::common::LoadPriority::LOW
                : proto::common::LoadPriority::HIGH));
        static_cast<void>(file.ReleasePath());
        plan.entries.push_back(
            {BITMAP_INDEX_DATA,
             storage::FileEntryTarget{state->file, 0, bytes}});
    } else {
        state->data = std::make_shared<std::vector<uint8_t>>(bytes);
        plan.entries.push_back({BITMAP_INDEX_DATA,
                                storage::MemoryEntryTarget{
                                    state->data, state->data->data(), bytes}});
    }
    return plan;
}

folly::coro::Task<IIndexReaderBasePtr>
BitmapIndexLoader::FinishPacked(IndexLoadPlan& plan,
                                const storage::LoadOptions& opts,
                                bool use_async,
                                folly::CancellationToken token) {
    const auto operation_token = opts.op_ctx ? opts.op_ctx->cancellation_token
                                             : folly::CancellationToken{};
    token = folly::cancellation_token_merge(token, operation_token);
    const auto state =
        std::any_cast<std::shared_ptr<PackedBitmapState>>(plan.load_context);
    if (state->meta.count % 8 != 0) {
        // Match legacy decoding: CRC covers all bytes, unused tail bits are ignored.
        auto* bytes = reinterpret_cast<uint8_t*>(state->validity->data());
        bytes[state->meta.count / 8] &=
            static_cast<uint8_t>((1u << (state->meta.count % 8)) - 1u);
    }
    IIndexReaderBasePtr inner;
    if (state->file) {
        const auto priority =
            opts.op_ctx && opts.op_ctx->runtime_load_priority.value_or(0) != 0
                ? proto::common::LoadPriority::LOW
                : proto::common::LoadPriority::HIGH;
        int fd = -1;
        char* mapped = nullptr;
        std::exception_ptr failure;
        try {
            auto local_io = [&] {
                const auto& path = state->file->path;
                fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) {
                    ThrowInfo(FileOpenFailed,
                              "failed to open bitmap target {}: {}",
                              path,
                              std::strerror(errno));
                }
                mapped = static_cast<char*>(::mmap(nullptr,
                                                   state->file->file_size,
                                                   PROT_READ,
                                                   MAP_PRIVATE,
                                                   fd,
                                                   0));
                if (mapped == MAP_FAILED) {
                    ThrowInfo(MmapError,
                              "failed to map bitmap target {}: {}",
                              path,
                              std::strerror(errno));
                }

                // Conversion scans portable postings once, in order. Best-effort only.
                (void)::madvise(
                    mapped, state->file->file_size, MADV_SEQUENTIAL);
            };
            if (use_async) {
                co_await storage::RunLocalFileIOAsync(local_io, priority);
            } else {
                local_io();
            }
            inner = co_await DispatchMmapOpen(
                state->params.value_type,
                reinterpret_cast<const uint8_t*>(mapped),
                state->file->file_size,
                state->meta,
                state->params,
                std::move(*state->validity),
                state->rebuild_validity,
                opts,
                token,
                use_async);
        } catch (...) {
            failure = std::current_exception();
        }
        auto cleanup = [&] {
            if (mapped != nullptr && mapped != MAP_FAILED) {
                (void)::munmap(mapped, state->file->file_size);
            }
            if (fd >= 0) {
                (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                (void)::close(fd);
            }
        };
        if (use_async) {
            co_await storage::RunLocalFileIOAsync(cleanup, priority);
        } else {
            cleanup();
        }
        if (failure)
            std::rethrow_exception(failure);
    } else {
        inner = DispatchOpen(state->params.value_type,
                             state->data->data(),
                             state->data->size(),
                             state->meta,
                             state->params,
                             std::move(*state->validity),
                             state->rebuild_validity,
                             state->layout);
    }
    co_return FinishPackedJsonProjectedOpen(std::move(state->projection),
                                            std::move(inner));
}

namespace {

const bool kBitmapLoaderRegistered = [] {
    LoaderRegistry::Instance().Register<BitmapIndexLoader>();
    return true;
}();

}  // namespace

}  // namespace milvus::index
