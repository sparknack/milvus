// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thread>
#include <gtest/gtest.h>
#include <numeric>
#include <cstring>
#include "folly/executors/ManualExecutor.h"
#include "folly/coro/BlockingWait.h"
#include "storage/FileWriter.h"
#include "index/contracts/query/IScalarPredicateReader.h"

#include <cstdlib>
#include <filesystem>
#include "index/IndexTypeAdapter.h"
#include "index/Meta.h"
#include "index/contracts/query/IPatternMatchReader.h"
#include "index/scalar/bitmap/BitmapIndexLoader.h"
#include "index/scalar/marisa/MarisaIndexLoader.h"
#include "index/scalar/ngram/NgramIndexLoader.h"
#include "index/scalar/sort/SortedIndexLoader.h"
#include "index/scalar/sort/SortedIndexFormat.h"
#include "index/scalar/sort/SortedIndexReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index::test {
namespace {
storage::LoadOptions
ScalarOptions(DataType type) {
    storage::LoadOptions opts;
    opts.params = {{"field_type", type},
                   {"value_type", type},
                   {"nested", false},
                   {"nullable", true}};
    return opts;
}

void
PlanProjection(IndexLoadPlan& plan,
               const storage::IndexEntryDirectory& directory,
               const nlohmann::json& metadata) {
    storage::LoadOptions opts;
    opts.params = AnnotateJsonProjectionCompleteness(
        ScalarReaderBackends()
            .Get<double>("JsonProjectedSortedDouble")
            .LoadParams({}),
        directory,
        metadata);
    static_cast<void>(PreparePackedJsonProjectedOpen(
        families::kSort, directory, metadata, opts, plan));
}

milvus::storage::IndexEntryDirectory
ErrorTestDirectory(
    std::initializer_list<std::pair<std::string, size_t>> entries) {
    nlohmann::json directory = {{"entries", nlohmann::json::array()}};
    size_t offset = 0;
    for (const auto& [name, bytes] : entries) {
        directory["entries"].push_back({{"name", name},
                                        {"offset", offset},
                                        {"size", bytes},
                                        {"crc32", "00000000"}});
        offset += bytes;
    }
    const auto json = directory.dump();
    return milvus::storage::ParseIndexEntryDirectory(
               std::span(reinterpret_cast<const uint8_t*>(json.data()),
                         json.size()),
               json.size() + offset + milvus::storage::MILVUS_V3_MAGIC_SIZE +
                   milvus::storage::MILVUS_V3_FOOTER_SIZE)
        .first;
}

template <typename F>
void
ExpectPackedLoadError(milvus::ErrorCode expected, F&& load) {
    try {
        load();
        FAIL() << "expected classified packed load error";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), expected);
        auto status = milvus::FailureCStatus(&error);
        EXPECT_EQ(status.error_code, static_cast<int>(expected));
        free(const_cast<char*>(status.error_msg));
    }
}
}  // namespace

TEST(ScalarIndexV3AsyncTest, HeapAndMmapPreserveValuesAndNulls) {
    for (const auto* name : {"BitmapVarchar",
                             "BitmapVarcharMmap",
                             "SortedVarchar",
                             "SortedVarcharMmap",
                             "InvertedVarchar",
                             "InvertedVarcharMmap",
                             "MarisaVarchar",
                             "MarisaVarcharMmap",
                             "FmIndexVarchar",
                             "FmIndexVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        ScalarTestData<std::string_view> data(
            {"alpha", "beta", "beta", "null"});
        data.validity.reset(3);
        const ScalarTestInput<std::string_view> input(data);
        auto artifact = backend.Build(input.View(), {.row_count = 4});
        auto reader =
            OpenV3(backend, SerializeV3(*artifact), {.row_count = 4}, true);
        const auto* pattern =
            dynamic_cast<const IPatternMatchReader*>(reader.get());
        ASSERT_NE(pattern, nullptr);
        ExpectHits(
            pattern->PatternMatch("beta", PatternOp::PrefixMatch), 4, {1, 2});
        ExpectHits(
            pattern->PatternMatch("null", PatternOp::PrefixMatch), 4, {});
        const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
        ASSERT_NE(nulls, nullptr);
        ExpectHits(nulls->IsNull(), 4, {3});
    }
}

TEST(ScalarIndexV3ErrorCodeTest, CorruptPayloadCleansSyncAndAsyncTargets) {
    const auto& backend =
        ScalarReaderBackends().Get<std::string_view>("MarisaVarcharMmap");
    ScalarTestData<std::string_view> data({"alpha", "beta"});
    const ScalarTestInput<std::string_view> values(data);
    auto artifact = backend.Build(values.View(), {.row_count = 2});
    for (const bool use_async : {false, true}) {
        SCOPED_TRACE(use_async);
        auto staging = storage::LocalDirectory::CreateOwned(
            std::filesystem::temp_directory_path().string(),
            "packed-crc-XXXXXX",
            "packed CRC test");
        auto bytes = MakePackedArtifactBuffer(SerializeV3(*artifact));
        auto input = std::make_shared<storage::RemoteInputStream>(
            std::make_shared<arrow::io::BufferReader>(bytes));
        auto source = storage::IndexEntryReader::Open(input, input->Size());
        const auto& payload = std::get<storage::PlainEntrySource>(
            source->Directory().At(MARISA_TRIE_INDEX).source);
        bytes->mutable_data()[payload.remote_offset] ^= 1;
        storage::LoadOptions options;
        options.enable_mmap = true;
        options.mmap_dir_path = staging->Path();
        options.params = backend.LoadParams({.row_count = 2});
        const auto loader = LoaderRegistry::Instance().Lookup(backend.Family());
        ExpectPackedLoadError(DataFormatBroken, [&] {
            if (use_async) {
                auto load = [&]() -> folly::coro::Task<IIndexReaderBasePtr> {
                    const auto priority = proto::common::LoadPriority::HIGH;
                    auto async_source =
                        co_await storage::AsyncIndexEntryReader::Open(
                            input, 0, priority);
                    co_return co_await LoadPackedIndexAsync(
                        loader, *async_source, options, priority);
                };
                static_cast<void>(folly::coro::blockingWait(
                    load().scheduleOn(storage::ResolveAsyncLoadExecutor(
                        {}, proto::common::LoadPriority::HIGH))));
            } else {
                static_cast<void>(LoadPackedIndex(loader, *source, options));
            }
        });
        EXPECT_TRUE(std::filesystem::is_empty(staging->Path()));
    }
}

TEST(ScalarIndexV3ErrorCodeTest, PersistedLengthsAreDataFormatErrors) {
    using namespace milvus;
    using namespace milvus::index;
    const auto numeric_opts = ScalarOptions(DataType::INT64);

    nlohmann::json sort_meta{
        {"index_length", 1}, {"num_rows", 1}, {"is_nested", false}};
    ExpectPackedLoadError(DataFormatBroken, [&] {
        SortedIndexLoader::PlanPacked(
            ErrorTestDirectory({{"index_data", 1}}), sort_meta, numeric_opts);
    });
    ExpectPackedLoadError(DataFormatBroken, [&] {
        SortedIndexLoader::PlanPacked(
            ErrorTestDirectory({{"index_data", sizeof(IndexStructure<int64_t>)},
                                {"idx_to_offsets", 1},
                                {"valid_bitset", 1}}),
            sort_meta,
            numeric_opts);
    });

    const auto string_opts = ScalarOptions(DataType::VARCHAR);
    const nlohmann::json string_meta{{"version", sort_format::kStringVersion},
                                     {"num_rows", 8},
                                     {"is_nested", false}};
    ExpectPackedLoadError(DataFormatBroken, [&] {
        SortedIndexLoader::PlanPacked(
            ErrorTestDirectory({{"index_data", 1}, {"valid_bitset", 2}}),
            string_meta,
            string_opts);
    });
    const auto bitmap_opts = ScalarOptions(DataType::INT64);
    const nlohmann::json bitmap_meta{{BITMAP_INDEX_LENGTH, 1},
                                     {BITMAP_INDEX_NUM_ROWS, 8}};
    ExpectPackedLoadError(DataFormatBroken, [&] {
        BitmapIndexLoader::PlanPacked(
            ErrorTestDirectory(
                {{BITMAP_INDEX_DATA, 1}, {BITMAP_INDEX_VALID_BITSET, 2}}),
            bitmap_meta,
            bitmap_opts);
    });
    const auto marisa_opts = ScalarOptions(DataType::VARCHAR);
    ExpectPackedLoadError(DataFormatBroken, [&] {
        MarisaIndexLoader::PlanPacked(
            ErrorTestDirectory({{MARISA_TRIE_INDEX, 1},
                                {MARISA_STR_IDS, sizeof(int64_t)},
                                {MARISA_CSR_INDEX, sizeof(uint32_t)}}),
            {},
            marisa_opts);
    });
    auto ngram_opts = ScalarOptions(DataType::VARCHAR);
    ngram_opts.params[MIN_GRAM] = 2;
    ngram_opts.params[MAX_GRAM] = 3;
    ngram_opts.params[SCALAR_INDEX_ENGINE_VERSION] = 3;
    ExpectPackedLoadError(DataFormatBroken, [&] {
        NgramIndexLoader::PlanPacked(
            ErrorTestDirectory({{"engine_file", 1}, {"ngram_avg_row_size", 1}}),
            {{"has_null", false}, {"file_names", {"engine_file"}}},
            ngram_opts);
    });
    IndexLoadPlan plan;
    ExpectPackedLoadError(DataFormatBroken, [&] {
        PlanProjection(
            plan,
            ErrorTestDirectory({{INDEX_NON_EXIST_OFFSET_FILE_NAME, 1}}),
            {{"has_non_exist", true}});
    });
}

TEST(ScalarIndexV3ErrorCodeTest, OptionalMetadataValidatesPresentTypes) {
    using namespace milvus;
    using namespace milvus::index;
    const auto bitmap_opts = ScalarOptions(DataType::INT64);
    const auto directory = ErrorTestDirectory({{BITMAP_INDEX_DATA, 0}});
    nlohmann::json metadata{{BITMAP_INDEX_LENGTH, 0},
                            {BITMAP_INDEX_NUM_ROWS, 0}};
    EXPECT_NO_THROW(
        BitmapIndexLoader::PlanPacked(directory, metadata, bitmap_opts));
    metadata["is_nested"] = "true";
    ExpectPackedLoadError(DataFormatBroken, [&] {
        BitmapIndexLoader::PlanPacked(directory, metadata, bitmap_opts);
    });
    IndexLoadPlan plan;
    EXPECT_NO_THROW(PlanProjection(plan, directory, {}));
    ExpectPackedLoadError(DataFormatBroken, [&] {
        PlanProjection(plan, directory, {{"has_non_exist", "false"}});
    });
    ExpectPackedLoadError(DataFormatBroken, [&] {
        PlanProjection(plan, directory, {{"has_non_exist", true}});
    });
}

TEST(ScalarIndexV3ErrorCodeTest, UnsupportedFormatVersionKeepsItsCode) {
    using namespace milvus;
    using namespace milvus::index;
    const auto string_opts = ScalarOptions(DataType::VARCHAR);
    ExpectPackedLoadError(Unsupported, [&] {
        SortedIndexLoader::PlanPacked(ErrorTestDirectory({}),
                                      {{"num_rows", 0}, {"version", 9999}},
                                      string_opts);
    });
    const auto marisa_opts = ScalarOptions(DataType::VARCHAR);
    ExpectPackedLoadError(Unsupported, [&] {
        MarisaIndexLoader::PlanPacked(
            ErrorTestDirectory({{MARISA_TRIE_INDEX, 1},
                                {MARISA_STR_IDS, sizeof(int64_t)},
                                {MARISA_CSR_INDEX, sizeof(uint32_t)},
                                {MARISA_CSR_OFFSETS, 0}}),
            {{"csr_num_keys", 0}, {"marisa_csr_format_version", 9999}},
            marisa_opts);
    });
}

TEST(ScalarIndexV3ErrorCodeTest, UnrecognizedHybridMetadataIsDataFormatError) {
    using namespace milvus;
    using namespace milvus::index;
    const Config config{{INDEX_FILES, {"milvus_packed_hybrid_index.v3"}}};
    ExpectPackedLoadError(DataFormatBroken, [&] {
        ResolvePackedLoadFamily(families::kHybrid, {}, config);
    });
    // Old standalone files still identify their type without hybrid metadata.
    EXPECT_EQ(ResolvePackedLoadFamily(
                  families::kHybrid,
                  {},
                  {{INDEX_FILES, {"milvus_packed_stlsort_index.v3"}}}),
              families::kSort);
}

TEST(ScalarIndexV3ErrorCodeTest, MmapFailureKeepsItsCodeAndCleansTargets) {
    auto staging = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "packed-error-XXXXXX",
        "packed mmap failure test");
    auto opts = ScalarOptions(DataType::INT64);
    opts.enable_mmap = true;
    opts.mmap_dir_path = staging->Path();
    std::vector<std::string> paths;
    {
        auto plan = SortedIndexLoader::PlanPacked(
            ErrorTestDirectory(
                {{"index_data", sizeof(IndexStructure<int64_t>)},
                 {"idx_to_offsets", sizeof(int32_t)},
                 {"valid_bitset", TargetBitmap(1, false).size_in_bytes()}}),
            {{"index_length", 1}, {"num_rows", 1}, {"is_nested", false}},
            opts);
        for (const auto& entry : plan.entries) {
            if (const auto* file =
                    std::get_if<storage::FileEntryTarget>(&entry.target)) {
                paths.push_back(file->staging->path);
                file->staging->Prepare(storage::io::Priority::HIGH);
                file->staging->Finish();
            }
        }
        ASSERT_EQ(paths.size(), 2);
        // /dev/null opens successfully but cannot be mapped. This reaches the
        // real mmap failure without relying on obsolete zero-row behavior.
        const auto& data =
            std::get<storage::FileEntryTarget>(plan.At("index_data").target);
        ASSERT_TRUE(std::filesystem::remove(data.staging->path));
        std::filesystem::create_symlink("/dev/null", data.staging->path);
        ExpectPackedLoadError(MmapError, [&] {
            folly::coro::blockingWait(
                SortedIndexLoader::FinishPacked(plan, opts, false, {}));
        });
    }
    for (const auto& path : paths) {
        EXPECT_FALSE(std::filesystem::exists(path));
    }
    EXPECT_TRUE(std::filesystem::is_empty(staging->Path()));
}
TEST(ScalarIndexV3LoadTest, BitmapFrozenConversionYieldsBetweenBatches) {
    // Singleton postings consume 32 aligned frozen bytes each: just over 16 MiB.
    constexpr size_t count = (16 * 1024 * 1024 / 32) + 1;
    const auto& backend = ScalarReaderBackends().Get<int64_t>("BitmapInt64");
    std::vector<int64_t> values(count);
    std::iota(values.begin(), values.end(), int64_t{0});
    ScalarTestData<int64_t> data(std::move(values));
    const ScalarTestInput<int64_t> rows(data);
    auto artifact = backend.Build(rows.View(), {.row_count = count});
    auto input = std::make_shared<storage::RemoteInputStream>(
        std::make_shared<arrow::io::BufferReader>(
            MakePackedArtifactBuffer(SerializeV3(*artifact))));
    auto source = storage::IndexEntryReader::Open(input, input->Size());
    auto staging = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "bitmap-batches-XXXXXX",
        "bitmap batch scheduling test");
    auto options = ScalarOptions(DataType::INT64);
    options.enable_mmap = true;
    options.mmap_dir_path = staging->Path();
    for (bool use_async : {false, true}) {
        SCOPED_TRACE(use_async);
        {
            auto plan = BitmapIndexLoader::PlanPacked(
                source->Directory(), source->IndexMeta(), options);
            for (const auto& entry : plan.entries) {
                const auto bytes = source->ReadEntry(entry.name).data;
                if (const auto* memory =
                        std::get_if<storage::MemoryEntryTarget>(
                            &entry.target)) {
                    std::memcpy(memory->data, bytes.data(), bytes.size());
                } else {
                    const auto& file =
                        std::get<storage::FileEntryTarget>(entry.target);
                    file.staging->Prepare(storage::io::Priority::HIGH);
                    file.staging->WriteAt(
                        file.offset, bytes.data(), bytes.size());
                    file.staging->Finish();
                }
            }
            auto task =
                BitmapIndexLoader::FinishPacked(plan, options, use_async, {});
            IIndexReaderBasePtr reader;
            if (use_async) {
                folly::ManualExecutor executor;
                auto future =
                    folly::coro::co_withExecutor(&executor, std::move(task))
                        .start();
                bool observed_batch = false;
                // File phases suspend onto another pool. Drive each resumed
                // conversion batch and observe progress before completion.
                while (!future.isReady()) {
                    executor.drive();
                    if (!future.isReady()) {
                        for (const auto& file :
                             std::filesystem::directory_iterator(
                                 staging->Path())) {
                            if (file.path().filename().string().find(
                                    "bitmap_frozen") != std::string::npos &&
                                file.file_size() >= 16 * 1024 * 1024) {
                                observed_batch = true;
                            }
                        }
                    }
                }
                EXPECT_TRUE(observed_batch);
                reader = std::move(future).get();
            } else {
                // No executor: sync conversion must not acquire a scheduling dependency.
                reader = folly::coro::blockingWait(std::move(task));
            }
            ASSERT_NE(reader, nullptr);
            EXPECT_EQ(reader->Count(), count);
            const auto* predicate =
                dynamic_cast<const IScalarPredicateReader<int64_t>*>(
                    reader.get());
            ASSERT_NE(predicate, nullptr);
            for (int64_t key : {int64_t{0}, int64_t{count - 1}}) {
                ExpectHits(
                    predicate->In(1, &key), count, {static_cast<size_t>(key)});
            }
        }
        EXPECT_TRUE(std::filesystem::is_empty(staging->Path()));
    }
}

TEST(ScalarIndexV3LoadTest, FileTargetsDoNotMoveFamilyFinalizationToFilePool) {
    const auto& backend = ScalarReaderBackends().Get<int64_t>("BitmapInt64");
    ScalarTestData<int64_t> data({10, 20, 10});
    const ScalarTestInput<int64_t> rows(data);
    auto artifact = backend.Build(rows.View(), {.row_count = 3});
    auto input = std::make_shared<storage::RemoteInputStream>(
        std::make_shared<arrow::io::BufferReader>(
            MakePackedArtifactBuffer(SerializeV3(*artifact))));
    auto staging = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "packed-executor-XXXXXX",
        "packed executor test");
    auto options = ScalarOptions(DataType::INT64);
    options.enable_mmap = true;
    options.mmap_dir_path = staging->Path();
    options.params["test_loading_thread"] =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto load = [&]() -> folly::coro::Task<IIndexReaderBasePtr> {
        auto opened = co_await storage::AsyncIndexEntryReader::Open(
            input, 0, proto::common::LoadPriority::HIGH);
        LoaderEntry::FinishPackedAsyncFn finish =
            [](IndexLoadPlan& plan,
               const storage::LoadOptions& opts,
               bool use_async,
               folly::CancellationToken token)
            -> folly::coro::Task<IIndexReaderBasePtr> {
            EXPECT_EQ(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                      opts.params.at("test_loading_thread").get<size_t>());
            auto reader = co_await BitmapIndexLoader::FinishPacked(
                plan, opts, use_async, token);
            EXPECT_EQ(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                      opts.params.at("test_loading_thread").get<size_t>());
            co_return reader;
        };
        auto loader = LoaderRegistry::Instance().Lookup(
            std::string(BitmapIndexLoader::kFamily));
        loader.finish_packed = finish;
        co_return co_await LoadPackedIndexAsync(
            loader, *opened, options, proto::common::LoadPriority::HIGH);
    };
    folly::ManualExecutor executor;
    auto future = folly::coro::co_withExecutor(&executor, load()).start();
    while (!future.isReady()) executor.drive();
    auto reader = std::move(future).get();
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->Count(), 3);
    reader.reset();
    EXPECT_TRUE(std::filesystem::is_empty(staging->Path()));
}

}  // namespace milvus::index::test
