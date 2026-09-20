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

#include "index/FstTermDictionary.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstlib.h>

#include "common/EasyAssert.h"

namespace milvus::index {
namespace {
constexpr std::string_view kMagic = "MVFST001";
constexpr size_t kHeader = 24, kChecksum = 8;
uint64_t
Read64(const char* p) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v |= uint64_t(uint8_t(p[i])) << (8 * i);
    return v;
}
void
Append64(std::string& s, uint64_t v) {
    for (size_t i = 0; i < 8; ++i) s.push_back(char(v >> (8 * i)));
}
struct Region {
    std::string owned;
    void* mapping = MAP_FAILED;
    size_t length = 0;
    ~Region() {
        if (mapping != MAP_FAILED)
            munmap(mapping, length);
    }
    std::string_view
    Bytes() const {
        return mapping == MAP_FAILED
                   ? std::string_view(owned)
                   : std::string_view(static_cast<const char*>(mapping),
                                      length);
    }
};
}  // namespace

struct FstTermDictionary::State {
    std::shared_ptr<Region> region;
    size_t count;
    bool empty_key;
    std::unique_ptr<fst::map<uint32_t>> reader;
    explicit State(std::shared_ptr<Region> r) : region(std::move(r)) {
        static_assert(std::endian::native == std::endian::little,
                      "Pinned cpp-fstlib bytecode is little endian");
        auto bytes = region->Bytes();
        if (bytes.size() < kHeader + kChecksum ||
            bytes.substr(0, 8) != kMagic ||
            Read64(bytes.data() + bytes.size() - kChecksum) !=
                fst::xxh64(bytes.data(), bytes.size() - kChecksum))
            ThrowInfo(ErrorCode::DataFormatBroken,
                      "invalid FST dictionary header or checksum");
        count = Read64(bytes.data() + 8);
        auto empty = Read64(bytes.data() + 16);
        if (count > UINT32_MAX || empty > 1 || empty > count)
            ThrowInfo(ErrorCode::DataFormatBroken,
                      "invalid FST dictionary count");
        empty_key = empty;
        auto payload =
            bytes.substr(kHeader, bytes.size() - kHeader - kChecksum);
        if (count == empty) {
            if (!payload.empty())
                ThrowInfo(ErrorCode::DataFormatBroken,
                          "unexpected empty FST payload");
        } else {
            // verify() checks bytecode header and body checksum, not a general
            // adversarial graph validator. Load our own immutable build artifacts.
            if (!fst::verify(payload.data(), payload.size()) ||
                fst::get_output_type(payload.data(), payload.size()) !=
                    fst::OutputType::uint32_t)
                ThrowInfo(ErrorCode::DataFormatBroken, "invalid FST bytecode");
            reader = std::make_unique<fst::map<uint32_t>>(payload.data(),
                                                          payload.size());
            if (!*reader)
                ThrowInfo(ErrorCode::DataFormatBroken,
                          "cannot open FST bytecode");
        }
    }
};

FstTermDictionary
FstTermDictionary::Build(const std::vector<std::string>& terms) {
    AssertInfo(terms.size() <= UINT32_MAX, "FST term count exceeds uint32");
    AssertInfo(
        std::is_sorted(terms.begin(), terms.end()) &&
            std::adjacent_find(terms.begin(), terms.end()) == terms.end(),
        "FST terms must be sorted and unique");
    auto region = std::make_shared<Region>();
    region->owned.assign(kMagic);
    Append64(region->owned, terms.size());
    bool empty = !terms.empty() && terms.front().empty();
    Append64(region->owned, empty);
    if (terms.size() > size_t(empty)) {
        std::vector<std::pair<std::string, uint32_t>> input;
        input.reserve(terms.size() - empty);
        for (size_t i = empty; i < terms.size(); ++i)
            input.emplace_back(terms[i], i);
        std::ostringstream out(std::ios::binary);
        auto [result, index] = fst::compile<uint32_t>(input, out, true);
        AssertInfo(result == fst::Result::Success && out.good(),
                   "FST compilation failed at term {}",
                   index);
        auto payload = out.str();
        AssertInfo(payload.size() < UINT32_MAX,
                   "FST bytecode exceeds 32-bit address space");
        region->owned += payload;
    }
    Append64(region->owned,
             fst::xxh64(region->owned.data(), region->owned.size()));
    FstTermDictionary result;
    result.state_ = std::make_shared<State>(std::move(region));
    return result;
}

FstTermDictionary
FstTermDictionary::MapFile(const std::string& path) {
    struct File {
        int fd;
        ~File() {
            if (fd >= 0)
                close(fd);
        }
    } file{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (file.fd < 0)
        ThrowInfo(ErrorCode::FileReadFailed,
                  "open FST dictionary failed: errno {}",
                  errno);
    struct stat info {};
    if (fstat(file.fd, &info) != 0)
        ThrowInfo(ErrorCode::FileReadFailed,
                  "stat FST dictionary failed: errno {}",
                  errno);
    if (!S_ISREG(info.st_mode) || info.st_size < int64_t(kHeader + kChecksum))
        ThrowInfo(ErrorCode::DataFormatBroken,
                  "invalid FST dictionary file size/type");
    auto region = std::make_shared<Region>();
    region->length = info.st_size;
    region->mapping =
        mmap(nullptr, region->length, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (region->mapping == MAP_FAILED)
        ThrowInfo(ErrorCode::FileReadFailed,
                  "mmap FST dictionary failed: errno {}",
                  errno);
    FstTermDictionary result;
    result.state_ = std::make_shared<State>(std::move(region));
    return result;
}

std::optional<uint32_t>
FstTermDictionary::Lookup(std::string_view term) const {
    if (!state_)
        return std::nullopt;
    if (term.empty())
        return state_->empty_key ? std::optional<uint32_t>(0) : std::nullopt;
    uint32_t id;
    if (!state_->reader || !state_->reader->exact_match_search(term, id))
        return std::nullopt;
    if (id >= state_->count)
        ThrowInfo(ErrorCode::DataFormatBroken,
                  "FST term ordinal outside metadata");
    return id;
}
size_t
FstTermDictionary::Count() const {
    return state_ ? state_->count : 0;
}
bool
FstTermDictionary::IsMapped() const {
    return state_ && state_->region->mapping != MAP_FAILED;
}
std::string_view
FstTermDictionary::SerializedBytes() const {
    return state_ ? state_->region->Bytes() : std::string_view{};
}
size_t
FstTermDictionary::ByteSize() const {
    if (!state_)
        return 0;
    return sizeof(State) + sizeof(Region) +
           (state_->reader ? sizeof(fst::map<uint32_t>) : 0) +
           (IsMapped() ? state_->region->length
                       : state_->region->owned.capacity());
}
std::vector<std::pair<std::string, uint32_t>>
FstTermDictionary::Enumerate() const {
    std::vector<std::pair<std::string, uint32_t>> result;
    if (!state_)
        return result;
    if (state_->empty_key)
        result.emplace_back("", 0);
    if (state_->reader) {
        auto terms = state_->reader->predictive_search("");
        result.insert(result.end(),
                      std::make_move_iterator(terms.begin()),
                      std::make_move_iterator(terms.end()));
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return result;
}
}  // namespace milvus::index
