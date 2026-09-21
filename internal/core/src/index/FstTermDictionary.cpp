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
#include <array>
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
// Small LIKE NFA: one bit per token plus the accepting state. Programs
// are immutable and live on the querying stack; branch states only hold a
// pointer and small integers. Longer patterns use the caller's RE2 fallback.
struct LikeProgram {
    static constexpr uint32_t kMany = 0x110000, kOne = 0x110001;
    std::array<uint32_t, 63> tokens{};
    size_t size = 0;
    uint64_t many = 0, one = 0;
};
struct Utf8State {
    uint32_t code = 0, minimum = 0;
    uint8_t remaining = 0;
    bool valid = true;
    bool
    Feed(uint8_t b) {
        if (!valid)
            return false;
        if (remaining) {
            if ((b & 0xc0) != 0x80) {
                valid = false;
                return false;
            }
            code = (code << 6) | (b & 63);
            if (--remaining)
                return false;
            valid = code >= minimum && code <= 0x10ffff &&
                    !(code >= 0xd800 && code <= 0xdfff);
            return valid;
        }
        if (b < 128) {
            code = b;
            return true;
        }
        if (b >= 0xc2 && b <= 0xdf) {
            code = b & 31;
            minimum = 0x80;
            remaining = 1;
        } else if (b >= 0xe0 && b <= 0xef) {
            code = b & 15;
            minimum = 0x800;
            remaining = 2;
        } else if (b >= 0xf0 && b <= 0xf4) {
            code = b & 7;
            minimum = 0x10000;
            remaining = 3;
        } else
            valid = false;
        return false;
    }
};
bool
CompileLike(std::string_view pattern, LikeProgram& p) {
    bool escaped = false;
    Utf8State utf8;
    for (uint8_t b : pattern) {
        if (!utf8.Feed(b)) {
            if (!utf8.valid)
                return false;
            continue;
        }
        uint32_t token = utf8.code;
        if (!escaped && token == '\\') {
            escaped = true;
            continue;
        }
        if (!escaped && token == '%')
            token = LikeProgram::kMany;
        else if (!escaped && token == '_')
            token = LikeProgram::kOne;
        escaped = false;
        if (token == LikeProgram::kMany && p.size &&
            p.tokens[p.size - 1] == token)
            continue;
        if (p.size == p.tokens.size())
            return false;
        if (token == LikeProgram::kMany)
            p.many |= uint64_t(1) << p.size;
        if (token == LikeProgram::kOne)
            p.one |= uint64_t(1) << p.size;
        p.tokens[p.size++] = token;
    }
    return !escaped && utf8.valid && !utf8.remaining;
}
struct LikeState {
    const LikeProgram* program;
    uint64_t active = 1;
    Utf8State utf8;
    explicit LikeState(const LikeProgram& p) : program(&p) {
        Close();
    }
    void
    Close() {
        active |= (active & program->many) << 1;
    }
    void
    step(char c) {
        if (!utf8.Feed(uint8_t(c))) {
            if (!utf8.valid)
                active = 0;
            return;
        }
        uint64_t next = active & program->many;
        next |= (active & program->one) << 1;
        uint64_t literals = active & ~(program->many | program->one);
        // The accepting bit has no outgoing transition.
        literals &= (uint64_t(1) << program->size) - 1;
        while (literals) {
            unsigned bit = std::countr_zero(literals);
            if (program->tokens[bit] == utf8.code)
                next |= uint64_t(1) << (bit + 1);
            literals &= literals - 1;
        }
        active = next;
        Close();
    }
    bool
    is_match() const {
        return !utf8.remaining && (active & (uint64_t(1) << program->size));
    }
    bool
    can_match() const {
        return active != 0;
    }
};
// Prefix DP intersected with FST arcs. Any cell outside |depth-column|<=k
// costs more than k even with transpositions, which preserve string length.
// Store only that band (at most five cells for k<=2), saturated at k+1.
// Two rows preserve OSA transpositions; UTF-8 advances DP only per scalar.
struct FuzzyRow {
    size_t begin = 0;
    std::array<uint8_t, 5> values{};
    uint8_t size = 0;
    uint8_t Get(size_t column, uint8_t cap) const {
        return column >= begin && column - begin < size
                   ? values[column - begin]
                   : cap;
    }
};
struct FuzzyState {
    const std::vector<uint32_t>* query;
    FuzzyRow row, previous;
    Utf8State utf8;
    uint32_t last = 0;
    size_t depth = 0;
    uint8_t limit, minimum = 0, previous_minimum = 0;
    FuzzyState(const std::vector<uint32_t>& q, uint32_t k)
        : query(&q), limit(k) {
        row.size = std::min<size_t>(q.size(), k) + 1;
        for (size_t j = 0; j < row.size; ++j) row.values[j] = j;
        previous = row;
    }
    void step(char byte) {
        if (!utf8.Feed(uint8_t(byte))) return;
        const auto cp = utf8.code;
        const auto cap = uint8_t(limit + 1);
        ++depth;
        FuzzyRow next;
        next.begin = depth > limit ? depth - limit : 0;
        const size_t end = std::min(query->size(), depth + limit);
        auto next_minimum = cap;
        if (next.begin <= end) {
            next.size = end - next.begin + 1;
            for (size_t j = next.begin; j <= end; ++j) {
                unsigned value;
                if (j == 0) {
                    value = std::min<size_t>(depth, cap);
                } else {
                    value = std::min({unsigned(row.Get(j, cap)) + 1,
                                      unsigned(next.Get(j - 1, cap)) + 1,
                                      unsigned(row.Get(j - 1, cap)) +
                                          (cp != (*query)[j - 1])});
                    if (depth > 1 && j > 1 && cp == (*query)[j - 2] &&
                        last == (*query)[j - 1])
                        value = std::min(value,
                                         unsigned(previous.Get(j - 2, cap)) + 1);
                }
                next.values[j - next.begin] = std::min<unsigned>(value, cap);
                next_minimum = std::min(next_minimum, next.values[j - next.begin]);
            }
        }
        previous = row;
        row = next;
        last = cp;
        previous_minimum = minimum;
        minimum = next_minimum;
    }
    bool is_match() const {
        return utf8.valid && !utf8.remaining &&
               row.Get(query->size(), limit + 1) <= limit;
    }
    bool can_match() const {
        // Retain the previous row conservatively: a future transposition reads
        // it, so pruning only by the current row would need a separate proof.
        return utf8.valid && depth <= query->size() + limit &&
               (minimum <= limit || previous_minimum < limit);
    }
};
/*
The MIT License (MIT)

Copyright (c) 2015 yhirose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Bytecode arc iteration adapted from cpp-fstlib (MIT),
// Copyright (c) 2022 Yuji Hirose. The same opcode/output semantics are used,
// but traversal keeps pending siblings on the heap and one mutable word buffer.
// No call-stack frame or copied partial string is retained per token byte.
class PatternFst : public fst::map<uint32_t> {
 public:
    using fst::map<uint32_t>::map;
    template <typename Automaton>
    void Visit(Automaton initial, const FstTermDictionary::Visitor& visitor,
               std::string_view prefix = {}) const {
        struct Frame {
            uint32_t address;
            size_t depth;
            uint32_t output;
            Automaton automaton;
            const char* labels = nullptr;
            size_t label_index = 0;
        };
        std::vector<Frame> stack;
        stack.reserve(64);
        stack.push_back({this->header_.start_address, 0, 0, std::move(initial)});
        std::string word;
        while (!stack.empty()) {
            auto& frame = stack.back();
            const char* end = this->byte_code_ + frame.address;
            auto p = end;
            const fst::FstOpe op(*p--);
            if (op.has_jump_table()) {
                size_t count = 0;
                p -= fst::vb_decode_value_reverse(p, count);
                p -= count * op.jump_table_element_size();
                if (this->header_.flags.data.jump_table_labels) {
                    frame.labels = p + 1 - count;
                    frame.label_index = 0;
                    p -= count;
                }
                frame.address -= std::distance(p, end);
                continue;
            }
            const char arc = frame.labels ? frame.labels[frame.label_index++]
                                          : this->read_arc(op, p);
            uint32_t delta, hub;
            bool has_hub;
            this->read_delta(op, p, delta, hub, has_hub);
            uint32_t suffix = 0, state_output = 0;
            if (op.data.has_output)
                p -= fst::OutputTraits<uint32_t>::read_byte_value(p, suffix);
            if (this->header_.need_state_output && op.data.has_state_output)
                p -= fst::OutputTraits<uint32_t>::read_byte_value(p, state_output);
            const auto size = std::distance(p, end);
            uint32_t child = 0;
            if (op.data.no_address) child = frame.address - size;
            else if (has_hub) child = hub;
            else if (delta) child = frame.address - size - delta + 1;
            const auto depth = frame.depth;
            const auto output = frame.output + suffix;
            auto automaton = frame.automaton;
            // Advance/pop the parent before pushing a child: vector growth must
            // never invalidate a still-live reference used by the traversal.
            if (op.data.last_transition) stack.pop_back();
            else frame.address -= size;
            if (depth < prefix.size() && prefix[depth] != arc) continue;
            automaton.step(arc);
            word.resize(depth);
            word.push_back(arc);
            if (op.data.final && word.size() >= prefix.size() && automaton.is_match())
                visitor(word, output + state_output);
            if (child && automaton.can_match())
                stack.push_back({child, word.size(), output, std::move(automaton)});
        }
    }
    void Prefix(std::string_view prefix,
                const FstTermDictionary::Visitor& visitor) const {
        Visit(fst::DummyAutomaton(), visitor, prefix);
    }
    void Fuzzy(const std::vector<uint32_t>& query, uint32_t edits,
               const FstTermDictionary::Visitor& visitor) const {
        Visit(FuzzyState(query, edits), visitor);
    }
    void Like(const LikeProgram& program,
              const FstTermDictionary::Visitor& visitor) const {
        Visit(LikeState(program), visitor);
    }
};
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
    std::unique_ptr<PatternFst> reader;
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
            reader =
                std::make_unique<PatternFst>(payload.data(), payload.size());
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
           (state_->reader ? sizeof(PatternFst) : 0) +
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
        state_->reader->Prefix("", [&](std::string_view term, uint32_t id) {
            result.emplace_back(term, id);
        });
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return result;
}

void
FstTermDictionary::ForEachPrefix(std::string_view prefix,
                                 const Visitor& visitor) const {
    if (!state_)
        return;
    if (prefix.empty() && state_->empty_key)
        visitor("", 0);
    if (state_->reader)
        state_->reader->Prefix(prefix, visitor);
}
bool
FstTermDictionary::ForEachLike(std::string_view pattern,
                               const Visitor& visitor) const {
    LikeProgram program;
    if (!CompileLike(pattern, program))
        return false;
    if (!state_)
        return true;
    if (state_->empty_key && LikeState(program).is_match())
        visitor("", 0);
    if (state_->reader)
        state_->reader->Like(program, visitor);
    return true;
}
void
FstTermDictionary::ForEachFuzzy(std::string_view term,
                               uint32_t max_edits,
                               const Visitor& visitor) const {
    if (max_edits > 2)
        ThrowInfo(ErrorCode::InvalidParameter,
                  "max_edit_distance must be within [0, 2]");
    std::vector<uint32_t> query;
    Utf8State utf8;
    for (uint8_t byte : term) {
        if (utf8.Feed(byte)) query.push_back(utf8.code);
        if (!utf8.valid)
            ThrowInfo(ErrorCode::InvalidParameter, "invalid UTF-8 fuzzy term");
    }
    if (utf8.remaining)
        ThrowInfo(ErrorCode::InvalidParameter, "incomplete UTF-8 fuzzy term");
    if (!state_) return;
    if (state_->empty_key && query.size() <= max_edits) visitor("", 0);
    if (state_->reader) state_->reader->Fuzzy(query, max_edits, visitor);
}
}  // namespace milvus::index
