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

#include "index/KnowherePositionIndex.h"
#include "common/EasyAssert.h"
#include "index/KnowherePoCIO.h"
#include "index/KnowhereSparsePostingCodec.h"
#include "index/sparse/codec/adaptive.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace milvus::index {
thread_local KnowherePositionIndex::DecodeStats
    KnowherePositionIndex::decode_stats_for_ut{};

void
KnowherePositionIndex::AppendTerm(const std::vector<uint32_t>& frequencies,
                                  const std::vector<uint32_t>& deltas) {
    AssertInfo(!sealed_, "position builder already sealed");
    AssertInfo(frequencies.size() <= INT32_MAX, "invalid position doc count");
    uint64_t total = 0;
    for (auto freq : frequencies) {
        AssertInfo(freq != 0, "posting frequency must be positive");
        total += freq;  // <= INT32_MAX * UINT32_MAX, fits uint64
    }
    AssertInfo(total == deltas.size(), "TF and positions length mismatch");
    terms_.push_back({doc_blocks_.size(),
                      position_offsets_.size(),
                      total,
                      static_cast<uint32_t>(frequencies.size())});
    knowhere::sparse::inverted::AdaptiveBlockCodec codec;
    uint64_t base = 0;
    for (size_t i = 0; i < frequencies.size(); i += 256) {
        const auto n = std::min<size_t>(256, frequencies.size() - i);
        doc_blocks_.push_back({frequencies_.size(), base});
        codec.encode(frequencies.data() + i, n, frequencies_);
        for (size_t j = 0; j < n; ++j) base += frequencies[i + j];
    }
    for (size_t i = 0; i < deltas.size(); i += 256) {
        position_offsets_.push_back(positions_.size());
        codec.encode(deltas.data() + i,
                     std::min<size_t>(256, deltas.size() - i),
                     positions_);
    }
}
void
KnowherePositionIndex::Seal() {
    AssertInfo(!sealed_, "position builder already sealed");
    frequencies_.resize(
        frequencies_.size() + KnowhereSparsePostingCodec::kPadding, 0);
    positions_.resize(positions_.size() + KnowhereSparsePostingCodec::kPadding,
                      0);
    terms_.shrink_to_fit();
    doc_blocks_.shrink_to_fit();
    position_offsets_.shrink_to_fit();
    frequencies_.shrink_to_fit();
    positions_.shrink_to_fit();
    sealed_ = true;
}
size_t
KnowherePositionIndex::ByteSize() const {
    return sizeof(*this) + terms_.capacity() * sizeof(Term) +
           doc_blocks_.capacity() * sizeof(DocBlock) +
           position_offsets_.capacity() * sizeof(uint64_t) +
           frequencies_.capacity() + positions_.capacity();
}
size_t
KnowherePositionIndex::LogicalBytes() const {
    return terms_.size() * sizeof(Term) +
           doc_blocks_.size() * sizeof(DocBlock) +
           position_offsets_.size() * sizeof(uint64_t) + frequencies_.size() +
           positions_.size();
}
void
KnowherePositionIndex::Reader::Read(size_t ordinal,
                                    uint64_t offset,
                                    std::vector<uint64_t>& output) {
    AssertInfo(owner_.sealed_ && ordinal < term_.docs,
               "invalid position reader ordinal");
    knowhere::sparse::inverted::AdaptiveBlockCodec codec;
    const size_t block = ordinal / 256, in_block = ordinal % 256;
    const auto& meta = owner_.doc_blocks_[term_.doc_block_begin + block];
    if (freq_block_ != block) {
        std::array<uint32_t, 256> freqs;
        const size_t n = std::min<size_t>(256, term_.docs - block * 256);
        codec.decode(
            owner_.frequencies_.data() + meta.freq_offset, freqs.data(), n);
        ++decode_stats_for_ut.freq_blocks;
        prefix_[0] = meta.position_base;
        for (size_t i = 0; i < n; ++i) prefix_[i + 1] = prefix_[i] + freqs[i];
        freq_block_ = block;
    }
    const uint64_t start = prefix_[in_block], end = prefix_[in_block + 1];
    output.clear();
    output.reserve(end - start);
    uint64_t position = offset;
    for (uint64_t i = start; i < end;) {
        const size_t pb = i / 256;
        if (pos_block_ != pb) {
            codec.decode(
                owner_.positions_.data() +
                    owner_.position_offsets_[term_.pos_block_begin + pb],
                deltas_.data(),
                std::min<uint64_t>(256, term_.positions - pb * 256));
            pos_block_ = pb;
            ++decode_stats_for_ut.position_blocks;
        }
        const uint64_t stop = std::min(end, (uint64_t(pb) + 1) * 256);
        for (; i < stop; ++i) {
            position += deltas_[i % 256];
            output.push_back(position);
        }
    }
}

namespace {
constexpr std::string_view kPositionMagic = "KHPOS001";
constexpr uint32_t kPositionType = 1;
constexpr uint32_t kPositionCodec = 1;  // Adaptive integer blocks.
void
PositionRequire(bool ok, const char* detail) {
    if (!ok)
        ThrowInfo(
            ErrorCode::DataFormatBroken, "invalid PoC positions: {}", detail);
}
size_t
PositionBlocks(uint64_t values) {
    return values / 256 + (values % 256 != 0);
}
}  // namespace

std::vector<uint8_t>
KnowherePositionIndex::SerializeForPoC() const {
    AssertInfo(sealed_, "cannot serialize unsealed positions");
    poc_io::Writer out;
    out.U64(terms_.size());
    out.U64(doc_blocks_.size());
    out.U64(position_offsets_.size());
    for (const auto& term : terms_) {
        out.U64(term.doc_block_begin);
        out.U64(term.pos_block_begin);
        out.U64(term.positions);
        out.U32(term.docs);
    }
    for (const auto& block : doc_blocks_) {
        out.U64(block.freq_offset);
        out.U64(block.position_base);
    }
    for (auto offset : position_offsets_) out.U64(offset);
    // SIMD over-read padding is an allocation detail, not persisted payload.
    out.Bytes(
        std::span(frequencies_)
            .first(frequencies_.size() - KnowhereSparsePostingCodec::kPadding));
    out.Bytes(
        std::span(positions_)
            .first(positions_.size() - KnowhereSparsePostingCodec::kPadding));
    return poc_io::Pack(
        kPositionMagic, kPositionType, kPositionCodec, out.data);
}

void
KnowherePositionIndex::LoadForPoC(std::span<const uint8_t> bytes,
                                  const std::vector<uint32_t>& expected_dfs) {
    uint32_t codec;
    const auto payload =
        poc_io::Unpack(bytes, kPositionMagic, kPositionType, &codec);
    PositionRequire(codec == kPositionCodec, "unsupported codec");
    poc_io::Reader input(payload);
    const auto terms = input.U64(), blocks = input.U64(), offsets = input.U64();
    PositionRequire(terms == expected_dfs.size(),
                    "term count differs from postings");
    // Validate each count against the bytes remaining before allocating.
    PositionRequire(terms <= input.Remaining() / 28, "term metadata truncated");
    KnowherePositionIndex next;
    next.terms_.reserve(terms);
    uint64_t expected_blocks = 0, expected_offsets = 0;
    for (size_t i = 0; i < terms; ++i) {
        Term term{input.U64(), input.U64(), input.U64(), input.U32()};
        PositionRequire(term.docs == expected_dfs[i] && term.docs <= INT32_MAX,
                        "document frequency differs from postings");
        PositionRequire(term.positions <= uint64_t(term.docs) * UINT32_MAX &&
                            term.positions >= term.docs,
                        "invalid total term frequency");
        PositionRequire(term.doc_block_begin == expected_blocks &&
                            term.pos_block_begin == expected_offsets,
                        "non-contiguous term block ranges");
        const auto db = PositionBlocks(term.docs),
                   pb = PositionBlocks(term.positions);
        PositionRequire(
            expected_blocks <= blocks && db <= blocks - expected_blocks &&
                expected_offsets <= offsets && pb <= offsets - expected_offsets,
            "term block range exceeds metadata");
        expected_blocks += db;
        expected_offsets += pb;
        next.terms_.push_back(term);
    }
    PositionRequire(expected_blocks == blocks && expected_offsets == offsets,
                    "unowned blocks");
    PositionRequire(blocks <= input.Remaining() / 16, "doc metadata truncated");
    next.doc_blocks_.reserve(blocks);
    for (size_t i = 0; i < blocks; ++i)
        next.doc_blocks_.push_back({input.U64(), input.U64()});
    PositionRequire(offsets <= input.Remaining() / 8,
                    "position offsets truncated");
    next.position_offsets_.reserve(offsets);
    for (size_t i = 0; i < offsets; ++i)
        next.position_offsets_.push_back(input.U64());
    const auto freq_bytes = input.Bytes(), pos_bytes = input.Bytes();
    input.Finish();
    auto validate_offsets = [&](size_t count, auto offset_at, size_t size) {
        PositionRequire((count == 0) == (size == 0),
                        "empty block/stream mismatch");
        if (!count)
            return;
        PositionRequire(offset_at(0) == 0, "unowned stream prefix");
        for (size_t i = 0; i < count; ++i) {
            const auto current = offset_at(i);
            const uint64_t end = i + 1 < count ? offset_at(i + 1) : size;
            PositionRequire(current < end && end <= size,
                            "invalid block byte interval");
        }
    };
    validate_offsets(
        blocks,
        [&](size_t i) { return next.doc_blocks_[i].freq_offset; },
        freq_bytes.size());
    validate_offsets(
        offsets,
        [&](size_t i) { return next.position_offsets_[i]; },
        pos_bytes.size());
    auto checked_freq = [&](size_t index, size_t n) {
        const auto begin = next.doc_blocks_[index].freq_offset;
        const auto end = index + 1 < blocks
                             ? next.doc_blocks_[index + 1].freq_offset
                             : freq_bytes.size();
        return KnowhereSparsePostingCodec::DecodeChecked(
            freq_bytes.subspan(begin, end - begin),
            n,
            KnowhereSparsePostingCodec::Format::Adaptive);
    };
    auto checked_pos = [&](size_t index, size_t n) {
        const auto begin = next.position_offsets_[index];
        const auto end = index + 1 < offsets ? next.position_offsets_[index + 1]
                                             : pos_bytes.size();
        return KnowhereSparsePostingCodec::DecodeChecked(
            pos_bytes.subspan(begin, end - begin),
            n,
            KnowhereSparsePostingCodec::Format::Adaptive);
    };
    // Validate with bounded block decoders before any trusted Reader is exposed.
    // Only two 256-value buffers are needed, even for a huge/high-TF term.
    for (const auto& term : next.terms_) {
        uint64_t consumed = 0;
        size_t decoded_position_block = SIZE_MAX;
        std::array<uint32_t, 256> deltas{};
        for (size_t b = 0; b < PositionBlocks(term.docs); ++b) {
            const auto global = term.doc_block_begin + b;
            PositionRequire(
                next.doc_blocks_[global].position_base == consumed,
                "doc block position base differs from frequency prefix");
            const size_t count = std::min<size_t>(256, term.docs - b * 256);
            const auto frequencies = checked_freq(global, count);
            for (size_t d = 0; d < count; ++d) {
                const auto frequency = frequencies[d];
                PositionRequire(
                    frequency > 0 && frequency <= term.positions - consumed,
                    "frequency sum differs from position count");
                uint64_t position = 0;
                for (uint32_t j = 0; j < frequency; ++j, ++consumed) {
                    const auto pb = consumed / 256;
                    if (decoded_position_block != pb) {
                        deltas = checked_pos(
                            term.pos_block_begin + pb,
                            std::min<uint64_t>(256, term.positions - pb * 256));
                        decoded_position_block = pb;
                    }
                    position += deltas[consumed % 256];
                    PositionRequire(position <= UINT32_MAX,
                                    "document token position overflows uint32");
                }
            }
        }
        PositionRequire(consumed == term.positions, "unused term positions");
    }
    next.frequencies_.reserve(freq_bytes.size() + KnowhereSparsePostingCodec::kPadding);
    next.positions_.reserve(pos_bytes.size() + KnowhereSparsePostingCodec::kPadding);
    next.frequencies_.assign(freq_bytes.begin(), freq_bytes.end());
    next.positions_.assign(pos_bytes.begin(), pos_bytes.end());
    next.Seal();
    // Publication is atomic with respect to load failure: old index untouched.
    *this = std::move(next);
}

// The span matcher below is adapted from Tantivy (MIT):
// Copyright (c) 2018 by the project authors, as listed in the AUTHORS file.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

bool
KnowherePhraseExists(const std::vector<std::vector<uint64_t>>& positions,
                     uint32_t slop) {
    KnowherePhraseScratch scratch;
    return KnowherePhraseExists(positions, slop, scratch);
}

bool
KnowherePhraseExists(const std::vector<std::vector<uint64_t>>& positions,
                     uint32_t slop,
                     KnowherePhraseScratch& scratch) {
    if (positions.empty())
        return false;
    for (const auto& p : positions)
        if (p.empty())
            return false;
    if (positions.size() == 1)
        return true;
    // Two-clause phrases need no span allocation, including repeated terms.
    if (positions.size() == 2) {
        const auto& left = positions[0];
        const auto& right = positions[1];
        size_t a = 0, b = 0;
        while (a < left.size() && b < right.size()) {
            if (left[a] <= right[b]) {
                if (right[b] - left[a] <= slop)
                    return true;
                ++a;
            } else {
                if (left[a] - right[b] <= slop)
                    return true;
                ++b;
            }
        }
        return false;
    }
    if (!slop) {
        // No per-candidate allocation. Start with the shortest positional list;
        // all lists already carry their own query-offset alignment.
        auto lead = std::min_element(
            positions.begin(),
            positions.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });
        for (auto p : *lead) {
            bool found = true;
            for (const auto& list : positions) {
                if (!std::binary_search(list.begin(), list.end(), p)) {
                    found = false;
                    break;
                }
            }
            if (found)
                return true;
        }
        return false;
    }
    // Compatibility translation of the no-score span path in Zilliz Tantivy
    // 96f3335/src/query/phrase_query/phrase_scorer.rs (MIT license).
    // Preserve iteration/pruning/ties; this is NOT Lucene's slop algorithm.
    // Only the multi-clause slop path uses spans. Reset here so exact and
    // two-clause candidates do not pay for touching unused scratch buffers.
    auto& spans = scratch.spans;
    auto& next = scratch.next;
    auto& best = scratch.best;
    spans.clear();
    next.clear();
    best.clear();
    using Span = KnowherePhraseScratch::Span;
    for (auto p : positions.front()) spans.push_back({p, p});
    for (size_t t = 1; t + 1 < positions.size(); ++t) {
        const auto& values = positions[t];
        next.clear();
        size_t start = 0;
        for (auto span : spans) {
            best.clear();
            uint64_t distance = UINT64_MAX;
            bool no_expansion = false;
            for (size_t i = start; i < values.size(); ++i) {
                const auto p = values[i];
                if (p >= span.left && p <= span.right) {
                    if (p == span.left)
                        start = i;
                    distance = span.right - span.left;
                    if (!no_expansion) {
                        best.clear();
                        best.push_back(span);
                        no_expansion = true;
                    }
                    continue;
                }
                uint64_t d;
                Span candidate;
                if (p < span.left) {
                    start = i;
                    d = span.right - p;
                    candidate = {p, span.right};
                } else {
                    d = p - span.left;
                    candidate = {span.left, p};
                    if (d > slop)
                        break;
                }
                if (d <= slop) {
                    if (d < distance) {
                        distance = d;
                        best.clear();
                    }
                    if (d == distance)
                        best.push_back(candidate);
                }
            }
            next.insert(next.end(), best.begin(), best.end());
        }
        spans.swap(next);
        if (spans.empty())
            return false;
    }
    const auto& last = positions.back();
    size_t a = 0, b = 0;
    while (a < spans.size() && b < last.size()) {
        const auto span = spans[a];
        const auto p = last[b];
        if (p < span.left) {
            if (span.right - p <= slop)
                return true;
            ++b;
        } else if (p > span.right) {
            if (p - span.left <= slop)
                return true;
            ++a;
        } else
            return true;
    }
    return false;
}
}  // namespace milvus::index
