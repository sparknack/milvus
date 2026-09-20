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
#include <algorithm>
#include <limits>
#include <numeric>
#include "common/EasyAssert.h"
#include "index/KnowhereSparsePostingCodec.h"
#include "index/sparse/codec/adaptive.h"

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
