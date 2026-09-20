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

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "index/KnowhereSparsePostingCodec.h"

namespace milvus::index {
// Experimental heap-only scalar backend. Only the UT builder creates one.
// Kept separate from ScalarIndex: persistence and production build are absent.
template <typename T, typename TargetBitmap, typename OpType>
class InvertedIndexKnowhereCore {
 public:
    using PostingFormat = KnowhereSparsePostingCodec::Format;
    explicit InvertedIndexKnowhereCore(
        PostingFormat format = PostingFormat::StreamVByte)
        : format_(format) {
    }
    size_t
    PostingLogicalBytes() const {
        return posting_bytes_.size() >= KnowhereSparsePostingCodec::kPadding
                   ? posting_bytes_.size() -
                         KnowhereSparsePostingCodec::kPadding
                   : 0;
    }
    struct TermPostingMeta {
        uint64_t doc_stream_offset;
        uint64_t doc_stream_length;
        uint32_t doc_freq;
        uint32_t format_version =
            1;  // space for future parallel position streams
    };
    struct MemoryStats {
        size_t terms, metadata, postings, nulls;
        size_t
        Total() const {
            return terms + metadata + postings + nulls;
        }
    };

    void
    Build(size_t n, const T* values, const std::vector<size_t>& nulls = {}) {
        AssertInfo(n <= size_t(INT32_MAX),
                   "PoC row count exceeds signed docID limit");
        AssertInfo(n == 0 || values != nullptr, "missing scalar values");
        AssertInfo(
            std::is_sorted(nulls.begin(), nulls.end()) &&
                std::adjacent_find(nulls.begin(), nulls.end()) == nulls.end() &&
                (nulls.empty() || nulls.back() < n),
            "PoC null offsets must be unique, sorted and in range");
        // Build transactionally, releasing temporary pairs before publishing.
        InvertedIndexKnowhereCore next(format_);
        next.count_ = n;
        next.null_offsets_ = nulls;
        {
            std::vector<std::pair<T, uint32_t>> pairs;
            pairs.reserve(n - nulls.size());
            size_t null = 0;
            for (size_t i = 0; i < n; ++i) {
                if (null < nulls.size() && nulls[null] == i) {
                    ++null;
                    continue;
                }
                CheckValue(values[i]);
                pairs.emplace_back(values[i], i);
            }
            std::sort(pairs.begin(), pairs.end());
            std::vector<uint32_t> ids;
            for (size_t i = 0; i < pairs.size();) {
                next.terms_.push_back(pairs[i].first);
                ids.clear();
                size_t j = i;
                do {
                    ids.push_back(pairs[j++].second);
                } while (j < pairs.size() && pairs[j].first == pairs[i].first);
                uint64_t offset = next.posting_bytes_.size();
                KnowhereSparsePostingCodec::Append(
                    ids.data(), ids.size(), next.posting_bytes_, format_);
                next.posting_metas_.push_back(
                    {offset,
                     next.posting_bytes_.size() - offset,
                     static_cast<uint32_t>(ids.size()),
                     1});
                i = j;
            }
        }
        next.posting_bytes_.resize(
            next.posting_bytes_.size() + KnowhereSparsePostingCodec::kPadding,
            0);
        *this = std::move(next);
    }
    // Text builders provide a sorted dictionary of unique increasing docIDs per term.
    // Publishing is transactional, as for scalar Build(). No positions or weights.
    template <typename PostingMap>
    void
    BuildFromPostings(size_t n,
                      const PostingMap& postings,
                      const std::vector<size_t>& nulls = {}) {
        AssertInfo(n <= size_t(INT32_MAX),
                   "PoC row count exceeds signed docID limit");
        AssertInfo(
            std::is_sorted(nulls.begin(), nulls.end()) &&
                std::adjacent_find(nulls.begin(), nulls.end()) == nulls.end() &&
                (nulls.empty() || nulls.back() < n),
            "invalid text null offsets");
        InvertedIndexKnowhereCore next(format_);
        next.count_ = n;
        next.null_offsets_ = nulls;
        for (const auto& [term, ids] : postings) {
            CheckValue(term);
            AssertInfo(next.terms_.empty() || next.terms_.back() < term,
                       "text dictionary must be strictly sorted");
            AssertInfo(!ids.empty() && ids.back() < n,
                       "invalid text docID range");
            uint64_t offset = next.posting_bytes_.size();
            KnowhereSparsePostingCodec::Append(
                ids.data(), ids.size(), next.posting_bytes_, format_);
            next.terms_.push_back(term);
            next.posting_metas_.push_back({offset,
                                           next.posting_bytes_.size() - offset,
                                           static_cast<uint32_t>(ids.size()),
                                           1});
        }
        next.posting_bytes_.resize(
            next.posting_bytes_.size() + KnowhereSparsePostingCodec::kPadding,
            0);
        *this = std::move(next);
    }
    template <typename Callback>
    void
    ForEachDoc(size_t term, Callback&& callback) const {
        if (term == terms_.size())
            return;
        const auto& meta = posting_metas_.at(term);
        KnowhereSparsePostingCodec::View posting(
            posting_bytes_.data() + meta.doc_stream_offset, format_);
        std::array<uint32_t, KnowhereSparsePostingCodec::kBlockSize> ids;
        for (size_t b = 0; b < posting.Blocks(); ++b) {
            auto n = posting.DecodeBlock(b, ids.data());
            for (size_t i = 0; i < n; ++i) callback(ids[i]);
        }
    }
    // Add the intersection of existing term ordinals to a caller-owned bitmap.
    // Sort by posting length; cursor seeks skip compressed blocks using max IDs.
    void
    IntersectInto(std::vector<size_t> terms, TargetBitmap& result) const {
        AssertInfo(result.size() == count_, "PoC result bitmap size mismatch");
        if (terms.empty())
            return;
        std::sort(terms.begin(), terms.end());
        terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
        for (auto term : terms)
            AssertInfo(term < posting_metas_.size(),
                       "invalid intersection term ordinal");
        if (terms.size() == 1) {
            DecodeInto(terms.front(), result);
            return;
        }
        std::sort(terms.begin(), terms.end(), [&](size_t a, size_t b) {
            return std::make_pair(DocFreq(a), a) <
                   std::make_pair(DocFreq(b), b);
        });
        // Above roughly one candidate per bitmap word, repeated per-doc seeks
        // become expensive. Use sequential decode + word-wise AND for dense
        // conjunctions. This is a conservative heuristic, not a codec property.
        if (DocFreq(terms.front()) > std::max<size_t>(1, count_ / 64)) {
            TargetBitmap intersection(count_), scratch(count_);
            DecodeInto(terms.front(), intersection);
            for (size_t i = 1; i < terms.size(); ++i) {
                scratch.reset();
                DecodeInto(terms[i], scratch);
                intersection &= scratch;
                if (intersection.count() == 0)
                    return;
            }
            result |= intersection;
            return;
        }
        using Cursor = KnowhereSparsePostingCodec::Cursor;
        std::vector<Cursor> cursors;
        cursors.reserve(terms.size());
        for (auto term : terms)
            cursors.emplace_back(
                posting_bytes_.data() + posting_metas_[term].doc_stream_offset,
                format_);
        auto& lead = cursors.front();
        auto candidate = lead.Seek(0);
        while (candidate != Cursor::kEnd) {
            bool matched = true;
            for (size_t i = 1; i < cursors.size(); ++i) {
                auto doc = cursors[i].Seek(candidate);
                if (doc == Cursor::kEnd)
                    return;
                if (doc != candidate) {
                    candidate = lead.Seek(doc);
                    matched = false;
                    break;
                }
            }
            if (matched) {
                result.set(candidate);
                candidate = lead.Next();
            }
        }
    }
    size_t
    Count() const {
        return count_;
    }
    MemoryStats
    Memory() const {
        size_t terms = terms_.capacity() * sizeof(T);
        if constexpr (std::is_same_v<T, bool>)
            terms = (terms_.capacity() + 7) / 8;
        if constexpr (std::is_same_v<T, std::string>) {
            // Include only out-of-object storage (exclude short-string storage).
            for (const auto& term : terms_) {
                auto p = reinterpret_cast<uintptr_t>(term.data());
                auto object = reinterpret_cast<uintptr_t>(&term);
                if (p < object || p >= object + sizeof(term))
                    terms += term.capacity() + 1;
            }
        }
        return {terms,
                posting_metas_.capacity() * sizeof(TermPostingMeta),
                posting_bytes_.capacity(),
                null_offsets_.capacity() * sizeof(size_t)};
    }
    size_t
    ByteSize() const {
        return sizeof(*this) + Memory().Total();
    }

    // PoC microbenchmark hooks; not exposed through production ScalarIndex.
    size_t
    Lookup(const T& value) const {
        CheckValue(value);
        auto it = std::lower_bound(terms_.begin(), terms_.end(), value);
        return it != terms_.end() && *it == value ? it - terms_.begin()
                                                  : terms_.size();
    }
    size_t
    TermCount() const {
        return terms_.size();
    }
    T
    Term(size_t i) const {
        return terms_.at(i);
    }
    uint32_t
    DocFreq(size_t i) const {
        return posting_metas_.at(i).doc_freq;
    }
    std::pair<size_t, size_t>
    Bounds(const T& lower, bool li, const T& upper, bool ui) const {
        CheckValue(lower);
        CheckValue(upper);
        size_t begin =
            (li ? std::lower_bound(terms_.begin(), terms_.end(), lower)
                : std::upper_bound(terms_.begin(), terms_.end(), lower)) -
            terms_.begin();
        size_t end =
            (ui ? std::upper_bound(terms_.begin(), terms_.end(), upper)
                : std::lower_bound(terms_.begin(), terms_.end(), upper)) -
            terms_.begin();
        return {begin, std::max(begin, end)};
    }
    void
    DecodeInto(size_t term, TargetBitmap& result) const {
        AssertInfo(result.size() == count_, "PoC result bitmap size mismatch");
        if (term == terms_.size())
            return;
        const auto& meta = posting_metas_.at(term);
        KnowhereSparsePostingCodec::View posting(
            posting_bytes_.data() + meta.doc_stream_offset, format_);
        std::array<uint32_t, KnowhereSparsePostingCodec::kBlockSize> ids;
        for (size_t b = 0; b < posting.Blocks(); ++b) {
            size_t n = posting.DecodeBlock(b, ids.data());
            for (size_t i = 0; i < n; ++i) result.set(ids[i]);
        }
    }
    TargetBitmap
    In(size_t n, const T* values) const {
        AssertInfo(n == 0 || values != nullptr, "missing query terms");
        TargetBitmap result(count_);
        for (size_t i = 0; i < n; ++i) DecodeInto(Lookup(values[i]), result);
        return result;
    }
    TargetBitmap
    NotIn(size_t n, const T* values) const {
        auto result = In(n, values);
        result.flip();
        for (auto i : null_offsets_) result.reset(i);
        return result;
    }
    TargetBitmap
    IsNull() const {
        TargetBitmap result(count_);
        for (auto i : null_offsets_) result.set(i);
        return result;
    }
    TargetBitmap
    IsNotNull() const {
        auto result = IsNull();
        result.flip();
        return result;
    }
    TargetBitmap
    Range(const T& lower, bool li, const T& upper, bool ui) const {
        auto [begin, end] = Bounds(lower, li, upper, ui);
        return Materialize(begin, end);
    }
    TargetBitmap
    Range(const T& value, OpType op) const {
        CheckValue(value);
        size_t lower = std::lower_bound(terms_.begin(), terms_.end(), value) -
                       terms_.begin();
        size_t upper = std::upper_bound(terms_.begin(), terms_.end(), value) -
                       terms_.begin();
        switch (op) {
            case OpType::LessThan:
                return Materialize(0, lower);
            case OpType::LessEqual:
                return Materialize(0, upper);
            case OpType::GreaterThan:
                return Materialize(upper, terms_.size());
            case OpType::GreaterEqual:
                return Materialize(lower, terms_.size());
            default:
                ThrowInfo(ErrorCode::OpTypeInvalid,
                          "invalid PoC range operator");
        }
    }

 private:
    PostingFormat format_;
    static void
    CheckValue(const T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            if (value.find('\0') != std::string::npos)
                ThrowInfo(ErrorCode::Unsupported,
                          "PoC does not support embedded NUL strings");
        }
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(value))
                ThrowInfo(ErrorCode::Unsupported,
                          "PoC does not support NaN terms");
        }
    }
    TargetBitmap
    Materialize(size_t begin, size_t end) const {
        TargetBitmap result(count_);
        for (size_t i = begin; i < end; ++i) DecodeInto(i, result);
        return result;
    }
    size_t count_ = 0;
    std::vector<T> terms_;
    std::vector<TermPostingMeta> posting_metas_;
    std::vector<uint8_t> posting_bytes_;
    std::vector<size_t> null_offsets_;
};
}  // namespace milvus::index
