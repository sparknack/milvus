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
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "index/KnowhereSparsePostingCodec.h"
#include "index/KnowherePoCIO.h"
#include "index/KnowhereTermOrder.h"
#include "index/KnowhereTermStorage.h"

namespace milvus::index {
// Experimental heap-only scalar backend. Only the UT builder creates one.
// Kept separate from ScalarIndex; versioned local snapshots are supported.
template <typename T, typename TargetBitmap, typename OpType>
class InvertedIndexKnowhereCore {
 public:
    using PostingFormat = KnowhereSparsePostingCodec::Format;
    explicit InvertedIndexKnowhereCore(
        PostingFormat format = PostingFormat::StreamVByte)
        : format_(format) {
    }
    PostingFormat
    PostingFormatForPoC() const {
        return format_;
    }

    // Persists the actual compressed bytes. Load validates them in place and
    // never reconstructs postings from source values or a temporary map.
    std::vector<uint8_t>
    SerializeForPoC() const {
        poc_io::Writer writer;
        writer.U64(count_);
        writer.U64(terms_.size());
        terms_.Save(writer);
        posting_offsets_.Save(writer);
        doc_freqs_.Save(writer);
        writer.U64(null_offsets_.size());
        for (auto row : null_offsets_) writer.U32(row);
        writer.Bytes(std::span<const uint8_t>(posting_bytes_.data(),
                                              PostingLogicalBytes()));
        return poc_io::Pack(
            "KWPCCORE",
            poc_io::TypeTag<T>(),
            2 | (format_ != PostingFormat::StreamVByte ? 1 : 0) |
                (format_ == PostingFormat::Adaptive ? 4 : 0),
            writer.data);
    }

    void
    LoadForPoC(std::span<const uint8_t> blob) {
        using poc_io::Check;
        uint32_t codec = 0;
        poc_io::Reader reader(
            poc_io::Unpack(blob, "KWPCCORE", poc_io::TypeTag<T>(), &codec));
        Check(
            codec == 0 || codec == 1 || codec == 2 || codec == 3 || codec == 7,
            "unknown persisted posting format");
        const bool compact = codec & 2;
        Check(std::endian::native == std::endian::little,
              "persisted posting codec requires little endian");
        InvertedIndexKnowhereCore next((codec & 4) ? PostingFormat::Adaptive
                                       : (codec & 1)
                                           ? PostingFormat::AdaptiveLegacy
                                           : PostingFormat::StreamVByte);
        const uint64_t count = reader.U64(), term_count = reader.U64();
        Check(count <= INT32_MAX && term_count <= UINT32_MAX,
              "persisted core domain exceeds limits");
        next.count_ = count;
        std::vector<TermPostingMeta> wire_metas;
        if (compact) {
            next.terms_.Load(reader, term_count);
            next.posting_offsets_.Load(reader, term_count);
            next.doc_freqs_.Load(reader, term_count);
            for (size_t i = 0; i < term_count; ++i) {
                const auto value = next.terms_[i];
                if constexpr (std::is_same_v<T, std::string>)
                    Check(value.find('\0') == std::string::npos,
                          "persisted term contains NUL");
                Check(i == 0 || TermLess(next.terms_[i - 1], value),
                      "persisted terms are not strictly ordered");
                Check(next.doc_freqs_[i] > 0 && next.doc_freqs_[i] <= count,
                      "invalid persisted posting metadata");
            }
        } else {
            // Every entry has at least one term byte and 24 metadata bytes.
            Check(term_count <= reader.Remaining() / 25,
                  "persisted term count exceeds payload");
            next.terms_.reserve(term_count);
            for (size_t i = 0; i < term_count; ++i) {
                T value;
                if constexpr (std::is_same_v<T, std::string>) {
                    value = reader.String();
                    Check(value.find('\0') == std::string::npos,
                          "persisted term contains NUL");
                } else if constexpr (std::is_same_v<T, bool>) {
                    auto raw = reader.U8();
                    Check(raw <= 1, "invalid persisted bool term");
                    value = raw;
                } else if constexpr (std::is_same_v<T, float>) {
                    value = std::bit_cast<float>(reader.U32());

                } else if constexpr (std::is_same_v<T, double>) {
                    value = std::bit_cast<double>(reader.U64());

                } else {
                    using Unsigned = std::make_unsigned_t<T>;
                    auto raw = reader.U64();
                    Check(raw <= std::numeric_limits<Unsigned>::max(),
                          "persisted integer term exceeds type width");
                    value = std::bit_cast<T>(static_cast<Unsigned>(raw));
                }
                Check(
                    next.terms_.empty() || TermLess(next.terms_.back(), value),
                    "persisted terms are not strictly ordered");
                next.terms_.push_back(std::move(value));
            }
            Check(term_count <= reader.Remaining() / 24,
                  "truncated persisted term metadata");
            wire_metas.reserve(term_count);
            next.posting_offsets_.reserve(term_count);
            next.doc_freqs_.reserve(term_count);
            for (size_t i = 0; i < term_count; ++i) {
                TermPostingMeta meta;
                meta.doc_stream_offset = reader.U64();
                meta.doc_stream_length = reader.U64();
                meta.doc_freq = reader.U32();
                meta.format_version = reader.U32();
                Check(meta.format_version == 1 && meta.doc_freq > 0 &&
                          meta.doc_freq <= count,
                      "invalid persisted posting metadata");
                wire_metas.push_back(meta);
                next.posting_offsets_.push_back(meta.doc_stream_offset);
                next.doc_freqs_.push_back(meta.doc_freq);
            }
        }
        const uint64_t null_count = reader.U64();
        Check(null_count <= count && null_count <= reader.Remaining() / 4,
              "invalid persisted null count");
        next.null_offsets_.reserve(null_count);
        for (size_t i = 0; i < null_count; ++i) {
            auto row = reader.U32();
            Check(row < count && (next.null_offsets_.empty() ||
                                  next.null_offsets_.back() < row),
                  "invalid persisted null offset");
            next.null_offsets_.push_back(row);
        }
        auto bytes = reader.Bytes();
        reader.Finish();
        if (compact) {
            wire_metas.reserve(term_count);
            for (size_t i = 0; i < term_count; ++i) {
                auto offset = next.posting_offsets_[i];
                auto end = i + 1 < term_count ? next.posting_offsets_[i + 1]
                                              : bytes.size();
                Check(offset <= end && end <= bytes.size(),
                      "invalid compact posting extent");
                wire_metas.push_back(
                    {offset, end - offset, uint32_t(next.doc_freqs_[i]), 1});
            }
        }
        uint64_t expected_offset = 0;
        for (const auto& meta : wire_metas) {
            Check(meta.doc_stream_offset == expected_offset &&
                      expected_offset <= bytes.size() &&
                      meta.doc_stream_length <= bytes.size() - expected_offset,
                  "persisted posting extent is invalid");
            auto posting =
                bytes.subspan(expected_offset, meta.doc_stream_length);
            expected_offset += meta.doc_stream_length;
            size_t header = 0;
            uint32_t df = 0;
            for (unsigned group = 0;; ++group) {
                Check(group < 5 && header < posting.size(),
                      "invalid persisted posting count vint");
                const auto byte = posting[header++];
                Check(group < 4 || (byte & 0x78) == 0,
                      "overflowing persisted posting count");
                df |= uint32_t(byte & 127) << (group * 7);
                if (byte & 128)
                    break;
            }
            Check(df == meta.doc_freq, "persisted posting DF mismatch");
            const size_t blocks = (size_t(df) + 255) / 256;
            const bool short_list =
                next.format_ == PostingFormat::Adaptive && df <= 256;
            const size_t directory = short_list ? 0 : (2 * blocks - 1) * 4;
            Check(directory <= posting.size() - header,
                  "truncated persisted block directory");
            auto word = [&](size_t position) {
                poc_io::Reader word_reader(posting.subspan(position, 4));
                return word_reader.U32();
            };
            const size_t data_begin = header + directory;
            const size_t data_size = posting.size() - data_begin;
            size_t start = 0;
            uint64_t next_id = 0;
            for (size_t block = 0; block < blocks; ++block) {
                const uint64_t end =
                    block + 1 == blocks ? data_size
                                        : word(header + blocks * 4 + block * 4);
                Check(end > start && end <= data_size,
                      "invalid persisted block extent");
                const size_t n = std::min<size_t>(256, df - block * 256);
                auto gaps = KnowhereSparsePostingCodec::DecodeChecked(
                    posting.subspan(data_begin + start, end - start),
                    n,
                    next.format_);
                uint32_t last = 0;
                for (size_t j = 0; j < n; ++j) {
                    const uint64_t id = next_id + gaps[j];
                    Check(id < count, "persisted document ID out of range");
                    Check(!std::binary_search(next.null_offsets_.begin(),
                                              next.null_offsets_.end(),
                                              id),
                          "persisted posting contains NULL row");
                    last = id;
                    next_id = id + 1;
                }
                Check(short_list || last == word(header + block * 4),
                      "persisted block maximum mismatch");
                start = end;
            }
        }
        Check(expected_offset == bytes.size(),
              "unreferenced persisted posting bytes");
        next.posting_bytes_.reserve(bytes.size() +
                                    KnowhereSparsePostingCodec::kPadding);
        next.posting_bytes_.assign(bytes.begin(), bytes.end());
        next.posting_bytes_.resize(
            next.posting_bytes_.size() + KnowhereSparsePostingCodec::kPadding,
            0);
        next.Compact();
        *this = std::move(next);
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
            std::sort(
                pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                    if (KnowhereTermOrder<T>::Equal(a.first, b.first))
                        return a.second < b.second;
                    return KnowhereTermOrder<T>{}(a.first, b.first);
                });
            std::vector<uint32_t> ids;
            for (size_t i = 0; i < pairs.size();) {
                next.terms_.push_back(pairs[i].first);
                ids.clear();
                size_t j = i;
                do {
                    ids.push_back(pairs[j++].second);
                } while (j < pairs.size() &&
                         KnowhereTermOrder<T>::Equal(pairs[j].first,
                                                     pairs[i].first));
                uint64_t offset = next.posting_bytes_.size();
                KnowhereSparsePostingCodec::Append(
                    ids.data(), ids.size(), next.posting_bytes_, format_);
                next.posting_offsets_.push_back(offset);
                next.doc_freqs_.push_back(static_cast<uint32_t>(ids.size()));
                i = j;
            }
        }
        next.posting_bytes_.resize(
            next.posting_bytes_.size() + KnowhereSparsePostingCodec::kPadding,
            0);
        next.Compact();
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
            AssertInfo(
                next.terms_.empty() || TermLess(next.terms_.back(), term),
                "text dictionary must be strictly sorted");
            AssertInfo(!ids.empty() && ids.back() < n,
                       "invalid text docID range");
            uint64_t offset = next.posting_bytes_.size();
            KnowhereSparsePostingCodec::Append(
                ids.data(), ids.size(), next.posting_bytes_, format_);
            next.terms_.push_back(term);
            next.posting_offsets_.push_back(offset);
            next.doc_freqs_.push_back(static_cast<uint32_t>(ids.size()));
        }
        next.posting_bytes_.resize(
            next.posting_bytes_.size() + KnowhereSparsePostingCodec::kPadding,
            0);
        next.Compact();
        *this = std::move(next);
    }
    template <typename Callback>
    void
    ForEachDoc(size_t term, Callback&& callback) const {
        if (term == terms_.size())
            return;
        const auto offset = posting_offsets_.at(term);
        KnowhereSparsePostingCodec::View posting(posting_bytes_.data() + offset,
                                                 format_);
        std::array<uint32_t, KnowhereSparsePostingCodec::kBlockSize> ids;
        for (size_t b = 0; b < posting.Blocks(); ++b) {
            auto n = posting.DecodeBlock(b, ids.data());
            for (size_t i = 0; i < n; ++i) callback(ids[i]);
        }
    }
    KnowhereSparsePostingCodec::Cursor
    NewCursor(size_t term) const {
        const auto offset = posting_offsets_.at(term);
        return KnowhereSparsePostingCodec::Cursor(
            posting_bytes_.data() + offset, format_);
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
            AssertInfo(term < posting_offsets_.size(),
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
            cursors.emplace_back(posting_bytes_.data() + posting_offsets_[term],
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
        return {terms_.Bytes(),
                posting_offsets_.Bytes() + doc_freqs_.Bytes(),
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
        auto i = Bound(value, false);
        return i < terms_.size() && !TermLess(value, terms_[i]) ? i
                                                                : terms_.size();
    }
    size_t
    TermCount() const {
        return terms_.size();
    }
    T
    Term(size_t i) const {
        return T(terms_.at(i));
    }
    // Callback-only term views backed by query-local scratch; do not retain them.
    // Do not return const T& generically: vector<bool> has proxy elements.
    template <typename Visitor>
    void
    ForEachStringPrefix(std::string_view prefix, Visitor&& visitor) const
        requires(std::is_same_v<T, std::string>) {
        std::string scratch;
        for (size_t i = Bound(prefix, false); i < terms_.size(); ++i) {
            terms_.Get(i, scratch);
            const std::string_view term(scratch);
            if (!term.starts_with(prefix))
                break;
            visitor(term, static_cast<uint32_t>(i));
        }
    }
    uint32_t
    DocFreq(size_t i) const {
        return doc_freqs_.at(i);
    }
    std::pair<size_t, size_t>
    Bounds(const T& lower, bool li, const T& upper, bool ui) const {
        CheckValue(lower);
        CheckValue(upper);
        size_t begin = Bound(lower, !li);
        size_t end = Bound(upper, ui);
        return {begin, std::max(begin, end)};
    }
    void
    DecodeInto(size_t term, TargetBitmap& result) const {
        AssertInfo(result.size() == count_, "PoC result bitmap size mismatch");
        if (term == terms_.size())
            return;
        const auto offset = posting_offsets_.at(term);
        KnowhereSparsePostingCodec::View posting(posting_bytes_.data() + offset,
                                                 format_);
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
        size_t lower = Bound(value, false);
        size_t upper = Bound(value, true);
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
    template <typename A, typename B>
    static bool
    TermLess(const A& a, const B& b) {
        if constexpr (std::is_same_v<T, std::string>)
            return std::string_view(a) < std::string_view(b);
        else
            return KnowhereTermOrder<T>{}(a, b);
    }
    template <typename Value>
    size_t
    Bound(const Value& value, bool upper) const {
        size_t lo = 0, hi = terms_.size();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const bool advance = upper ? !TermLess(value, terms_[mid])
                                       : TermLess(terms_[mid], value);
            if (advance)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }
    void
    Compact() {
        terms_.Compact();
        posting_offsets_.Compact();
        doc_freqs_.Compact();
        posting_bytes_.shrink_to_fit();
        null_offsets_.shrink_to_fit();
    }
    PostingFormat format_;
    static void
    CheckValue(const T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            if (value.find('\0') != std::string::npos)
                ThrowInfo(ErrorCode::Unsupported,
                          "PoC does not support embedded NUL strings");
        }
    }
    TargetBitmap
    Materialize(size_t begin, size_t end) const {
        TargetBitmap result(count_);
        for (size_t i = begin; i < end; ++i) DecodeInto(i, result);
        return result;
    }
    size_t count_ = 0;
    KnowhereTermStorage<T> terms_;
    KnowherePackedVector posting_offsets_;
    KnowherePackedVector doc_freqs_;
    std::vector<uint8_t> posting_bytes_;
    std::vector<size_t> null_offsets_;
};
}  // namespace milvus::index
