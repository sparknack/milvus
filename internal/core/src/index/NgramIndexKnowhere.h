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
#include "index/KnowherePoCAdapterIO.h"
#include "index/ScalarIndex.h"
#include "index/NgramIndexBase.h"
#include "index/NgramInvertedIndex.h"  // shared conservative regex literal extractor
#include "index/InvertedIndexKnowhereCore.h"
#include "index/FstTermDictionary.h"
#include "common/RegexQuery.h"
#include "exec/expression/Expr.h"
#include "tantivy/tokenizer.h"

namespace milvus::index {
class NgramIndexKnowhere : public ScalarIndex<std::string>,
                           public NgramIndexBase {
 public:
    virtual std::vector<uint8_t>
    SerializeForPoC() const {
        poc_io::Writer w;
        w.U64(min_);
        w.U64(max_);
        w.U64(average_row_bytes_);
        w.Bytes(core_.SerializeForPoC());
        auto terms = dictionary_.Enumerate();
        w.U64(terms.size());
        for (const auto& [term, id] : terms) w.String(term);
        return poc_io::Pack("KNGRAM01", 4, 0, w.data);
    }
    virtual void
    LoadForPoC(std::span<const uint8_t> bytes) {
        poc_io::Reader r(poc_io::UnpackAdapter(bytes, "KNGRAM01", 4));
        poc_io::Check(r.U64() == min_, "ngram minimum mismatch");
        poc_io::Check(r.U64() == max_, "ngram maximum mismatch");
        const auto average = r.U64();
        Core next;
        next.LoadForPoC(r.Bytes());
        const auto count = r.U64();
        poc_io::Check(count == next.TermCount() && count <= r.Remaining()/8,
                      "ngram dictionary count mismatch");
        std::vector<std::string> terms;
        terms.reserve(count);
        for (size_t i=0; i<count; ++i) {
            terms.push_back(r.String());
            poc_io::Check(next.Term(i)==i && (i==0 || terms[i-1]<terms[i]) &&
                          terms[i].find('\0') == std::string::npos,
                          "invalid ngram dictionary");
        }
        r.Finish();
        auto dictionary = FstTermDictionary::Build(terms);
        core_ = std::move(next);
        dictionary_ = std::move(dictionary);
        average_row_bytes_ = average;
        ComputeByteSize();
    }
 protected:
    void
    SwapStateForPoC(NgramIndexKnowhere& other) noexcept {
        using std::swap;
        swap(core_, other.core_);
        swap(dictionary_, other.dictionary_);
        swap(average_row_bytes_, other.average_row_bytes_);
        swap(min_, other.min_);
        swap(max_, other.max_);
        swap(analyzer_, other.analyzer_);
        swap(query_analyzer_, other.query_analyzer_);
        swap(cached_byte_size_, other.cached_byte_size_);
    }
 public:

    using Core = InvertedIndexKnowhereCore<uint32_t, TargetBitmap, OpType>;
    NgramIndexKnowhere(size_t min_gram, size_t max_gram)
        : ScalarIndex<std::string>(NGRAM_INDEX_TYPE),
          min_(min_gram),
          max_(max_gram),
          core_(KnowhereSparsePostingCodec::Format::Adaptive) {
        auto make = [](size_t min, size_t max) {
            auto res = tantivy::RustResultWrapper(
                tantivy_create_ngram_analyzer(min, max));
            AssertTantivyOk(
                res, "ngram analyzer creation failed: {}", res.result_->error);
            return std::make_unique<tantivy::Tokenizer>(
                res.result_->value.ptr._0);
        };
        analyzer_ = make(min_, max_);
        query_analyzer_ = make(max_, max_);
    }
    ScalarIndexType
    GetIndexType() const override {
        return ScalarIndexType::NGRAM;
    }
    bool
    ShouldUseOp(proto::plan::OpType, const std::string& = "") const override {
        return false;
    }
    bool
    SupportPatternMatch() const override {
        return false;
    }
    const bool
    HasRawData() const override {
        return false;
    }
    bool
    IsMmapSupported() const override {
        return false;
    }
    int64_t
    Count() override {
        return core_.Count();
    }
    int64_t
    Size() override {
        return Count();
    }
    void
    ComputeByteSize() override {
        cached_byte_size_ = core_.ByteSize() + dictionary_.ByteSize();
    }
    TargetBitmap
    IsNotNull() override {
        return core_.IsNotNull();
    }
    const TargetBitmap
    IsNull() override {
        return core_.IsNull();
    }
    void
    Build(size_t n,
          const std::string* values,
          const bool* valid = nullptr) override {
        std::vector<size_t> nulls;
        if (valid)
            for (size_t i = 0; i < n; ++i)
                if (!valid[i])
                    nulls.push_back(i);
        BuildWithNullOffsetsForUT(n, values, nulls);
    }
    void
    BuildWithRawDataForUT(size_t n,
                          const void* values,
                          const Config& = {}) override {
        Build(n, static_cast<const std::string*>(values));
    }
    void
    BuildWithNullOffsetsForUT(size_t n,
                              const std::string* values,
                              const std::vector<size_t>& nulls = {}) {
        AssertInfo(n <= size_t(INT32_MAX) && (n == 0 || values),
                   "invalid ngram build row domain");
        AssertInfo(
            std::is_sorted(nulls.begin(), nulls.end()) &&
                std::adjacent_find(nulls.begin(), nulls.end()) == nulls.end() &&
                (nulls.empty() || nulls.back() < n),
            "invalid ngram null offsets");
        std::map<std::string, std::vector<uint32_t>> postings;
        auto tokenizer = analyzer_->Clone();
        size_t null = 0, total_bytes = 0;
        for (size_t row = 0; row < n; ++row) {
            if (null < nulls.size() && nulls[null] == row) {
                ++null;
                continue;
            }
            CheckText(values[row]);
            total_bytes += values[row].size();
            auto stream = tokenizer->CreateTokenStreamCopyText(values[row]);
            while (stream->advance()) {
                auto& docs = postings[stream->get_token()];
                if (docs.empty() || docs.back() != row)
                    docs.push_back(row);
            }
        }
        std::vector<std::string> terms;
        std::map<uint32_t, std::vector<uint32_t>> ordinal_postings;
        for (auto& [term, docs] : postings) {
            AssertInfo(terms.size() < UINT32_MAX,
                       "ngram term ordinal overflow");
            ordinal_postings.emplace(terms.size(), std::move(docs));
            terms.push_back(term);
        }
        auto dictionary = FstTermDictionary::Build(terms);
        Core next(KnowhereSparsePostingCodec::Format::Adaptive);
        next.BuildFromPostings(n, ordinal_postings, nulls);
        core_ = std::move(next);
        dictionary_ = std::move(dictionary);
        average_row_bytes_ =
            n > nulls.size() ? total_bytes / (n - nulls.size()) : 0;
        ComputeByteSize();
    }
    bool
    CanHandleLiteral(const std::string& literal,
                     proto::plan::OpType op) const override {
        if (op == OpType::Match) {
            auto parts = split_by_wildcard(literal);
            return !parts.empty() &&
                   std::all_of(parts.begin(), parts.end(), [&](const auto& p) {
                       return Length(p) >= min_;
                   });
        }
        if (op == OpType::RegexMatch) {
            auto parts = extract_literals_from_regex(literal);
            return std::any_of(parts.begin(), parts.end(), [&](const auto& p) {
                return Length(p) >= min_;
            });
        }
        return (op == OpType::InnerMatch || op == OpType::PrefixMatch ||
                op == OpType::PostfixMatch) &&
               Length(literal) >= min_;
    }
    void
    ExecutePhase1(const std::string& literal,
                  proto::plan::OpType op,
                  TargetBitmap& candidates) override {
        AssertInfo(candidates.size() == size_t(Count()),
                   "ngram candidate row domain mismatch");
        AssertInfo(CanHandleLiteral(literal, op),
                   "unsupported ngram phase1 literal");
        if (candidates.none())
            return;
        std::vector<std::string> parts;
        if (op == OpType::Match)
            parts = split_by_wildcard(literal);
        else if (op == OpType::RegexMatch) {
            for (auto& p : extract_literals_from_regex(literal))
                if (Length(p) >= min_)
                    parts.push_back(p);
        } else
            parts.push_back(literal);
        std::unique_ptr<tantivy::Tokenizer> tokenizer;
        std::vector<size_t> ids;
        auto resolve = [&](const std::string& term) {
            auto id = dictionary_.Lookup(term);
            if (!id)
                return false;
            ids.push_back(*id);
            return true;
        };
        for (const auto& part : parts) {
            CheckText(part);
            if (Length(part) <= max_) {
                if (!resolve(part)) {
                    candidates.reset();
                    return;
                }
            } else {
                if (!tokenizer)
                    tokenizer = query_analyzer_->Clone();
                auto stream = tokenizer->CreateTokenStreamCopyText(part);
                while (stream->advance())
                    if (!resolve(stream->get_token())) {
                        candidates.reset();
                        return;
                    }
            }
        }
        // All required grams have been resolved before any decode/early stop.
        // NGRAM produces a superset, unlike TEXT_MATCH's exact conjunction.
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        std::sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
            return std::make_pair(core_.DocFreq(a), a) <
                   std::make_pair(core_.DocFreq(b), b);
        });
        const size_t dense_budget = average_row_bytes_ < 100    ? 2
                                    : average_row_bytes_ < 1000 ? 3
                                                                : 5;
        size_t count = candidates.count();
        TargetBitmap scratch;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0 && (count <= 8 || i >= (count <= 128 ? 8 : dense_budget)))
                break;
            const size_t previous = count;
            if (i == 0 && count == size_t(Count())) {
                candidates.reset();
                core_.DecodeInto(ids[i], candidates);
            } else if (count * 8 < core_.DocFreq(ids[i])) {
                // Probe only incoming candidates; do not materialize a dense
                // posting just to discard almost all of its document IDs.
                auto cursor = core_.NewCursor(ids[i]);
                for (auto row = candidates.find_first(); row;
                     row = candidates.find_next(*row)) {
                    if (cursor.Seek(*row) != *row)
                        candidates.reset(*row);
                }
            } else {
                if (scratch.size() == 0)
                    scratch = TargetBitmap(Count());
                else
                    scratch.reset();
                core_.DecodeInto(ids[i], scratch);
                candidates &= scratch;
            }
            count = candidates.count();
            // No reduction from a second gram: let the exact predicate finish.
            if (i > 0 && count > 128 && count == previous)
                break;
        }
    }
    void
    ExecutePhase2(const std::string& literal,
                  proto::plan::OpType op,
                  exec::SegmentExpr* segment,
                  TargetBitmap& candidates,
                  int64_t offset,
                  int64_t size) override {
        // Match the existing Tantivy phase2 fast path, including its byte-length check.
        if (op == OpType::InnerMatch && literal.size() <= max_)
            return;
        if (candidates.none())
            return;
        AssertInfo(size >= 0 && candidates.size() == size_t(size),
                   "ngram batch size mismatch");
        auto apply = [&](auto&& predicate) {
            // A dense input cannot benefit from constructing an offset vector.
            // Keep its original contiguous scan, with no per-row gather probe.
            if (candidates.count() * 8 > size_t(size)) {
                auto batch = [&](const std::string_view* data,
                                 int64_t n,
                                 TargetBitmapView bits) {
                    for (auto i = bits.find_first(); i && *i < size_t(n);
                         i = bits.find_next(*i))
                        if (!predicate(data[*i]))
                            bits[*i] = false;
                };
                segment->ProcessDataChunkForRange<std::string_view>(
                    batch, TargetBitmapView(candidates), offset, size);
                return;
            }
            segment->FilterStringCandidatesForRange(
                predicate, TargetBitmapView(candidates), offset, size);
        };
        switch (op) {
            case OpType::InnerMatch:
                apply([&](auto s) {
                    return s.find(literal) != std::string_view::npos;
                });
                break;
            case OpType::PrefixMatch:
                apply([&](auto s) { return s.starts_with(literal); });
                break;
            case OpType::PostfixMatch:
                apply([&](auto s) { return s.ends_with(literal); });
                break;
            case OpType::Match: {
                LikePatternMatcher matcher(literal);
                // Keep the character/segment matcher in this predicate's
                // compiled body. The two gather paths otherwise make GCC
                // outline SegmentMatchesAt inside its per-character loop.
                apply([&](auto s) __attribute__((flatten)) { return matcher(s); });
                break;
            }
            case OpType::RegexMatch: {
                PartialRegexMatcher matcher(literal);
                apply([&](auto s) { return matcher(s); });
                break;
            }
            default:
                ThrowInfo(Unsupported, "unsupported ngram phase2 operation");
        }
    }
    const TargetBitmap
    In(size_t, const std::string*) override {
        ThrowInfo(Unsupported, "ngram is a candidate index");
    }
    const TargetBitmap
    NotIn(size_t, const std::string*) override {
        ThrowInfo(Unsupported, "ngram is a candidate index");
    }
    const TargetBitmap
    Range(const std::string&, OpType) override {
        ThrowInfo(Unsupported, "ngram is a candidate index");
    }
    const TargetBitmap
    Range(const std::string&, bool, const std::string&, bool) override {
        ThrowInfo(Unsupported, "ngram is a candidate index");
    }
    std::optional<std::string>
    Reverse_Lookup(size_t) const override {
        ThrowInfo(Unsupported, "ngram has no raw data");
    }
    BinarySet
    Serialize(const Config& = {}) override {
        return poc_io::ToBinarySet(SerializeForPoC());
    }
    void
    Load(const BinarySet& set, const Config& = {}) override {
        LoadForPoC(poc_io::FromBinarySet(set));
    }
    void
    Load(tracer::TraceContext, const Config& = {}) override {
        ThrowInfo(Unsupported, "resident ngram PoC has no persistence");
    }
    void
    Build(const Config& = {}) override {
        ThrowInfo(Unsupported, "resident ngram PoC uses explicit row build");
    }
    IndexStatsPtr
    Upload(const Config& = {}) override {
        ThrowInfo(Unsupported, "resident ngram PoC has no persistence");
    }

 private:
    static size_t
    Length(const std::string& s) {
        return Utf8CharCount(s.data(), s.size());
    }
    static void
    CheckText(const std::string& s) {
        if (s.find('\0') != std::string::npos)
            ThrowInfo(Unsupported, "resident ngram PoC rejects embedded NUL");
    }
    size_t min_, max_;
    size_t average_row_bytes_ = 0;
    Core core_;
    FstTermDictionary dictionary_;
    std::unique_ptr<tantivy::Tokenizer> analyzer_, query_analyzer_;
};
}  // namespace milvus::index
