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
#include <map>
#include <memory>
#include "index/TextMatchIndexBase.h"
#include "index/FstTermDictionary.h"
#include "index/KnowherePositionIndex.h"
#include "index/InvertedIndexKnowhereCore.h"
#include "common/Tracer.h"
#include "tantivy/tokenizer.h"

namespace milvus::index {
// Resident sealed TEXT_MATCH PoC. Milvus retains the Tantivy analyzer;
// term dictionary, docID compression and query execution are independent.
class TextMatchIndexKnowhere final : public TextMatchIndexBase {
 public:
    using Core = InvertedIndexKnowhereCore<uint32_t, TargetBitmap, OpType>;
    explicit TextMatchIndexKnowhere(
        const std::string& analyzer_params = "{}",
        KnowhereSparsePostingCodec::Format format =
            KnowhereSparsePostingCodec::Format::Adaptive)
        : analyzer_(std::make_unique<milvus::tantivy::Tokenizer>(
              std::string(analyzer_params))),
          core_(format),
          format_(format) {
    }
    void
    Build(size_t n,
          const std::string* texts,
          const std::vector<size_t>& nulls = {}) {
        AssertInfo(n <= size_t(INT32_MAX) && (n == 0 || texts != nullptr),
                   "invalid text build row domain");
        AssertInfo(
            std::is_sorted(nulls.begin(), nulls.end()) &&
                std::adjacent_find(nulls.begin(), nulls.end()) == nulls.end() &&
                (nulls.empty() || nulls.back() < n),
            "invalid text null offsets");
        auto tokenizer = analyzer_->Clone();
        struct PostingBuilder {
            std::vector<uint32_t> docs, frequencies, deltas;
            uint32_t last_position = 0;
        };
        std::map<std::string, PostingBuilder> postings;
        size_t null = 0;
        for (size_t row = 0; row < n; ++row) {
            if (null < nulls.size() && nulls[null] == row) {
                ++null;
                continue;
            }
            CheckText(texts[row]);
            auto stream = tokenizer->CreateTokenStreamCopyText(texts[row]);
            while (stream->advance()) {
                auto [text, position] = DetailedToken(*stream);
                auto& posting = postings[text];
                if (posting.docs.empty() || posting.docs.back() != row) {
                    posting.docs.push_back(row);
                    posting.frequencies.push_back(0);
                    posting.last_position = 0;
                }
                AssertInfo(position >= posting.last_position &&
                               posting.frequencies.back() < UINT32_MAX,
                           "invalid analyzer position sequence or frequency");
                ++posting.frequencies.back();
                posting.deltas.push_back(position - posting.last_position);
                posting.last_position = position;
            }
        }
        std::vector<std::string> terms;
        terms.reserve(postings.size());
        std::map<uint32_t, std::vector<uint32_t>> ordinal_postings;
        KnowherePositionIndex next_positions;
        for (auto& [term, posting] : postings) {
            AssertInfo(terms.size() < UINT32_MAX,
                       "text term count exceeds uint32");
            ordinal_postings.emplace(terms.size(), std::move(posting.docs));
            next_positions.AppendTerm(posting.frequencies, posting.deltas);
            posting.frequencies.clear();
            posting.deltas.clear();
            terms.push_back(term);
        }
        next_positions.Seal();
        auto dictionary = FstTermDictionary::Build(terms);
        Core next(format_);
        next.BuildFromPostings(n, ordinal_postings, nulls);
        // Publish both together only after every build stage succeeds.
        core_ = std::move(next);
        dictionary_ = std::move(dictionary);
        positions_ = std::move(next_positions);
    }
    int64_t
    Count() override {
        return core_.Count();
    }
    int64_t
    ByteSize() const override {
        return core_.ByteSize() + dictionary_.ByteSize() +
               positions_.ByteSize();
    }
    int64_t
    ValidityBitmapByteSize() const override {
        return 0;
    }
    TargetBitmap
    IsNotNull() override {
        return core_.IsNotNull();
    }
    TargetBitmap
    MatchQuery(const std::string& query, uint32_t minimum) override {
        tracer::AutoSpan span("TextMatchIndexKnowhere::MatchQuery",
                              tracer::GetRootSpan());
        CheckText(query);
        auto tokenizer = analyzer_->Clone();
        auto stream = tokenizer->CreateTokenStreamCopyText(query);
        std::vector<std::string> terms;
        while (stream->advance()) terms.push_back(stream->get_token());
        if (minimum <= 1) {
            std::sort(terms.begin(), terms.end());
            terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
            TargetBitmap result(core_.Count());
            for (const auto& term : terms) {
                if (auto id = dictionary_.Lookup(term))
                    core_.DecodeInto(*id, result);
            }
            return result;
        }
        TargetBitmap result(core_.Count());
        if (minimum > terms.size())
            return result;
        // Resolve all clauses before touching postings. Missing dictionary terms
        // cannot contribute; even matching every remaining clause may be insufficient.
        std::vector<size_t> postings;
        postings.reserve(terms.size());
        for (const auto& term : terms) {
            if (auto id = dictionary_.Lookup(term))
                postings.push_back(*id);
        }
        if (postings.size() < minimum)
            return result;
        // If every remaining clause is required, intersection is sufficient.
        // Decide BEFORE deduplicating: repeated clauses retain their weight for
        // general thresholds. Missing clauses can reduce a threshold to AND.
        if (postings.size() == minimum) {
            core_.IntersectInto(std::move(postings), result);
            return result;
        }
        // Tantivy counts matching query clauses, including repeated query terms.
        // Repeated occurrences in a document do not count as extra clauses.
        std::vector<uint32_t> matched(core_.Count(), 0);
        for (auto id : postings) {
            core_.ForEachDoc(id, [&](uint32_t row) {
                if (matched[row] < minimum && ++matched[row] == minimum)
                    result.set(row);
            });
        }
        return result;
    }
    TargetBitmap
    FuzzyMatchQuery(const std::string& query, uint32_t max_edits) override {
        if (max_edits > 2)
            ThrowInfo(ErrorCode::InvalidParameter,
                      "max_edit_distance must be within [0, 2]");
        if (max_edits == 0)
            return MatchQuery(query, 1);
        CheckText(query);
        auto tokenizer = analyzer_->Clone();
        auto stream = tokenizer->CreateTokenStreamCopyText(query);
        TargetBitmap result(core_.Count());
        std::vector<uint32_t> expanded;
        std::vector<std::string> queries;
        while (stream->advance()) {
            auto term = stream->get_token();
            if (std::find(queries.begin(), queries.end(), term) != queries.end())
                continue;
            queries.push_back(term);
            dictionary_.ForEachFuzzy(term, max_edits,
                                    [&](std::string_view, uint32_t id) {
                                        expanded.push_back(id);
                                    });
        }
        std::sort(expanded.begin(), expanded.end());
        expanded.erase(std::unique(expanded.begin(), expanded.end()),
                       expanded.end());
        for (auto term : expanded)
            core_.DecodeInto(term, result);
        return result;
    }
    TargetBitmap
    PhraseMatchQuery(const std::string& query, uint32_t slop) override {
        CheckText(query);
        auto tokenizer = analyzer_->Clone();
        auto stream = tokenizer->CreateTokenStreamCopyText(query);
        struct Clause {
            size_t term;
            uint32_t position;
        };
        std::vector<Clause> clauses;
        uint32_t max_position = 0;
        TargetBitmap result(core_.Count());
        while (stream->advance()) {
            auto [text, position] = DetailedToken(*stream);
            auto id = dictionary_.Lookup(text);
            if (!id)
                return result;
            clauses.push_back({*id, position});
            max_position = std::max(max_position, position);
        }
        if (clauses.empty())
            return result;
        if (clauses.size() == 1) {
            core_.DecodeInto(clauses[0].term, result);
            return result;
        }
        // Match the pinned PhraseQuery's offset ordering, then Intersection's
        // stable DF ordering. Keep repeated clauses and their position offsets.
        std::stable_sort(clauses.begin(), clauses.end(), [](auto a, auto b) {
            return a.position < b.position;
        });
        std::stable_sort(clauses.begin(), clauses.end(), [&](auto a, auto b) {
            return core_.DocFreq(a.term) < core_.DocFreq(b.term);
        });
        std::vector<size_t> terms;
        std::vector<KnowhereSparsePostingCodec::Cursor> docs;
        std::vector<KnowherePositionIndex::Reader> readers;
        struct DecodeClause {
            size_t clause;
            uint64_t offset;
        };
        struct CopyClause {
            size_t source, target;
            uint64_t source_offset, target_offset;
        };
        std::vector<DecodeClause> decode;
        std::vector<CopyClause> copies;
        // Resolve sharing once per query. The row loop decodes unique terms
        // without a per-clause duplicate branch, then aligns duplicate buffers.
        for (size_t i = 0; i < clauses.size(); ++i) {
            const uint64_t offset =
                uint64_t(max_position) - clauses[i].position;
            auto found = std::find(terms.begin(), terms.end(), clauses[i].term);
            if (found == terms.end()) {
                terms.push_back(clauses[i].term);
                decode.push_back({i, offset});
            } else {
                const auto first = decode[found - terms.begin()];
                copies.push_back({first.clause, i, first.offset, offset});
            }
        }
        docs.reserve(terms.size());
        readers.reserve(terms.size());
        for (auto term : terms) {
            docs.push_back(core_.NewCursor(term));
            readers.emplace_back(positions_, term);
        }
        TargetBitmap candidates(core_.Count());
        core_.IntersectInto(std::move(terms), candidates);
        std::vector<std::vector<uint64_t>> positions(clauses.size());
        KnowherePhraseScratch scratch;
        for (auto row = candidates.find_first(); row;
             row = candidates.find_next(*row)) {
            for (size_t i = 0; i < decode.size(); ++i) {
                AssertInfo(docs[i].Seek(*row) == *row,
                           "phrase candidate missing from posting");
                readers[i].Read(docs[i].PostingOrdinal(),
                                decode[i].offset,
                                positions[decode[i].clause]);
            }
            for (auto copy : copies) {
                const auto& source = positions[copy.source];
                auto& target = positions[copy.target];
                target.resize(source.size());
                for (size_t j = 0; j < source.size(); ++j)
                    target[j] =
                        source[j] - copy.source_offset + copy.target_offset;
            }
            if (KnowherePhraseExists(positions, slop, scratch))
                result.set(*row);
        }
        return result;
    }
    size_t
    PositionBytesForUT() const {
        return positions_.LogicalBytes();
    }
    const FstTermDictionary&
    Dictionary() const {
        return dictionary_;
    }
    // Dictionary-only reload acceptance hook. The complete text index does not
    // yet have a persisted format. Reject a dictionary from another build.
    void
    UseMappedDictionaryForUT(const std::string& path) {
        auto mapped = FstTermDictionary::MapFile(path);
        if (mapped.SerializedBytes() != dictionary_.SerializedBytes())
            ThrowInfo(ErrorCode::DataFormatBroken,
                      "FST does not belong to this text index");
        dictionary_ = std::move(mapped);
    }
    const Core&
    CoreForUT() const {
        return core_;
    }

 private:
    static std::pair<std::string, uint32_t>
    DetailedToken(milvus::tantivy::TokenStream& stream) {
        auto token = stream.get_detailed_token();
        auto release = [](const char* ptr) { free_rust_string(ptr); };
        std::unique_ptr<const char, decltype(release)> text(token.token,
                                                            release);
        AssertInfo(
            token.position >= 0 && uint64_t(token.position) <= UINT32_MAX,
            "analyzer position exceeds uint32 domain");
        return {std::string(text.get()), static_cast<uint32_t>(token.position)};
    }
    static void
    CheckText(const std::string& text) {
        if (text.find('\0') != std::string::npos)
            ThrowInfo(Unsupported, "text PoC does not support embedded NUL");
    }
    std::unique_ptr<milvus::tantivy::Tokenizer> analyzer_;
    Core core_;
    KnowhereSparsePostingCodec::Format format_;
    FstTermDictionary dictionary_;
    KnowherePositionIndex positions_;
};
}  // namespace milvus::index
