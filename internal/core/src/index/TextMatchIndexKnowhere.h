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
        std::map<std::string, std::vector<uint32_t>> postings;
        size_t null = 0;
        for (size_t row = 0; row < n; ++row) {
            if (null < nulls.size() && nulls[null] == row) {
                ++null;
                continue;
            }
            CheckText(texts[row]);
            auto stream = tokenizer->CreateTokenStreamCopyText(texts[row]);
            while (stream->advance()) {
                auto& ids = postings[stream->get_token()];
                if (ids.empty() || ids.back() != row)
                    ids.push_back(row);
            }
        }
        std::vector<std::string> terms;
        terms.reserve(postings.size());
        std::map<uint32_t, std::vector<uint32_t>> ordinal_postings;
        for (auto& [term, ids] : postings) {
            AssertInfo(terms.size() < UINT32_MAX,
                       "text term count exceeds uint32");
            ordinal_postings.emplace(terms.size(), std::move(ids));
            terms.push_back(term);
        }
        auto dictionary = FstTermDictionary::Build(terms);
        Core next(format_);
        next.BuildFromPostings(n, ordinal_postings, nulls);
        // Publish both together only after every build stage succeeds.
        core_ = std::move(next);
        dictionary_ = std::move(dictionary);
    }
    int64_t
    Count() override {
        return core_.Count();
    }
    int64_t
    ByteSize() const override {
        return core_.ByteSize() + dictionary_.ByteSize();
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
    static void
    CheckText(const std::string& text) {
        if (text.find('\0') != std::string::npos)
            ThrowInfo(Unsupported, "text PoC does not support embedded NUL");
    }
    std::unique_ptr<milvus::tantivy::Tokenizer> analyzer_;
    Core core_;
    KnowhereSparsePostingCodec::Format format_;
    FstTermDictionary dictionary_;
};
}  // namespace milvus::index
