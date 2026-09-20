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
#include "index/ScalarIndex.h"
#include "index/InvertedIndexKnowhereCore.h"
#include "index/InvertedIndexUtil.h"
#include "index/FstTermDictionary.h"
#include "common/RegexQuery.h"

namespace milvus::index {
// Resident scalar PoC. Intentionally independent of Tantivy and not factory registered.
template <typename T>
class InvertedIndexKnowhere : public ScalarIndex<T> {
 public:
    using Core = InvertedIndexKnowhereCore<T, TargetBitmap, OpType>;
    using ScalarIndex<T>::IsNotNull;
    explicit InvertedIndexKnowhere(
        KnowhereSparsePostingCodec::Format format =
            KnowhereSparsePostingCodec::Format::StreamVByte)
        : ScalarIndex<T>(INVERTED_INDEX_TYPE), core_(format), format_(format) {
    }
    ScalarIndexType
    GetIndexType() const override {
        return ScalarIndexType::INVERTED;
    }
    void
    Build(size_t n, const T* values, const bool* valid = nullptr) override {
        std::vector<size_t> nulls;
        if (valid)
            for (size_t i = 0; i < n; ++i)
                if (!valid[i])
                    nulls.push_back(i);
        BuildWithNullOffsetsForUT(n, values, nulls);
    }
    void
    BuildWithNullOffsetsForUT(size_t n,
                              const T* values,
                              const std::vector<size_t>& nulls = {}) {
        Core next(format_);
        next.Build(n, values, nulls);
        FstTermDictionary dictionary;
        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<std::string> terms;
            terms.reserve(next.TermCount());
            for (size_t i = 0; i < next.TermCount(); ++i)
                terms.push_back(next.Term(i));
            dictionary = FstTermDictionary::Build(terms);
        }
        core_ = std::move(next);
        dictionary_ = std::move(dictionary);
        ComputeByteSize();
    }
    void
    BuildWithRawDataForUT(size_t n,
                          const void* values,
                          const Config& config = {}) override {
        if (config.contains("is_array") || config.contains("is_nested_index"))
            ThrowInfo(Unsupported,
                      "Knowhere scalar PoC supports scalar rows only");
        Build(n, static_cast<const T*>(values));
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
        this->cached_byte_size_ = core_.ByteSize() + dictionary_.ByteSize();
    }
    const bool
    HasRawData() const override {
        return false;
    }
    bool
    IsMmapSupported() const override {
        return false;
    }
    bool
    SupportFastReverseLookup() const override {
        return false;
    }
    const Core*
    CoreForUT() const {
        return &core_;
    }
    const TargetBitmap
    In(size_t n, const T* values) override {
        tracer::AutoSpan span("InvertedIndexKnowhere::In",
                              tracer::GetRootSpan());
        return core_.In(n, values);
    }
    const TargetBitmap
    NotIn(size_t n, const T* values) override {
        tracer::AutoSpan span("InvertedIndexKnowhere::NotIn",
                              tracer::GetRootSpan());
        return core_.NotIn(n, values);
    }
    const TargetBitmap
    IsNull() override {
        tracer::AutoSpan span("InvertedIndexKnowhere::IsNull",
                              tracer::GetRootSpan());
        return core_.IsNull();
    }
    TargetBitmap
    IsNotNull() override {
        tracer::AutoSpan span("InvertedIndexKnowhere::IsNotNull",
                              tracer::GetRootSpan());
        return core_.IsNotNull();
    }
    const TargetBitmap
    Range(const T& value, OpType op) override {
        tracer::AutoSpan span("InvertedIndexKnowhere::Range",
                              tracer::GetRootSpan());
        return core_.Range(value, op);
    }
    const TargetBitmap
    Range(const T& lo, bool li, const T& hi, bool ui) override {
        tracer::AutoSpan span("InvertedIndexKnowhere::Range",
                              tracer::GetRootSpan());
        return core_.Range(lo, li, hi, ui);
    }
    const TargetBitmap
    InApplyFilter(size_t n,
                  const T* values,
                  const std::function<bool(size_t)>& filter) override {
        auto result = In(n, values);
        apply_hits_with_filter(result, filter);
        return result;
    }
    void
    InApplyCallback(size_t n,
                    const T* values,
                    const std::function<void(size_t)>& callback) override {
        auto result = In(n, values);
        apply_hits_with_callback(result, callback);
    }
    bool
    SupportPatternMatch() const override {
        return std::is_same_v<T, std::string>;
    }
    const TargetBitmap
    PatternMatch(const std::string& pattern, proto::plan::OpType op) override {
        if constexpr (!std::is_same_v<T, std::string>) {
            return ScalarIndex<T>::PatternMatch(pattern, op);
        } else {
            TargetBitmap result(Count());
            auto emit = [&](std::string_view, uint32_t id) {
                core_.DecodeInto(id, result);
            };
            if (op == proto::plan::OpType::PrefixMatch) {
                core_.ForEachStringPrefix(pattern, emit);
            } else if (op == proto::plan::OpType::RegexMatch) {
                PartialRegexMatcher matcher(pattern);
                core_.ForEachStringPrefix(
                    "", [&](std::string_view term, uint32_t id) {
                        if (matcher(term))
                            core_.DecodeInto(id, result);
                    });
            } else {
                std::string like;
                switch (op) {
                    case proto::plan::OpType::Match:
                        like = pattern;
                        break;
                    case proto::plan::OpType::PostfixMatch:
                        like = "%" + EscapeLikePattern(pattern);
                        break;
                    case proto::plan::OpType::InnerMatch:
                        like = "%" + EscapeLikePattern(pattern) + "%";
                        break;
                    default:
                        return ScalarIndex<T>::PatternMatch(pattern, op);
                }
                // Validate the WHOLE pattern before any empty-index/prefix exit.
                RegexMatcher matcher(PatternMatchTranslator{}(like));
                // The resident core already owns sorted unique values. Scan
                // views inside the literal prefix range, without reconstructing
                // long strings through the FST. This does not read raw rows.
                core_.ForEachStringPrefix(
                    extract_fixed_prefix_from_pattern(like),
                    [&](std::string_view term, uint32_t id) {
                        if (matcher(term))
                            core_.DecodeInto(id, result);
                    });
            }
            return result;
        }
    }
    std::optional<T>
    Reverse_Lookup(size_t) const override {
        ThrowInfo(Unsupported,
                  "Knowhere scalar PoC does not support Reverse_Lookup");
    }
    BinarySet
    Serialize(const Config&) override {
        ThrowInfo(Unsupported,
                  "Knowhere scalar PoC does not support Serialize");
    }
    void
    Load(const BinarySet&, const Config& = {}) override {
        ThrowInfo(Unsupported, "Knowhere scalar PoC does not support Load");
    }
    void
    Load(tracer::TraceContext, const Config& = {}) override {
        ThrowInfo(Unsupported, "Knowhere scalar PoC does not support Load");
    }
    void
    Build(const Config& = {}) override {
        ThrowInfo(Unsupported, "Knowhere scalar PoC does not support Build");
    }
    IndexStatsPtr
    Upload(const Config& = {}) override {
        ThrowInfo(Unsupported, "Knowhere scalar PoC does not support Upload");
    }
    IndexStatsPtr
    UploadUnified(const Config&) override {
        ThrowInfo(Unsupported,
                  "Knowhere scalar PoC does not support UploadUnified");
    }
    void
    LoadUnified(const Config&, milvus::OpContext* = nullptr) override {
        ThrowInfo(Unsupported,
                  "Knowhere scalar PoC does not support LoadUnified");
    }

 private:
    Core core_;
    KnowhereSparsePostingCodec::Format format_;
    FstTermDictionary dictionary_;
};
}  // namespace milvus::index
