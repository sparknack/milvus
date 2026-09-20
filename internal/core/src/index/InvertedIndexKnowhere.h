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

namespace milvus::index {
// Resident scalar PoC. Intentionally independent of Tantivy and not factory registered.
template <typename T>
class InvertedIndexKnowhere : public ScalarIndex<T> {
 public:
    using Core = InvertedIndexKnowhereCore<T, TargetBitmap, OpType>;
    using ScalarIndex<T>::IsNotNull;
    InvertedIndexKnowhere() : ScalarIndex<T>(INVERTED_INDEX_TYPE) {
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
        core_.Build(n, values, nulls);
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
        this->cached_byte_size_ = core_.ByteSize();
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
};
}  // namespace milvus::index
