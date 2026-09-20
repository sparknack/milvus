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

#include "index/InvertedIndexKnowhere.h"
#include "index/JsonIndexBuilder.h"

namespace milvus::index {
// Resident, one-path scalar JSON PoC. Extraction and casting deliberately use
// the same production converter as JsonScalarIndexWrapper; no Tantivy base.
template <typename T>
class JsonPathIndexKnowhere : public InvertedIndexKnowhere<T> {
 public:
    JsonPathIndexKnowhere(const JsonCastType& cast_type,
                          std::string path,
                          const proto::schema::FieldSchema& schema,
                          JsonCastFunction cast_function =
                              JsonCastFunction::FromString("unknown"))
        : InvertedIndexKnowhere<T>(
              KnowhereSparsePostingCodec::Format::Adaptive),
          cast_type_(cast_type),
          path_(std::move(path)),
          schema_(schema),
          cast_function_(cast_function) {
        if (cast_type.data_type() == JsonCastType::DataType::ARRAY)
            ThrowInfo(Unsupported,
                      "JSON path scalar PoC does not support ARRAY casts");
    }

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        auto converted = ConvertJsonToTypedFieldData<T>(
            fields, schema_, path_, cast_type_, cast_function_);
        auto& field = converted.field_data;
        const auto n = field->get_num_rows();
        FixedVector<T> values(n);
        std::vector<size_t> invalid;
        TargetBitmap exists(n, true), valid(n, true);
        for (size_t row = 0; row < n; ++row) {
            if (field->is_valid(row))
                values[row] = *static_cast<const T*>(field->RawValue(row));
            else {
                invalid.push_back(row);
                valid.reset(row);
            }
        }
        for (auto row : converted.non_exist_offsets) exists.reset(row);
        // All allocations precede publication. Base construction is transactional;
        // bitmap moves below cannot allocate. Rebuilds require exclusive access.
        InvertedIndexKnowhere<T>::BuildWithNullOffsetsForUT(
            n, values.data(), invalid);
        exists_ = std::move(exists);
        valid_ = std::move(valid);
        ComputeByteSize();
    }

    // JSON extraction must own every build so its two validity domains cannot
    // become stale after a caller supplies only typed scalar values.
    void
    Build(size_t, const T*, const bool* = nullptr) override {
        ThrowInfo(Unsupported, "JSON path PoC requires JSON FieldData build");
    }
    void
    BuildWithRawDataForUT(size_t, const void*, const Config& = {}) override {
        ThrowInfo(Unsupported, "JSON path PoC requires JSON FieldData build");
    }
    TargetBitmap
    IsNotNull() override {
        return valid_.clone();
    }
    const TargetBitmap
    IsNull() override {
        auto result = valid_.clone();
        result.flip();
        return result;
    }
    const TargetBitmap
    NotIn(size_t n, const T* values) override {
        auto result = InvertedIndexKnowhere<T>::In(n, values);
        result.flip();
        result &= valid_;
        return result;
    }
    JsonCastType
    GetCastType() const override {
        return cast_type_;
    }
    TargetBitmap
    Exists() override {
        return exists_.clone();
    }
    void
    ComputeByteSize() override {
        InvertedIndexKnowhere<T>::ComputeByteSize();
        this->cached_byte_size_ +=
            exists_.size_in_bytes() + valid_.size_in_bytes();
    }

 private:
    // Hide nonvirtual scalar-only PoC entry points on the JSON adapter.
    using InvertedIndexKnowhere<T>::BuildWithNullOffsetsForUT;
    using InvertedIndexKnowhere<T>::BuildFromPostingsForPoC;
    JsonCastType cast_type_;
    std::string path_;
    proto::schema::FieldSchema schema_;
    JsonCastFunction cast_function_;
    TargetBitmap exists_;
    TargetBitmap valid_;
};
}  // namespace milvus::index
