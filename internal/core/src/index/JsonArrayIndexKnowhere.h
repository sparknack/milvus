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
#include "index/ArrayConjunctionIndex.h"
#include <map>

namespace milvus::index {
// Resident JSON path ARRAY membership with row-domain flat adaptive postings.
// Separate array-shape validity from path existence for exact NOT semantics.
template <typename T>
class JsonArrayIndexKnowhere : public InvertedIndexKnowhere<T>,
                               public ArrayConjunctionIndex<T> {
 public:
    JsonArrayIndexKnowhere(const JsonCastType& cast_type,
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
        const auto expected = std::is_same_v<T, bool>     ? "ARRAY_BOOL"
                              : std::is_same_v<T, double> ? "ARRAY_DOUBLE"
                                                          : "ARRAY_VARCHAR";
        static_assert(std::is_same_v<T, bool> || std::is_same_v<T, double> ||
                      std::is_same_v<T, std::string>);
        if (cast_type.ToString() != expected)
            ThrowInfo(Unsupported,
                      "JSON array cast must match its typed backend");
    }

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        size_t n = 0;
        for (const auto& field : fields) n += field->get_num_rows();
        TargetBitmap exists(n, true), valid(n);
        std::vector<size_t> invalid;
        std::map<T, std::vector<uint32_t>> postings;
        size_t row = 0;
        for (const auto& field : fields) {
            AssertInfo(field->get_data_type() == DataType::JSON,
                       "JSON array builder requires JSON FieldData");
            for (int64_t i = 0; i < field->get_num_rows(); ++i, ++row) {
                if (schema_.nullable() && !field->is_valid(i)) {
                    invalid.push_back(row);
                    continue;
                }
                const auto* json = static_cast<const Json*>(field->RawValue(i));
                auto array = json->dom_doc().at_pointer(path_).get_array();
                if (array.error() == simdjson::SUCCESS)
                    valid.set(row);
                else
                    invalid.push_back(row);
            }
        }
        // Production extraction defines numeric conversion and ignores mixed
        // elements that do not match this path's array cast.
        ProcessJsonFieldData<T>(
            fields,
            schema_,
            path_,
            cast_type_,
            cast_function_,
            [&](const T* values, int64_t size, int64_t offset) {
                for (int64_t i = 0; i < size; ++i) {
                    auto& docs = postings[values[i]];
                    if (docs.empty() || docs.back() != offset)
                        docs.push_back(offset);
                }
            },
            [](int64_t) {},
            [&](int64_t offset) { exists.reset(offset); },
            [](const Json&, const std::string&, simdjson::error_code) {});
        InvertedIndexKnowhere<T>::BuildFromPostingsForPoC(n, postings, invalid);
        exists_ = std::move(exists);
        valid_ = std::move(valid);
        ComputeByteSize();
    }

    TargetBitmap
    All(size_t n, const T* values) override {
        if (n == 0)
            return valid_.clone();
        const auto* core = this->CoreForUT();
        TargetBitmap result(core->Count());
        std::vector<size_t> terms;
        terms.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto term = core->Lookup(values[i]);
            if (term == core->TermCount())
                return result;
            terms.push_back(term);
        }
        core->IntersectInto(std::move(terms), result);
        return result;
    }

    // JSON extraction must own every build so its two validity domains cannot
    // become stale after a caller supplies only typed scalar values.
    void
    Build(size_t, const T*, const bool* = nullptr) override {
        ThrowInfo(Unsupported, "JSON array PoC requires JSON FieldData build");
    }
    void
    BuildWithRawDataForUT(size_t, const void*, const Config& = {}) override {
        ThrowInfo(Unsupported, "JSON array PoC requires JSON FieldData build");
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
