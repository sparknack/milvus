// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "index/InvertedIndexKnowhere.h"
#include "index/ArrayConjunctionIndex.h"
#include "common/Array.h"
#include <map>
#include <cmath>

namespace milvus::index {
template <typename T>
constexpr DataType
KnowhereArrayElementType() {
    if constexpr (std::is_same_v<T, bool>)
        return DataType::BOOL;
    if constexpr (std::is_same_v<T, int8_t>)
        return DataType::INT8;
    if constexpr (std::is_same_v<T, int16_t>)
        return DataType::INT16;
    if constexpr (std::is_same_v<T, int32_t>)
        return DataType::INT32;
    if constexpr (std::is_same_v<T, int64_t>)
        return DataType::INT64;
    if constexpr (std::is_same_v<T, float>)
        return DataType::FLOAT;
    if constexpr (std::is_same_v<T, double>)
        return DataType::DOUBLE;
    return DataType::VARCHAR;
}
// Ordinary ARRAY membership: one document per row, not per element.
// The resident payload is the same flat adaptive postings as scalar/text PoCs.
template <typename T>
class InvertedIndexKnowhereArray : public InvertedIndexKnowhere<T>,
                                   public ArrayConjunctionIndex<T> {
    static_assert(std::is_same_v<T, bool> || std::is_same_v<T, int8_t> ||
                      std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, int64_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::string>,
                  "unsupported ARRAY scalar element type");

 public:
    explicit InvertedIndexKnowhereArray(bool nested = false)
        : InvertedIndexKnowhere<T>(
              KnowhereSparsePostingCodec::Format::Adaptive),
          nested_(nested) {
    }

    bool
    IsNestedIndex() const override {
        return nested_;
    }

    TargetBitmap
    All(size_t n, const T* values) override {
        AssertInfo(!nested_,
                   "row conjunction cannot query an element-domain index");
        if (n == 0)
            return this->IsNotNull();
        const auto* core = this->CoreForUT();
        TargetBitmap result(core->Count());
        std::vector<size_t> terms;
        terms.reserve(n);
        // Resolve every term before decoding any posting. A missing term
        // makes the exact conjunction empty regardless of query term order.
        for (size_t i = 0; i < n; ++i) {
            auto term = core->Lookup(values[i]);
            if (term == core->TermCount())
                return result;
            terms.push_back(term);
        }
        // IntersectInto deduplicates ordinals, orders by document frequency,
        // and chooses exact cursor seeking or dense bitmap intersection.
        // It ORs hits into its destination, so the destination must be zero.
        core->IntersectInto(std::move(terms), result);
        return result;
    }

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        std::map<T, std::vector<uint32_t>> postings;
        std::vector<size_t> nulls;
        size_t row = 0, element = 0;
        for (const auto& field : fields) {
            AssertInfo(field->get_data_type() == DataType::ARRAY,
                       "ARRAY builder requires ARRAY FieldData");
            for (int64_t i = 0; i < field->get_num_rows(); ++i, ++row) {
                if (!field->is_valid(i)) {
                    if (!nested_)
                        nulls.push_back(row);
                    continue;
                }
                const auto& array =
                    *static_cast<const Array*>(field->RawValue(i));
                AssertInfo(
                    !array.is_element_nullable(),
                    "resident ARRAY PoC does not support nullable elements");
                // Protobuf ScalarField string_data constructs an Array with
                // physical STRING, even when its schema is ARRAY<VARCHAR>.
                if constexpr (std::is_same_v<T, std::string>) {
                    AssertInfo(IsStringDataType(array.get_element_type()),
                               "ARRAY builder requires string elements");
                } else {
                    // INT8/INT16 arrays use the protobuf int_data / INT32
                    // physical payload; Array performs the typed conversion.
                    constexpr auto physical_type =
                        (std::is_same_v<T, int8_t> ||
                         std::is_same_v<T, int16_t>)
                            ? DataType::INT32
                            : KnowhereArrayElementType<T>();
                    AssertInfo(array.get_element_type() == physical_type,
                               "ARRAY builder element physical type mismatch");
                }
                for (int j = 0; j < array.length(); ++j) {
                    auto value = array.template get_data_unchecked<T>(j);
                    // Validate before std::map comparison: NaN compares
                    // neither less nor greater than an existing finite key,
                    // so inserting first could silently merge their postings.
                    if constexpr (std::is_floating_point_v<T>) {
                        if (std::isnan(value))
                            ThrowInfo(Unsupported,
                                      "PoC does not support NaN terms");
                    }
                    auto& docs = postings[value];
                    // The row IDs are increasing. Checking the last ID avoids
                    // a per-row hash set and makes duplicate values idempotent.
                    if (nested_) {
                        // Preserve duplicate occurrences as different element
                        // IDs. Segcore maps the field's offsets back to rows.
                        docs.push_back(element++);
                    } else if (docs.empty() || docs.back() != row) {
                        docs.push_back(row);
                    }
                }
            }
        }
        this->BuildFromPostingsForPoC(nested_ ? element : row, postings, nulls);
    }

 private:
    const bool nested_;
};
}  // namespace milvus::index
