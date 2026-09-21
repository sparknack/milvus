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
#include "index/ScalarIndex.h"

namespace milvus::index {
// Optional query capability for exact row-domain ARRAY membership conjunctions.
// Implementations must explicitly provide row-domain membership semantics.
template <typename T>
class ArrayConjunctionIndex {
 public:
    virtual ~ArrayConjunctionIndex() = default;
    // Empty conjunction returns all valid rows; no approximate candidates.
    virtual TargetBitmap
    All(size_t n, const T* values) = 0;
};

template <typename T>
ArrayConjunctionIndex<T>*
GetArrayConjunctionIndex(ScalarIndex<T>* index,
                         DataType field_type,
                         int64_t active_count) {
    const bool row_array =
        field_type == DataType::ARRAY ||
        (field_type == DataType::JSON &&
         index->GetCastType().data_type() == JsonCastType::DataType::ARRAY);
    if (!row_array || index->IsNestedIndex() || index->Count() != active_count)
        return nullptr;
    return dynamic_cast<ArrayConjunctionIndex<T>*>(index);
}
}  // namespace milvus::index
