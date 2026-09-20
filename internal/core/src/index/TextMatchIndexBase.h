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
#include "common/Types.h"
#include "common/FieldData.h"
#include "common/EasyAssert.h"

namespace milvus::index {
// Shared text-index contract. Storage and analyzer ownership belong to implementations.
class TextMatchIndexBase {
 public:
    virtual ~TextMatchIndexBase() = default;
    virtual int64_t
    Count() = 0;
    virtual int64_t
    ByteSize() const = 0;
    virtual int64_t
    ValidityBitmapByteSize() const = 0;
    virtual TargetBitmap
    IsNotNull() = 0;
    virtual TargetBitmap
    MatchQuery(const std::string& query, uint32_t minimum) = 0;
    virtual TargetBitmap
    PhraseMatchQuery(const std::string&, uint32_t) {
        ThrowInfo(Unsupported,
                  "phrase match is not supported by this text index");
    }
    virtual TargetBitmap
    FuzzyMatchQuery(const std::string&, uint32_t) {
        ThrowInfo(Unsupported,
                  "fuzzy match is not supported by this text index");
    }
    // Growing ingestion is an optional capability; sealed-only implementations reject it.
    virtual void
    AddTextsGrowing(size_t, const std::string*, const bool*, int64_t) {
        ThrowInfo(Unsupported, "growing text ingestion is not supported");
    }
    virtual void
    BuildIndexFromFieldData(const std::vector<FieldDataPtr>&, bool, int64_t) {
        ThrowInfo(Unsupported, "field-data text ingestion is not supported");
    }
    virtual void
    Commit() {
        ThrowInfo(Unsupported, "text index commit is not supported");
    }
    virtual void
    Reload() {
        ThrowInfo(Unsupported, "text index reload is not supported");
    }
};
}  // namespace milvus::index
