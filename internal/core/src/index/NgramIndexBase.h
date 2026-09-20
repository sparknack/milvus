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
#include "pb/plan.pb.h"
namespace milvus::exec {
class SegmentExpr;
}
namespace milvus::index {
// Query-only contract. Ownership remains with the cached ScalarIndex object.
class NgramIndexBase {
 public:
    virtual ~NgramIndexBase() = default;
    virtual int64_t
    Count() = 0;
    virtual TargetBitmap
    IsNotNull() = 0;
    virtual bool
    CanHandleLiteral(const std::string&, proto::plan::OpType) const = 0;
    virtual void
    ExecutePhase1(const std::string&, proto::plan::OpType, TargetBitmap&) = 0;
    virtual void
    ExecutePhase2(const std::string&,
                  proto::plan::OpType,
                  exec::SegmentExpr*,
                  TargetBitmap&,
                  int64_t,
                  int64_t) = 0;
};
}  // namespace milvus::index
