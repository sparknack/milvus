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

#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"
#include <string_view>

#include "index/Families.h"
#include "index/IndexLoadPlan.h"
#include "storage/IndexEntryFormat.h"
#include "index/contracts/query/IIndexReaderBase.h"
#include "storage/artifact/FileSource.h"
#include "storage/artifact/LoadOptions.h"

// The LOADER of the spatial family. The legacy wire has no row-count
// field, so Open requires authoritative runtime `num_rows` (or the
// existing `index_num_rows` alias) in LoadOptions::params.

namespace milvus::index {

class RTreeIndexLoader final {
 public:
    static constexpr std::string_view kFamily = families::kRTree;

    static ReaderCaps
    DeriveCaps(const Config& index_meta);

    static IIndexReaderBasePtr
    Open(storage::FileSource& source, const storage::LoadOptions& opts);
    // Describe direct V3 destinations; caller performs and drains all reads.
    static IndexLoadPlan
    PlanPacked(const storage::IndexEntryDirectory& directory,
               const nlohmann::json& metadata,
               const storage::LoadOptions& opts);

    // Initialize from completed targets. Directory remains owned on failure.
    // Async switches only blocking file phases; synchronous loading stays inline.
    static folly::coro::Task<IIndexReaderBasePtr>
    FinishPacked(IndexLoadPlan& plan,
                 const storage::LoadOptions& opts,
                 bool use_async,
                 folly::CancellationToken token);
};

}  // namespace milvus::index
