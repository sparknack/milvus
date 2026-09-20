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

#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"
#include "index/contracts/Registry.h"
#include "storage/AsyncIndexEntryReader.h"
#include "storage/IndexEntryReader.h"
#include "storage/FileManager.h"

namespace milvus::index {

// Open metadata using the context's pinned load mode.
std::unique_ptr<storage::AsyncIndexEntryReader>
InspectPackedIndexFile(const std::vector<std::string>& files,
                       const storage::FileManagerContext& context,
                       bool is_index_file = true);

IIndexReaderBasePtr
LoadPackedIndexFile(const LoaderEntry& loader,
                    const storage::FileManagerContext& context,
                    const std::string& path,
                    const storage::LoadOptions& options,
                    bool is_index_file = true);

// Both paths consume the same family plan and construct the same reader.
// The caller keeps the opened packed reader and options alive until completion.
// Failure drains reads and destroys uncommitted file targets before returning.
IIndexReaderBasePtr
LoadPackedIndex(const LoaderEntry& loader,
                storage::IndexEntryReader& source,
                const storage::LoadOptions& options,
                folly::CancellationToken token = {});

folly::coro::Task<IIndexReaderBasePtr>
LoadPackedIndexAsync(const LoaderEntry& loader,
                     storage::AsyncIndexEntryReader& source,
                     const storage::LoadOptions& options,
                     proto::common::LoadPriority priority,
                     folly::CancellationToken token = {});

}  // namespace milvus::index
