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

#include "index/contracts/Registry.h"
#include "storage/FileManager.h"

namespace milvus::index {

// Open the logical V1/V2 entry directory, fetching slice metadata with the
// context's pinned transport mode. Does not prefetch engine payloads.
std::unique_ptr<storage::V1RemoteSource>
OpenLegacyIndexSource(const storage::FileManagerContext& context,
                      const std::vector<std::string>& paths,
                      const storage::LoadOptions& options,
                      storage::V1SourceLayout layout);

// Execute a legacy loader on the loading executor. Remote reads/admission
// suspend; families offload blocking file phases and retain their owners.
folly::coro::Task<IIndexReaderBasePtr>
LoadLegacyIndexAsync(const LoaderEntry& loader,
                     storage::FileSource& source,
                     const storage::LoadOptions& options);

// Cache boundary: choose the pinned mode once, open the source and load it.
IIndexReaderBasePtr
LoadLegacyIndexFile(const LoaderEntry& loader,
                    const storage::FileManagerContext& context,
                    const std::vector<std::string>& paths,
                    const storage::LoadOptions& options,
                    storage::V1SourceLayout layout);

}  // namespace milvus::index
