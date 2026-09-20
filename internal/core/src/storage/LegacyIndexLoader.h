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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>

#include "filemanager/InputStream.h"
#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"
#include "milvus-storage/filesystem/fs.h"
#include "pb/common.pb.h"
#include "storage/ChunkManager.h"
#include "storage/IndexEntryTarget.h"

namespace milvus::storage {

struct LegacyIndexFileInfo {
    size_t file_bytes{0};
    size_t payload_bytes{0};
    size_t payload_offset{0};
    size_t max_transient_bytes{0};
    bool raw_payload{false};
};

/** @brief Inspect envelope metadata with inline range reads under admission. */
LegacyIndexFileInfo
InspectLegacyIndexFile(const ChunkManagerPtr& chunk_manager,
                       const std::string& path,
                       proto::common::LoadPriority priority,
                       folly::CancellationToken token = {});

struct LegacyIndexFile {
    std::string path;
    LegacyIndexFileInfo info;
};
}  // namespace milvus::storage
