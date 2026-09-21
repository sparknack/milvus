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

#include <string_view>

#include "index/Families.h"
#include "index/contracts/query/IIndexReaderBase.h"
#include "storage/artifact/FileSource.h"
#include "storage/artifact/LoadOptions.h"

namespace milvus::index {

// Stateless V1/V2 disk-vector loader. Non-streaming backends are materialized
// into one unique DiskFileManager generation before knowhere Deserialize.
// Streaming backends open only exact raw, unsliced objects advertised by the
// current artifact inventory; small legacy sidecars still use FileSource.
class VectorDiskLoader final {
 public:
    static constexpr std::string_view kFamily = families::kVectorDisk;

    static ReaderCaps
    DeriveCaps(const Config& index_meta);

    static IIndexReaderBasePtr
    Open(storage::FileSource& source, const storage::LoadOptions& opts);

    // Borrow source/options until completion. Run on the local-file executor:
    // remote reads suspend, while native initialization and mmap remain local.
    static folly::coro::Task<IIndexReaderBasePtr>
    OpenAsync(storage::FileSource& source, const storage::LoadOptions& opts);

    // Select the native backend before inspecting remote envelopes. Stream
    // backends prepare only sidecars; engine files remain lazily accessible.
    static std::vector<std::string>
    AsyncEntryNames(storage::FileSource& source,
                    const storage::LoadOptions& opts);

 private:
    // Share validation, decoding and ownership across both transport modes.
    static folly::coro::Task<IIndexReaderBasePtr>
    OpenImpl(bool use_async,
             storage::FileSource& source,
             const storage::LoadOptions& opts);
};

}  // namespace milvus::index
