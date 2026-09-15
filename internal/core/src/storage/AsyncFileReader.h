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
#include <string>
#include "arrow/io/interfaces.h"
#include "arrow/filesystem/filesystem.h"
#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"

namespace milvus::storage {

// Filesystem open can issue a synchronous HEAD, even via OpenInputFileAsync().
// Run it on the filesystem's I/O executor and drain it before honoring cancel.
[[nodiscard]] folly::coro::Task<std::shared_ptr<arrow::io::RandomAccessFile>>
OpenInputFileAsync(std::shared_ptr<arrow::fs::FileSystem> fs,
                   std::string path,
                   folly::CancellationToken token);

// Resolves file size through the native async API when available. Generic
// files use their I/O executor. Issued requests drain on cancel.
[[nodiscard]] folly::coro::Task<int64_t>
GetFileSizeAsync(arrow::io::RandomAccessFile& file,
                 folly::CancellationToken token);

// Reads exactly bytes into caller-owned storage on the awaiting executor.
// The caller validates file bounds and owns admission and destination lifetime.
// Cancellation drains an issued read before returning; typed storage status
// details are retained through the existing bounded read retries.
[[nodiscard]] folly::coro::Task<void>
ReadFileRangeAsync(arrow::io::RandomAccessFile& file,
                   uint64_t offset,
                   uint8_t* destination,
                   size_t bytes,
                   folly::CancellationToken token);

}  // namespace milvus::storage
