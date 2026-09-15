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

#include "storage/AsyncFileReader.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include "arrow/buffer.h"
#include "arrow/util/thread_pool.h"
#include "folly/coro/Promise.h"
#include "folly/Try.h"
#include "folly/coro/WithCancellation.h"
#include "folly/futures/Future.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "storage/EntryStreamUtils.h"

namespace milvus::storage {
namespace {

// Keep the Arrow status intact for retry classification. The coroutine does
// not abandon a caller-owned destination while the Arrow future can write it.
template <typename T>
folly::coro::Future<arrow::Result<T>>
AwaitFileResult(arrow::Future<T> arrow_future) {
    auto [promise, future] =
        folly::coro::makePromiseContract<arrow::Result<T>>();
    auto completion = std::make_shared<folly::coro::Promise<arrow::Result<T>>>(
        std::move(promise));
    arrow_future.AddCallback([completion](const arrow::Result<T>& result) {
        completion->setValue(result);
    });
    return future;
}

// Arrow's task runner does not transport thrown C++ exceptions. Complete a
// Folly promise so filesystem exceptions retain their types across the I/O hop.
template <typename Fn>
auto
RunFileIO(arrow::internal::Executor& executor, Fn fn) {
    using Result = std::invoke_result_t<Fn>;
    auto [promise, future] = folly::coro::makePromiseContract<Result>();
    auto completion =
        std::make_shared<folly::coro::Promise<Result>>(std::move(promise));
    auto status = executor.Spawn([completion, fn = std::move(fn)]() mutable {
        completion->setResult(folly::makeTryWith(std::move(fn)));
    });
    if (!status.ok()) {
        throw milvus_storage::ToSegcoreError(status);
    }
    return future;
}

bool
IsRetryableRead(const arrow::Status& status) {
    if (auto detail =
            milvus_storage::ExtendStatusDetail::UnwrapStatus(status)) {
        return detail->retryable();
    }
    return status.ToString().find("Failed to flush response stream") !=
           std::string::npos;
}

}  // namespace

folly::coro::Task<std::shared_ptr<arrow::io::RandomAccessFile>>
OpenInputFileAsync(std::shared_ptr<arrow::fs::FileSystem> fs,
                   std::string path,
                   folly::CancellationToken token) {
    ThrowIfCancelled(token, "OpenInputFileAsync");
    auto pending = RunFileIO(
        *fs->io_context().executor(),
        [fs, path = std::move(path)] { return fs->OpenInputFile(path); });
    auto result = co_await folly::coro::co_withCancellation(
        folly::CancellationToken{}, std::move(pending));
    ThrowIfCancelled(token, "OpenInputFileAsync");
    if (!result.ok()) {
        throw milvus_storage::ToSegcoreError(result.status());
    }
    co_return std::move(result).ValueOrDie();
}

folly::coro::Task<int64_t>
GetFileSizeAsync(arrow::io::RandomAccessFile& file,
                 folly::CancellationToken token) {
    ThrowIfCancelled(token, "GetFileSizeAsync");
    auto* native =
        dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(&file);
    auto pending = native != nullptr
                       ? AwaitFileResult(native->GetSizeAsync())
                       : RunFileIO(*file.io_context().executor(),
                                   [&file] { return file.GetSize(); });
    auto result = co_await folly::coro::co_withCancellation(
        folly::CancellationToken{}, std::move(pending));
    ThrowIfCancelled(token, "GetFileSizeAsync");
    if (!result.ok()) {
        throw milvus_storage::ToSegcoreError(result.status());
    }
    AssertInfo(*result >= 0, "Negative async file size: {}", *result);
    co_return *result;
}

folly::coro::Task<void>
ReadFileRangeAsync(arrow::io::RandomAccessFile& file,
                   uint64_t offset,
                   uint8_t* destination,
                   size_t bytes,
                   folly::CancellationToken token) {
    if (bytes == 0) {
        co_return;
    }
    auto* native =
        dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(&file);
    for (int attempt = 0;; ++attempt) {
        ThrowIfCancelled(token, "ReadFileRangeAsync");
        auto future =
            native != nullptr
                ? native->ReadAtAsyncInto(offset, bytes, destination)
                : file.ReadAsync(arrow::io::default_io_context(), offset, bytes)
                      .Then([destination, bytes](
                                const std::shared_ptr<arrow::Buffer>& buffer)
                                -> arrow::Result<int64_t> {
                          if (buffer == nullptr || buffer->size() != bytes) {
                              return arrow::Status::IOError(
                                  "Short buffered async range read");
                          }
                          std::memcpy(destination, buffer->data(), bytes);
                          return bytes;
                      });
        auto result = co_await folly::coro::co_withCancellation(
            folly::CancellationToken{}, AwaitFileResult(std::move(future)));
        ThrowIfCancelled(token, "ReadFileRangeAsync");
        if (result.ok()) {
            AssertInfo(*result == bytes,
                       "Short native async range read: expected {}, got {}",
                       bytes,
                       *result);
            co_return;
        }
        if (!IsRetryableRead(result.status()) || attempt == 5) {
            throw milvus_storage::ToSegcoreError(result.status());
        }
        co_await folly::futures::sleep(std::chrono::milliseconds(1 << attempt));
    }
}

}  // namespace milvus::storage
