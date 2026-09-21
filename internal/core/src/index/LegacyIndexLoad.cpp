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

#include "index/LegacyIndexLoad.h"

#include "common/OpContext.h"
#include "folly/coro/BlockingWait.h"
#include "folly/coro/WithCancellation.h"
#include "segcore/storagev2translator/StorageV2Config.h"
#include "storage/AsyncLoadExecutor.h"
#include "storage/EntryStreamUtils.h"
#include "storage/LocalFileIOPool.h"

namespace milvus::index {
namespace {
proto::common::LoadPriority
LoadPriority(const storage::LoadOptions& options) {
    return options.op_ctx &&
                   options.op_ctx->runtime_load_priority.value_or(0) != 0
               ? proto::common::LoadPriority::LOW
               : proto::common::LoadPriority::HIGH;
}

bool
UseAsync(const storage::FileManagerContext& context) {
    return context.use_async_load.value_or(
        segcore::storagev2translator::StorageV2AsyncLoadEnabled());
}
}  // namespace

std::unique_ptr<storage::V1RemoteSource>
OpenLegacyIndexSource(const storage::FileManagerContext& context,
                      const std::vector<std::string>& paths,
                      const storage::LoadOptions& options,
                      storage::V1SourceLayout layout) {
    if (!UseAsync(context)) {
        return std::make_unique<storage::V1RemoteSource>(
            context,
            paths,
            options,
            storage::ArtifactStoragePath::Index,
            layout);
    }
    return folly::coro::blockingWait(folly::coro::co_withExecutor(
        storage::ResolveAsyncLoadExecutor({}, LoadPriority(options)),
        storage::V1RemoteSource::OpenAsync(context,
                                           paths,
                                           options,
                                           storage::ArtifactStoragePath::Index,
                                           layout)));
}

folly::coro::Task<IIndexReaderBasePtr>
LoadLegacyIndexAsync(const LoaderEntry& loader,
                     storage::FileSource& source,
                     const storage::LoadOptions& options) {
    AssertInfo(loader.open_async != nullptr,
               "index family has no legacy async loader");
    // Keep the conditional out of the co_await expression: GCC 12 can
    // evaluate its null branch incorrectly while lowering the coroutine.
    const auto operation_token = options.op_ctx
                                     ? options.op_ctx->cancellation_token
                                     : folly::CancellationToken{};
    const auto token = folly::cancellation_token_merge(
        operation_token, co_await folly::coro::co_current_cancellation_token);
    auto load = [&]() -> folly::coro::Task<IIndexReaderBasePtr> {
        storage::ThrowIfCancelled(token, "open legacy index");
        auto reader = co_await loader.open_async(source, options);
        AssertInfo(reader != nullptr, "legacy loader returned a null reader");
        storage::ThrowIfCancelled(token, "publish legacy index");
        co_return std::move(reader);
    };
    co_return co_await folly::coro::co_withCancellation(
        token,
        folly::coro::co_withExecutor(
            storage::ResolveAsyncLoadExecutor({}, LoadPriority(options)),
            load()));
}

IIndexReaderBasePtr
LoadLegacyIndexFile(const LoaderEntry& loader,
                    const storage::FileManagerContext& context,
                    const std::vector<std::string>& paths,
                    const storage::LoadOptions& options,
                    storage::V1SourceLayout layout) {
    if (!UseAsync(context)) {
        auto source = OpenLegacyIndexSource(context, paths, options, layout);
        return loader.open(*source, options);
    }
    auto load = [&]() -> folly::coro::Task<IIndexReaderBasePtr> {
        auto source = co_await storage::V1RemoteSource::OpenAsync(
            context,
            paths,
            options,
            storage::ArtifactStoragePath::Index,
            layout);
        co_return co_await LoadLegacyIndexAsync(loader, *source, options);
    };
    return folly::coro::blockingWait(folly::coro::co_withExecutor(
        storage::ResolveAsyncLoadExecutor({}, LoadPriority(options)), load()));
}

}  // namespace milvus::index
