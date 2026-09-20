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

#include "index/PackedIndexLoad.h"

#include <cstring>
#include <exception>
#include <optional>

#include "folly/coro/WithCancellation.h"
#include "storage/EntryStreamUtils.h"
#include "storage/LocalFileIOPool.h"

#include "folly/coro/BlockingWait.h"
#include "common/OpContext.h"
#include "segcore/storagev2translator/StorageV2Config.h"
#include "storage/AsyncLoadExecutor.h"
#include "storage/MemFileManagerImpl.h"

namespace milvus::index {
namespace {
folly::coro::Task<std::unique_ptr<storage::AsyncIndexEntryReader>>
InspectPackedScalarIndex(const std::vector<std::string>& files,
                         const storage::FileManagerContext& context,
                         bool is_index_file,
                         bool use_async_load) {
    AssertInfo(files.size() == 1 && context.Valid(),
               "Async scalar load requires one V3 file and a valid context");
    storage::MemFileManagerImpl manager(context);
    std::shared_ptr<InputStream> input;
    if (use_async_load) {
        input = co_await folly::coro::co_withCancellation(
            folly::CancellationToken{},
            manager.OpenInputStreamAsync(files.front(), is_index_file));
    } else {
        input = manager.OpenInputStream(files.front(), is_index_file);
    }
    AssertInfo(input != nullptr, "Failed to open packed scalar index");
    co_return co_await storage::AsyncIndexEntryReader::Open(
        std::move(input),
        context.fieldDataMeta.collection_id,
        proto::common::LoadPriority::HIGH,
        {});
}
}  // namespace

std::unique_ptr<storage::AsyncIndexEntryReader>
InspectPackedIndexFile(const std::vector<std::string>& files,
                       const storage::FileManagerContext& context,
                       bool is_index_file) {
    const bool use_async = context.use_async_load.value_or(
        segcore::storagev2translator::StorageV2AsyncLoadEnabled());
    auto task =
        InspectPackedScalarIndex(files, context, is_index_file, use_async);
    return use_async ? folly::coro::blockingWait(folly::coro::co_withExecutor(
                           storage::ResolveAsyncLoadExecutor(
                               {}, proto::common::LoadPriority::HIGH),
                           std::move(task)))
                     : folly::coro::blockingWait(std::move(task));
}

IIndexReaderBasePtr
LoadPackedIndexFile(const LoaderEntry& loader,
                    const storage::FileManagerContext& context,
                    const std::string& path,
                    const storage::LoadOptions& options,
                    bool is_index_file) {
    const auto priority =
        options.op_ctx && options.op_ctx->runtime_load_priority.value_or(0) != 0
            ? proto::common::LoadPriority::LOW
            : proto::common::LoadPriority::HIGH;
    const auto token = options.op_ctx ? options.op_ctx->cancellation_token
                                      : folly::CancellationToken{};
    storage::ThrowIfCancelled(token, "open packed index");
    storage::MemFileManagerImpl manager(context);
    if (context.use_async_load.value_or(
            segcore::storagev2translator::StorageV2AsyncLoadEnabled())) {
        auto load = [&]() -> folly::coro::Task<IIndexReaderBasePtr> {
            auto opened = co_await folly::coro::co_awaitTry(
                folly::coro::co_withCancellation(
                    folly::CancellationToken{},
                    manager.OpenInputStreamAsync(path, is_index_file)));
            storage::ThrowIfCancelled(token, "open packed index");
            auto input = std::move(opened).value();
            AssertInfo(
                input != nullptr, "Failed to open packed index {}", path);
            auto source = co_await storage::AsyncIndexEntryReader::Open(
                std::move(input),
                context.fieldDataMeta.collection_id,
                priority,
                token);
            co_return co_await LoadPackedIndexAsync(
                loader, *source, options, priority, token);
        };
        return folly::coro::blockingWait(folly::coro::co_withExecutor(
            storage::ResolveAsyncLoadExecutor({}, priority), load()));
    }
    auto input = manager.OpenInputStream(path, is_index_file);
    AssertInfo(input != nullptr, "Failed to open packed index {}", path);
    auto source = storage::IndexEntryReader::Open(
        input,
        input->Size(),
        context.fieldDataMeta.collection_id,
        priority == proto::common::LoadPriority::HIGH ? ThreadPoolPriority::HIGH
                                                      : ThreadPoolPriority::LOW,
        token);
    return LoadPackedIndex(loader, *source, options, token);
}

namespace {

// Finalization owns its await: mappings and destinations survive batch yields.
folly::coro::Task<IIndexReaderBasePtr>
FinishPackedLoad(const LoaderEntry& loader,
                 IndexLoadPlan& plan,
                 const storage::LoadOptions& options,
                 bool use_async,
                 folly::CancellationToken token) {
    if (const auto* sync = std::get_if<LoaderEntry::FinishPackedSyncFn>(
            &loader.finish_packed)) {
        co_return (*sync)(plan, options);
    }
    co_return co_await std::get<LoaderEntry::FinishPackedAsyncFn>(
        loader.finish_packed)(plan, options, use_async, std::move(token));
}

void
RequirePackedLoader(const LoaderEntry& loader) {
    AssertInfo(
        loader.plan_packed && std::visit([](auto fn) { return fn != nullptr; },
                                         loader.finish_packed),
        "index family has no packed load implementation");
}

}  // namespace

IIndexReaderBasePtr
LoadPackedIndex(const LoaderEntry& loader,
                storage::IndexEntryReader& source,
                const storage::LoadOptions& options,
                folly::CancellationToken token) {
    RequirePackedLoader(loader);
    storage::ThrowIfCancelled(token, "plan packed index");
    auto plan =
        loader.plan_packed(source.Directory(), source.IndexMeta(), options);
    const auto files = storage::CollectIndexFileTargets(plan.entries);
    for (const auto& file : files) {
        file->Prepare(storage::io::GetPriorityFromLoadPriority(
            options.op_ctx &&
                    options.op_ctx->runtime_load_priority.value_or(0) != 0
                ? proto::common::LoadPriority::LOW
                : proto::common::LoadPriority::HIGH));
    }
    for (const auto& entry : plan.entries) {
        storage::ThrowIfCancelled(token, "read packed index");
        const auto expected = source.Directory().At(entry.name).plaintext_size;
        AssertInfo(expected <= storage::EntryTargetSize(entry.target),
                   "packed entry exceeds planned destination: {}",
                   entry.name);
        size_t offset = 0;
        source.ReadEntryStream(
            entry.name, [&](const uint8_t* bytes, size_t size) {
                storage::ThrowIfCancelled(token, "read packed index");
                AssertInfo(offset <= expected && size <= expected - offset,
                           "packed entry exceeds declared size: {}",
                           entry.name);
                if (const auto* memory =
                        std::get_if<storage::MemoryEntryTarget>(
                            &entry.target)) {
                    if (size != 0) {
                        std::memcpy(memory->data + offset, bytes, size);
                    }
                } else {
                    const auto& file =
                        std::get<storage::FileEntryTarget>(entry.target);
                    file.staging->WriteAt(file.offset + offset, bytes, size);
                }
                offset += size;
            });
        AssertInfo(offset == expected, "short packed entry: {}", entry.name);
    }
    for (const auto& file : files) {
        file->Finish();
    }
    storage::ThrowIfCancelled(token, "initialize packed index");
    auto reader = folly::coro::blockingWait(
        FinishPackedLoad(loader, plan, options, false, token));
    AssertInfo(reader != nullptr, "packed loader returned a null reader");
    storage::ThrowIfCancelled(token, "publish packed index");
    plan.Commit();
    return reader;
}

folly::coro::Task<IIndexReaderBasePtr>
LoadPackedIndexAsync(const LoaderEntry& loader,
                     storage::AsyncIndexEntryReader& source,
                     const storage::LoadOptions& options,
                     proto::common::LoadPriority priority,
                     folly::CancellationToken token) {
    RequirePackedLoader(loader);
    token = folly::cancellation_token_merge(
        token, co_await folly::coro::co_current_cancellation_token);
    storage::ThrowIfCancelled(token, "plan packed index");
    std::optional<IndexLoadPlan> plan;
    IIndexReaderBasePtr reader;
    std::exception_ptr failure;
    try {
        // Planning can create directories. Await the complete local phase even
        // after cancellation so its captured references remain valid.
        co_await storage::RunLocalFileIOAsync(
            [&] {
                plan.emplace(loader.plan_packed(
                    source.Directory(), source.IndexMeta(), options));
            },
            priority);
        storage::ThrowIfCancelled(token, "read packed index");
        co_await source.ReadEntriesAsync(plan->entries, priority, token);
        storage::ThrowIfCancelled(token, "initialize packed index");
        // Families offload their blocking file phases; CPU conversion stays
        // on the loading executor even when the plan contains file targets.
        reader = co_await FinishPackedLoad(loader, *plan, options, true, token);
        AssertInfo(reader != nullptr, "packed loader returned a null reader");
        storage::ThrowIfCancelled(token, "publish packed index");
    } catch (...) {
        failure = std::current_exception();
    }
    // A failed reader may own mappings into planned files. Destroy it before
    // removing those files; release the plan's context on the local executor.
    co_await storage::RunLocalFileIOAsync(
        [&] {
            // Cancellation can arrive while this final local task is queued.
            if (!failure) {
                try {
                    storage::ThrowIfCancelled(token, "publish packed index");
                } catch (...) {
                    failure = std::current_exception();
                }
            }
            if (failure) {
                reader.reset();
            } else {
                plan->Commit();
            }
            plan.reset();
        },
        priority);
    if (failure) {
        std::rethrow_exception(failure);
    }
    co_return std::move(reader);
}

}  // namespace milvus::index
