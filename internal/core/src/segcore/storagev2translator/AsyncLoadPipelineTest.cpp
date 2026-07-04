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

#include "segcore/storagev2translator/AsyncLoadPipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "arrow/util/thread_pool.h"
#include "folly/ScopeGuard.h"
#include "folly/coro/BlockingWait.h"
#include "folly/coro/FutureUtil.h"
#include "gtest/gtest.h"
#include "common/GroupChunk.h"
#include "milvus-storage/async_read_options.h"
#include "pb/common.pb.h"

using namespace milvus::segcore::storagev2translator;
using milvus::proto::common::LoadPriority;
using namespace std::chrono_literals;

namespace {

folly::coro::Task<int>
AwaitSubmittedLoadTaskForTest() {
    auto value = co_await folly::coro::toTask(
        SubmitStorageV3LoadTask(LoadPriority::HIGH, [] { return 41; }));
    co_return value + 1;
}

std::shared_ptr<arrow::RecordBatch>
MakeEmptyRecordBatchForTest() {
    return arrow::RecordBatch::Make(
        std::make_shared<arrow::Schema>(
            std::vector<std::shared_ptr<arrow::Field>>{}),
        0,
        std::vector<std::shared_ptr<arrow::Array>>{});
}

std::vector<std::shared_ptr<arrow::RecordBatch>>
MakeEmptyRecordBatchesForTest(size_t count) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    batches.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        batches.push_back(MakeEmptyRecordBatchForTest());
    }
    return batches;
}

class AsyncTestChunkReader final : public milvus_storage::api::ChunkReader {
 public:
    using ReadImpl = std::function<folly::SemiFuture<
        arrow::Result<milvus_storage::api::RecordBatchVector>>(
        const std::vector<int64_t>&,
        const milvus_storage::api::AsyncReadOptions&)>;

    explicit AsyncTestChunkReader(ReadImpl read_impl, size_t chunk_count = 1024)
        : read_impl_(std::move(read_impl)), chunk_count_(chunk_count) {
    }

    size_t
    total_number_of_chunks() const override {
        return chunk_count_;
    }

    arrow::Result<std::vector<int64_t>>
    get_chunk_indices(const std::vector<int64_t>& row_indices) override {
        return row_indices;
    }

    arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    get_chunk(int64_t) override {
        return MakeEmptyRecordBatchForTest();
    }

    arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
    get_chunks(const std::vector<int64_t>& chunk_indices,
               size_t /*parallelism*/) override {
        return MakeEmptyRecordBatchesForTest(chunk_indices.size());
    }

    folly::SemiFuture<arrow::Result<milvus_storage::api::RecordBatchVector>>
    get_chunks_async(
        const std::vector<int64_t>& chunk_indices,
        const milvus_storage::api::AsyncReadOptions& options) override {
        return read_impl_(chunk_indices, options);
    }

    arrow::Result<std::vector<uint64_t>>
    get_chunk_size() override {
        return std::vector<uint64_t>(chunk_count_, 1);
    }

    arrow::Result<std::vector<uint64_t>>
    get_chunk_rows() override {
        return std::vector<uint64_t>(chunk_count_, 1);
    }

 private:
    ReadImpl read_impl_;
    size_t chunk_count_;
};

}  // namespace

TEST(ManifestGroupTranslatorAsyncPipelineTest, CoroSmoke) {
    auto value = folly::coro::blockingWait(StorageV3AsyncLoadCoroSmokeTest());
    EXPECT_EQ(value, 1);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     SubmitLoadTaskRunsAsynchronouslyOnStoragePool) {
    std::promise<void> started;
    auto started_future = started.get_future();
    std::promise<void> release;
    auto release_future = release.get_future().share();

    auto result = SubmitStorageV3LoadTask(LoadPriority::HIGH, [&]() {
        started.set_value();
        release_future.wait();
        return 42;
    });

    ASSERT_EQ(started_future.wait_for(5s), std::future_status::ready);
    EXPECT_FALSE(result.isReady());

    release.set_value();
    EXPECT_EQ(std::move(result).get(), 42);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     SubmitLoadTaskPropagatesExceptionsToSemiFuture) {
    auto result = SubmitStorageV3LoadTask(LoadPriority::HIGH, []() -> int {
        throw std::runtime_error("storage v3 load failed");
    });

    EXPECT_THROW(std::move(result).get(), std::runtime_error);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     SubmittedLoadTaskCanBeAwaitedByFollyCoroTask) {
    auto value = folly::coro::blockingWait(AwaitSubmittedLoadTaskForTest());

    EXPECT_EQ(value, 42);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmitAndSubmitDoesNotStartTaskBeforeBudgetIsGranted) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/1, /*high_reserved_bytes=*/0});
    auto existing = scheduler.Admit(LoadPriority::HIGH, 1);
    ASSERT_TRUE(existing.isReady());
    auto existing_lease = std::move(existing).get();
    std::atomic<bool> task_started{false};

    auto result = std::async(std::launch::async, [&] {
        return folly::coro::blockingWait(AdmitAndSubmitStorageV3LoadTask(
            scheduler, LoadPriority::HIGH, 1, [&] {
                task_started.store(true);
                return 42;
            }));
    });

    EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
    EXPECT_FALSE(task_started.load());

    existing_lease.Release();
    EXPECT_EQ(result.get(), 42);
    EXPECT_TRUE(task_started.load());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionFulfillsPendingRequestsOutsideSchedulerLock) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/1, /*high_reserved_bytes=*/0});
    auto running = scheduler.Admit(LoadPriority::HIGH, 1);
    ASSERT_TRUE(running.isReady());
    auto running_lease = std::move(running).get();

    auto waiting =
        scheduler.Admit(LoadPriority::HIGH, 1)
            .deferValue([](StorageV3LoadBudgetLease&& admitted_lease) {
                admitted_lease.Release();
                return 7;
            });
    EXPECT_FALSE(waiting.isReady());

    auto release =
        std::async(std::launch::async, [&] { running_lease.Release(); });
    ASSERT_EQ(release.wait_for(5s), std::future_status::ready);
    release.get();
    EXPECT_EQ(std::move(waiting).get(), 7);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     BuildReadTasksSortsAndMergesContiguousRowGroups) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/2, /*rg_offset=*/2, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/0, /*rg_offset=*/0, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/1, /*rg_offset=*/1, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/3, /*rg_offset=*/4, /*rg_count=*/1, /*memory_size=*/16 * MB},
    };

    auto tasks = BuildStorageV3ReadTasks(std::move(units), 64 * MB);

    ASSERT_EQ(tasks.size(), 2);
    EXPECT_EQ(tasks[0].rg_offset, 0);
    EXPECT_EQ(tasks[0].rg_count, 3);
    EXPECT_EQ(tasks[0].task_memory, 48 * MB);
    ASSERT_EQ(tasks[0].units.size(), 3);
    EXPECT_EQ(tasks[0].units[0].cid, 0);
    EXPECT_EQ(tasks[0].units[1].cid, 1);
    EXPECT_EQ(tasks[0].units[2].cid, 2);

    EXPECT_EQ(tasks[1].rg_offset, 4);
    EXPECT_EQ(tasks[1].rg_count, 1);
    ASSERT_EQ(tasks[1].units.size(), 1);
    EXPECT_EQ(tasks[1].units[0].cid, 3);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     BuildReadTasksSplitsWhenReadWindowTargetWouldBeExceeded) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/0, /*rg_offset=*/0, /*rg_count=*/1, /*memory_size=*/8 * MB},
        {/*cid=*/1, /*rg_offset=*/1, /*rg_count=*/1, /*memory_size=*/8 * MB},
        {/*cid=*/2, /*rg_offset=*/2, /*rg_count=*/1, /*memory_size=*/8 * MB},
    };

    auto tasks = BuildStorageV3ReadTasks(std::move(units), 16 * MB);

    ASSERT_EQ(tasks.size(), 2);
    EXPECT_EQ(tasks[0].rg_offset, 0);
    EXPECT_EQ(tasks[0].rg_count, 2);
    EXPECT_EQ(tasks[0].task_memory, 16 * MB);
    ASSERT_EQ(tasks[0].units.size(), 2);
    EXPECT_EQ(tasks[0].units[0].cid, 0);
    EXPECT_EQ(tasks[0].units[1].cid, 1);

    EXPECT_EQ(tasks[1].rg_offset, 2);
    EXPECT_EQ(tasks[1].rg_count, 1);
    EXPECT_EQ(tasks[1].task_memory, 8 * MB);
    ASSERT_EQ(tasks[1].units.size(), 1);
    EXPECT_EQ(tasks[1].units[0].cid, 2);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     LoadCellsAsyncSchedulesReadWindowTasksAndReturnsRequestedOrder) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/2,
         /*rg_offset=*/2,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
        {/*cid=*/0,
         /*rg_offset=*/0,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
        {/*cid=*/1,
         /*rg_offset=*/1,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
    };
    std::vector<std::tuple<int64_t, int64_t>> reads;
    std::mutex reads_mutex;
    std::vector<int64_t> finalized_cids;
    std::mutex finalized_cids_mutex;
    StorageV3AsyncLoadFn load =
        [&](milvus::OpContext*,
            StorageV3ReadTask task) -> folly::SemiFuture<StorageV3LoadResult> {
        {
            std::lock_guard<std::mutex> lock(reads_mutex);
            reads.emplace_back(task.rg_offset, task.rg_count);
        }
        auto loaded_cells = std::make_shared<StorageV3LoadedCells>();
        loaded_cells->reserve(task.units.size());
        for (const auto& unit : task.units) {
            std::lock_guard<std::mutex> lock(finalized_cids_mutex);
            finalized_cids.push_back(unit.cid);
            loaded_cells->push_back(
                {unit.cid, std::make_unique<milvus::GroupChunk>()});
        }
        return folly::makeSemiFuture(StorageV3LoadResult(loaded_cells));
    };
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/64 * MB, /*high_reserved_bytes=*/0});

    auto cells =
        folly::coro::blockingWait(LoadStorageV3CellsAsync(nullptr,
                                                          std::move(units),
                                                          std::move(load),
                                                          16 * MB,
                                                          LoadPriority::HIGH,
                                                          scheduler));

    ASSERT_EQ(reads.size(), 3);
    EXPECT_EQ(reads[0], (std::tuple<int64_t, int64_t>{0, 1}));
    EXPECT_EQ(reads[1], (std::tuple<int64_t, int64_t>{1, 1}));
    EXPECT_EQ(reads[2], (std::tuple<int64_t, int64_t>{2, 1}));
    std::sort(finalized_cids.begin(), finalized_cids.end());
    EXPECT_EQ(finalized_cids, std::vector<int64_t>({0, 1, 2}));
    ASSERT_EQ(cells.size(), 3);
    EXPECT_EQ(cells[0].cid, 2);
    EXPECT_EQ(cells[1].cid, 0);
    EXPECT_EQ(cells[2].cid, 1);
    EXPECT_NE(cells[0].chunk, nullptr);
    EXPECT_NE(cells[1].chunk, nullptr);
    EXPECT_NE(cells[2].chunk, nullptr);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     LoadCellsAsyncWaitsForAsyncReadBeforeFinalizing) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/0,
         /*rg_offset=*/0,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
    };

    folly::Promise<arrow::Result<milvus_storage::api::RecordBatchVector>>
        read_promise;
    auto read_future = read_promise.getSemiFuture();
    std::promise<void> read_started;
    std::atomic<int> finalize_calls{0};

    auto reader = std::make_shared<AsyncTestChunkReader>(
        [&](const std::vector<int64_t>& chunk_indices,
            const milvus_storage::api::AsyncReadOptions& options)
            -> folly::SemiFuture<
                arrow::Result<milvus_storage::api::RecordBatchVector>> {
            EXPECT_EQ(chunk_indices, std::vector<int64_t>({0}));
            EXPECT_NE(options.materialize_executor, nullptr);
            EXPECT_NE(options.materialize_executor,
                      arrow::internal::GetCpuThreadPool());
            read_started.set_value();
            return std::move(read_future);
        });
    auto finalizer =
        [&](const std::vector<std::shared_ptr<arrow::Table>>& tables,
            int64_t cid) {
            EXPECT_EQ(cid, 0);
            EXPECT_EQ(tables.size(), 1);
            finalize_calls.fetch_add(1);
            return std::make_unique<milvus::GroupChunk>();
        };
    auto load = MakeStorageV3ChunkLoadFn(reader, std::move(finalizer));
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/64 * MB, /*high_reserved_bytes=*/0});

    auto result = std::async(std::launch::async, [&] {
        return folly::coro::blockingWait(
            LoadStorageV3CellsAsync(nullptr,
                                    std::move(units),
                                    std::move(load),
                                    16 * MB,
                                    LoadPriority::HIGH,
                                    scheduler));
    });

    ASSERT_EQ(read_started.get_future().wait_for(5s),
              std::future_status::ready);
    EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
    EXPECT_EQ(finalize_calls.load(), 0);

    read_promise.setValue(MakeEmptyRecordBatchesForTest(1));

    auto cells = result.get();
    ASSERT_EQ(cells.size(), 1);
    EXPECT_EQ(cells[0].cid, 0);
    EXPECT_NE(cells[0].chunk, nullptr);
    EXPECT_EQ(finalize_calls.load(), 1);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     LoadCellsAsyncFinalizesOnMaterializeExecutorThread) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/0,
         /*rg_offset=*/0,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
    };

    folly::Promise<arrow::Result<milvus_storage::api::RecordBatchVector>>
        read_promise;
    auto read_future = read_promise.getSemiFuture();
    std::promise<void> read_started;
    std::atomic<arrow::internal::Executor*> materialize_executor{nullptr};
    std::atomic<bool> finalizer_on_materialize_executor{false};

    auto reader = std::make_shared<AsyncTestChunkReader>(
        [&](const std::vector<int64_t>& chunk_indices,
            const milvus_storage::api::AsyncReadOptions& options)
            -> folly::SemiFuture<
                arrow::Result<milvus_storage::api::RecordBatchVector>> {
            EXPECT_EQ(chunk_indices, std::vector<int64_t>({0}));
            EXPECT_NE(options.materialize_executor, nullptr);
            EXPECT_NE(options.materialize_executor,
                      arrow::internal::GetCpuThreadPool());
            materialize_executor.store(options.materialize_executor);
            read_started.set_value();
            return std::move(read_future);
        });
    auto finalizer =
        [&](const std::vector<std::shared_ptr<arrow::Table>>& tables,
            int64_t cid) {
            EXPECT_EQ(cid, 0);
            EXPECT_EQ(tables.size(), 1);
            auto* executor = materialize_executor.load();
            EXPECT_NE(executor, nullptr);
            if (executor != nullptr) {
                finalizer_on_materialize_executor.store(
                    executor->OwnsThisThread());
            }
            EXPECT_FALSE(StorageV3DiskExecutor()->OwnsThisThread());
            return std::make_unique<milvus::GroupChunk>();
        };
    auto load = MakeStorageV3ChunkLoadFn(reader, std::move(finalizer));
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/64 * MB, /*high_reserved_bytes=*/0});

    auto result = std::async(std::launch::async, [&] {
        return folly::coro::blockingWait(
            LoadStorageV3CellsAsync(nullptr,
                                    std::move(units),
                                    std::move(load),
                                    16 * MB,
                                    LoadPriority::HIGH,
                                    scheduler));
    });

    ASSERT_EQ(read_started.get_future().wait_for(5s),
              std::future_status::ready);
    auto completer = std::async(std::launch::async, [&] {
        read_promise.setValue(MakeEmptyRecordBatchesForTest(1));
    });
    ASSERT_EQ(completer.wait_for(5s), std::future_status::ready);
    completer.get();

    auto cells = result.get();
    ASSERT_EQ(cells.size(), 1);
    EXPECT_EQ(cells[0].cid, 0);
    EXPECT_NE(cells[0].chunk, nullptr);
    EXPECT_TRUE(finalizer_on_materialize_executor.load());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     MmapLoadCellsAsyncFinalizesOnDiskExecutorThread) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/0,
         /*rg_offset=*/0,
         /*rg_count=*/1,
         /*memory_size=*/16 * MB,
         /*loading_overhead_size=*/16 * MB},
    };

    folly::Promise<arrow::Result<milvus_storage::api::RecordBatchVector>>
        read_promise;
    auto read_future = read_promise.getSemiFuture();
    std::promise<void> read_started;
    std::atomic<arrow::internal::Executor*> materialize_executor{nullptr};
    std::atomic<bool> finalizer_on_materialize_executor{true};
    std::atomic<bool> finalizer_on_disk_executor{false};

    auto reader = std::make_shared<AsyncTestChunkReader>(
        [&](const std::vector<int64_t>& chunk_indices,
            const milvus_storage::api::AsyncReadOptions& options)
            -> folly::SemiFuture<
                arrow::Result<milvus_storage::api::RecordBatchVector>> {
            EXPECT_EQ(chunk_indices, std::vector<int64_t>({0}));
            EXPECT_NE(options.materialize_executor, nullptr);
            EXPECT_NE(options.materialize_executor,
                      arrow::internal::GetCpuThreadPool());
            materialize_executor.store(options.materialize_executor);
            read_started.set_value();
            return std::move(read_future);
        });
    auto finalizer =
        [&](const std::vector<std::shared_ptr<arrow::Table>>& tables,
            int64_t cid) {
            EXPECT_EQ(cid, 0);
            EXPECT_EQ(tables.size(), 1);
            auto* executor = materialize_executor.load();
            EXPECT_NE(executor, nullptr);
            if (executor != nullptr) {
                finalizer_on_materialize_executor.store(
                    executor->OwnsThisThread());
            }
            finalizer_on_disk_executor.store(
                StorageV3DiskExecutor()->OwnsThisThread());
            return std::make_unique<milvus::GroupChunk>();
        };
    auto load = MakeStorageV3ChunkLoadFn(
        reader, std::move(finalizer), StorageV3LocalizeExecutor::Disk);
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/64 * MB, /*high_reserved_bytes=*/0});

    auto result = std::async(std::launch::async, [&] {
        return folly::coro::blockingWait(
            LoadStorageV3CellsAsync(nullptr,
                                    std::move(units),
                                    std::move(load),
                                    16 * MB,
                                    LoadPriority::HIGH,
                                    scheduler));
    });

    ASSERT_EQ(read_started.get_future().wait_for(5s),
              std::future_status::ready);
    auto completer = std::async(std::launch::async, [&] {
        read_promise.setValue(MakeEmptyRecordBatchesForTest(1));
    });
    ASSERT_EQ(completer.wait_for(5s), std::future_status::ready);
    completer.get();

    auto cells = result.get();
    ASSERT_EQ(cells.size(), 1);
    EXPECT_EQ(cells[0].cid, 0);
    EXPECT_NE(cells[0].chunk, nullptr);
    EXPECT_FALSE(finalizer_on_materialize_executor.load());
    EXPECT_TRUE(finalizer_on_disk_executor.load());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     DiskExecutorNumThreadsCanBeConfigured) {
    auto reset =
        folly::makeGuard([] { SetStorageV3DiskExecutorNumThreads(0); });

    SetStorageV3DiskExecutorNumThreads(1);
    EXPECT_EQ(StorageV3DiskExecutor()->GetCapacity(), 1);

    SetStorageV3DiskExecutorNumThreads(2);
    EXPECT_EQ(StorageV3DiskExecutor()->GetCapacity(), 2);

    SetStorageV3DiskExecutorNumThreads(0);
    EXPECT_GE(StorageV3DiskExecutor()->GetCapacity(), 1);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionPrefersQueuedHighPriorityAfterRelease) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/1, /*high_reserved_bytes=*/0});

    auto low_running = scheduler.Admit(LoadPriority::LOW, 1);
    ASSERT_TRUE(low_running.isReady());
    auto low_lease = std::move(low_running).get();

    auto low_waiting = scheduler.Admit(LoadPriority::LOW, 1);
    EXPECT_FALSE(low_waiting.isReady());
    auto high_waiting = scheduler.Admit(LoadPriority::HIGH, 1);
    EXPECT_FALSE(high_waiting.isReady());

    low_lease.Release();
    EXPECT_TRUE(high_waiting.isReady());
    EXPECT_FALSE(low_waiting.isReady());

    auto high_lease = std::move(high_waiting).get();
    high_lease.Release();
    EXPECT_TRUE(low_waiting.isReady());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionDoesNotLetNewLowBypassQueuedHigh) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/2, /*high_reserved_bytes=*/0});

    auto low_running = scheduler.Admit(LoadPriority::LOW, 1);
    ASSERT_TRUE(low_running.isReady());
    auto low_lease = std::move(low_running).get();

    auto high_waiting = scheduler.Admit(LoadPriority::HIGH, 2);
    EXPECT_FALSE(high_waiting.isReady());

    auto low_waiting = scheduler.Admit(LoadPriority::LOW, 1);
    EXPECT_FALSE(low_waiting.isReady());

    low_lease.Release();
    EXPECT_TRUE(high_waiting.isReady());
    EXPECT_FALSE(low_waiting.isReady());

    auto high_lease = std::move(high_waiting).get();
    high_lease.Release();
    EXPECT_TRUE(low_waiting.isReady());
    auto low_waiting_lease = std::move(low_waiting).get();
    low_waiting_lease.Release();
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionDoesNotLetLowBorrowHighReservedBytes) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/2, /*high_reserved_bytes=*/1});

    auto low_running = scheduler.Admit(LoadPriority::LOW, 1);
    ASSERT_TRUE(low_running.isReady());
    auto low_lease = std::move(low_running).get();

    auto low_waiting = scheduler.Admit(LoadPriority::LOW, 1);
    EXPECT_FALSE(low_waiting.isReady());
    auto high_running = scheduler.Admit(LoadPriority::HIGH, 1);
    ASSERT_TRUE(high_running.isReady());
    auto high_lease = std::move(high_running).get();

    high_lease.Release();
    EXPECT_FALSE(low_waiting.isReady());

    low_lease.Release();
    EXPECT_TRUE(low_waiting.isReady());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionTreatsZeroCapacityAsUnlimited) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/0, /*high_reserved_bytes=*/0});

    auto high = scheduler.Admit(LoadPriority::HIGH, 100);
    auto low = scheduler.Admit(LoadPriority::LOW, 100);

    EXPECT_TRUE(high.isReady());
    EXPECT_TRUE(low.isReady());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionAllowsOversizedRequestToRunExclusivelyWhenIdle) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/10, /*high_reserved_bytes=*/4});

    auto oversized_low = scheduler.Admit(LoadPriority::LOW, 11);
    ASSERT_TRUE(oversized_low.isReady());
    auto oversized_lease = std::move(oversized_low).get();

    auto high_waiting = scheduler.Admit(LoadPriority::HIGH, 1);
    EXPECT_FALSE(high_waiting.isReady());

    oversized_lease.Release();
    EXPECT_TRUE(high_waiting.isReady());
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     AdmissionUpdateConfigWakesPendingRequest) {
    StorageV3AdmissionScheduler scheduler(
        {/*total_bytes=*/1, /*high_reserved_bytes=*/0});

    auto running = scheduler.Admit(LoadPriority::HIGH, 1);
    ASSERT_TRUE(running.isReady());
    auto running_lease = std::move(running).get();

    auto waiting = scheduler.Admit(LoadPriority::HIGH, 1);
    EXPECT_FALSE(waiting.isReady());

    scheduler.UpdateConfig({/*total_bytes=*/2, /*high_reserved_bytes=*/0});
    EXPECT_TRUE(waiting.isReady());
}
