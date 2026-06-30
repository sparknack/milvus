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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "folly/coro/BlockingWait.h"
#include "folly/coro/FutureUtil.h"
#include "gtest/gtest.h"
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
     BuildBatchesSortsAndMergesContiguousRowGroups) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/2, /*rg_offset=*/2, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/0, /*rg_offset=*/0, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/1, /*rg_offset=*/1, /*rg_count=*/1, /*memory_size=*/16 * MB},
        {/*cid=*/3, /*rg_offset=*/4, /*rg_count=*/1, /*memory_size=*/16 * MB},
    };

    auto batches = BuildStorageV3LoadBatches(std::move(units), 64 * MB);

    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0].rg_offset, 0);
    EXPECT_EQ(batches[0].rg_count, 3);
    EXPECT_EQ(batches[0].batch_memory, 48 * MB);
    ASSERT_EQ(batches[0].units.size(), 3);
    EXPECT_EQ(batches[0].units[0].cid, 0);
    EXPECT_EQ(batches[0].units[1].cid, 1);
    EXPECT_EQ(batches[0].units[2].cid, 2);

    EXPECT_EQ(batches[1].rg_offset, 4);
    EXPECT_EQ(batches[1].rg_count, 1);
    ASSERT_EQ(batches[1].units.size(), 1);
    EXPECT_EQ(batches[1].units[0].cid, 3);
}

TEST(ManifestGroupTranslatorAsyncPipelineTest,
     BuildBatchesSplitsWhenTargetWouldBeExceeded) {
    constexpr int64_t MB = 1 << 20;
    std::vector<StorageV3LoadUnit> units = {
        {/*cid=*/0, /*rg_offset=*/0, /*rg_count=*/1, /*memory_size=*/24 * MB},
        {/*cid=*/1, /*rg_offset=*/1, /*rg_count=*/1, /*memory_size=*/24 * MB},
        {/*cid=*/2, /*rg_offset=*/2, /*rg_count=*/1, /*memory_size=*/24 * MB},
    };

    auto batches = BuildStorageV3LoadBatches(std::move(units), 48 * MB);

    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0].rg_offset, 0);
    EXPECT_EQ(batches[0].rg_count, 2);
    EXPECT_EQ(batches[0].batch_memory, 48 * MB);
    ASSERT_EQ(batches[0].units.size(), 2);
    EXPECT_EQ(batches[0].units[0].cid, 0);
    EXPECT_EQ(batches[0].units[1].cid, 1);

    EXPECT_EQ(batches[1].rg_offset, 2);
    EXPECT_EQ(batches[1].rg_count, 1);
    EXPECT_EQ(batches[1].batch_memory, 24 * MB);
    ASSERT_EQ(batches[1].units.size(), 1);
    EXPECT_EQ(batches[1].units[0].cid, 2);
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
