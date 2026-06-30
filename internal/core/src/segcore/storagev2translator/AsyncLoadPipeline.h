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

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "cachinglayer/Utils.h"
#include "pb/common.pb.h"
#include "segcore/storagev2translator/GroupCTMeta.h"
#include "storage/ThreadPools.h"

#include <folly/coro/FutureUtil.h>
#include <folly/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

namespace milvus::segcore::storagev2translator {

struct StorageV3LoadUnit {
    milvus::cachinglayer::cid_t cid;
    int64_t rg_offset;
    int64_t rg_count;
    int64_t memory_size;
    int64_t loading_overhead_size = 0;
};

struct StorageV3LoadBatch {
    int64_t rg_offset;
    int64_t rg_count;
    int64_t batch_memory = 0;
    std::vector<StorageV3LoadUnit> units;
};

std::vector<StorageV3LoadBatch>
BuildStorageV3LoadBatches(std::vector<StorageV3LoadUnit> units,
                          int64_t split_target_bytes);

using StorageV3LoadingOverheadFunc = std::function<int64_t(int64_t)>;

std::vector<StorageV3LoadUnit>
BuildStorageV3LoadUnitsForCells(
    const GroupCTMeta& meta,
    const std::vector<milvus::cachinglayer::cid_t>& cids,
    const StorageV3LoadingOverheadFunc& loading_overhead_bytes);

template <typename Func>
auto
SubmitStorageV3LoadTask(milvus::proto::common::LoadPriority priority,
                        Func&& func)
    -> folly::SemiFuture<std::invoke_result_t<std::decay_t<Func>&>> {
    using TaskFunc = std::decay_t<Func>;
    using Result = std::invoke_result_t<TaskFunc&>;

    folly::Promise<Result> promise;
    auto future = promise.getSemiFuture();
    auto shared_promise =
        std::make_shared<folly::Promise<Result>>(std::move(promise));
    auto shared_func = std::make_shared<TaskFunc>(std::forward<Func>(func));

    try {
        auto& pool = milvus::ThreadPools::GetThreadPool(
            milvus::PriorityForLoad(priority));
        pool.Submit([shared_promise, shared_func]() mutable {
            try {
                if constexpr (std::is_void_v<Result>) {
                    std::invoke(*shared_func);
                    shared_promise->setValue();
                } else {
                    shared_promise->setValue(std::invoke(*shared_func));
                }
            } catch (...) {
                shared_promise->setException(
                    folly::exception_wrapper(std::current_exception()));
            }
        });
    } catch (...) {
        shared_promise->setException(
            folly::exception_wrapper(std::current_exception()));
    }

    return future;
}

struct StorageV3AdmissionConfig {
    size_t total_bytes;
    size_t high_reserved_bytes = 0;
};

class StorageV3AdmissionScheduler;

class StorageV3LoadBudgetLease {
 public:
    StorageV3LoadBudgetLease() = default;
    StorageV3LoadBudgetLease(const StorageV3LoadBudgetLease&) = delete;
    StorageV3LoadBudgetLease&
    operator=(const StorageV3LoadBudgetLease&) = delete;
    StorageV3LoadBudgetLease(StorageV3LoadBudgetLease&& other) noexcept;
    StorageV3LoadBudgetLease&
    operator=(StorageV3LoadBudgetLease&& other) noexcept;
    ~StorageV3LoadBudgetLease();

    void
    Release();

 private:
    friend class StorageV3AdmissionScheduler;

    StorageV3LoadBudgetLease(StorageV3AdmissionScheduler* scheduler,
                             milvus::proto::common::LoadPriority priority,
                             size_t bytes);

    StorageV3AdmissionScheduler* scheduler_{nullptr};
    milvus::proto::common::LoadPriority priority_{
        milvus::proto::common::LoadPriority::LOW};
    size_t bytes_{0};
};

class StorageV3AdmissionScheduler {
 public:
    explicit StorageV3AdmissionScheduler(StorageV3AdmissionConfig config);

    folly::SemiFuture<StorageV3LoadBudgetLease>
    Admit(milvus::proto::common::LoadPriority priority, size_t bytes);

 private:
    friend class StorageV3LoadBudgetLease;

    struct PendingAdmission {
        milvus::proto::common::LoadPriority priority;
        size_t bytes;
        folly::Promise<StorageV3LoadBudgetLease> promise;
    };

    void
    Release(milvus::proto::common::LoadPriority priority, size_t bytes);

    void
    TryScheduleLocked();

    bool
    CanAdmitLocked(milvus::proto::common::LoadPriority priority,
                   size_t bytes) const;

    void
    MarkAdmittedLocked(milvus::proto::common::LoadPriority priority,
                       size_t bytes);

    StorageV3AdmissionConfig config_;
    size_t used_high_bytes_{0};
    size_t used_low_bytes_{0};
    std::deque<PendingAdmission> high_pending_;
    std::deque<PendingAdmission> low_pending_;
    mutable std::mutex mutex_;
};

template <typename Func>
auto
AdmitAndSubmitStorageV3LoadTask(StorageV3AdmissionScheduler& scheduler,
                                milvus::proto::common::LoadPriority priority,
                                size_t bytes,
                                Func&& func)
    -> folly::coro::Task<std::invoke_result_t<std::decay_t<Func>&>> {
    using TaskFunc = std::decay_t<Func>;
    using Result = std::invoke_result_t<TaskFunc&>;

    auto shared_func = std::make_shared<TaskFunc>(std::forward<Func>(func));
    auto lease = co_await folly::coro::toTask(scheduler.Admit(priority, bytes));
    (void)lease;

    if constexpr (std::is_void_v<Result>) {
        co_await folly::coro::toTask(SubmitStorageV3LoadTask(
            priority, [shared_func]() mutable { std::invoke(*shared_func); }));
        co_return;
    } else {
        co_return co_await folly::coro::toTask(SubmitStorageV3LoadTask(
            priority, [shared_func]() mutable -> Result {
                return std::invoke(*shared_func);
            }));
    }
}

folly::coro::Task<int>
StorageV3AsyncLoadCoroSmokeTest();

}  // namespace milvus::segcore::storagev2translator
