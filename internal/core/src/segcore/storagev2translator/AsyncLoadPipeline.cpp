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
#include <utility>

#include "common/EasyAssert.h"
#include "folly/coro/FutureUtil.h"

namespace milvus::segcore::storagev2translator {

namespace {

bool
IsHighPriority(milvus::proto::common::LoadPriority priority) {
    return priority == milvus::proto::common::LoadPriority::HIGH;
}

}  // namespace

std::vector<StorageV3LoadBatch>
BuildStorageV3LoadBatches(std::vector<StorageV3LoadUnit> units,
                          int64_t split_target_bytes) {
    std::sort(
        units.begin(),
        units.end(),
        [](const StorageV3LoadUnit& left, const StorageV3LoadUnit& right) {
            if (left.rg_offset != right.rg_offset) {
                return left.rg_offset < right.rg_offset;
            }
            return left.cid < right.cid;
        });

    std::vector<StorageV3LoadBatch> batches;
    StorageV3LoadBatch current{};

    for (auto& unit : units) {
        bool should_split = false;
        if (!current.units.empty()) {
            auto next_expected_rg_offset = current.rg_offset + current.rg_count;
            auto batch_full =
                split_target_bytes > 0 &&
                current.batch_memory + unit.memory_size > split_target_bytes;
            if (unit.rg_offset != next_expected_rg_offset || batch_full) {
                should_split = true;
            }
        }
        if (should_split) {
            batches.push_back(std::move(current));
            current = {};
        }
        if (current.units.empty()) {
            current.rg_offset = unit.rg_offset;
            current.rg_count = 0;
            current.batch_memory = 0;
        }
        current.rg_count += unit.rg_count;
        current.batch_memory += unit.memory_size;
        current.units.push_back(std::move(unit));
    }

    if (!current.units.empty()) {
        batches.push_back(std::move(current));
    }

    return batches;
}

std::vector<StorageV3LoadUnit>
BuildStorageV3LoadUnitsForCells(
    const GroupCTMeta& meta,
    const std::vector<milvus::cachinglayer::cid_t>& cids,
    const StorageV3LoadingOverheadFunc& loading_overhead_bytes) {
    std::vector<StorageV3LoadUnit> units;
    units.reserve(cids.size());

    for (auto cid : cids) {
        AssertInfo(cid >= 0 && static_cast<size_t>(cid) <
                                   meta.chunk_memory_size_.size(),
                   "[StorageV3] cid {} is out of range. Total cells: {}",
                   cid,
                   meta.chunk_memory_size_.size());
        auto [start, end] = meta.get_row_group_range(cid);
        auto memory_size = meta.chunk_memory_size_[cid];
        units.push_back({cid,
                         static_cast<int64_t>(start),
                         static_cast<int64_t>(end - start),
                         memory_size,
                         loading_overhead_bytes(memory_size)});
    }

    return units;
}

StorageV3LoadBudgetLease::StorageV3LoadBudgetLease(
    StorageV3AdmissionScheduler* scheduler,
    milvus::proto::common::LoadPriority priority,
    size_t bytes)
    : scheduler_(scheduler), priority_(priority), bytes_(bytes) {
}

StorageV3LoadBudgetLease::StorageV3LoadBudgetLease(
    StorageV3LoadBudgetLease&& other) noexcept
    : scheduler_(other.scheduler_),
      priority_(other.priority_),
      bytes_(other.bytes_) {
    other.scheduler_ = nullptr;
    other.bytes_ = 0;
}

StorageV3LoadBudgetLease&
StorageV3LoadBudgetLease::operator=(StorageV3LoadBudgetLease&& other) noexcept {
    if (this != &other) {
        Release();
        scheduler_ = other.scheduler_;
        priority_ = other.priority_;
        bytes_ = other.bytes_;
        other.scheduler_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

StorageV3LoadBudgetLease::~StorageV3LoadBudgetLease() {
    Release();
}

void
StorageV3LoadBudgetLease::Release() {
    if (scheduler_ == nullptr || bytes_ == 0) {
        return;
    }
    auto* scheduler = scheduler_;
    auto priority = priority_;
    auto bytes = bytes_;
    scheduler_ = nullptr;
    bytes_ = 0;
    scheduler->Release(priority, bytes);
}

StorageV3AdmissionScheduler::StorageV3AdmissionScheduler(
    StorageV3AdmissionConfig config)
    : config_(config) {
    config_.high_reserved_bytes =
        std::min(config_.high_reserved_bytes, config_.total_bytes);
}

folly::SemiFuture<StorageV3LoadBudgetLease>
StorageV3AdmissionScheduler::Admit(milvus::proto::common::LoadPriority priority,
                                   size_t bytes) {
    folly::Promise<StorageV3LoadBudgetLease> promise;
    auto future = promise.getSemiFuture();

    std::lock_guard<std::mutex> lock(mutex_);
    if (CanAdmitLocked(priority, bytes)) {
        MarkAdmittedLocked(priority, bytes);
        promise.setValue(StorageV3LoadBudgetLease(this, priority, bytes));
        return future;
    }

    auto& queue = IsHighPriority(priority) ? high_pending_ : low_pending_;
    queue.push_back(PendingAdmission{priority, bytes, std::move(promise)});
    return future;
}

void
StorageV3AdmissionScheduler::Release(
    milvus::proto::common::LoadPriority priority, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& used_bytes =
        IsHighPriority(priority) ? used_high_bytes_ : used_low_bytes_;
    used_bytes = bytes > used_bytes ? 0 : used_bytes - bytes;
    TryScheduleLocked();
}

void
StorageV3AdmissionScheduler::TryScheduleLocked() {
    while (!high_pending_.empty() &&
           CanAdmitLocked(high_pending_.front().priority,
                          high_pending_.front().bytes)) {
        auto pending = std::move(high_pending_.front());
        high_pending_.pop_front();
        MarkAdmittedLocked(pending.priority, pending.bytes);
        pending.promise.setValue(
            StorageV3LoadBudgetLease(this, pending.priority, pending.bytes));
    }

    while (!low_pending_.empty() &&
           CanAdmitLocked(low_pending_.front().priority,
                          low_pending_.front().bytes)) {
        auto pending = std::move(low_pending_.front());
        low_pending_.pop_front();
        MarkAdmittedLocked(pending.priority, pending.bytes);
        pending.promise.setValue(
            StorageV3LoadBudgetLease(this, pending.priority, pending.bytes));
    }
}

bool
StorageV3AdmissionScheduler::CanAdmitLocked(
    milvus::proto::common::LoadPriority priority, size_t bytes) const {
    auto used_total = used_high_bytes_ + used_low_bytes_;
    if (bytes > config_.total_bytes || used_total > config_.total_bytes ||
        bytes > config_.total_bytes - used_total) {
        return false;
    }

    if (IsHighPriority(priority)) {
        return true;
    }

    auto low_capacity = config_.total_bytes - config_.high_reserved_bytes;
    return used_low_bytes_ <= low_capacity &&
           bytes <= low_capacity - used_low_bytes_;
}

void
StorageV3AdmissionScheduler::MarkAdmittedLocked(
    milvus::proto::common::LoadPriority priority, size_t bytes) {
    if (IsHighPriority(priority)) {
        used_high_bytes_ += bytes;
        return;
    }
    used_low_bytes_ += bytes;
}

folly::coro::Task<int>
StorageV3AsyncLoadCoroSmokeTest() {
    co_return co_await folly::coro::toTask(SubmitStorageV3LoadTask(
        milvus::proto::common::LoadPriority::HIGH, [] { return 1; }));
}

}  // namespace milvus::segcore::storagev2translator
