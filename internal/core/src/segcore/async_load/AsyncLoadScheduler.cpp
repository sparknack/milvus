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

#include "segcore/async_load/AsyncLoadScheduler.h"

#include <algorithm>
#include <utility>

#include "storage/EntryStreamUtils.h"

namespace milvus::segcore::async_load {
namespace {

bool
IsHighPriority(milvus::proto::common::LoadPriority priority) {
    return priority == milvus::proto::common::LoadPriority::HIGH;
}

}  // namespace

LoadBudgetLease::LoadBudgetLease(LoadAdmissionScheduler* scheduler,
                                 milvus::proto::common::LoadPriority priority,
                                 size_t bytes)
    : scheduler_(scheduler), priority_(priority), bytes_(bytes) {
}

LoadBudgetLease::LoadBudgetLease(LoadBudgetLease&& other) noexcept
    : scheduler_(other.scheduler_),
      priority_(other.priority_),
      bytes_(other.bytes_) {
    other.scheduler_ = nullptr;
    other.bytes_ = 0;
}

LoadBudgetLease&
LoadBudgetLease::operator=(LoadBudgetLease&& other) noexcept {
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

LoadBudgetLease::~LoadBudgetLease() {
    Release();
}

void
LoadBudgetLease::Release() {
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

LoadAdmissionScheduler::LoadAdmissionScheduler(LoadAdmissionConfig config)
    : config_(config) {
    config_.high_reserved_bytes =
        std::min(config_.high_reserved_bytes, config_.total_bytes);
}

void
LoadAdmissionScheduler::UpdateConfig(LoadAdmissionConfig config) {
    std::vector<PendingAdmission> admitted;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config.high_reserved_bytes =
            std::min(config.high_reserved_bytes, config.total_bytes);
        config_ = config;
        admitted = TakeAdmittedLocked();
    }
    FulfillAdmissions(std::move(admitted));
}

folly::SemiFuture<LoadBudgetLease>
LoadAdmissionScheduler::Admit(milvus::proto::common::LoadPriority priority,
                              size_t bytes) {
    folly::Promise<LoadBudgetLease> promise;
    auto future = promise.getSemiFuture();

    bool admitted = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (CanAdmitLocked(priority, bytes)) {
            MarkAdmittedLocked(priority, bytes);
            admitted = true;
        } else {
            auto& queue =
                IsHighPriority(priority) ? high_pending_ : low_pending_;
            queue.push_back(
                PendingAdmission{priority, bytes, std::move(promise)});
        }
    }

    if (admitted) {
        promise.setValue(LoadBudgetLease(this, priority, bytes));
    }
    return future;
}

void
LoadAdmissionScheduler::Release(milvus::proto::common::LoadPriority priority,
                                size_t bytes) {
    std::vector<PendingAdmission> admitted;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& used_bytes =
            IsHighPriority(priority) ? used_high_bytes_ : used_low_bytes_;
        used_bytes = bytes > used_bytes ? 0 : used_bytes - bytes;
        admitted = TakeAdmittedLocked();
    }
    FulfillAdmissions(std::move(admitted));
}

std::vector<LoadAdmissionScheduler::PendingAdmission>
LoadAdmissionScheduler::TakeAdmittedLocked() {
    std::vector<PendingAdmission> admitted;
    while (!high_pending_.empty() &&
           CanAdmitLocked(high_pending_.front().priority,
                          high_pending_.front().bytes)) {
        auto pending = std::move(high_pending_.front());
        high_pending_.pop_front();
        MarkAdmittedLocked(pending.priority, pending.bytes);
        admitted.push_back(std::move(pending));
    }

    while (!low_pending_.empty() &&
           CanAdmitLocked(low_pending_.front().priority,
                          low_pending_.front().bytes)) {
        auto pending = std::move(low_pending_.front());
        low_pending_.pop_front();
        MarkAdmittedLocked(pending.priority, pending.bytes);
        admitted.push_back(std::move(pending));
    }
    return admitted;
}

void
LoadAdmissionScheduler::FulfillAdmissions(
    std::vector<PendingAdmission> admissions) {
    for (auto& admission : admissions) {
        admission.promise.setValue(
            LoadBudgetLease(this, admission.priority, admission.bytes));
    }
}

bool
LoadAdmissionScheduler::CanAdmitLocked(
    milvus::proto::common::LoadPriority priority, size_t bytes) const {
    if (config_.total_bytes == 0) {
        return true;
    }

    auto used_total = used_high_bytes_ + used_low_bytes_;
    if (bytes > config_.total_bytes) {
        return used_total == 0;
    }
    if (used_total > config_.total_bytes ||
        bytes > config_.total_bytes - used_total) {
        return false;
    }

    if (IsHighPriority(priority)) {
        return true;
    }
    if (!high_pending_.empty()) {
        return false;
    }

    auto low_capacity = config_.total_bytes - config_.high_reserved_bytes;
    if (bytes > low_capacity) {
        return used_total == 0;
    }

    return used_low_bytes_ <= low_capacity &&
           bytes <= low_capacity - used_low_bytes_;
}

void
LoadAdmissionScheduler::MarkAdmittedLocked(
    milvus::proto::common::LoadPriority priority, size_t bytes) {
    if (IsHighPriority(priority)) {
        used_high_bytes_ += bytes;
        return;
    }
    used_low_bytes_ += bytes;
}

LoadAdmissionConfig
LoadAdmissionConfigFromLoadBudget() {
    auto total_bytes =
        milvus::storage::TransientMemoryBudget::GetLoadTransientBudget()
            .CapacityBytes();
    auto high_reserved_bytes =
        total_bytes == 0 ? size_t{0} : std::max<size_t>(1, total_bytes / 4);
    return {total_bytes, high_reserved_bytes};
}

LoadAdmissionScheduler&
GetLoadAdmissionScheduler() {
    static LoadAdmissionScheduler scheduler(
        LoadAdmissionConfigFromLoadBudget());
    scheduler.UpdateConfig(LoadAdmissionConfigFromLoadBudget());
    return scheduler;
}

}  // namespace milvus::segcore::async_load
