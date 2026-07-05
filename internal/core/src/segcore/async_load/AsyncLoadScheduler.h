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
#include <deque>
#include <mutex>
#include <vector>

#include "folly/futures/Future.h"
#include "folly/futures/Promise.h"
#include "pb/common.pb.h"

namespace milvus::segcore::async_load {

struct LoadAdmissionConfig {
    size_t total_bytes;
    size_t high_reserved_bytes = 0;
};

class LoadAdmissionScheduler;

class LoadBudgetLease {
 public:
    LoadBudgetLease() = default;
    LoadBudgetLease(const LoadBudgetLease&) = delete;
    LoadBudgetLease&
    operator=(const LoadBudgetLease&) = delete;
    LoadBudgetLease(LoadBudgetLease&& other) noexcept;
    LoadBudgetLease&
    operator=(LoadBudgetLease&& other) noexcept;
    ~LoadBudgetLease();

    void
    Release();

 private:
    friend class LoadAdmissionScheduler;

    LoadBudgetLease(LoadAdmissionScheduler* scheduler,
                    milvus::proto::common::LoadPriority priority,
                    size_t bytes);

    LoadAdmissionScheduler* scheduler_{nullptr};
    milvus::proto::common::LoadPriority priority_{
        milvus::proto::common::LoadPriority::LOW};
    size_t bytes_{0};
};

class LoadAdmissionScheduler {
 public:
    explicit LoadAdmissionScheduler(LoadAdmissionConfig config);

    void
    UpdateConfig(LoadAdmissionConfig config);

    folly::SemiFuture<LoadBudgetLease>
    Admit(milvus::proto::common::LoadPriority priority, size_t bytes);

 private:
    friend class LoadBudgetLease;

    struct PendingAdmission {
        milvus::proto::common::LoadPriority priority;
        size_t bytes;
        folly::Promise<LoadBudgetLease> promise;
    };

    void
    Release(milvus::proto::common::LoadPriority priority, size_t bytes);

    std::vector<PendingAdmission>
    TakeAdmittedLocked();

    void
    FulfillAdmissions(std::vector<PendingAdmission> admissions);

    bool
    CanAdmitLocked(milvus::proto::common::LoadPriority priority,
                   size_t bytes) const;

    void
    MarkAdmittedLocked(milvus::proto::common::LoadPriority priority,
                       size_t bytes);

    LoadAdmissionConfig config_;
    size_t used_high_bytes_{0};
    size_t used_low_bytes_{0};
    std::deque<PendingAdmission> high_pending_;
    std::deque<PendingAdmission> low_pending_;
    mutable std::mutex mutex_;
};

LoadAdmissionConfig
LoadAdmissionConfigFromLoadBudget();

LoadAdmissionScheduler&
GetLoadAdmissionScheduler();

}  // namespace milvus::segcore::async_load
