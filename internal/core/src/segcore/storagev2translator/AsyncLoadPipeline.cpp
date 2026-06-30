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
#include <limits>
#include <unordered_map>
#include <utility>

#include "common/EasyAssert.h"
#include "folly/coro/Collect.h"
#include "folly/coro/FutureUtil.h"
#include "segcore/Utils.h"
#include "storage/EntryStreamUtils.h"

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

namespace {

size_t
StorageV3LoadUnitBudgetBytes(const StorageV3LoadUnit& unit) {
    auto overhead_size = unit.loading_overhead_size > 0
                             ? unit.loading_overhead_size
                             : unit.memory_size;
    AssertInfo(overhead_size > 0,
               "[StorageV3] Load unit cid={} has invalid budget bytes {}",
               unit.cid,
               overhead_size);
    return static_cast<size_t>(overhead_size);
}

int64_t
StorageV3BatchReaderMemoryLimit(const StorageV3LoadBatch& batch,
                                int64_t split_target_bytes) {
    auto capped = split_target_bytes > 0
                      ? std::min(batch.batch_memory, split_target_bytes)
                      : batch.batch_memory;
    return std::max<int64_t>(capped,
                             milvus::segcore::FieldDataReadWindowBytes());
}

using StorageV3LoadedCellsPtr = std::shared_ptr<StorageV3LoadedCells>;

StorageV3LoadedCellsPtr
ReadAndFinalizeStorageV3Batch(
    milvus::OpContext* op_ctx,
    StorageV3LoadBatch batch,
    const std::shared_ptr<milvus::segcore::BatchReaderFactory>& reader_factory,
    const std::shared_ptr<milvus::segcore::CellFinalizeFunc>& finalize_cell,
    int64_t split_target_bytes) {
    CheckCancellation(op_ctx, -1, "LoadStorageV3CellsAsync");

    auto tables_result = (*reader_factory)(
        /*batch_key=*/0,
        batch.rg_offset,
        batch.rg_count,
        StorageV3BatchReaderMemoryLimit(batch, split_target_bytes),
        /*read_parallelism=*/1);
    AssertInfo(tables_result.ok(),
               "[StorageV3] Failed to read batch: {}",
               tables_result.status().ToString());
    auto all_tables = std::move(tables_result).ValueOrDie();
    AssertInfo(all_tables.size() == static_cast<size_t>(batch.rg_count),
               "[StorageV3] reader returns less tables than expected, batch rg "
               "count: {}, result size: {}",
               batch.rg_count,
               all_tables.size());
    CheckCancellation(op_ctx, -1, "LoadStorageV3CellsAsync");

    auto loaded_cells = std::make_shared<StorageV3LoadedCells>();
    loaded_cells->reserve(batch.units.size());
    int64_t table_offset = 0;
    for (const auto& unit : batch.units) {
        CheckCancellation(op_ctx, -1, "LoadStorageV3CellsAsync");
        std::vector<std::shared_ptr<arrow::Table>> cell_tables;
        cell_tables.reserve(unit.rg_count);
        for (int64_t i = 0; i < unit.rg_count; ++i) {
            cell_tables.push_back(std::move(all_tables[table_offset + i]));
        }
        table_offset += unit.rg_count;
        loaded_cells->push_back(
            {unit.cid, (*finalize_cell)(cell_tables, unit.cid)});
    }

    return loaded_cells;
}

}  // namespace

size_t
StorageV3LoadBatchBudgetBytes(const StorageV3LoadBatch& batch) {
    size_t total = 0;
    for (const auto& unit : batch.units) {
        auto bytes = StorageV3LoadUnitBudgetBytes(unit);
        if (bytes > std::numeric_limits<size_t>::max() - total) {
            return std::numeric_limits<size_t>::max();
        }
        total += bytes;
    }
    return total;
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

folly::coro::Task<StorageV3LoadedCells>
LoadStorageV3CellsAsync(milvus::OpContext* op_ctx,
                        std::vector<StorageV3LoadUnit> units,
                        milvus::segcore::BatchReaderFactory reader_factory,
                        milvus::segcore::CellFinalizeFunc finalize_cell,
                        int64_t split_target_bytes,
                        milvus::proto::common::LoadPriority priority,
                        StorageV3AdmissionScheduler& scheduler) {
    if (units.empty()) {
        co_return StorageV3LoadedCells{};
    }

    std::vector<milvus::cachinglayer::cid_t> requested_cids;
    requested_cids.reserve(units.size());
    for (const auto& unit : units) {
        requested_cids.push_back(unit.cid);
    }

    auto batches =
        BuildStorageV3LoadBatches(std::move(units), split_target_bytes);
    auto shared_reader = std::make_shared<milvus::segcore::BatchReaderFactory>(
        std::move(reader_factory));
    auto shared_finalizer = std::make_shared<milvus::segcore::CellFinalizeFunc>(
        std::move(finalize_cell));

    std::vector<folly::coro::Task<StorageV3LoadedCellsPtr>> tasks;
    tasks.reserve(batches.size());
    for (auto& batch : batches) {
        auto budget_bytes = StorageV3LoadBatchBudgetBytes(batch);
        tasks.push_back(AdmitAndSubmitStorageV3LoadTask(
            scheduler,
            priority,
            budget_bytes,
            [op_ctx,
             batch = std::move(batch),
             shared_reader,
             shared_finalizer,
             split_target_bytes]() mutable {
                return ReadAndFinalizeStorageV3Batch(op_ctx,
                                                     std::move(batch),
                                                     shared_reader,
                                                     shared_finalizer,
                                                     split_target_bytes);
            }));
    }

    auto batch_results =
        co_await folly::coro::collectAllRange(std::move(tasks));

    std::unordered_map<milvus::cachinglayer::cid_t,
                       std::unique_ptr<milvus::GroupChunk>>
        cells_by_cid;
    cells_by_cid.reserve(requested_cids.size());
    for (auto& batch_result : batch_results) {
        for (auto& cell : *batch_result) {
            cells_by_cid.emplace(cell.cid, std::move(cell.chunk));
        }
    }

    StorageV3LoadedCells ordered_cells;
    ordered_cells.reserve(requested_cids.size());
    for (auto cid : requested_cids) {
        auto it = cells_by_cid.find(cid);
        AssertInfo(
            it != cells_by_cid.end(), "[StorageV3] cell {} not loaded", cid);
        ordered_cells.push_back({cid, std::move(it->second)});
    }

    co_return ordered_cells;
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

void
StorageV3AdmissionScheduler::UpdateConfig(StorageV3AdmissionConfig config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config.high_reserved_bytes =
        std::min(config.high_reserved_bytes, config.total_bytes);
    config_ = config;
    TryScheduleLocked();
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
StorageV3AdmissionScheduler::MarkAdmittedLocked(
    milvus::proto::common::LoadPriority priority, size_t bytes) {
    if (IsHighPriority(priority)) {
        used_high_bytes_ += bytes;
        return;
    }
    used_low_bytes_ += bytes;
}

StorageV3AdmissionConfig
StorageV3AdmissionConfigFromLoadBudget() {
    auto total_bytes =
        milvus::storage::TransientMemoryBudget::GetLoadTransientBudget()
            .CapacityBytes();
    auto high_reserved_bytes =
        total_bytes == 0 ? size_t{0} : std::max<size_t>(1, total_bytes / 4);
    return {total_bytes, high_reserved_bytes};
}

StorageV3AdmissionScheduler&
GetStorageV3LoadAdmissionScheduler() {
    static StorageV3AdmissionScheduler scheduler(
        StorageV3AdmissionConfigFromLoadBudget());
    scheduler.UpdateConfig(StorageV3AdmissionConfigFromLoadBudget());
    return scheduler;
}

folly::coro::Task<int>
StorageV3AsyncLoadCoroSmokeTest() {
    co_return co_await folly::coro::toTask(SubmitStorageV3LoadTask(
        milvus::proto::common::LoadPriority::HIGH, [] { return 1; }));
}

}  // namespace milvus::segcore::storagev2translator
