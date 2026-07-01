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
#include <exception>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "arrow/util/thread_pool.h"
#include "common/EasyAssert.h"
#include "folly/coro/Collect.h"
#include "folly/coro/FutureUtil.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/thread_factory/NamedThreadFactory.h"
#include "segcore/Utils.h"
#include "storage/EntryStreamUtils.h"

namespace milvus::segcore::storagev2translator {

namespace {

bool
IsHighPriority(milvus::proto::common::LoadPriority priority) {
    return priority == milvus::proto::common::LoadPriority::HIGH;
}

class MilvusThreadPoolArrowExecutor final : public arrow::internal::Executor {
 public:
    explicit MilvusThreadPoolArrowExecutor(milvus::ThreadPoolPriority priority)
        : priority_(priority) {
    }

    int
    GetCapacity() override {
        return static_cast<int>(
            milvus::ThreadPools::GetThreadPool(priority_).GetMaxThreadNum());
    }

    bool
    OwnsThisThread() override {
        return current_executor_ == this;
    }

 protected:
    arrow::Status
    SpawnReal(arrow::internal::TaskHints,
              arrow::internal::FnOnce<void()> task,
              arrow::StopToken stop_token,
              StopCallback&& stop_callback) override {
        auto task_ptr =
            std::make_shared<arrow::internal::FnOnce<void()>>(std::move(task));
        auto stop_callback_ptr =
            std::make_shared<StopCallback>(std::move(stop_callback));
        try {
            milvus::ThreadPools::GetThreadPool(priority_).Submit(
                [this,
                 task_ptr,
                 stop_token = std::move(stop_token),
                 stop_callback_ptr]() mutable {
                    if (stop_token.IsStopRequested()) {
                        if (*stop_callback_ptr) {
                            std::move (*stop_callback_ptr)(stop_token.Poll());
                        }
                        return;
                    }
                    auto* previous_executor = current_executor_;
                    current_executor_ = this;
                    try {
                        std::move (*task_ptr)();
                    } catch (...) {
                        current_executor_ = previous_executor;
                        throw;
                    }
                    current_executor_ = previous_executor;
                });
        } catch (const std::exception& e) {
            return arrow::Status::IOError(e.what());
        } catch (...) {
            return arrow::Status::IOError(
                "failed to submit task to Milvus thread pool");
        }
        return arrow::Status::OK();
    }

 private:
    milvus::ThreadPoolPriority priority_;
    static thread_local MilvusThreadPoolArrowExecutor* current_executor_;
};

thread_local MilvusThreadPoolArrowExecutor*
    MilvusThreadPoolArrowExecutor::current_executor_ = nullptr;

arrow::internal::Executor*
StorageV3HighPriorityExecutor() {
    static MilvusThreadPoolArrowExecutor executor(milvus::HIGH);
    return &executor;
}

class StorageV3DiskArrowExecutor final : public arrow::internal::Executor {
 public:
    StorageV3DiskArrowExecutor()
        : thread_count_(std::max(1, milvus::CPU_NUM)),
          executor_(
              thread_count_,
              std::make_shared<folly::NamedThreadFactory>("STORAGEV3_DISK_")) {
    }

    int
    GetCapacity() override {
        return static_cast<int>(thread_count_);
    }

    bool
    OwnsThisThread() override {
        return current_executor_ == this;
    }

 protected:
    arrow::Status
    SpawnReal(arrow::internal::TaskHints,
              arrow::internal::FnOnce<void()> task,
              arrow::StopToken stop_token,
              StopCallback&& stop_callback) override {
        auto task_ptr =
            std::make_shared<arrow::internal::FnOnce<void()>>(std::move(task));
        auto stop_callback_ptr =
            std::make_shared<StopCallback>(std::move(stop_callback));
        try {
            executor_.add([this,
                           task_ptr,
                           stop_token = std::move(stop_token),
                           stop_callback_ptr]() mutable {
                if (stop_token.IsStopRequested()) {
                    if (*stop_callback_ptr) {
                        std::move (*stop_callback_ptr)(stop_token.Poll());
                    }
                    return;
                }
                auto* previous_executor = current_executor_;
                current_executor_ = this;
                try {
                    std::move (*task_ptr)();
                } catch (...) {
                    current_executor_ = previous_executor;
                    throw;
                }
                current_executor_ = previous_executor;
            });
        } catch (const std::exception& e) {
            return arrow::Status::IOError(e.what());
        } catch (...) {
            return arrow::Status::IOError(
                "failed to submit task to StorageV3 disk executor");
        }
        return arrow::Status::OK();
    }

 private:
    size_t thread_count_;
    folly::CPUThreadPoolExecutor executor_;
    static thread_local StorageV3DiskArrowExecutor* current_executor_;
};

thread_local StorageV3DiskArrowExecutor*
    StorageV3DiskArrowExecutor::current_executor_ = nullptr;

template <typename Result, typename Func>
folly::SemiFuture<Result>
SubmitStorageV3ArrowExecutorTask(arrow::internal::Executor* executor,
                                 Func&& func) {
    if (executor == nullptr) {
        return folly::makeSemiFuture(
            Result(arrow::Status::Invalid("StorageV3 executor is null")));
    }

    using TaskFunc = std::decay_t<Func>;
    auto task = std::make_shared<TaskFunc>(std::forward<Func>(func));

    if (executor->OwnsThisThread()) {
        try {
            return folly::makeSemiFuture(std::invoke(*task));
        } catch (...) {
            return folly::makeSemiFuture<Result>(
                folly::exception_wrapper(std::current_exception()));
        }
    }

    folly::Promise<Result> promise;
    auto future = promise.getSemiFuture();
    auto shared_promise =
        std::make_shared<folly::Promise<Result>>(std::move(promise));
    auto status = executor->Spawn([shared_promise, task]() mutable {
        try {
            shared_promise->setValue(std::invoke(*task));
        } catch (...) {
            shared_promise->setException(
                folly::exception_wrapper(std::current_exception()));
        }
    });
    if (!status.ok()) {
        shared_promise->setValue(Result(std::move(status)));
    }
    return future;
}

}  // namespace

arrow::internal::Executor*
StorageV3DiskExecutor() {
    static StorageV3DiskArrowExecutor executor;
    return &executor;
}

std::vector<StorageV3ReadTask>
BuildStorageV3ReadTasks(std::vector<StorageV3LoadUnit> units,
                        int64_t read_task_target_bytes) {
    std::sort(
        units.begin(),
        units.end(),
        [](const StorageV3LoadUnit& left, const StorageV3LoadUnit& right) {
            if (left.rg_offset != right.rg_offset) {
                return left.rg_offset < right.rg_offset;
            }
            return left.cid < right.cid;
        });

    std::vector<StorageV3ReadTask> tasks;
    StorageV3ReadTask current{};

    for (auto& unit : units) {
        bool should_split = false;
        if (!current.units.empty()) {
            auto next_expected_rg_offset = current.rg_offset + current.rg_count;
            auto task_full =
                read_task_target_bytes > 0 &&
                current.task_memory + unit.memory_size > read_task_target_bytes;
            if (unit.rg_offset != next_expected_rg_offset || task_full) {
                should_split = true;
            }
        }
        if (should_split) {
            tasks.push_back(std::move(current));
            current = {};
        }
        if (current.units.empty()) {
            current.rg_offset = unit.rg_offset;
            current.rg_count = 0;
            current.task_memory = 0;
        }
        current.rg_count += unit.rg_count;
        current.task_memory += unit.memory_size;
        current.units.push_back(std::move(unit));
    }

    if (!current.units.empty()) {
        tasks.push_back(std::move(current));
    }

    return tasks;
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

arrow::Result<std::vector<std::shared_ptr<arrow::Table>>>
RecordBatchesToTables(
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches) {
    std::vector<std::shared_ptr<arrow::Table>> tables;
    tables.reserve(batches.size());
    for (auto& batch : batches) {
        auto table_result = arrow::Table::FromRecordBatches({std::move(batch)});
        if (!table_result.ok()) {
            return table_result.status();
        }
        tables.push_back(std::move(table_result).ValueOrDie());
    }
    return tables;
}

StorageV3LoadResult
FinalizeStorageV3ReadTask(
    milvus::OpContext* op_ctx,
    StorageV3ReadTask task,
    milvus_storage::api::RecordBatchVector batches,
    const std::shared_ptr<milvus::segcore::CellFinalizeFunc>& finalize_cell,
    const char* cancellation_scope) {
    CheckCancellation(op_ctx, -1, "LoadStorageV3CellsAsync");

    auto tables_result = RecordBatchesToTables(std::move(batches));
    if (!tables_result.ok()) {
        return tables_result.status();
    }
    auto all_tables = std::move(tables_result).ValueOrDie();

    AssertInfo(
        all_tables.size() == static_cast<size_t>(task.rg_count),
        "[StorageV3] reader returns less tables than expected, read task rg "
        "count: {}, result size: {}",
        task.rg_count,
        all_tables.size());
    CheckCancellation(op_ctx, -1, cancellation_scope);

    auto loaded_cells = std::make_shared<StorageV3LoadedCells>();
    loaded_cells->reserve(task.units.size());
    int64_t table_offset = 0;
    for (const auto& unit : task.units) {
        CheckCancellation(op_ctx, -1, cancellation_scope);
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

folly::coro::Task<StorageV3LoadedCellsPtr>
ReadAndFinalizeStorageV3TaskAsync(
    milvus::OpContext* op_ctx,
    StorageV3ReadTask task,
    const std::shared_ptr<StorageV3AsyncLoadFn>& load,
    milvus::proto::common::LoadPriority priority,
    StorageV3AdmissionScheduler& scheduler) {
    constexpr const char* cancellation_scope = "LoadStorageV3CellsAsync";
    CheckCancellation(op_ctx, -1, cancellation_scope);

    auto budget_bytes = StorageV3ReadTaskBudgetBytes(task);
    auto lease =
        co_await folly::coro::toTask(scheduler.Admit(priority, budget_bytes));
    (void)lease;

    CheckCancellation(op_ctx, -1, cancellation_scope);
    auto loaded_cells_result =
        co_await folly::coro::toTask((*load)(op_ctx, std::move(task)));
    AssertInfo(loaded_cells_result.ok(),
               "[StorageV3] Failed to load task: {}",
               loaded_cells_result.status().ToString());
    co_return std::move(loaded_cells_result).ValueOrDie();
}

}  // namespace

size_t
StorageV3ReadTaskBudgetBytes(const StorageV3ReadTask& task) {
    size_t total = 0;
    for (const auto& unit : task.units) {
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

StorageV3AsyncLoadFn
MakeStorageV3ChunkLoadFn(
    std::shared_ptr<milvus_storage::api::ChunkReader> chunk_reader,
    milvus::segcore::CellFinalizeFunc finalize_cell,
    StorageV3LocalizeExecutor localize_executor) {
    auto shared_finalizer = std::make_shared<milvus::segcore::CellFinalizeFunc>(
        std::move(finalize_cell));
    return [chunk_reader = std::move(chunk_reader),
            shared_finalizer,
            localize_executor](milvus::OpContext* op_ctx,
                               StorageV3ReadTask task)
               -> folly::SemiFuture<StorageV3LoadResult> {
        constexpr const char* cancellation_scope = "LoadStorageV3CellsAsync";
        std::vector<int64_t> rg_indices(task.rg_count);
        std::iota(rg_indices.begin(), rg_indices.end(), task.rg_offset);

        milvus_storage::api::AsyncReadOptions options;
        options.read_parallelism = 1;
        options.materialize_executor = StorageV3HighPriorityExecutor();

        try {
            if (localize_executor == StorageV3LocalizeExecutor::Materialize) {
                return milvus_storage::api::get_chunks_async_then(
                    chunk_reader,
                    std::move(rg_indices),
                    options,
                    [op_ctx, task = std::move(task), shared_finalizer](
                        milvus_storage::api::RecordBatchVector&&
                            batches) mutable -> StorageV3LoadResult {
                        return FinalizeStorageV3ReadTask(op_ctx,
                                                         std::move(task),
                                                         std::move(batches),
                                                         shared_finalizer,
                                                         cancellation_scope);
                    });
            }

            auto read_future =
                chunk_reader->get_chunks_async(std::move(rg_indices), options);
            return std::move(read_future)
                .deferValue(
                    [chunk_reader,
                     op_ctx,
                     task = std::move(task),
                     shared_finalizer](
                        arrow::Result<milvus_storage::api::RecordBatchVector>&&
                            read_result) mutable
                    -> folly::SemiFuture<StorageV3LoadResult> {
                        (void)chunk_reader;
                        if (!read_result.ok()) {
                            return folly::makeSemiFuture(
                                StorageV3LoadResult(read_result.status()));
                        }

                        auto batches = read_result.MoveValueUnsafe();
                        return SubmitStorageV3ArrowExecutorTask<
                            StorageV3LoadResult>(
                            StorageV3DiskExecutor(),
                            [op_ctx,
                             task = std::move(task),
                             batches = std::move(batches),
                             shared_finalizer]() mutable
                            -> StorageV3LoadResult {
                                return FinalizeStorageV3ReadTask(
                                    op_ctx,
                                    std::move(task),
                                    std::move(batches),
                                    shared_finalizer,
                                    cancellation_scope);
                            });
                    });
        } catch (...) {
            return folly::makeSemiFuture<StorageV3LoadResult>(
                folly::exception_wrapper(std::current_exception()));
        }
    };
}

folly::coro::Task<StorageV3LoadedCells>
LoadStorageV3CellsAsync(milvus::OpContext* op_ctx,
                        std::vector<StorageV3LoadUnit> units,
                        StorageV3AsyncLoadFn load,
                        int64_t read_task_target_bytes,
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

    auto read_tasks =
        BuildStorageV3ReadTasks(std::move(units), read_task_target_bytes);
    auto shared_loader =
        std::make_shared<StorageV3AsyncLoadFn>(std::move(load));

    std::vector<folly::coro::Task<StorageV3LoadedCellsPtr>> tasks;
    tasks.reserve(read_tasks.size());
    for (auto& read_task : read_tasks) {
        tasks.push_back(ReadAndFinalizeStorageV3TaskAsync(
            op_ctx, std::move(read_task), shared_loader, priority, scheduler));
    }

    auto task_results = co_await folly::coro::collectAllRange(std::move(tasks));

    std::unordered_map<milvus::cachinglayer::cid_t,
                       std::unique_ptr<milvus::GroupChunk>>
        cells_by_cid;
    cells_by_cid.reserve(requested_cids.size());
    for (auto& task_result : task_results) {
        for (auto& cell : *task_result) {
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

folly::SemiFuture<StorageV3LoadBudgetLease>
StorageV3AdmissionScheduler::Admit(milvus::proto::common::LoadPriority priority,
                                   size_t bytes) {
    folly::Promise<StorageV3LoadBudgetLease> promise;
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
        promise.setValue(StorageV3LoadBudgetLease(this, priority, bytes));
    }
    return future;
}

void
StorageV3AdmissionScheduler::Release(
    milvus::proto::common::LoadPriority priority, size_t bytes) {
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

std::vector<StorageV3AdmissionScheduler::PendingAdmission>
StorageV3AdmissionScheduler::TakeAdmittedLocked() {
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
StorageV3AdmissionScheduler::FulfillAdmissions(
    std::vector<PendingAdmission> admissions) {
    for (auto& admission : admissions) {
        admission.promise.setValue(StorageV3LoadBudgetLease(
            this, admission.priority, admission.bytes));
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
