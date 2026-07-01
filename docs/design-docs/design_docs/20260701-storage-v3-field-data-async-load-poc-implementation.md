# StorageV3 Field Data Async Load PoC 实现文档

本文档整理 StorageV3 field data load 的 `folly::coro` PoC 实现。对应实现只接入新建的 `ManifestGroupTranslatorV2`，不改动原有 `ManifestGroupTranslator` 和 `GroupChunkTranslator` 的生产路径。

## 目标和边界

目标：

- 为 StorageV3 field data load 增加一套独立的 async load pipeline PoC。
- 保留 `cachinglayer::Translator<GroupChunk>::get_cells()` 的同步接口，由 V2 内部 `blockingWait` coroutine 结果。
- 在 cell 粒度做 admission，IO 侧仍可把连续 row group 合并成 batch。
- materialize/read task 复用 `storage::ThreadPools` 的 HIGH/LOW pool。
- 使用 HIGH/LOW pending queue 实现 HIGH 优先和 LOW 不抢 HIGH reserved 的调度语义。
- 用 V1/V2 parity test 锁住行为，便于后续 TDD 演进。

非目标：

- 不删除或重写现有 `ManifestGroupTranslator`。
- 不改 `cachinglayer::Translator` public ABI。
- 不实现完整动态 QoS governor。
- 不拆独立 mmap write executor；mmap materialize 仍沿用 `create_group_chunk(..., load_priority_)`。
- 当前 PoC 没有接入 milvus-storage native async API；`BatchReaderFactory` 仍是同步读取接口，coroutine 主要编排 admission、thread-pool submit、batch result collect。

## 文件结构

新增/改造的核心文件：

- `internal/core/src/segcore/storagev2translator/ManifestGroupTranslatorV2.h`
- `internal/core/src/segcore/storagev2translator/ManifestGroupTranslatorV2.cpp`
- `internal/core/src/segcore/storagev2translator/ManifestGroupTranslatorV2Test.cpp`
- `internal/core/src/segcore/storagev2translator/AsyncLoadPipeline.h`
- `internal/core/src/segcore/storagev2translator/AsyncLoadPipeline.cpp`
- `internal/core/src/segcore/storagev2translator/AsyncLoadPipelineTest.cpp`

旧路径仍保留：

- `ManifestGroupTranslator.cpp` 继续使用 `LoadCellBatchAsync(...)`
- `GroupChunkTranslator.cpp` 继续使用 `LoadCellBatchAsync(...)`
- `segcore/memory_planner.cpp` 内的 `LoadCellBatchAsync(...)` 未被删除

## 总体流程

V2 `get_cells()` 的执行链路：

```text
ManifestGroupTranslatorV2::get_cells(cids)
  -> BuildStorageV3LoadUnitsForCells(meta_, cids, loading_overhead_bytes)
  -> MakeChunkReaderFactory(chunk_reader_)
  -> blockingWait(LoadStorageV3CellsAsync(...))
       -> BuildStorageV3LoadBatches(units, FieldDataLoadBatchSplitTargetBytes())
       -> for each batch:
            AdmitAndSubmitStorageV3LoadTask(scheduler, priority, budget, task)
              -> scheduler.Admit(priority, budget)
              -> SubmitStorageV3LoadTask(priority, task)
                   -> ThreadPools::GetThreadPool(PriorityForLoad(priority))
                   -> ReadAndFinalizeStorageV3Batch(...)
       -> collectAllRange(batch tasks)
       -> reorder loaded cells by requested cids
  -> return vector<pair<cid, GroupChunk>>
```

V2 的同步边界在 `ManifestGroupTranslatorV2::get_cells()` 内：

```cpp
auto loaded_cells = folly::coro::blockingWait(LoadStorageV3CellsAsync(...));
```

这样不需要改 `Translator` 接口，也不影响 cache layer 上层调用方式。

## LoadUnit 和 Batch Planning

PoC 引入两个中间结构：

```cpp
struct StorageV3LoadUnit {
    cachinglayer::cid_t cid;
    int64_t rg_offset;
    int64_t rg_count;
    int64_t memory_size;
    int64_t loading_overhead_size;
};

struct StorageV3LoadBatch {
    int64_t rg_offset;
    int64_t rg_count;
    int64_t batch_memory;
    std::vector<StorageV3LoadUnit> units;
};
```

`BuildStorageV3LoadUnitsForCells(...)` 从 `GroupCTMeta` 把 requested cids 转成 row-group range：

- `cid`
- `rg_offset`
- `rg_count`
- `memory_size`
- `loading_overhead_size`

`BuildStorageV3LoadBatches(...)` 负责 IO batch 合并：

- 先按 `rg_offset` 排序。
- 连续 row group 且累计 `batch_memory` 不超过 split target 时合并。
- row group 不连续或超过 split target 时拆 batch。
- batch 内部按 row group 顺序读取，最终结果再按原始 requested cids reorder。

这解决了“异步粒度”和“IO 粒度”不完全对齐的问题：admission 和返回语义按 cell，实际 read 可以按 batch 合并。

## Coroutine 和 ThreadPool Bridge

`SubmitStorageV3LoadTask(...)` 把 `storage::ThreadPools` 的 `Submit` 包装成 `folly::SemiFuture<T>`：

- 根据 `LoadPriority` 映射到 `ThreadPoolPriority::HIGH` 或 `ThreadPoolPriority::LOW`。
- task 内捕获 `folly::Promise`，成功时 `setValue`，异常时 `setException`。
- 外层 coroutine 通过 `folly::coro::toTask(...)` await 这个 `SemiFuture`。

`AdmitAndSubmitStorageV3LoadTask(...)` 是 admission + submit 的 coroutine helper：

```cpp
auto lease = co_await folly::coro::toTask(scheduler.Admit(priority, bytes));
co_return co_await folly::coro::toTask(SubmitStorageV3LoadTask(priority, task));
```

注意：这里的 callable 参数按值接收，而不是 forwarding reference。原因是 coroutine body 不是调用时立即执行；如果接收 rvalue 引用，lambda 可能在 coroutine 真正启动前已经失效。这个问题在测试中表现为 ASAN bad-free，修复方式是让 callable 进入 coroutine frame 后再转成 `shared_ptr`。

## Admission Scheduler

`StorageV3AdmissionScheduler` 是 PoC 级调度器，提供：

- HIGH/LOW 两个 pending queue。
- HIGH pending 优先调度。
- LOW 不能占用 HIGH reserved bytes。
- 新来的 LOW 不能绕过已经排队的 HIGH。
- `total_bytes == 0` 表示 unlimited。
- oversized batch 可在空闲时独占运行，避免大 cell 永久无法准入。
- `UpdateConfig(...)` 支持跟随外部 budget capacity 变化并唤醒 pending request。

默认 scheduler：

```cpp
StorageV3AdmissionScheduler& GetStorageV3LoadAdmissionScheduler();
```

默认 config 来自现有进程级 load transient budget：

```cpp
auto total_bytes =
    storage::TransientMemoryBudget::GetLoadTransientBudget().CapacityBytes();
auto high_reserved_bytes =
    total_bytes == 0 ? 0 : max<size_t>(1, total_bytes / 4);
```

这只是 PoC 默认值。后续如果要产品化，HIGH reserved 比例和 admission capacity 应该变成可配置项，并接入 metrics。

## Read 和 Materialize

`ReadAndFinalizeStorageV3Batch(...)` 执行单个 batch：

1. 检查 cancellation。
2. 调用 `BatchReaderFactory(batch_key=0, rg_offset, rg_count, reader_memory_limit, read_parallelism=1)`。
3. 校验返回 table 数量等于 batch row group 数。
4. 按每个 unit 的 `rg_count` 把 batch tables 切回 cell tables。
5. 调用 `CellFinalizeFunc` 生成 `GroupChunk`。

当前 `ManifestGroupTranslatorV2` 使用：

```cpp
auto factory = milvus::segcore::MakeChunkReaderFactory(chunk_reader_);
```

materialize 使用原有 `load_group_chunk(...)`，包括：

- field id 解析
- external field name fallback
- Arrow array normalize
- memory/mmap `create_group_chunk(...)`
- mmap 文件名 generation suffix

因此 V2 的数据 materialize 结果通过 V1/V2 parity test 做行为对照。

## ManifestGroupTranslatorV2 接入

V2 初始从 V1 shadow copy 而来，后续仅 V2 改造成新 pipeline。

`ManifestGroupTranslatorV2::get_cells()` 关键变化：

- 空 `cids` 早返回，避免 `max_element` 空输入。
- cid range 校验仍保留。
- 构建 `StorageV3LoadUnit`，不再转成 `CellSpec`。
- 不再创建 `CellReaderChannel`。
- 不再调用 `LoadCellBatchAsync(...)`。
- 改为 `blockingWait(LoadStorageV3CellsAsync(...))`。
- 返回结果保持 requested cids 顺序。

V1 仍作为 fallback 和 parity baseline。

## TDD 覆盖

`ManifestGroupTranslatorAsyncPipelineTest` 覆盖：

- coroutine smoke test。
- ThreadPool submit 可异步执行并返回 `SemiFuture`。
- 异常能传播到 `SemiFuture`。
- `SubmitStorageV3LoadTask` 可被 `folly::coro::Task` await。
- admission 未 granted 前 task 不会启动。
- batch planner sort/merge contiguous row groups。
- batch planner 按 split target 拆分。
- `LoadStorageV3CellsAsync` 合并 IO batch，并按 requested order 返回。
- HIGH pending 优先于 LOW。
- 新 LOW 不能绕过 queued HIGH。
- LOW 不能占用 HIGH reserved bytes。
- capacity 0 unlimited。
- oversized batch 空闲时独占运行。
- `UpdateConfig` 唤醒 pending request。

`ManifestGroupTranslatorV2ParityTest` 覆盖：

- V2 和 V1 的 `num_cells/key/cell_id/estimated_byte_size/get_cells` parity。
- 从 requested cells 构建 `StorageV3LoadUnit` 的 row group range 正确。
- tiny admission budget 下 V2 仍能完成真实 StorageV3 load。
- 空 `cids` 返回空结果。

`ManifestGroupTranslatorTest` 继续覆盖旧 V1 路径，确认没有被新 PoC 破坏。

## 已验证命令

构建：

```bash
ninja -C cmake_build all_tests
```

结果：通过。

本次改动相关测试：

```bash
source ./scripts/setenv.sh >/dev/null
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
LD_LIBRARY_PATH="$PWD/cmake_build/src:$PWD/cmake_build/thirdparty/milvus-storage/milvus-storage-build:${LD_LIBRARY_PATH:-}" \
./cmake_build/unittest/all_tests \
  --gtest_filter="ManifestGroupTranslatorAsyncPipelineTest.*:*ManifestGroupTranslatorV2ParityTest*:*ManifestGroupTranslatorTest*"
```

结果：34/34 通过。

全量 C++ `all_tests` 曾运行但未完成，且在中途已出现非本次相关失败：

- `ArithmeticCheck/TypedScalarIndexCreatorTest/*.Codec`
- `CBoolIndexTest.All`
- `CInt64IndexTest.All`

失败原因是本机 `/tmp/milvus` 目录属于其他用户且权限为 `700`，导致测试打开 `/tmp/milvus/mmap_test` permission denied。该失败不在 storagev2translator 路径。

## 当前限制和后续工作

当前限制：

- 还没有接入 milvus-storage native async public API。
- `ReadAndFinalizeStorageV3Batch` 内部 `read_parallelism` 固定为 1。
- admission config 只从 `TransientMemoryBudget` 推导，HIGH reserved 比例固定为 1/4。
- admission 无 metrics。
- cancellation 只在 pipeline 的关键阶段检查，pending admission 本身还没有 timeout/TTL。
- V2 通过 `blockingWait` 兼容同步 `get_cells()`，还不是端到端 async API。

建议后续 TDD 顺序：

1. 给 `BatchReaderFactory` 增加 async variant 或引入 milvus-storage `get_chunks_async` adapter。
2. 增加 read_parallelism 计算，复用 `FieldDataReadWindowBytes()` / `FieldDataMaxReadParallelism()`。
3. 增加 admission metrics：pending queue length、admitted bytes、wait duration、priority。
4. 增加 pending admission cancellation/timeout 语义。
5. 加 feature flag，在真实 load path 里选择 V1/V2。
6. 跑固定 workload benchmark，对比 V1 `LoadCellBatchAsync` 和 V2 coroutine pipeline 的 P50/P99、峰值 transient memory、HIGH load tail latency。

## Commit 列表

- `8b7b2a0586 enhance: add storage v3 async load poc scaffold`
- `b235dba50f enhance: add async load cell orchestrator`
- `2d3a759bd6 enhance: refine storage v3 load admission`
- `171c5610f8 enhance: route manifest translator v2 through async load`
- `f2912ee9ae fix: preserve queued high load admission priority`
- `fe1087b47e fix: handle empty manifest translator v2 cell loads`
