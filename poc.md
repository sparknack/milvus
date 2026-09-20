# InvertedIndexKnowhere 纯内存过滤 PoC 设计与实施计划

## 文档状态

- 当前阶段：PoC 实现与验证。
- 第一阶段只做 docID posting，不包含 position 链。
- position、词频、phrase/proximity 和 BM25 作为后续独立阶段。

实现入口、可执行命令及当前验证边界见
[PoC 实现说明](tools/knowhere-scalar-poc/README.md)。独立 codec/core 测试已通过；
Milvus C++ 依赖及完整链接已完成，17 项回归通过；百万行直接 API 与 segcore
Build/LoadIndex/Search 对比已运行，结果见
[实测报告](tools/knowhere-scalar-poc/results/2026-09-18-arm64/README.md)。
范围过滤有收益，但当前内存口径不支持“内存节省”结论；持久化文件重载仍不在本 PoC 范围内。

2026-09-20 补充 [AG News 真实文本实测](tools/knowhere-scalar-poc/results/2026-09-20-ag-news/README.md)：
12 万标题、12 万正文和 100 万单词值，18 项回归及 36 个 segcore 查询场景通过。
这是 VARCHAR 完整值的标量过滤，不是 BM25/分词全文检索。

## 1. 背景与目标

Milvus 当前 scalar inverted index 使用 Tantivy。这个 PoC 希望验证另一条路径：
复用 Knowhere sparse inverted index 当前采用的 block-compressed posting 格式，
实现一个面向 scalar filter 的纯内存倒排索引，并通过现有
`ScalarIndex<T>` 统一接口调用独立实现，对比两者在索引完全驻留内存时的性能。

需要回答四个问题：

1. Knowhere 的 256-entry StreamVByte posting block 是否适合 scalar filter？
2. 等值、`IN`、`NOT IN` 和范围查询与 Tantivy 相比表现如何？
3. 两种索引的常驻内存、bytes/row 和构建成本分别是多少？
4. 如果性能不理想，瓶颈来自 term 字典、posting 解码，还是结果 bitmap 物化？

## 2. 方案摘要

1. 新增 `InvertedIndexKnowhere<T>`，但不直接调用 Knowhere sparse ANN Search。
2. 复用 Knowhere 的 posting header、256-entry block、docID delta 和
   StreamVByte codec。
3. 删除 sparse value/score payload，形成 filter 专用的 docID-only 格式。
4. term 字典使用有序数组，同时支持 exact 和 range。
5. 只在 `BuildWithRawDataForUT` 中通过显式配置开启，不修改生产构建、加载或
   `IndexFactory`。
6. Tantivy 以 `load_in_mmap=false` 重新打开并预热，确保比较 RAM versus RAM。
7. 第一阶段不存 position，但 term metadata 为后续平行 position stream 预留
   扩展点。

## 3. 术语和粒度

- **term**：scalar field 的一个唯一值。VARCHAR scalar index 使用完整字符串，
  本阶段不分词。
- **docID / row ID**：term 所属的 Milvus 行偏移。
- **posting list**：同一 term 对应的有序 docID 列表。
- **codec block**：最多包含 256 个 docID 的压缩单元。
- **posting blob**：保存所有 posting list 的逻辑连续字节区。
- **I/O chunk**：未来从对象存储独立读取的数据单元，目标不超过 4 MiB。
- **position 链**：term 在文档中的位置序列，用于 phrase、proximity、词频和
  后续 BM25；本阶段不实现。

三种粒度的关系是：

```text
4 MiB I/O chunk（后续）
  └── 多个 posting list
        └── 多个 256-entry codec block
```

长 posting 将来允许跨越多个 4 MiB chunk；单个 codec block 不应跨 chunk。
当前纯内存 PoC 不实现 chunk，只要求 offset 设计不能阻碍后续拆分。

## 4. Knowhere 现状分析

当前仓库固定的 Knowhere 版本，其推荐 index version 是 8。sealed sparse index
在非 Cardinal 构建中，版本 8/9 默认使用 flat codec；本 PoC 显式选择以下
已有编码路径（不依赖默认选择）：

- `BlockInvertedIndex`；
- `block_streamvbyte`；
- 每个压缩 block 最多 256 条 posting。

Knowhere 单个 posting list 当前布局如下：

```text
varint posting_count
int32  block_max_doc_ids[num_blocks]
uint32 block_end_offsets[num_blocks - 1]
block_0
block_1
...

block := StreamVByte(delta_doc_ids) + sparse_value_payload
```

Knowhere 的公开 sparse API 面向 ANN：输入 sparse vector，执行维度遍历、分数
累加和 top-k；它没有公开的 `term -> posting cursor`。因此用 one-hot sparse
vector 调 ANN Search 并不能公平衡量 scalar filter。PoC 应复用底层 posting
表示，而不是复用 ANN 查询接口。

## 5. 范围与非目标

### 5.1 第一阶段包含

- 主要 benchmark 类型为 `int64_t` scalar field。
- 实现允许时支持其他可排序 primitive 和 string 类型。
- `IN`、`NOT IN`、单边 range、双边 range、`IsNull`、`IsNotNull`。
- `InApplyFilter` 和 `InApplyCallback` 保持 Tantivy 当前语义。
- 纯内存构建、查询、内存统计和 Tantivy 逐位结果对比。
- end-to-end benchmark 以及必要的分层 microbenchmark。

### 5.2 第一阶段不包含

- position 链、term frequency、phrase 和 proximity。
- BM25、WAND、MaxScore、top-k 和打分。
- 分词、`TextMatch`、正则、前缀、后缀和 contains。
- Array、JSON 和 nested index。
- 生产级 Serialize、Upload、Load、mmap 和 S3。
- 4 MiB chunk 的实际切分和按需 I/O。
- 用户可见参数、默认后端变更和生产发布。

## 6. 总体架构

```text
segcore LoadIndex / Search
           |
     ScalarIndex<T>
      /           \
InvertedIndexTantivy<T>   InvertedIndexKnowhere<T>
                                  |
                         InvertedIndexKnowhereCore
                                  |
                         有序 term 表 + metadata
                                  |
                         StreamVByte docID blocks
```

两者是独立实现。生产 Tantivy 和 Meta.h 不增加实验分支；测试通过统一接口
验证查询链。Tantivy RAM/null 构建准备仅存在于测试辅助类中。

## 7. 内存数据模型

`InvertedIndexKnowhereCore` 持有：

```text
count_                 总行数
terms_                 排序后的唯一 term
posting_metas_         与 terms_ 一一对应
posting_bytes_         docID posting 的逻辑连续字节区
null_offsets_          null row offsets
cached_byte_size_      构建结束后的常驻内存
```

每个 `TermPostingMeta` 逻辑上包含：

```text
doc_stream_offset      posting 起点
doc_stream_length      posting 字节长度
doc_freq               posting 中的 docID 数量
flags / format_version 后续扩展
```

实现时也可以用 `posting_offsets_[i]` 和 `posting_offsets_[i + 1]` 表示边界，但
逻辑模型必须允许以后增加 position stream、chunk ID 和不同编码类型。

## 8. DocID posting 字节格式

单个 posting 使用：

```text
+----------------------+--------------------------------------+
| posting_count        | Knowhere 风格 varint32               |
+----------------------+--------------------------------------+
| block_max_doc_ids    | int32[num_blocks]                    |
+----------------------+--------------------------------------+
| block_end_offsets    | uint32[num_blocks - 1]               |
+----------------------+--------------------------------------+
| encoded_doc_blocks   | StreamVByte delta docIDs             |
+----------------------+--------------------------------------+
```

规则：

- `num_blocks = ceil(posting_count / 256)`；
- `block_max_doc_ids[b]` 是第 `b` 个 block 的最后一个 docID；
- `block_end_offsets[b]` 是第 `b` 个 block 结束位置相对于 doc block 区起点的
  offset，也等于下一个 block 的起点；
- 最后一个 block 不保存 end offset；
- 整数 header 沿用 Knowhere 当前的 native little-endian 表示；
- Knowhere varint 的最后一个字节最高位为 1，之前字节最高位为 0，PoC 必须兼容
  这一规则。

docID 必须单调递增，delta 规则为：

```text
previous_doc_id = -1
delta[i] = doc_id[i] - previous_doc_id - 1
previous_doc_id = doc_id[i]
```

新 block 的第一个 delta 仍相对前一个 block 的最后一个 docID。解码时可以利用
前一 block 的 max docID 恢复 base，从而支持 block 级跳转。

### 8.1 为什么去掉 sparse value

Knowhere sparse value 用于分数计算，而 scalar filter 只需要 docID。保留 value
至少多消耗 4 bytes/posting，并污染对 docID 格式 cache 行为的评估。因此本阶段
使用 docID-only 格式，并明确称为 **Knowhere-derived docID block format**，
不宣称与完整 Knowhere sparse index 文件字节兼容。

## 9. Position 链的后续扩展点

position 不进入第一阶段，但当前设计必须保证以后无需重写 docID stream。建议
后续使用平行 stream：

```text
TermPostingMeta
  ├── doc_stream_offset / length
  ├── position_index_offset / length
  └── position_data_offset / length

doc stream:
  docID_0, docID_1, ...

position index stream:
  每个 posting ordinal 对应的 position 起止位置或 term frequency

position data stream:
  每个文档内部 delta-encoded positions
```

第一阶段需要保持以下约束：

1. term 内 docID 严格有序。
2. 每个 docID 对应稳定的 posting ordinal。
3. doc stream 与未来 position stream 使用独立 offset。
4. filter query 只访问 doc stream。
5. phrase/proximity 先由 doc stream 求候选，再按需访问 position。
6. 做 4 MiB chunk 时，doc 和 position stream 可以独立切块、独立拉取。

这样未来增加 position 后，纯 filter 不会被迫读取 position 数据。

## 10. Term 字典

第一阶段采用排序后的 `std::vector<T>`：

- exact：`lower_bound`，复杂度 `O(log U)`；
- range：`lower_bound`/`upper_bound` 定位连续 term 区间；
- term 与 posting metadata 使用相同下标。

不直接使用 Knowhere sparse dimension map，因为它面向无序 `uint32_t` dimension，
不能直接表示任意 int64/double/string，也不能提供 range 枚举。

这意味着 benchmark 测到的是“有序数组字典 + Knowhere posting”的整体效果。
exact 和 range 必须分开报告。如果 exact 被字典限制，可追加 hash lookup；如果
range term 枚举成为瓶颈，再评估 FST 或 SortedBlock。

## 11. 构建与查询流程

### 11.1 构建

1. 遍历 scalar 数据，对有效行生成 `(term, row_id)`。
2. 按 term、row ID 排序。
3. 按 term 分组，生成 term 表和 posting metadata。
4. 将每组 row ID 编码为 256-entry block，追加到 posting blob。
5. 记录 null row。
6. 释放临时 pair/posting vector。
7. 基于容器 capacity 统计查询阶段常驻内存。

构建耗时会记录，但不是第一阶段的主要优化目标。查询计时前不得保留临时构建
数据，避免内存结果失真。

### 11.2 查询

- `IN`：分配结果 bitmap，逐 term 二分查找，直接将 posting 解码进同一 bitmap。
- `NOT IN`：执行 `IN`、翻转 bitmap、清除 null row。
- Range：定位 term 区间，依次将区间内 posting 解码进同一 bitmap。
- Null：保持现有 Tantivy/SQL 三值语义。
- Filter/callback：先生成 bitmap 再应用，与当前 Tantivy 保持相同口径。

Range 结果必须同时报告枚举的 term 数和命中的 row 数，避免把 term 枚举成本与
posting decode 成本混为一谈。

## 12. 独立 ScalarIndex 实现

- `InvertedIndexKnowhere<T>` 直接继承 `ScalarIndex<T>`，不继承或调用 Tantivy。
- 实现计数、大小、IN/NOT IN、范围、null 和 filter/callback 接口。
- pattern 和持久化接口明确返回不支持。
- 测试显式构造两种实现，不使用 backend 配置开关。
- `ResidentTantivyIndexForTest` 仅负责基线的 null/RAM 构建准备。
- 不修改 IndexFactory、生产 Tantivy、Meta.h 和 persisted metadata。

## 13. 纯内存对比口径

计时区域必须满足：

- 没有 S3、网络、文件读取和显式 I/O。
- Knowhere 的 term、metadata 和 posting blob 全部在 heap。
- Tantivy 构建后关闭 writer，以 `load_in_mmap=false` 重新打开。
- 两种索引执行覆盖全部 case 的预热。
- 构建、加载、预热和正确性校验不计入查询延迟。
- 主 benchmark 单线程、release/optimized 构建。
- 输出记录 CPU、机器信息、构建模式和数据种子。

如果 Tantivy non-mmap 模式仍有不可控文件访问，要通过 page-fault 或系统调用
信息确认；在口径确认前不能称为严格 RAM 对比。

## 14. Benchmark 设计

### 14.1 数据集

默认使用确定性的 100 万行 `int64_t` 数据：

| 数据集 | 唯一 term 数 | 主要观察点 |
|---|---:|---|
| Low cardinality | 100 | 长 posting、bitmap 写入密集 |
| Medium cardinality | 10,000 | 常见 scalar filter |
| High cardinality | 接近 1,000,000 | 短 posting、字典成本高 |
| Zipf-like | 约 10,000 | 热点和冷门 posting 并存 |

### 14.2 End-to-end workload

最终验收增加 segcore 层的同路径对比：
`BuildWithRawDataForUT -> segcore::LoadIndex -> Segment::Search`。
两边使用相同 schema、数据、查询计划、查询向量和参数，只切换 scalar 后端。
过滤字段不加载 raw data，避免退回扫描。向量部分使用相同的 exact L2 路径，
分别记录构建、segment 装载和 Search 耗时，并比较行偏移和距离。
这里的 Load 是 segcore 装载已构建索引对象，不是持久化文件反序列化；
纯内存 PoC 仍不包含生产 Serialize/Upload/Load。
scalar public API benchmark 和以下 microbenchmark 保留，用于解释端到端瓶颈。


每个数据集至少执行：

- rare、medium-frequency、hot term 的 exact；
- 8 term 和 64 term 的 `IN`；
- 约 0.1%、1%、10%、50% 命中的 range；
- 小 term 集合的 `NOT IN`。

每个 case 输出 term 数、hit 数、selectivity、median、p95、两者延迟比和 bitmap
一致性。

### 14.3 分层 microbenchmark

为区分瓶颈，除 public API 的完整查询外，再记录：

1. 空 `TargetBitmap(count)` 分配和清零成本。
2. term lookup only。
3. 已知 posting offset 下的 decode + bitmap set。
4. 字典、解码和 bitmap 全部包含的完整查询。

这些 helper 只用于 PoC/测试，不进入生产公共接口。

### 14.4 测量方法

- 先做结果一致性校验，再计时。
- 预热后采集多批样本，主指标为 median，p95 用于观察抖动。
- 每批运行足够次数，使耗时高于计时器分辨率。
- query 顺序随机化，减少固定 cache 顺序偏差。
- 使用 hit count 或 bitmap checksum 防止结果被优化掉。
- 不设性能测试阈值；正确性是 gate，性能数据用于决策。

### 14.5 内存统计

分别报告 term 容量、posting metadata、posting blob、null offsets、总 bytes、
bytes/row、bytes/posting 和 Tantivy `index_size_bytes()`。vector 按 capacity
计入；构建峰值单独记录，不混入查询常驻内存。

## 15. 正确性测试

### 15.1 Codec

- 0、1、255、256、257、511、512、513 个 docID。
- 连续 docID、大 gap、多 block、varint 边界。
- block max docID、block end offset 和 StreamVByte round-trip。

### 15.2 Scalar

- 重复值、不存在 term、空查询、空索引。
- 单 term、多 term、`NOT IN`。
- 四种单边 range 和开闭双边区间。
- null、non-null、全 null。
- 至少一个跨 256-entry block 的 term。

### 15.3 独立实现对比

- 所有支持查询逐位比较 bitmap。
- 默认配置确认仍使用 Tantivy。
- 两种实现通过 `ScalarIndex<T>` 接入相同 segcore LoadIndex/Search。
- 不支持接口确认明确报错而不是崩溃。

## 16. 实施阶段

### 阶段 0：Codec 依赖检查

验证 macOS/Linux 是否能链接 Knowhere 的 `streamvbyte_encode_0124` 和
`streamvbyte_decode_0124`。如果不可用，再决定是否引入 Apache-licensed codec
源文件。在依赖确定之前不修改 `InvertedIndexTantivy`。

### 阶段 1：Posting codec

计划新增：

- `internal/core/src/index/KnowhereSparsePostingCodec.h`
- `internal/core/src/index/KnowhereSparsePostingCodec.cpp`
- `internal/core/src/index/KnowhereSparsePostingCodecTest.cpp`

### 阶段 2：InvertedIndexKnowhere

计划新增：

- `internal/core/src/index/InvertedIndexKnowhere.h`
- `internal/core/src/index/InvertedIndexKnowhereCore.h`
- `internal/core/src/index/InvertedIndexKnowhereTest.cpp`

### 阶段 3：独立适配与 segcore 验证

新增独立 ScalarIndex 适配类和测试辅助类；生产 Tantivy 与 Meta.h 保持不变。

### 阶段 4：Benchmark

计划新增：

- `internal/core/src/index/InvertedIndexKnowhereBenchmarkTest.cpp`

输出 human-readable 表格以及可复制的 CSV/JSON 数据。

### 阶段 5：验证

建议命令：

```bash
cmake --build cmake_build --target all_tests -j2

internal/core/output/unittest/all_tests \
  --gtest_filter='KnowhereSparsePostingCodec*:*InvertedIndexKnowhere*'

internal/core/output/unittest/all_tests \
  --gtest_filter='InvertedIndexKnowhereBenchmark.*'
```

另外运行现有 `InvertedIndex` 测试，验证默认 Tantivy 路径无回归。实际二进制路径
以当前 CMake 配置为准。

## 17. 交付物与决策标准

PoC 交付物：

1. 实现和单元测试。
2. 可单独运行的 benchmark。
3. 不同数据分布和查询类型的结果表。
4. Tantivy/Knowhere 常驻内存对比。
5. term lookup、posting decode 和 bitmap 物化成本拆解。
6. 是否继续持久化、4 MiB chunk 和 position 链的建议。

正确性是硬门槛：bitmap 必须与 Tantivy 完全一致，默认路径不能变化，不支持功能
必须明确报错。

性能根据瓶颈分类决策：

- exact/IN 好、range 差：保留 posting，评估 FST 或 SortedBlock 字典。
- 短 posting 差：小 posting 存原始 docID，大 posting 使用 StreamVByte。
- 长 posting 差：检查解码吞吐、bitmap 写入和 block size。
- 两者都被 bitmap 主导：优先优化结果物化或压缩 bitmap。
- 整体有优势：进入持久化格式和 4 MiB chunk 设计。

## 18. 风险与控制

- **私有 codec 符号不稳定：** 限制在一个 translation unit，并先做链接验证。
- **格式兼容表述错误：** 只称 Knowhere-derived docID format。
- **Tantivy 内存口径不公平：** non-mmap reload、预热并检查 page fault。
- **term 字典干扰判断：** 同时提供 end-to-end 和分层 microbenchmark。
- **PoC 被误用于生产：** 仅 `BuildWithRawDataForUT` 生效。
- **过早加入 position：** 第一阶段只留 metadata 扩展点，不存 position 数据。
- **结果不可复现：** 固定种子，记录机器、CPU、构建模式和查询参数。

## 19. 后续 position 链阶段

确认 docID posting 值得继续后，再单独设计 position 链。建议下一阶段只增加：

- 每个 posting 的 term frequency；
- 每个 posting 的 position 起止 offset；
- 文档内 delta position 编码；
- phrase/proximity 的 position 交集；
- position chunk 的按需加载。

BM25 top-k、WAND/MaxScore 再作为其后的独立阶段。这样 docID 格式、position 和
打分算法不会同时变化，benchmark 结果才能清楚归因。

## 后续文本索引调研边界

已确认先替换 Tantivy index，保留现有 Tantivy analyzer；analyzer 的替换另行考虑。
Phrase match 尚未实现，先依据锁定版本的 Tantivy 源码确定改造项，见
[调研记录](tools/knowhere-scalar-poc/tantivy-phrase-review.md)。

## TEXT_MATCH 扩展

已新增独立的 `TextMatchIndexKnowhere`：保留 Tantivy analyzer，复用本 PoC 的
docID codec，实现分词倒排、OR 和 minimum_should_match。`TextMatchIndexBase`
用于 segcore 共用文本执行入口；标量 InvertedIndexTantivy 原文件不变。
这是 sealed 驻留 PoC，文本 load 使用同一个 text-index holder/runtime 挂载，
不是 scalar LoadIndex，也不是持久化文件重载。

[设计](docs/design-docs/knowhere-text-match-poc.md) /
[正确性和性能结果](tools/knowhere-scalar-poc/results/2026-09-20-text-match/README.md)。


### Knowhere（adaptive）与 Tantivy 对照

文本 PoC 现可选择 Knowhere 原生 `AdaptiveBlockCodec::encode_doc_ids/decode`，
默认 adaptive；scalar 默认仍为 StreamVByte。两种格式使用相同查询算法及
缺词提前返回。[Knowhere（adaptive）与 Tantivy 的两轮正确性和性能对照](tools/knowhere-scalar-poc/results/2026-09-20-block-adaptive/README.md)
包含逻辑 posting 字节数、构建时间及 segcore Search；
[workload 完整定义](tools/knowhere-scalar-poc/results/2026-09-20-text-match/workloads.md)
列明实际查询词、门槛、命中数和选择率。
