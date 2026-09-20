# Knowhere PHRASE_MATCH：positions 压缩与遍历设计

状态：已按方案 C 实现 sealed resident PoC；TEXT_MATCH 基线为 `61f6424e26`。
继续使用 Tantivy analyzer，支持 exact 和 slop phrase。下文保留设计取舍，
末节说明实际实现与后续优化边界。
范围仍是 sealed resident PoC，不扩展 growing、BM25、远程加载或生产存储路径。

## 1. 已核对的源码

不是把 latest 的说明套到当前依赖上：

- Lucene 固定 `releases/lucene/10.3.1`，阅读 postings writer/reader、exact/sloppy matcher。
- Milvus 实际 Tantivy v7 依赖为 `96f3335ab5f061926c5b44cf246e81243e1dedc5`。
  已阅读本地 Cargo checkout 的 positions serializer/reader、SegmentPostings、
  PhraseScorer、Intersection，以及 Milvus 的 index_reader_text.rs、token_stream_c.rs。
- 当前编译使用 `/tmp/milvus-poc-knowhere`，HEAD 为
  `8103a1f6feded53f3bc784c5b97abed4f6ffdd56`；adaptive.h 的 encode/decode
  处理原始无符号整数，本身不执行 docID 差分。

### Lucene 的可借鉴部分

Lucene103 将 docID/TF、positions、可选 payload/offset 分流；完整整数块大小为 128，
尾部采用 VInt。positions 的 delta 在每篇文档重新从零开始，压缩块可跨文档。
跳表包含 positions 文件指针及块内位置。reader 在需要 nextPosition 时才补齐 TF、
跳过未消费的位置，恢复文档内累加状态。精确匹配先求 docID 交集，再按查询 offset
推进各个位置迭代器。sloppy matcher 另有优先队列及重复词碰撞处理。

来源：[格式](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/lucene103/Lucene103PostingsFormat.html)、
[writer](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/codecs/lucene103/Lucene103PostingsWriter.java)、
[reader](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/codecs/lucene103/Lucene103PostingsReader.java)、
[exact matcher](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/ExactPhraseMatcher.java)、
[sloppy matcher](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/SloppyPhraseMatcher.java)。

### Tantivy 的兼容基准

当前 fork 同样将 docID/TF 与 positions 分开。positions 每 128 个差分整数做
bitpacking，尾部 VInt；term 保存 bit-width 表，reader 累计块长度来跳过 payload。
当前文档的位置序号由 doc block 的 position_offset 加块内此前 TF 之和得到。
读取 TF 个位置差分，然后从文档起点累加。此处 TF 是切分位置列表的长度，不是 BM25 权重。

PhraseScorer 对 positions 加查询 offset 后求交；slop 路径维护 spans。
Milvus 对零/单个 token 转为普通词查询，collector 不需要评分。
**兼容 oracle 是这个 fork 和 Milvus binding，不是 Lucene。** 两者代码结构不同，
不能未经差分测试就假定所有 slop、换序、重复词行为等价。

来源与逐文件解释见[现有 Tantivy 调研](../../../knowhere-scalar-poc/tantivy-phrase-review.md)。

## 2. 三种可行布局

以下是本 PoC 的设计选项，不是声称上游采用了所有这些格式。

| 方案 | positions 的压缩边界 | doc 跳转后的定位 | 优点 | 代价 |
|---|---|---|---|---|
| A：每篇文档独立 | 每个 term/doc 一段 | 每文档 offset + TF | 实现简单，随机读取直接 | TF=1/2 时大量短块和 offsets，难利用 SIMD |
| B：每个 doc block 独立 | 每 256 篇命中文档重新开始 positions 分组 | doc block 的位置字节起点 + 块内 TF 前缀 | doc/pos block 生命周期容易对应 | 每个 doc block 产生位置尾块；高 TF 文档仍需要位置子块目录 |
| C：每个 term 连续流 | term 的全部 position delta 按固定个数分块 | doc block 的位置序号前缀 + TF 前缀 + 位置块目录 | TF 小也能形成整块，顺序解码和跳转兼顾 | 多维护位置块目录，doc/pos 边界独立 |

**推荐 C**：采用 Lucene/Tantivy 的逻辑分流和延迟解码思路，保留当前 adaptive codec。
不是照搬其字节格式，也不引入 Java/Rust postings 实现。A 可做测试参考解码器，
不建议作为最终驻留格式；B 是后续基于实际定位成本再比较的替代方案。

## 3. 推荐的压缩布局

所有 term 继续共享 flat byte arrays；不在驻留索引中维护 vector<vector<positions>>。
构建期可暂存 term/doc/position 聚合数据，发布前编码并释放。

```text
FST: term -> ordinal
term_meta[ordinal]: DF, total_positions, doc_range, freq_range, pos_range,
                    doc_aux_range, pos_directory_range, capabilities, codecs

existing doc blob:     256 docIDs/block + block max + byte offset
freq blob:             对应每个 doc block 的 TF 数组，独立压缩
position blob:         term 内按 doc 顺序拼接 delta，256 integers/block
per-doc-block aux:     freq_byte_offset, position_ordinal_base (u64)
per-position-block:    byte offset (u64)，term 元数据提供总数以确定尾块长度
```

元数据应作为带 positions 的文本 sidecar；标量 core 不需要无条件增加 TF/positions。
通用 doc cursor 只暴露 posting ordinal 等定位信息，文本层组合 PositionReader。

- docID 仍用当前 adaptive gap-1。positions 不能调用要求严格递增 docID 的 Append。
- 对单篇文档的位置 `[2,3,17]` 编码为 `[2,1,14]`。下一篇文档独立重置为零。
  首位置为零、相同位置产生零 delta 都必须合法；不做 gap-1。
- TF 为每个 term/doc 的位置个数，不做差分；先用 adaptive 通用 encode。
  全一块可用其 all-equal 表示，不额外引入 singleton 格式。
- positions 的差分整数也先用 adaptive 通用 encode/decode：支持 all-equal、
  bitpacking/patching 和尾块选择。256 沿用现有 codec 合约；不能声称比 128 最优。
  首版不再做 SVB/adaptive 横向比较，重点对比完整 Knowhere 与 Tantivy。
- 压缩块可以跨 doc，但不能跨 term。位置累计值在 doc 边界重置，不在压缩块边界重置。
- 字节偏移和总位置序号用 u64，TF/单个位置值用 u32；构建时检查累计溢出和 FFI
  的有符号 position 范围。查询 offset 对齐用足够宽的有符号类型，避免减法下溢。
- SIMD 可读 padding、所有 range 的长度和 codec/version 都要明确。未来落盘采用
  固定字节序的字段编码，不能把 C++ struct 内存直接写出。

例：term 在 doc 7 的位置为 `[2,5]`，在 doc 11 为 `[0,9,10]`：
TF=`[2,3]`，position deltas=`[2,3,0,9,1]`。
读取 doc 11 时从该文档的位置序号 2 开始，只累计 `[0,9,1]`，得到 `[0,9,10]`。
即使起点落在压缩块中间，也不把块中属于 doc 7 的前缀加进来。

粗略目录成本：每 256 docs 的两个 u64 约 0.0625 bytes/doc posting；每 256
positions 的 u64 约 0.03125 bytes/occurrence（不含 term 元数据、尾块和容器容量）。
实际低 DF term 的固定开销可能占主导，需要按 DF/TF 分桶统计，不能只引用大块均摊。

## 4. Cursor 合约和定位

```text
DocCursor: Doc(), Next(), SeekDoc(target), PostingOrdinal()
TextPostingCursor: 组合 DocCursor + 懒加载 TF/位置目录
                   Freq(), PositionsForCurrentDoc()
PositionCursor: NextPosition(), AdvancePosition(target), End()
```

DocCursor 的块号和块内下标给出 posting ordinal；不暴露全局 docID 来猜位置偏移。
当前 doc 在 block b 的第 i 项时：

```text
start = position_ordinal_base[b] + sum(TF[b][0..i))
count = TF[b][i]
pos_block = start / 256
within_block = start % 256
byte_address = pos_range.begin + pos_directory[pos_block]
```

TF 块第一次需要时才解码，并建立查询私有的 u64 前缀数组，成本最多 256 个 TF。
同一 doc block 内再次访问无需重复求前缀。SeekDoc 跳过 doc block 时读取新的
position_ordinal_base，**不扫描之前的 TF 或 positions**。

PositionCursor 从当前文档起始序号开始，最多消费 count 个整数；跨压缩块保持
当前文档的累加值，换文档重置。未读完就进入下一文档时，重新以 start/count 定位，
不必解码被丢弃的尾部。首版 AdvancePosition 只保证块内/顺序前进，不承诺按位置值
跨块二分：要实现后者还需文档内位置最大值和重启基值，不能复用 doc block maxima。

各查询独立维护解码 buffer，bytes 生命周期由索引 holder 保证。TEXT_MATCH 的
DocCursor 路径不解码 TF/positions，用计数器测试证明，而不只靠接口声明。

## 5. Phrase 遍历

1. 用同一个 analyzer 获得 `(term, query_position)`；保留停用词产生的空洞。
   读取 get_detailed_token，并以 RAII 释放其中 Rust 分配的 token 字符串。
   position_length 不擅自扩展为新的 token graph 语义，行为跟随现有 binding。
2. 缺任何必需 term 立即返回；空/单 token 对齐现有 Milvus 行为。
3. 先按 DF 求 docID 交集。稀疏场景直接复用 cursor；密集场景可复用 bitmap
   生成候选，再将各 text cursor 单调 SeekDoc 到候选。不能一律 cursor 化而重演
   当前 dense AND 的退步；候选生成与位置验证单独计数。
4. 仅对交集文档打开位置视图。slop=0 验证所有 `position - query_position`
   是否存在共同值；保留每个 clause 的 offset。找到一次即可输出此 doc，不统计评分。
5. 重复词可以共享不可变位置块，但每个 clause 要有独立遍历状态；不能像 TEXT_MATCH
   OR 一样直接删去重复 clause。doc 求交可去重，位置约束不可丢失。
6. slop>0 先以当前 Tantivy spans 逻辑建立等价参考实现，涵盖其执行顺序和重复词；
   再优化。不能简化为“窗口宽度减词数”或直接替换为 Lucene priority queue。

为先验证 slop 兼容性，可临时在候选文档内 materialize 每个 term 的位置；这是
query-local scratch，不是索引驻留的 vector-of-vectors。高 TF 文档的峰值内存
必须单独记录，后续再将等价算法改为流式读取。

## 6. 实施顺序和验收

1. 独立位置/TF codec、sidecar builder、ordinal cursor；文档和查询位置 roundtrip。
2. 精确 phrase，先低层扫描 oracle，再 Tantivy bitmap，再真实 segcore PHRASE_MATCH。
3. slop、重复词、停用词/Jieba 对齐，完整回归通过才称为支持 PHRASE_MATCH。
4. 同时复测原 TEXT_MATCH 74 场景，报告新增存储、build CPU 和 query 成本。
   不以只完成 slop=0 的性能代表完整 PHRASE_MATCH。

必测边界：TF=1/2/255/256/257，doc 和 pos 两种块错位、单 doc 跨多个位置块、
term 尾块、零 delta、稀有词命中靠后 doc、跳过大量高频 doc、不读完 positions
就 seek 下一 doc、重复短语、换序/slop 边界、空/null/中文、失败构建保留旧索引、
并发查询。跨块用 decode counters 证明跳过 payload，而不是仅看耗时。

性能 workload 要有：真实相邻 2/4/8 词、相同词但不相邻、停用词空洞、重复词、
高低频混合、高频无 phrase 命中、slop=0/1/2/更大值、长文高 TF。逐条给出 query、
slop、DF、doc 交集候选数和最终命中数；先全 bitmap/Search 结果对齐再计时。

当前设计需要的是逻辑分流，不要求立刻变成多个物理文件。未来文件布局和 mmap
接入另行按 Milvus 存储契约设计；本轮没有实现或承诺完整索引持久化。

## 7. 首版实现边界

- `KnowherePositionIndex` 是文本 sidecar，TF/positions 和目录均为 flat arrays。
  使用现有 adaptive 通用整数编码；不是标量索引的隐式字段。
- doc cursor 新增 PostingOrdinal；Position Reader 缓存当前 TF 前缀和位置块。
  它按候选文档读取位置到复用的 query-local buffers。首版尚未暴露逐位置
  Next/Advance API，exact 与 slop 都先 materialize 当前文档的所需位置。
  因此并不保证找到首个 phrase 后能停止解码该文档的剩余 positions。
- 两词精确/slop 验证采用无临时分配的双指针；多词 exact 从最短位置数组出发
  查找对齐值；多词 slop 保留 fork 的 span 推进与 tie/pruning 规则，附 MIT 声明。
- 候选生成复用当前 core 的稀疏 cursor/密集 bitmap 分派，之后各位置 reader 的
  doc cursor 再 seek 候选。稀疏查询存在两次 doc 遍历，这是后续可融合的开销。
- 初版 metadata 为进程内结构，未实现文件解析、codec version 持久化或 mmap
  positions。固定字节序/校验/损坏文件处理属于未来完整持久化工作。
- 新增检查涵盖 1,638 组枚举/随机 query+slop 对照，以及 standard、stop filter、
  Jieba 的 exact scan 和 Tantivy 对照。高 TF 功能测试包含单文档 1,400 次同词出现。
  高 TF 性能另有每词 64 次的可复现合成数据；尚未测 build 峰值 RSS。

[正确性、90 个 phrase workload 和 TEXT_MATCH 复测结果](../../../knowhere-scalar-poc/results/2026-09-20-phrase/README.md)。

## 8. 基于 perf 的首轮查询优化

- docID Seek 在当前块已解码、目标落在下一个 posting 以内时直接推进；
  远距离跳转仍按 block maxima 二分，块边界、倒退请求和结束状态合约保持不变。
- 查询准备阶段生成唯一 term 的解码计划和重复 clause 的 offset 复制计划。
  每个唯一 term 只持有一个 cursor/position reader；重复 clause 仍有独立位置
  对齐数组，保留原来的稳定 DF 顺序。未套用 Lucene 的重复词碰撞语义。
- 多词 slop 的 spans/next/best 由单次查询持有，跨候选复用 capacity，
  只在进入多词 slop 路径时清理逻辑内容。二词和 exact 路径不触碰这些 buffers。
  三组 spans 的推进顺序、平局和剪枝规则保持与首版一致。
- 验证新增相邻 Seek 的 posting ordinal、重复 term 的实际解码块计数、
  同 DF 且不相邻的重复 clause、stop-filter offset 空洞以及跨候选 scratch 复用。
  仍以 Tantivy 完整 bitmap 和 segcore Search 结果为兼容性基准。

[逐阶段 A/B、最终 PHRASE/TEXT_MATCH 复测及剩余热点](../../../knowhere-scalar-poc/results/2026-09-20-phrase-optimized/README.md)。
