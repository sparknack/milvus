# Knowhere JSON 路径 NGRAM resident PoC

范围：对一个显式 JSON pointer 指向的字符串建立 ngram，保留 Tantivy analyzer；
不是全 JSON flatten 索引，不添加持久化、growing 或生产 factory。双方均保留原始 JSON 字段。

## 实现依据

- Lucene 10.3.1 [NGramTokenizer](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/analysis/common/src/java/org/apache/lucene/analysis/ngram/NGramTokenizer.java)：按 Unicode codepoint 滑动生成 grams，不将字段值中的标点/空白忽略；候选 postings 不表示子串连续性。
- Milvus `NgramInvertedIndex.cpp::BuildWithJsonFieldData` 使用 `ProcessJsonFieldData<string>`，VARCHAR cast、unknown cast function；每路径一个索引，所以无须给 gram 再编码 JSON path。
- `NgramInvertedIndex.cpp::ExecutePhase2` 从原始 JSON 的指定 path 取 string_view，按原 LikePatternMatcher/RE2/find/prefix/suffix 判断。没有可用 gram 的模式由原 executor 回退 raw JSON。
- `JsonIndexBuilder.cpp::ProcessJsonFieldData` 对缺失路径/null/错误类型不发射 term，仅 JSON 整行 NULL 调用 null_adder。这个原有 NGRAM validity 契约与 raw 字符串谓词的 validity 不一致，NOT 差分暴露了下面记录的语义问题。
- Lucene 的[FieldExistsQuery](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/FieldExistsQuery.java)使用 doc values/norm/vector 的存在信息；不能自动替代 Milvus 的 JSON path/null/cast 语义。

## 实现

独立 `JsonNgramIndexKnowhere : NgramIndexKnowhere`。构建复用 ProcessJsonFieldData，将非字符串或不存在的路径保持为空占位，不产生 grams；所有不能提取字符串的行记录为 invalid。
首版三方 NOT 差分发现 Tantivy NGRAM 对缺失路径返回错误的 NOT 命中，Knowhere 因此采用上述 typed validity 对齐 raw executor。后续 NULL 审计修正 Tantivy JSON NGRAM 构建时的无效行记录，现在正向、NOT 及组合逻辑均要求三方一致。历史 benchmark 仍保留修复前的 Tantivy 基线，不追改历史数据。旧索引中缺失的无效行信息需要通过重建补齐。
父类保留 FST ordinal + adaptive flat postings 与稀有 gram 优先/提前停止。Phase2 在候选密度不超过 1/8 时通过现有 pinned string-view gather 只访问候选，在密集时批量读取 string view；两条路径都只为候选构造 Json、提取路径并执行精确 predicate。字符串视图随 pin 保持有效，parser-owned 结果在下一次解析前消费。
没有给缺失路径伪造字符串 sentinel，没有缓存原始 JSON 的解析结果。

## 校验与测量

对 T/K/raw 三方完整 bitmap、NOT bitmap、Search offsets/distances 差分，并统计 Phase1 调用验证路由。
数据包含 120K 合成低/高基数与 AG News 标题/正文；JSON 添加无关字段、缺失路径、null、数字、布尔、数组、对象，避免只测干净字符串。
沿用 29 个明确模式、11批随机顺序、CPU计时与独立Phase1统计。初版测完后再作独立评审与perf，依据瓶颈优化，并保留前后结果。

## 边界

valid UTF-8，无内嵌 NUL（父类原有PoC约束）。复用现有JSON解析异常路径，不catch/stringify重抛；optional路径不存在/类型不匹配保持正常不命中。
NULL语义以真实segcore结果校验，不声称原有标量、JSON路径索引与NGRAM的validity定义相同。

## 实测记录

[完整性能报告、正确性 XML、优化前后 perf 和逐项 workload](../../../knowhere-scalar-poc/results/2026-09-20-structured/README.md)。
