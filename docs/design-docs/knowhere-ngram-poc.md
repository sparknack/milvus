# Knowhere VARCHAR NGRAM resident PoC

状态：已实现并完成正确性、两轮性能、perf 和独立评审；
[首版结果](../../../knowhere-scalar-poc/results/2026-09-20-ngram/README.md)与
[查询优化结果](../../../knowhere-scalar-poc/results/2026-09-20-ngram-optimized/README.md)。

范围：只替换 VARCHAR NGRAM 的索引后端，保留 Tantivy analyzer 与 Milvus 的原文验证语义。
不注册生产 factory，不实现 JSON、ARRAY、growing、持久化或 mmap。本轮不自动提交。

## 对照实现

- Lucene 10.3.1 NGramTokenizer：Unicode codepoint 边界、流式生成；NGramTokenFilter
  在已有 token 内生成 grams。不能把 EdgeNGram 前缀语义用于任意子串。
  https://lucene.apache.org/core/10_3_1/analysis/common/org/apache/lucene/analysis/ngram/NGramTokenizer.html
- 本仓库 Tantivy binding `index_ngram_writer.rs` 使用 NgramTokenizer(min,max,false)，
  Basic postings，无 fieldnorm、词频评分或 positions。
- `index_reader.rs::ngram_match_query/ngram_tokenize`：长度在 [min,max] 内直接查询 term，
  更长字面量切固定 max-gram，支持按 DF 排序。
- `NgramInvertedIndex.cpp`：按模式安全提取字面量、Phase1 候选过滤、Phase2 原文复核。
  当前短行路径可能只查 2/3 个 gram，并按候选率提前退出；不能要求两个后端候选位图相同。

## 接入

抽取 query-only `NgramIndexBase`（Count、IsNotNull、CanHandleLiteral、Phase1、Phase2）。
现有 Tantivy 类实现该接口，segcore 的 pin 保持对 cache cell 的所有权，只将查询指针换成接口。
独立 `NgramIndexKnowhere : ScalarIndex<string>, NgramIndexBase` 使用现有 adaptive flat postings
与 FST gram -> ordinal 字典。原文字段由 segment 保存，不复制到索引。

通过小型 FFI 构造与现有 writer 完全相同的 Tantivy NgramTokenizer，复用已有 TokenStream。
不重新实现 tokenizer，也不添加新的用户 analyzer 配置。C binding 头由 cargo/cbindgen 生成。
构建临时按 gram 聚合 docIDs，逐行去重，发布前压缩成单一 blob 与 offset/length 元数据。
最终没有每个 posting 一个 vector；构建失败不替换旧索引。

Phase1 解析所有需要的 grams，先查词典，任一缺失立即返回空候选，再按 DF 升序过滤输入 candidates。
优化版仅在 NGRAM 内允许候选超集：候选不超过 8 行即可停止；9–128 行最多检查 8 个 gram；
更密集的候选按平均有效行字节数使用 2/3/5 个 gram 的预算（分界 100/1000 字节），
第二个及后续 gram 未减少超过 128 行的候选时停止。预算是 PoC 启发式，并非自动最优代价模型。
输入候选少于 posting DF 的 1/8 时使用 cursor Seek，否则 decode 成 bitmap 求交；
共享 TEXT_MATCH 精确求交接口未改变。单 gram InnerMatch 仍是精确候选。
Phase2 沿用 VARCHAR 的 find/prefix/suffix/LikePatternMatcher/RE2 原文判断，按 segment batch 执行。
输入候选密度超过 1/8 时直接保留原连续读取；否则逐 batch 判断，
候选密度不超过 1/8 的 batch 使用现有 get_views_by_offsets 收集字符串视图，空 batch 跳过；
密集 batch 保留连续视图读取。pin 生命周期覆盖 predicate，chunk offset 与结果 bitmap 的相对
offset 分别维护；超出 int32 gather offset 域时使用连续读取。本优化仅接入 Knowhere PoC。
LIKE predicate 局部使用 GCC/Clang `flatten` 属性，避免 gather 引入的多调用点使字符段匹配
退化为内层循环的独立函数调用；不改 matcher 算法或公共 matcher 声明，结果依赖编译器，
其他平台需要重新测量。
没有可用 grams 的短模式、全通配符、不安全 regex 提取由原 segcore 路由回退原文扫描。
非 NGRAM 标量表达式也继续走原文，禁止把 gram posting 当完整字段等值索引使用。

## 验收

先校验候选包含真实命中、最终完整位图及 Search offsets/distances 三方一致，再计时。
覆盖 min/max、重复 gram、非连续 gram 造成的假阳性、NULL/空串、Unicode/emoji、转义、短模式、
缺失词条、普通标量 fallback、NOT、失败 rebuild、空索引与跨 batch。
双方 segcore build -> LoadIndex attach -> Search，均加载原文供 Phase2 使用。
每条 query 记录是否路由 NGRAM、两种候选行数、最终命中行数、Search wall/CPU、Phase1 时间。
使用 120K 行合成数据与 AG News 原文标题/正文，adaptive codec，两轮随机后端顺序。
不将 resident attach 称为持久化重载，不把 ByteSize 当成同口径 RSS。

## 错误与边界审计

新增 analyzer 构造保留 Tantivy InvalidArgument -> binding discriminant -> AssertTantivyOk ->
SegcoreError 链；没有 stringify 后重抛或 catch-all 改写。min=0/min>max 应拒绝。
新增 build 指针/null offset、候选域不匹配为内部契约断言；NUL 与持久化为 PoC Unsupported。
有效 UTF-8、无嵌入 NUL 是比较输入域，非法 UTF-8 和服务端/cgo 全链路未验证。
并发 rebuild 不支持；只读查询使用 query-local tokenizer clone 和 scratch。

`CanHandleLiteral` 仅判断候选过滤是否适用，复用的 split_by_wildcard 并非完整 LIKE 语法校验器。
例如尾部反斜杠不会在此 helper 中报错；不能把 eligibility 接口当成 parser/matcher 的替代。
本轮不修改已有非法模式的校验时机，也不声称直接 Phase1 API 校验所有模式语法。
