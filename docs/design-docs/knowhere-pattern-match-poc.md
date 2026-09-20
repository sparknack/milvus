# Knowhere Prefix / LIKE / Regex 功能 PoC 计划

状态：resident PoC 已实现并通过正确性检查；性能与 perf 结果见
[优化后测试报告](../../../knowhere-scalar-poc/results/2026-09-20-pattern-optimized/README.md)。
本文件保留最初方案与后续扩展目标；实际覆盖及限制以报告为准。基线为 Milvus 0e28e56f97。
范围：resident 标量字符串；暂不扩展 ARRAY、JSON、NGRAM、growing、持久化或生产 factory。
使用现有 adaptive flat docID postings，不新增频次或位置链。

## 1. 先固定 Milvus 的语义

匹配对象是未分词字段的完整字符串值，不能用 TEXT_MATCH 的分词词典替代。

| 操作 | 语义 | 当前 Tantivy 接入 |
|---|---|---|
| PrefixMatch | 字面量前缀 | escape 前缀后附加可含换行的任意后缀，走 RegexQuery |
| Match / LIKE | 整串匹配，% 任意长度，_ 一个 Unicode 码点，反斜杠转义 | PatternMatchTranslator -> RegexQuery -> 自动机词典搜索 |
| PostfixMatch / InnerMatch | 字面量后缀/包含 | EscapeLikePattern 后生成 LIKE |
| RegexMatch | RE2 PartialMatch，UTF-8、dot_nl=true | Rust 枚举 term，回调 C++ PartialRegexMatcher，命中后读取 posting |

RegexMatch("abc") 可匹配 "xabcx"，LIKE "abc" 不可以。
不能把任意 regex 开头字面量当成前缀；anchors、flags、alternation 都会影响剪枝。

已核对源码：
- [InvertedIndexTantivy.h](../../internal/core/src/index/InvertedIndexTantivy.h)：PatternMatch、ShouldUseOp。
- [InvertedIndexTantivy.cpp](../../internal/core/src/index/InvertedIndexTantivy.cpp)：PrefixMatch、PatternQuery。
- [RegexQuery.h](../../internal/core/src/common/RegexQuery.h)、
  [RegexQuery.cpp](../../internal/core/src/common/RegexQuery.cpp)：LIKE 翻译、完整/部分匹配。
- [index_reader.rs](../../internal/core/thirdparty/tantivy/tantivy-binding/src/index_reader.rs)：
  prefix_query_keyword、regex_query、regex_match_query。

当前 ShouldUseOp 对普通倒排的 RegexMatch、PostfixMatch、InnerMatch 返回 false。
“存在索引方法”不等于 segcore 实际调用它；benchmark 必须证明双方走了索引。

## 2. Tantivy 与 Lucene 的实现

Tantivy 固定当前依赖 96f3335ab5f061926c5b44cf246e81243e1dedc5：
[RegexQuery](https://github.com/zilliztech/tantivy/blob/96f3335ab5f061926c5b44cf246e81243e1dedc5/src/query/regex_query.rs)
编译 tantivy_fst::Regex；
[AutomatonWeight](https://github.com/zilliztech/tantivy/blob/96f3335ab5f061926c5b44cf246e81243e1dedc5/src/query/automaton_weight.rs)
调用 term_dict.search(automaton)，将匹配 term 的 Basic block postings 合并进 bitmap。
默认非 quickwit 配置使用 FST 词典，联合遍历可剪去不可能匹配的分支。
这是 Milvus LIKE 的路径，不是当前 RegexMatch 的 RE2 callback 路径。

Lucene 固定 releases/lucene/10.3.1，不混用 main：
- [PrefixQuery](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/PrefixQuery.java)
  构造前缀字节链及任意字节后缀的接受状态。
- [WildcardQuery](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/WildcardQuery.java)
  将 literal、*、? 编成自动机，字符按 Unicode code point 处理。
- [RegexpQuery](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/RegexpQuery.java)
  使用 Lucene 自己的语法；支持确定化工作量限制及选择是否确定化。
- [CompiledAutomaton](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/util/automaton/CompiledAutomaton.java)
  区分空集、全集、单 term、一般自动机；一般情况走 Terms.intersect。
- [BlockTree IntersectTermsEnum](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/codecs/lucene103/blocktree/IntersectTermsEnum.java)
  在词典块遍历时推进自动机状态并剪枝，不宜概括成“整张 FST 求交”。

借鉴自动机引导词典遍历、简单模式专门处理和编译预算。
不移植评分/rewrite，也不以 Lucene 正则语法替代 RE2。
Lucene wildcard 的尾反斜杠处理与 Milvus LIKE 的报错不同，不能机械替换通配符后照搬。

## 3. 推荐方案

所有路径流式输出 term ordinal，再 DecodeInto 同一个结果 bitmap；
不收集全部匹配字符串、不为每个 term 建 bitmap，不改变 posting 布局。

### Prefix：直接使用现有 FST

FstTermDictionary 新增 ForEachPrefix(prefix, callback)、ForEachTerm(callback)，
包装 cpp-fstlib 的 callback predictive_search。不要使用会收集全部字符串并排序的 Enumerate。
空字符串由 existing empty_key 元数据单独处理；空前缀包含全部已索引值。
回调的 term view 仅在调用期间有效。

scalar string core 当前使用排序 terms，FST 主要接在 text PoC，尚未接在 scalar adapter。
新增 scalar string FST 时确保输出 ordinal 与 core 完全一致；成功后共同发布，
失败构建保留旧索引。先保留原 terms 并明确计入两份词典的空间；
本阶段不顺带重构标量存储。

### LIKE：参考路径加自动机路径

1. 先完整验证 pattern，复用 PatternMatchTranslator + RegexMatcher 作为参考实现。
   有安全固定前缀时仅枚举该前缀；否则枚举全部 term。
   不可因前缀无命中就漏报 pattern 尾部非法转义。
2. 将 literal、%、_ 编成 LIKE 专用 NFA，以活动状态集合配合 UTF-8 解码状态
   遍历 FST 字节边；_ 消耗一个码点，不是字节或字形簇。
   接受状态输出 ordinal，死状态剪枝。
3. literal-only、纯前缀专门处理；%abc% 等无前缀模式最坏仍可能遍历大部分词典。
4. 不急于完整 DFA 确定化。编译结构 query-local 共享，仅复制分支运行状态。
   优化/缓存预算用尽时回退到同语义参考路径，不能截断结果；阈值留待实测。
5. cpp-fstlib 固定 7b5a8f93105ddfdbc278613435c1230c17e084b6，
   已有 protected depth_first_visit 和 step/is_match/can_match 合约，
   可在本地子类封装，保持 vendored 源码不变。
   递归、partial_word 构造和状态复制仍有成本，要做长 term 与分配检查。

### RegexMatch：首版复用 RE2

每次查询编译一个 PartialRegexMatcher，遍历唯一 term 并匹配，命中后 OR posting。
这是词典扫描，不是逐行扫描；也不再通过 Rust 逐 term 回调。

不引入 std::regex、Lucene regex parser 或 RE2 私有 DFA 接口。
通用 regex 首版不承诺自动机剪枝；若成为瓶颈，再单独评估可证明正确的 anchored-prefix
优化或兼容编译器。无法证明 anchors/flags/alternation 下剪枝安全时保留全词典扫描。

## 4. 交付顺序

| 步骤 | 内容 | 验收 |
|---|---|---|
| P0 | 语义用例、原文 oracle、Tantivy 基线 | 完整结果及非法 pattern 行为明确 |
| P1 | scalar FST、流式 Prefix、LIKE/Regex 参考路径 | 三方 bitmap 对齐 |
| P2 | LIKE 自动机联合遍历 | 与参考路径和 Tantivy 随机差分，无漏匹配 |
| P3 | segcore benchmark 与报告 | 确认索引路径、先验证后计时、披露回退 |

不修改生产 Tantivy ShouldUseOp。需要强制索引时使用测试专用 adapter/路由，
加调用计数并检查 raw-column fallback；双方同样处理。
默认路由与 raw scan 另列，不混进索引性能表。

## 5. 正确性门槛

- 空表/空词典、空字符串、NULL、空 pattern、无匹配、全匹配、重复值。
- 转义后的 %/_/反斜杠、正则元字符、尾反斜杠、非法正则。
- ASCII、中文、emoji、组合字符、换行；_ 与 UTF-8 字节边界。
- Regex partial/full、anchors、字符类、重复、alternation、大小写 flags。
  RE2 不支持语法必须保持报错，不能悄悄返回无匹配。
- 嵌入 NUL / 非法 UTF-8 单列：先核对现有 FFI/build 的可往返输入域；
  被截断值不能作为合法的三方等价证据，差异须明确列为限制或修复。
- 随机字符串和合法 LIKE、有界 regex 生成；否定表达式的 NULL/validity。
- 构建失败保留旧索引、并发查询、现有 scalar/TEXT_MATCH/PHRASE_MATCH 回归。

Oracle：Tantivy 完整 bitmap + 不使用词典剪枝的原始行扫描。
LIKE 再用独立码点 wildcard matcher 对照；Regex 以现有 RE2 为语义标准。
比较完整 segcore bitmap 和 Search offsets/distances，不只比较命中数或 top-k。
发生分歧先定位，不跳过失败 workload 去报性能。

## 6. 性能矩阵

AG News 原始标题/正文按完整 keyword 值索引；增加合成字符串，
独立控制行数与 1K/10K/100K 基数、均匀/倾斜 DF、共享前缀、短/长值、UTF-8。

| 类别 | pattern 示例 | 关注点 |
|---|---|---|
| Prefix | abc、reuters、中文前缀、不存在前缀、空前缀 | 命中子树大小 |
| LIKE 有前缀 | abc%、abc_def%、abc%xyz | 前缀剪枝及中途拒绝 |
| LIKE 无前缀 | %abc、%abc%、_abc%、% | 最坏词典遍历 |
| 转义 LIKE | 字面量 %/_/反斜杠 | 避免误剪枝 |
| Regex | abc、^abc、abc$、^(abc|xyz)[0-9]+$、(?i)abc、.* | partial、anchors、flags、宽匹配 |

按数据生成实际命中 0/1/10/1000/大比例 term 的 query，不只给示例 pattern。
每行报告操作、query、行数、基数、遍历/验证 term 数、匹配 term 数、
匹配 posting DF 总和、最终命中行数、解码块数。

主表仍为 resident segcore build/attach/Search，查询时间包含 pattern 编译；
另测纯过滤和编译/词典/posting 开销，报告 wall、明确定义的分位数、
process/thread CPU、build 时间及内存。详细计数独立运行，避免插桩污染正常计时。
至少两轮；必要时分别对小前缀、宽 LIKE、Regex scan 采 perf。
本阶段不称为序列化后重载，也不预先保证自动机或 FST 一定快于参考扫描。

## 7. 本次落地范围

新增 ForEachPrefix 与 ForEachLike，空前缀承担 ForEachTerm，不另加同义 API。
LIKE 使用 63-token 位集 NFA，超出预算后按固定前缀枚举并用现有 RE2 判断；
一般 RegexMatch 使用现有 RE2 partial 语义，仍是词典扫描。
本次未加入 literal-only / match-all 专用快路径，也未替换 cpp-fstlib 的递归遍历。

已做 500 个随机 ASCII LIKE 差分、300 个随机 Unicode LIKE 独立 DP 验证，
以及完整 segcore 位图/结果比对和既有 text/phrase 回归。
性能矩阵包含 3 档合成基数及 AG News 标题/正文，各 21 个 workload。
原计划中的 Zipf 分布、visited arcs / decoded blocks 插桩、全长 65535 字节 term
压力与非法 UTF-8 等价性尚未覆盖，不作为本次已完成项。

## 8. Resident 查询策略优化（2026-09-20）

初版 perf 发现 cpp-fstlib 递归遍历中的字符串重建与分配占据大量 CPU。
当前 scalar PatternMatch 改用 core 已保留的排序唯一值，通过 string_view 输出原 ordinal：
Prefix 以 lower_bound 定位连续区间；LIKE 完整验证后取安全字面量前缀区间并调用
现有 RE2 FullMatch；RegexMatch 遍历唯一值并调用 RE2 PartialMatch。
命中后仍 DecodeInto 同一个 bitmap，adaptive flat postings 不变。

这没有新增词典副本，也没有读取原始行字段。FST 和原有唯一值均仍保留，空间计量不变；
FST 的 Unicode NFA API 仍有独立测试，但不再用于 scalar PatternMatch 热路径。
第 3、7 节记录初版方案，当前查询路径以本节为准。
未新增 match-all、任意 regex 前缀剪枝或非法 UTF-8 快路径。

该策略依赖 resident 唯一值已存在，不能把结果外推至未来仅 FST / mmap 词典。
若后续删除排序字符串，需要重新评估无字符串重建的 ordinal-only walker、
连续词条字节存储或按块解码；本轮没有实现这些存储改造。
