# Knowhere resident text analyzer compatibility

本轮仍只替换 index，保留 Tantivy analyzer。新增测试
[`KnowhereAnalyzerCompatTest.cpp`](../../internal/core/src/index/KnowhereAnalyzerCompatTest.cpp)
验证已经存在的配置穿过 Knowhere token dictionary、postings 和 positions 后，
仍得到相同的 TEXT_MATCH / PHRASE_MATCH / TEXT_MATCH_FUZZY 结果。

## 先确认实际契约

- [`analyzer.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/analyzer.rs)
  区分 `type` 内置模板和 `tokenizer` 加有序 `filter` 自定义 pipeline。
  [`build_in_analyzer.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/build_in_analyzer.rs)
  定义 standard、English、Chinese、Arabic、Thai 模板；本测试使用源码中的真实配置。
- [`synonym_filter.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/filter/synonym_filter.rs)
  克隆原 token 的 position/position_length。展开出的多个 token 在同一 position，
  不是递增的词序列；`expand:false` 的分组归一到第一个词，显式映射可以产生多个词。
  HashSet 导出的同位置词顺序没有保证，因此测试比较 token 集合，不假设输出顺序。
- [`filter.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/filter/filter.rs)
  和 [`stemmer_filter.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/filter/stemmer_filter.rs)
  确认 lowercase、asciifolding、English stemmer 和 stop 的配置名称及执行顺序。
  stop 删除 token 后保留原 position，不能把 hole 压平。
- [`lang_ident_tokenizer.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/analyzer/tokenizers/lang_ident_tokenizer.rs)
  每份输入分别识别语言，再按 mapping 选择 analyzer。文档与短查询可能走不同 analyzer；
  不应把构建时检测的语言绑定给查询。此处用 whatlang 和足够长的英/中文句子验证实际分派，
  用字母数字字符串验证 default 路径。
- [`tokenizer.h`](../../internal/core/thirdparty/tantivy/tokenizer.h) /
  [`tokenizer_c.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/tokenizer_c.rs)
  是 Knowhere 复用的 C++/Rust 适配器；查询克隆 analyzer，Build/phrase 读取 detailed token 的 position。
- [`index_reader_text.rs`](../../internal/core/thirdparty/tantivy/tantivy-binding/src/index_reader_text.rs)
  是当前 Tantivy 查询契约：TEXT 的 minimum_should_match 按 query clause 计数，重复词和
  同位置同义词不自动合并为一个 clause；FUZZY 是各 token fuzzy query 的 OR；PHRASE 将
  全部 `(position, term)` 传给 PhraseQuery。这里没有 Lucene synonym graph 的替代词路径重写。
  因此不能擅自将同位置词改为 OR，并宣称这是兼容性修复。

## 覆盖与验证方式

八种配置分别包含展开同义词、替换同义词叠加 stop gap、fold/stem/stop、四种语言模板，
以及 whatlang 动态分派。每种配置使用 259 行，含空字符串、标点、重复文本及三个 NULL 行；
NULL 行的物理字符串仍有内容，防止仅测试空 payload 而漏掉 validity。

每个查询覆盖：

- TEXT direct API minimum 0/1/2/3/7，包括重复查询词、缺词和不可能满足的阈值；
  public parser 要求 minimum>=1，因此 direct 0 的 Search 使用等价的 public 1；
- PHRASE slop 0/1/2/8，包括同位置同义词、stop position hole 和词序变化；
- FUZZY edit distance 0/1/2，Unicode codepoint 上的相邻换位代价为 1；
- 三种操作的 NOT，验证 NULL 不因补集而错误返回。

先比较完整 K/T index bitmap；TEXT 用原始 token 集的 clause 计数 oracle，exact phrase
用相对 position 扫描 oracle，FUZZY 用独立 Unicode 全矩阵 OSA oracle。非零 slop 与当前
Tantivy 比较，不声称另有独立 slop 语义证明。

之后将两种 index 通过相同 runtime holder 挂到 sealed segment，**不加载原 text 字段**，
关闭 expression cache，检查真实 FilterBitsNode 全 bitmap 和向量 Search 的完整 offsets/distances。
这防止 raw fallback 或 top-k 恰好相同掩盖 index 差异。

测试状态由主任务统一构建及运行记录给出。本文件不把静态审查当作运行通过的证据。

## 明确边界

本轮没有改 analyzer 实现，也没有引入新的 analyzer 语法。新增自包含覆盖见下节；
gRPC tokenizer 的成功连接、TLS/参数及服务故障仍需服务 fixture。
Lindera 的默认字典构建 feature 当前未开启（`TANTIVY_FEATURES_LIST` 为空），没有下载字典，
因此未覆盖。远程资源名解析及资源同步、任意 filter 排列也未穷尽。
已有接口可接收某个配置，不等于其所有外部资源及失败模式已验证。

当前测试为兼容性闭环，不重新运行所有语言的性能 benchmark，也不声称这些配置的吞吐量
已经超过 Tantivy。Growing 仍不在本文件范围。后续 snapshot 已绑定配置字符串；外部资源内容恢复的限制见下节。

## Executed results

See the [compatibility report](../../../knowhere-scalar-poc/results/2026-09-21-compat/README.md) for passing correctness runs, scope limits, cast benchmark/perf and unified regression (326 passed, 3 existing skips).

## 后续资源兼容性验证

在原八种配置之外增加 whitespace、char_group（Unicode 标点/空白）、ICU、regex+length、
inline decompounder、decimaldigit、pinyin 七种本地 pipeline；另有 Lingua 动态语言分派。
Lingua 使用当前 Cargo 已编入的依赖数据，验证长英文、中文与 default 输入，不代表每种语言均覆盖。
所有配置沿用 259 行、完整 bitmap/NOT/Search 与独立 oracle 的相同流程；
`POC_SNAPSHOT_ROUNDTRIP=1` 时还经过文件写入、旧 index 与缓冲区销毁、新实例加载。

本地资源 fixture 源于 Rust 自带 `analyzer/data/test` 的 synonym、stop、decompounder 三份小词典，
测试创建临时副本并使用真实 `{"type":"local","path":...}` 配置，避免依赖运行目录。
增加已构造 Knowhere analyzer 在文件删除后继续可用、删除文件后新构造 K/T 都失败的验证。
这不包含远程下载，也不改变 Rust 返回的错误分类。

**资源持久化有一个明确限制：** snapshot 绑定 analyzer 配置字符串，却没有绑定被引用文件的内容。
新增 `ChangedLocalResourceIsNotBoundBySnapshot` 用同一路径的 `car,auto` → `car,truck` 词典修改
证明旧索引可被加载，但查询分词变化会使 `auto` 从命中变为不命中。这个测试是缺口复现，不能
计入“资源重载完全兼容”的结论。集成之前需要资源版本/内容指纹绑定或自包含资源；缺文件则在
fresh constructor 失败，尚未进入 Load。未擅自增加新的 analyzer 公共语义。

本次新增测试的执行结果见[输入边界报告](../../../knowhere-scalar-poc/results/2026-09-21-boundaries/README.md)。
