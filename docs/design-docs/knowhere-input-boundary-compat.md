# Knowhere resident PoC：输入边界确认

本轮只补齐确认、测试和非有限浮点词项处理；生产 factory、远程存储、mmap、Growing 后续接入。
执行结果见[验证报告](../../../knowhere-scalar-poc/results/2026-09-21-boundaries/README.md)。

## 非有限浮点数

默认 proxy 标量 FLOAT/DOUBLE 校验拒绝 NaN 和正负 Inf；同一 validator 的 ARRAY 元素校验没有该检查，nested ARRAY 也复用此路径。因此不能把普通 scalar 的上游约束当作所有倒排输入的约束。Go 新测试直接覆盖这些 validator 方法；它不是运行服务的 SDK 写入测试，也未修改写入规则。

Tantivy 将 FLOAT 写入值和查询值都提升为 double，按可排序 IEEE 位编码组织词项，正负零归并。Knowhere 共用 `KnowhereTermOrder`，覆盖 build 排序/去重、ARRAY 临时 map、查询二分和加载时有序性校验。NaN 的符号和 payload 参与顺序；float signaling NaN 提升到 double 的行为以真实 Tantivy 路径为基准。

测试包含 FLOAT/DOUBLE、signed/payload/signaling NaN、正负 Inf、正负零、denorm/max、NULL、重复 ARRAY 元素；对比 IN/NOT IN、单边/双边范围，以及销毁旧实例和缓冲区后的加载。额外执行普通/nested ARRAY 的 segcore 完整过滤 bitmap 和 Search。JSON cast 继续保留已有 NaN side stream，加载器显式禁止把 NaN 塞进普通 stream。

**索引顺序不等于 IEEE 算术比较。** 含 NaN 的范围查询可能与原始字段扫描不同；本轮对齐当前 Tantivy 索引语义，没有改变 Milvus 既有 raw/index 语义差异。nested 范围测试使用显式 total-order oracle 和真实 Tantivy 对照，不能报告为三方 raw parity。补测还发现原生 nested `IN [0, 7]` 扫描会把 NaN 误匹配为命中，而 T/K 索引不命中；测试单独刻画这个既有 raw 差异，其他 membership case 继续三方验证。这两处原生扫描差异留待独立处理。IN 的来源是 `TermExpr.cpp` 小列表选择 `SimdBatchElement`，`Element.h` 未对齐头部及 `SimdFilterImpl.h` 尾部使用 `std::binary_search`，而 SIMD 中段用 `==`。NaN 误命中依赖对齐，测试严格要求差异只发生在 NaN 且 label 匹配的多命中上，并断言差异实际存在。

## Embedded NUL：已确认缺口，尚未支持

VARCHAR/TEXT/ARRAY string 的现有 validator 不拒绝 `\0`，但 Tantivy C ABI 并不统一：

- 字符串写入和 prefix 查询带长度，保留 NUL 后内容。
- keyword IN/range 查询使用 C 字符串，截断到首个 NUL。
- 共享 tokenizer 的输入接口使用 C 字符串；直接 Tantivy TEXT writer 却传递长度。

新增测试记录实际行为，并确认 Knowhere 拒绝 NUL 的失败构建不会破坏之前发布的索引。它们是差异复现，不是兼容通过。不能简单删除 Knowhere guard，否则共享 tokenizer 会静默丢掉后半段文本；也不能统一截断，因为这会改变长度敏感接口和原文语义。

接入前需要决定并统一这些接口的长度语义，再补 scalar/ARRAY/JSON、TEXT/PHRASE、NGRAM 与原文扫描的一致性测试。本轮没有修改 Tantivy analyzer 或现有公共查询语义。

## Analyzer 资源

新增本地 pipeline、Lingua 和本地 synonym/stop/decompounder 文件测试，见[analyzer 设计记录](knowhere-analyzer-compat.md)。

同路径文件内容变化是实测缺口：snapshot 只绑定配置字符串，尚未绑定资源内容/version。旧 postings 配上重新构造的新词典可改变查询结果。缺文件的新构造失败、已构造实例继续工作也有测试。后续应绑定资源指纹或保存自包含资源，并验证同步与恢复生命周期。

Lindera 字典 feature 未启用；gRPC tokenizer 成功服务/TLS/故障及远程资源同步没有成功 fixture，仍是待补项。不能据本地覆盖声称所有 analyzer 配置已兼容。

## 审查边界

本轮不修改错误映射、重试或 wire code。坏 JSON snapshot 的新检查在构造点使用现有 `poc_io::Check`（`DataFormatBroken`），事务性加载验证旧状态仍可查询。NUL 拒绝和 analyzer 缺文件是既有失败路径的复现；没有据此声称生产 cgo 错误传播或远程恢复已经验证。
