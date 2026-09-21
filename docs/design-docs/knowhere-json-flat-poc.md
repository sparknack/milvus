# Resident Knowhere JSON flat PoC

## Existing implementation and reusable design

Milvus `JsonFlatIndex.cpp::build_index_for_json` extracts the configured JSON root and submits it to the Tantivy JSON field. `JsonFlatIndex.h::JsonFlatIndexQueryExecutor` then binds a relative path and query type; its numeric methods query I64, U64 and F64 channels. The C++ wrapper `thirdparty/tantivy/tantivy-wrapper.h::json_terms_query` has cross-type equality conversions, while executor range helpers also compensate for integer-to-double collisions. These contracts cannot be replaced by blindly storing all numbers as double.

The pinned [Tantivy JSON indexing implementation](https://github.com/zilliztech/tantivy/blob/96f3335ab5f061926c5b44cf246e81243e1dedc5/src/core/json_utils.rs) indexes typed values under paths and tracks per-path positions for repeated JSON values. This resident scalar/filter PoC needs document membership rather than term frequency/positions. The binding's `json_exists_query.rs` reads typed dynamic fast-field columns and optionally their subpaths to evaluate existence. [Lucene StringField](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/StringField.java) provides the analogous DOCS-only exact-value representation. [FieldExistsQuery](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/FieldExistsQuery.java) relies on explicit field-presence information from doc values/norms; membership postings alone cannot recover every validity distinction.

Milvus has multiple notions of presence. `common/Json.h::exist` treats recursively empty objects/arrays and all-null containers as nonexistent. Scalar comparable validity additionally requires a value of the requested family at the exact path. JSON contains requires an array at that path, including a valid empty array. These masks must not be conflated: a scalar 42 and an array [42] have different scalar-query semantics even when a flattening index places both values under one path.

## Resident design

`JsonFlatIndexKnowhere` is independent of Tantivy and implements the shared query-only `JsonFlatIndexBase`. A build produces an immutable shared snapshot containing row validity and path dictionaries. Each path has separate scalar and array-element postings, with separate BOOL/I64/U64/F64/STRING channels. All resident postings use the existing adaptive flat compressed core. Transient posting maps are released after build; query executors share snapshot ownership rather than copying or rebuilding indexes. Rebuild publication preserves previously created executors' snapshots.

Paths remain escaped JSON pointers. Object member slashes and tildes are escaped rather than converted into ambiguous dot separators. Array element paths are represented explicitly by ordinal components, while immediate scalar elements also contribute to that array path's membership postings. Nested arrays/objects do not accidentally contribute their inner scalar values to immediate scalar membership. The document root can be the whole JSON value or an indexed subpath such as `/doc`; executor paths are relative to that configured root.

Executors derive from the scalar Knowhere adapter to reuse the query interface, but delegate operations to immutable path cores. `SetArrayQuery` selects array membership and array-shape validity. `ExactPathExists` exposes scalar family masks for scalar comparisons and array shape for contains. `Exists` follows recursive nonempty Milvus semantics. The executor resolves missing paths to empty results. It does not read or retain raw JSON rows.

## Numeric contract

The three physical numeric dictionaries are retained separately. Binary-search bounds operate on stored terms, without flattening integers into double at build time. Extended precision is used only as a comparison carrier on targets with at least a 64-bit integer mantissa; the code asserts this property at compile time. Query-double comparisons explicitly apply double conversion to integer terms to preserve existing raw semantics. Integer queries retain exact comparisons; binary range's U64 channel follows the current raw evaluator's U64-double fallback. This is intentionally sensitive to query type and operation, rather than claiming one mathematical comparison rule matches all existing Milvus paths.

Correctness cases include 2^53−1, 2^53, 2^53+1, an F64 representation at 2^53, INT64 extrema, U64 beyond INT64 and U64_MAX. Boundary behavior is compared with actual raw segcore evaluation; any Tantivy disagreement is emitted as a measured compatibility difference. The direct executor test also checks that adjacent I64 values were not irreversibly collapsed at build time.

## Validation and benchmark protocol

The harness creates three sealed segments with identical raw JSON and vectors. Tantivy uses its production builder and a RAM reader over finished files. Knowhere builds the snapshot and attaches through LoadIndex. The index-only workload group drops JSON raw fields from both indexed segments, proving those routed queries cannot silently scan them. A separate native-fallback group retains raw data where production query policy requires it. Root and nonempty `/doc` configurations cover equality, IN, numeric ranges, string/bool values, descendant existence, nested object values, NOT, and JSON contains/any/all. Full filter bitmaps and Search offsets/distances must match before timing.

Mixed scalar/array/type paths, empty containers, escaped path names and numeric boundaries are additional correctness workloads. Knowhere must match raw. Tantivy/raw differences are explicitly printed as `flat_compatibility` records with mismatch counts, not suppressed or reported as successful parity. Such cases are not included in the ordinary three-way performance comparison.

Four 120000-row benchmark groups cover root and `/doc` indexing, each split into index-only and `_native_fallback` datasets. Each index root has 30 timed queries split into a 22-query index-only dataset and an 8-query native-fallback dataset: the original 16 plus LIKE prefix/suffix/contains/NOT/missing, string range/NOT, missing-path NOT, three SQL row-null predicates, and three regex cases. The harness uses 4D exact L2, nq=1, topk=10, NULL every 257 rows, expression caching disabled, three warmups and eleven shuffled timing batches, reporting wall and process CPU medians. Gated perf starts only after build, correctness and warmup. Timing excludes persistence, disk/mmap and distributed execution. Validation results must be recorded separately; the existence of this design and tests is not proof they have executed successfully.

## Pattern, native fallback and row validity

String payloads additionally include absent, JSON null and numeric wrong-type rows, ensuring pattern negation cannot simply invert match bits. LIKE prefix, string ranges and ordinary NOT remain in the raw-dropped group and require strict Tantivy/Knowhere/raw parity. Suffix/contains LIKE and regex partial/anchored/NOT use a separate `_native_fallback` group: production Tantivy ShouldUseOp rejects PostfixMatch/InnerMatch/RegexMatch and executes raw JSON scanning. Knowhere supports those operators through the flat index. These cases require strict three-way equality, but their timing compares native execution routes, not the speed of two indexing algorithms. The dormant inherited Tantivy RegexMatch implementation is not evidence of a bug on its actual native route.

The retained-raw group also covers SQL row IsNull/IsNotNull/NOT IsNull, empty-array conjunction, and correctness-only unsafe-large-integer ranges. The latter are explicitly forced to raw by the production UnaryExpr/BinaryRange policy; a subroot index cannot independently satisfy every whole-row NULL query. Those cases must not be advertised as index-only performance. Row validity/constant-only evaluation may not enter a path executor.

Tantivy instrumentation counts actual executor query methods rather than CreateExecutor, because the selector can instantiate an executor and subsequently choose raw. Each Search emits a `flat_route` record and CSV includes `t_route`/`k_route`. Asymmetric pattern cases assert zero Tantivy index-query calls and positive Knowhere calls. Unsafe range fallback asserts neither backend calls the index. Raw-dropped cases require both backend counters. Cases with no query-method call but retained raw are conservatively labeled `raw_or_validity`, not falsely declared indexed. Full bitmap and Search equality remains mandatory for common cases; known mixed-shape/numeric compatibility diagnostics remain explicit.

## Verified compatibility exceptions

The completed `flat-correctness` run found three Tantivy/raw differences, in both index-root configurations: mixed scalar equality (205 rows), mixed array membership (205 rows), and an escaped path (1 row). Those verified differences retain explicit compatibility-exception status. Additional correctness-only mixed-shape negation and duplicate-key fixtures report Tantivy differences explicitly and still require Knowhere/raw equality. Knowhere/raw parity remains mandatory for them.

All numeric boundary cases, mixed NOT, empty-container Exists and empty-array contains-all produced zero three-way differences. Their tests now require strict three-way parity, so a later regression cannot be accepted as an unreviewed compatibility difference. This tightening changes assertions only, not timed workloads or implementation.

## Profile-directed range and pattern optimization

The initial wide numeric range profile spent 24.46% in integer core materialization, 5.74% in double materialization, 14.71% in block decoding, 13.82% in codec work and 9.13% constructing views (self samples). The optimized scalar Range computes the same lower/upper ordinal bounds for all three physical numeric channels first. Numeric columns retain cumulative document frequencies in dictionary order, included in resident byte accounting. Complement is selected only when its term count is strictly smaller and its summed document frequency is no greater than positive traversal; both estimates take constant time after finding bounds. When both conditions hold, it clones the exact scalar-numeric validity bitmap and removes postings from both outer intervals of all three dictionaries. Narrow/equal-half ranges still decode matching terms. This changes traversal cost without changing cross-type comparisons.

Complement execution is restricted to Range on scalar single-valued paths. In remains positive union, and array mode never subtracts out-of-range postings: a row such as `[-1, 0, 1, 999]` must survive a range that matches its first three elements. Tests verify the exact threshold, route counter, cross-channel results, NULL/wrong-type exclusion, empty/inverted intervals and the multi-valued counterexample.

LIKE compiles and validates the complete pattern before looking up a path, then scans only the fixed literal prefix interval of its string dictionary. Regex compilation likewise precedes missing-path return. Tests include escaped literal underscore, nonexistent prefix, and malformed LIKE/regex on both existing and absent paths.

Scalar single-value safety additionally requires duplicate-key handling. Object traversal now indexes only the first occurrence of each key, matching raw JSON pointer binding, rather than combining multiple values under the same row/path. A build-only seen-key set provides this invariant. Ancestor Exists still considers nonempty content of ignored duplicate children, because Milvus's recursive existence helper scans every object member. Raw-string duplicate-key fixtures compare first/later values, null-first, strings, ranges and existence across both index roots; any Tantivy differences are emitted explicitly and excluded from timings. A separate `/doc` versus `/doc2` fixture ensures root extraction does not use a string-prefix shortcut.

These implementation changes still require the next unified correctness run and performance measurement; profile attribution alone is not evidence of the achieved speedup.

Correctness-only `not (mixed == 99)` and `not (json_contains(mixed, 99))` additionally distinguish scalar and array validity when the searched number is absent. Unlike the existing `not (mixed == 42)` fixture, these cannot accidentally agree merely because the array contains the searched value. They are excluded from timing and keep explicit mixed-shape Tantivy compatibility diagnostics.

A skewed regression fixture has 900 rows of zero and one row of each value 1–100. Range [1,99] must retain positive traversal despite matching 99 of 101 terms, and its full bitmap is compared against raw JSON numeric extraction. Existing uniform wide-range fixtures must still enter complement traversal.

## Executed validation and measured results

The unified resident PoC build, correctness runs, two benchmark rounds, independent
reviews and gated perf evidence are recorded in the
[expanded-feature report](../../../knowhere-scalar-poc/results/2026-09-21-expanded/README.md).
That report distinguishes index-only timings, native raw fallbacks, known
Tantivy/raw compatibility differences and accounted payload sizes. Earlier
baseline/proposal sections describe the optimization process, not additional
claims of production coverage.
