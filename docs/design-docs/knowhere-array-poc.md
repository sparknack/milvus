# Resident Knowhere ARRAY membership PoC

## Scope and row semantics

Implement ordinary ARRAY membership for BOOL, INT8, INT16, INT32, INT64, FLOAT, DOUBLE and VARCHAR, with one posting docID per Milvus row. A constructor-selected nested mode also indexes STRUCT ARRAY scalar subfields in the element domain. The new `InvertedIndexKnowhereArray<T>` is independent of Tantivy and uses the existing adaptive flat posting store and string FST. It is a resident segcore PoC: BuildWithFieldData → LoadIndex attachment → native expression execution → exact vector Search. Persistence, factory registration, nullable elements, recursive ARRAY<ARRAY> indexing, and array equality/order/length acceleration are outside this iteration. Existing segcore offsets perform the nested element-to-row conversion.

A row may contribute many terms; repeated values contribute only one posting entry. Empty valid arrays have zero postings but remain valid. NULL rows have zero postings and are invalid. `contains` and `contains_any` union docIDs; `contains_all` intersects them. Empty any is false, empty all is true for valid rows, and NOT must preserve NULL invalidity. No term frequency or positional stream is necessary for these set-membership operations. Per-term vectors exist only during build; resident storage is flat adaptive bytes plus metadata.

## Source investigation before implementation

Milvus `InvertedIndexTantivy.cpp:824` (`build_index_for_array`) and `:857` (string specialization) submit one document for every row, including empty/NULL rows. `null_offset_` separately records NULL. Rust binding `src/index_writer_v7/index_writer.rs:167` creates a TantivyDocument, adds each element under the same field, then inserts that document. Cargo.lock pins v7 to `96f3335ab5f061926c5b44cf246e81243e1dedc5`, and the legacy 0.21.1-fix4 backend to `bc211a5a76930b120b1eeb25073be27f20ba0387`.

[Lucene Document](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/Document.java) allows repeated fields with one name. [StringField](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/StringField.java) indexes a whole value as one term using DOCS-only postings and omits norms. This supports the same membership representation, without score/position overhead.

[Lucene TermInSetQuery](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/TermInSetQuery.java) uses constant-score OR semantics; its blended strategy combines bitsets and limited live iterators for many terms. [ConjunctionDISI](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/ConjunctionDISI.java) sorts iterators by cost and advances the sparsest first. The pinned [Tantivy intersection](https://github.com/zilliztech/tantivy/blob/96f3335ab5f061926c5b44cf246e81243e1dedc5/src/query/intersection.rs) similarly orders size hints and seeks docIDs. These are useful exact intersection strategies if profiling identifies contains_all overhead.

Milvus currently implements contains_all in `JsonContainsExpr.cpp:2495` as repeated `In(1)` calls and bitmap AND. The initial PoC kept this route to establish a comparable baseline. The optimization now adds the optional query-only `ArrayConjunctionIndex<T>` interface; it does not change ScalarIndex or Tantivy. The expression takes this capability for ordinary ARRAY or explicitly ARRAY-cast JSON, a non-nested index, and Count equal to the active row count. Scalar/flat JSON, nested indexes, range mismatches and backends without the capability retain the existing path. This preserves nested-index element-to-row conversion instead of intersecting element IDs as row IDs.

## Build and segcore attachment

The builder reads ARRAY FieldData row validity before accessing the Array object, deduplicates each term/row through the last posting ID, and delegates transactional publication to `BuildFromPostingsForPoC`. Thus a failed build does not publish a partially populated resident core. NULL and empty rows still count toward Count(). The inherited scalar index interface supports native ordinary ARRAY expression routing without modifying the Tantivy implementation or segcore mounting contracts.

`KnowhereArrayTest.cpp` builds both backends from the same nullable FieldData. The Tantivy comparator uses its real production writer and reopens its finished files in RAM. All three sealed segments (Tantivy, Knowhere, raw) retain raw arrays and the identical 4D L2 vectors. Per-backend counters verify Search actually calls the inverted index for nonempty term queries. Expression result caching is disabled.

## Correctness and performance protocol

INT64 and VARCHAR use cardinalities 64, 1000, and 100000. Valid rows have variable array lengths (1–12 generated terms, a common term on 80% of nonempty rows, and periodic duplicates; every 97th eligible row also contains IDs 0–7 to provide positive all-eight matches). Every 257th row is NULL, every 19th row empty. Workloads explicitly cover common/rare/missing membership; any of two/eight terms; all of two/eight terms; duplicates; common+missing; empty any/all; and negations. Query values are printed in `term_ids`; VARCHAR maps ID to `tag/{six-digit ID}/中文`.

Before timing, compare every result bit against an independent row-level membership reference and against raw segcore execution; compare Search offsets and distances for all three backends. The timing run uses 120000 rows per dataset, nq=1, topk=10, three warmups, eleven shuffled timing batches, wall-clock and process CPU medians. Profile mode gates recording after build, checks and warmup. This document describes the protocol, not a claim that an unexecuted test passed. Results and any optimized follow-up must be recorded separately.

## Profile-directed conjunction optimization

The initial 100000-cardinality INT64 profiles attributed 29.88% of all-eight samples to core bitmap materialization and 11.71% to block decoding; common-plus-missing attributed 52.46% and 20.32%, respectively. These samples motivated removing per-term full result materialization.

Knowhere `All` first resolves every requested term to an ordinal and returns an empty bitmap on the first missing term, before decoding any posting. It then invokes the shared exact `IntersectInto`: deduplicate ordinals, sort by document frequency, and use cursor seeks for sparse candidates or dense bitmap intersection when warranted. `IntersectInto` adds bits to its destination, so All supplies an initially empty bitmap. Empty conjunction explicitly returns IsNotNull, keeping valid empty rows and rejecting NULLs. This changes neither the codec nor the exact intersection semantics shared with TEXT_MATCH.

The correctness suite adds reversed terms, missing-first and repeated reversed terms, proves the optional All route is called by Search, and checks dispatcher guards for JSON, nested IDs, row-count mismatch and a backend without the capability. The original twelve timed workloads are unchanged, allowing direct before/after comparison. Compilation, correctness execution and optimized timing remain separate validation steps; code inspection alone is not a performance claim.

## 实测记录

[完整性能报告、正确性 XML、优化前后 perf 和逐项 workload](../../../knowhere-scalar-poc/results/2026-09-20-structured/README.md)。

## Production type and nested-mode matrix (2026-09-21)

The production ordinary and STRUCT-subfield inverted factories support BOOL, INT8, INT16, INT32, INT64, FLOAT, DOUBLE and VARCHAR (`IndexFactory.cpp::CreateNestedIndexInverted`, `InvertedIndexArrayTest.cpp::ElementType`). The resident adapter now accepts these eight C++ types. INT8/INT16 use protobuf int_data with an INT32 physical payload; `Array::get_data_unchecked<T>` performs the typed conversion. VARCHAR uses STRING physical payload. Typed tests preserve domain-valid queries; BOOL has no invented missing-value query, and INT8 negative misses remain representable. FLOAT/DOUBLE workloads use fractional values ending in .25.

`proxy/util.go::validateElementNullable` currently rejects every element_nullable=true request with the explicit temporary-disable branch, including otherwise supported ordinary ARRAY types. Although core Array has element validity storage, Tantivy's ordinary/nested builders read unchecked payloads, so merely ignoring invalid elements in Knowhere would not establish three-way compatibility. This iteration therefore does not expand the disabled API. Recursive ARRAY<ARRAY> schema is allowed only inside STRUCT ARRAY (`proxy/util.go::validateFieldType`), but `internal/util/indexparamcheck/index_params_validation.go::ValidateFieldIndexParams` explicitly rejects indexing recursive ARRAY. That is different from the supported scalar-subfield nested index implemented here.

The nested resident mode follows `InvertedIndexTantivy.cpp::build_index_for_array_nested`: every valid row contributes one indexed document per element; duplicate values retain distinct element IDs; NULL and empty rows contribute no element documents. Count and IsNotNull refer to the element domain, while the sealed raw column retains row validity. `IsNestedIndex` causes existing segcore `GetArrayOffsets` and `ProcessIndexChunksWithRowLevel` to convert matches and use row validity. Contains-all must perform each term's element-to-row projection before the row bitmap AND: intersecting element IDs first would wrongly require different values at the same element position. The optional row All capability remains disabled for this mode. `NullExpr::DetermineExecPath` explicitly uses raw validity for nested indexes, covered by full bitmap and Search tests.

[Lucene ToParentBlockJoinQuery](https://github.com/apache/lucene/blob/main/lucene/join/src/java/org/apache/lucene/search/join/ToParentBlockJoinQuery.java) is the relevant analogue: child matches are projected into parent document space, with distinct identifiers and preserved block boundaries. Milvus already owns the row/element offsets, so the PoC reuses its existing projection instead of introducing a second join representation.

Additional correctness covers every newly supported ordinary type and all eight types in nested mode, with NULL/empty rows, duplicates, any/all/NOT, and index-route counters. Nested NULL/NOT NULL checks compare complete bitmaps and Search against raw rows. Benchmarks add BOOL (2), INT8 (64/100), INT16 (64/10000), INT32/FLOAT/DOUBLE (64/100000), and nested INT64/VARCHAR (64/100000); suffix numbers denote configured domain cardinality. Existing INT64/VARCHAR datasets and workloads remain unchanged. Execution results must be reported separately after the frozen source is built and tested.

The expanded-feature profile found nested contains-all dominated by the per-element row projection. Its shared segcore path now calls the existing `ElementBitsetToRowBitsetAny` reducer for each term before applying row-level AND. Both Tantivy and Knowhere use this optimization; final comparisons rebuild and measure both. Zero-initialized output, global element offset zero and active row range preserve empty/NULL rows and row/element domains.

Floating ARRAY builders now use the scalar core's Tantivy-compatible total-order comparator before ordered-map insertion, keeping NaN terms distinct from finite keys. FLOAT is promoted to double just like the Tantivy binding. Tests cover FLOAT/DOUBLE and both row/element domains, including signed/payload NaN, infinity, zero, NULL and snapshot reload; see [input boundaries](knowhere-input-boundary-compat.md). Split-FieldData tests also cover global IDs and NULL rows with retained nonempty physical payload.

## Executed validation and measured results

The unified resident PoC build, correctness runs, two benchmark rounds, independent
reviews and gated perf evidence are recorded in the
[expanded-feature report](../../../knowhere-scalar-poc/results/2026-09-21-expanded/README.md).
That report distinguishes index-only timings, native raw fallbacks, known
Tantivy/raw compatibility differences and accounted payload sizes. Earlier
baseline/proposal sections describe the optimization process, not additional
claims of production coverage.

## Resident space update (2026-09-21)

The later [space compaction](knowhere-space-poc.md) removes the unused scalar
FST and stores ordered scalar strings in a contiguous pool. Earlier FST/space
statements above describe the original implementation. TEXT/NGRAM retain their
actively used FSTs; query semantics and snapshot bytes are unchanged.
