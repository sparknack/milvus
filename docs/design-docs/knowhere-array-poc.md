# Resident Knowhere ARRAY membership PoC

## Scope and row semantics

Implement ordinary ARRAY<INT64> and ARRAY<VARCHAR> membership with one posting docID per Milvus row. The new `InvertedIndexKnowhereArray<T>` is independent of Tantivy and uses the existing adaptive flat posting store and string FST. It is a resident segcore PoC: BuildWithFieldData → LoadIndex attachment → native expression execution → exact vector Search. Persistence, factory registration, nullable elements, nested ARRAY, element offsets, array equality/order/length acceleration and other element types are outside this iteration.

A row may contribute many terms; repeated values contribute only one posting entry. Empty valid arrays have zero postings but remain valid. NULL rows have zero postings and are invalid. `contains` and `contains_any` union docIDs; `contains_all` intersects them. Empty any is false, empty all is true for valid rows, and NOT must preserve NULL invalidity. No term frequency or positional stream is necessary for these set-membership operations. Per-term vectors exist only during build; resident storage is flat adaptive bytes plus metadata.

## Source investigation before implementation

Milvus `InvertedIndexTantivy.cpp:824` (`build_index_for_array`) and `:857` (string specialization) submit one document for every row, including empty/NULL rows. `null_offset_` separately records NULL. Rust binding `src/index_writer_v7/index_writer.rs:167` creates a TantivyDocument, adds each element under the same field, then inserts that document. Cargo.lock pins v7 to `96f3335ab5f061926c5b44cf246e81243e1dedc5`, and the legacy 0.21.1-fix4 backend to `bc211a5a76930b120b1eeb25073be27f20ba0387`.

[Lucene Document](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/Document.java) allows repeated fields with one name. [StringField](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/StringField.java) indexes a whole value as one term using DOCS-only postings and omits norms. This supports the same membership representation, without score/position overhead.

[Lucene TermInSetQuery](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/TermInSetQuery.java) uses constant-score OR semantics; its blended strategy combines bitsets and limited live iterators for many terms. [ConjunctionDISI](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/ConjunctionDISI.java) sorts iterators by cost and advances the sparsest first. The pinned [Tantivy intersection](https://github.com/zilliztech/tantivy/blob/96f3335ab5f061926c5b44cf246e81243e1dedc5/src/query/intersection.rs) similarly orders size hints and seeks docIDs. These are useful exact intersection strategies if profiling identifies contains_all overhead.

Milvus currently implements contains_all in `JsonContainsExpr.cpp:2495` as repeated `In(1)` calls and bitmap AND. The initial PoC kept this route to establish a comparable baseline. The optimization now adds the optional query-only `ArrayConjunctionIndex<T>` interface; it does not change ScalarIndex or Tantivy. The expression takes this capability only for ARRAY, a non-nested index, and Count equal to the active row count. JSON, nested indexes, range mismatches and backends without the capability retain the existing path. This preserves nested-index element-to-row conversion instead of intersecting element IDs as row IDs.

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
