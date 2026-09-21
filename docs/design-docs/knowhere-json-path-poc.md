# Resident Knowhere JSON path scalar PoC

This PoC replaces the postings backend for a single explicitly typed JSON path.
It reuses Milvus extraction, casting, path escaping, plan routing, and vector
search. `JsonPathIndexKnowhere<T>` derives from the independent Knowhere scalar
index, not from `InvertedIndexTantivy` or its JSON wrapper. The scope is resident
DOUBLE, VARCHAR, and BOOL path indexes. JSON flat indexing, JSON NGRAM, arrays,
factory registration, and persisted reload are separate work.

## Prior implementation review

Milvus `JsonScalarIndexWrapper.h` and `JsonIndexBuilder.cpp` are the behavioral
reference. The scalar wrapper calls `ConvertJsonToTypedFieldData`, which marks
missing paths and unsuccessful casts invalid. It separately collects
`non_exist_offsets`: a present value that fails a cast must still satisfy
EXISTS. JSON null and SQL NULL are tracked according to the existing converter.
The Knowhere wrapper invokes precisely this converter, then builds adaptive
postings and a separate existence bitmap. No analyzer is involved in a typed
JSON scalar path.

The Tantivy binding (`tantivy-binding/src/index_reader.rs`) uses basic term
queries for small IN sets and `TermSetQuery` for larger sets; range queries use
typed term bounds. Knowhere reuses its typed sorted dictionary, direct bitmap
union, and adaptive postings for these operations. String terms also retain the
existing resident FST representation. Permanent postings are flattened; the
converter's typed column is temporary and released after build.

[Lucene FieldExistsQuery](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/FieldExistsQuery.java)
uses a field's doc-values, norms, or vector iterator to enumerate existence,
with all-documents shortcuts where provable. We borrow the principle that
existence is dedicated metadata rather than a union of all value terms. Milvus
JSON semantics require a separate bitmap because cast failure is different
from path absence.

[Lucene TermInSetQuery](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/TermInSetQuery.java)
sorts and deduplicates its terms and performs constant-score multi-term
matching. It also separates term enumeration from result collection. The PoC
uses the existing exact scalar IN/NOT IN implementation rather than introducing
scoring, BM25, or positional data for this scalar feature.

## Build and query

1. Convert input JSON FieldData with the production converter and requested
   JSON pointer/cast type.
2. Build scalar adaptive postings from valid cast values. Invalid offsets
   supply scalar validity metadata. Build the existence bitmap independently.
3. Publish existence only after the scalar build succeeds.
4. Attach via the native resident segcore `LoadIndex` path with `JSON_PATH` and
   `JSON_CAST_TYPE` metadata. Existing segcore expression dispatch uses the
   scalar interface and `GetCastType`.
5. Equality, IN, NOT IN, and range queries reuse the independent scalar index;
   EXISTS clones the dedicated bitmap. Logical NOT remains segcore's operation
   and preserves the existing nullable expression semantics.

There is no new persistence format or storage-path change. This benchmark's
load stage is resident attachment, not disk deserialization or mmap.

## Validation and workloads

`KnowhereJsonPath.SegcoreParity` creates independent Tantivy, Knowhere, and raw
JSON sealed segments. It checks every result bit, then complete Search offsets
and distances. Per-index counters verify queries actually dispatch to each
index. It includes numeric fractional values, hot and rare keys, missing keys,
JSON null, nullable rows, arrays/objects rejected by scalar casts, Unicode, and
an escaped JSON pointer (`/a~1b~0c`). It covers IN, NOT IN, narrow/wide ranges,
EXISTS, NOT EXISTS, and NOT equality. Correctness precedes timing.

`KnowhereJsonPath.Benchmark` uses 120,000 deterministic rows per DOUBLE,
VARCHAR, and BOOL dataset; every fifth generated value is hot, other keys have
up to 100,000 distinct values. Missing-path stride is 23, JSON-null stride 29,
array/object cast-failure strides 31/37, SQL NULL stride 257 after shuffling.
The same raw field remains available in all three segments. Search uses 4-D
exact L2, nq=1, topk=10. Expression result caching is disabled. After three
warmups per query/backend, eleven shuffled batches measure complete Search
wall time and process CPU. No timings from extraction, build, correctness, or
warmup enter the Search samples. Build/attach times are reported separately.

Perf mode uses the existing ready/go/done gate after selected-workload
correctness and warmups, enabling an eight-second isolated Search loop. Final
performance claims require the root report's actual benchmark and perf results;
this design document alone does not claim a speedup.

## Typed validity optimization

The initial DOUBLE `not_equal` profile attributed approximately 3.09% of samples
to Knowhere `IsNotNull` and 2.86% to `NotIn`; the shared L2 search accounted for
approximately 63%. The baseline Search median was about 492 us versus Tantivy's
474 us. These numbers identify a bounded optimization opportunity, not a claim
that this metadata change can accelerate the dominating vector search.

The adapter now materializes typed validity once during JSON conversion.
`IsNotNull` clones it, `IsNull` clones and flips it, and `NotIn` intersects the
complement of exact term hits with it. This matches Tantivy's existing sealed
validity-cache approach and avoids repeated iteration over invalid row offsets.
EXISTS still has its own bitmap: cast failure never becomes path absence.
Both bitmaps are accounted for in `ByteSize` and are moved into place only after
transactional postings construction succeeds. Builds require exclusive access;
this is failure-atomic publication, not a concurrent rebuild protocol.

Typed scalar build overrides reject entry without JSON extraction, and the
nonvirtual scalar-only PoC build helpers are hidden on this adapter. The tests
exercise different-size rebuilds, all-valid and all-invalid typed data, empty
build, caller mutation of cloned validity, rejected scalar entry points, and
embedded-NUL rejection after extraction while preserving the previous postings
and both bitmaps. Intentional casts to the implementation base class are outside
the adapter API. Benchmark and perf must be rerun before quantifying the gain.

## 实测记录

[完整性能报告、正确性 XML、优化前后 perf 和逐项 workload](../../../knowhere-scalar-poc/results/2026-09-20-structured/README.md)。

## Resident space update (2026-09-21)

The later [space compaction](knowhere-space-poc.md) removes the unused scalar
FST and stores ordered scalar strings in a contiguous pool. Earlier FST/space
statements above describe the original implementation. TEXT/NGRAM retain their
actively used FSTs; query semantics and snapshot bytes are unchanged.
