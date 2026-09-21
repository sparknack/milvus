# Resident Knowhere JSON path ARRAY PoC

This adapter replaces the postings backend of one typed JSON array path. It
inherits the independent `InvertedIndexKnowhere<T>` and reuses flat adaptive
posting storage and the resident string dictionary. It does not subclass or
modify Tantivy. Supported production cast types are ARRAY_BOOL, ARRAY_DOUBLE,
and ARRAY_VARCHAR. Milvus has no ARRAY_INT64 JSON cast; integer query literals
are tested through ARRAY_DOUBLE rather than inventing a new production type.
Growing segments, persistence, JSON flat indexing, and factory integration are
outside this resident PoC.

## Implementation review before coding

`JsonScalarIndexWrapper::BuildInvertedWithJsonFieldData` passes array casts to
`ProcessJsonFieldData`. That production helper extracts the JSON pointer,
filters elements through the requested scalar type, and supplies all accepted
values with the same document offset. Empty arrays and arrays with no accepted
elements still represent documents. Duplicate matching elements must not
multiply hits. Mixed-type elements that cannot be extracted by the requested
cast do not get indexed.

[Lucene SortedSetDocValuesField](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/document/SortedSetDocValuesField.java)
models multiple values per document as a set and distinguishes that representation
from indexed postings used for efficient membership search. We borrow the
document-domain set semantics; we do not copy doc-values scanning into the
Knowhere search path.

[Lucene ConjunctionDISI](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/ConjunctionDISI.java)
orders iterators by estimated cost so a sparse iterator leads the conjunction.
The adapter's optional `All` capability resolves all terms before decoding,
returns empty immediately for a missing term, and uses the existing exact
Knowhere intersection implementation. Initial segcore measurements retain the
existing per-term bitmap intersection; enabling the optional capability is a
separate measurable optimization.

## Independent existence and array validity

The adapter invokes production `ProcessJsonFieldData<T>` to construct temporary
term-to-document lists. Consecutive duplicate row IDs per term are removed, then
`BuildFromPostingsForPoC` flattens and compresses the postings. A separate
shape-validation pass records rows whose path is an actual array, including
empty arrays and arrays containing only other types. It excludes SQL NULL,
missing paths, JSON null, objects, and scalar values. This bitmap supplies
`IsNotNull` for membership-expression three-valued logic. EXISTS is independently
populated from the production helper's `non_exist_adder`: scalar values and
nonempty objects at the path still exist even though array membership is UNKNOWN.
The production `Json::exist` recursively treats empty arrays, empty objects,
and containers containing only null/empty containers as absent. We preserve
that convention: an empty array is valid for membership but false for EXISTS.
Thus neither bitmap can be reconstructed from the other.

All potentially failing extraction and postings work precedes publishing the
new existence and validity bitmaps. Scalar-only build entry points are rejected
or hidden. Rebuilds require exclusive access. Temporary maps and vectors are
released; permanent posting lists are flat, not per-term vectors.

The existing Tantivy ARRAY path wrapper recorded only SQL NULL in
`null_offset_`, so NOT membership could include missing or non-array paths
while raw JSON treated them as UNKNOWN. The wrapper now computes array-shape
validity before invoking the unchanged shared extraction helper. Its callback
no longer duplicates SQL NULL offsets. `non_exist_offsets_` remains controlled
by the production helper, preserving EXISTS for present scalar/object paths.
This is a baseline correctness fix, applied before claiming comparative speed.
The existing `JsonPathIndexTest` gains a direct Tantivy regression for missing,
null, wrong-shape, empty, and mixed-element arrays. Segcore tests require strict
Tantivy/Knowhere/raw parity for NOT and composed nullable expressions.

`ProcessJsonFieldData` itself is unchanged: its other callers include scalar
conversion and both NGRAM adapters. The fix is limited to the ARRAY branch of
`JsonScalarIndexWrapper`, leaving those scalar/NGRAM validity conventions intact.

## Workloads and verification

`KnowhereJsonArray.SegcoreParity` tests DOUBLE, VARCHAR, BOOL and an escaped path
`a/b~c`. All positive membership and EXISTS/NOT EXISTS workloads compare every
bitmap bit and Search offsets/distances across Tantivy, Knowhere, and raw sealed
segments. Negative membership, double NOT, `P or not P`, and SQL NULL combinations
exercise array-shape validity. Empty ANY/ALL and heterogeneous query lists are
checked through segcore too, preserving its current constant-result semantics
for empty JSON query lists and raw fallback for mixed query types.
`BuildAndMembershipContracts` covers empty arrays, duplicate elements, arrays
with no accepted elements, scalar/missing/null paths, exact multi-term
conjunction, empty conjunction, rejected raw build, smaller/empty rebuild, and
failure after JSON extraction preserving the previous index.

`KnowhereJsonArray.Benchmark` uses 120,000 rows per cast type. A nonempty array
has 1–8 rotating values from a 100,000-value domain; 80% receive common value 0,
every 97th generated row receives terms 0–7, and each row repeats a value to test
deduplication. Strides 19, 23, 29, 31, 37 produce empty arrays, absent paths, JSON
null, scalar values, and objects. Every eleventh eligible array includes mixed
unindexable elements and a nested array. Numeric data also contains fractional
42.5. Unicode appears in string terms. Rows are shuffled deterministically;
every 257th final row is SQL NULL.

Case names state the operation and supplied term count: `any_8` means OR over
terms 0–7, `all_8` means all eight terms must occur in the same row. Boolean
queries naturally collapse repeated literal values to at most two distinct
terms, so their labels do not imply eight distinct Boolean terms. Other cases
cover rare/common/missing membership, two-term conjunction/disjunction, duplicate
conjunction terms, and numeric fractional membership. There are 32 timed cases.

All segments follow build → resident LoadIndex attachment → complete Search.
Raw JSON remains loaded. Search is exact four-dimensional L2, nq=1, topk=10;
expression result caching is disabled. Correctness precedes three warmups and
11 shuffled timing batches. Wall/process CPU medians and build/attach times are
reported. Perf uses the ready/go/done gate after correctness and warmups. There
is no persisted reload or mmap claim. Performance claims await actual runs.

## Existing persisted indexes

The Tantivy validity correction applies to newly built indexes. Existing
persisted ARRAY indexes may contain only SQL NULL row offsets, without the
missing-path and wrong-shape information needed for correct membership
validity. Loading those files cannot reconstruct the missing information from
the old offset metadata alone; they require an index rebuild. This resident
PoC does not implement automatic repair, format migration, or promise corrected
semantics for previously persisted indexes.

The optimized query route enables `ArrayConjunctionIndex` only for an explicit JSON ARRAY cast, non-nested row domain and matching active row count. K-specific All counters verify dispatch. JSON scalar/flat casts and nested/count mismatches are tested as negative guards. Empty and mixed-query front-end paths stay unchanged. The initial repeated-In baseline is retained in the expanded-feature report for comparison.

## Executed validation and measured results

The unified resident PoC build, correctness runs, two benchmark rounds, independent
reviews and gated perf evidence are recorded in the
[expanded-feature report](../../../knowhere-scalar-poc/results/2026-09-21-expanded/README.md).
That report distinguishes index-only timings, native raw fallbacks, known
Tantivy/raw compatibility differences and accounted payload sizes. Earlier
baseline/proposal sections describe the optimization process, not additional
claims of production coverage.
