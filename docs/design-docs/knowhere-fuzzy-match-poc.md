# Knowhere resident FUZZY_MATCH PoC

## Scope and compatibility contract

Replace the sealed resident text index query implementation only. Keep the same
Tantivy analyzer and the real segcore TextMatchFuzzy expression/Search route.
No changes to the production index factory, growing support, index persistence,
Tantivy's implementation, or query parser are included.

The Milvus expression is `text_match_fuzzy(text, "alergy aple", max_edit_distance=1)`.
The public C++ interface is `FuzzyMatchQuery(query, max_edit_distance)`.

Source audit before implementation:

- `internal/core/thirdparty/tantivy/tantivy-binding/src/index_reader_text.rs`,
  `fuzzy_match_query`: validate distance 0/1/2 before tokenization, distance zero
  delegates to ordinary OR term matching, other distances OR whole-term fuzzy
  queries for analyzer tokens with `transposition_cost_one=true`.
- Pinned Tantivy `96f3335ab5f061926c5b44cf246e81243e1dedc5`,
  `src/query/fuzzy_query.rs`: non-prefix DFA, Unicode Levenshtein automata,
  constant-score/all matching terms through AutomatonWeight. No expansion cap
  is exposed by this Milvus binding.
- `levenshtein_automata` 0.2.1, `levenshtein_nfa.rs`: query/other `chars()` are
  Unicode scalar values, adjacent transposition costs one; the pending
  transposition consumes the reversed pair. The oracle uses optimal string
  alignment (OSA), not unrestricted Damerau distance.
- `internal/core/src/exec/expression/UnaryExpr.cpp`: validates required edit
  distance in [0,2] then calls the resident text holder's FuzzyMatchQuery.

[Lucene 10.3.1 FuzzyQuery](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/search/FuzzyQuery.java)
uses OSA when transpositions are enabled, but its default expansion/scoring and
short-word scaling rules differ. Do not import those product-level restrictions.
[Lucene LevenshteinAutomata](https://github.com/apache/lucene/blob/releases/lucene/10.3.1/lucene/core/src/java/org/apache/lucene/util/automaton/LevenshteinAutomata.java)
constructs DFAs in Unicode code points with a maximum edit distance of two.
The architecture to reuse is dictionary/automaton intersection; the current
implementation below is a bounded DP automaton rather than Lucene's parametric
DFA construction. No Lucene code is copied.

Examples: `ab`/`ba` costs one, `ca`/`abc` costs three under OSA; `cafe`/`café`
costs one code-point substitution. UTF-8 byte count is not edit distance. We do
not normalize accents or grapheme clusters beyond the configured analyzer.
Multiple tokens mean OR, not AND or a fuzzy phrase. There is no exact-prefix
restriction and no top-50 term cutoff. Missing exact dictionary entries may
still have fuzzy matches.

## Implementation

`FstTermDictionary::ForEachFuzzy(term, max_edits, visitor)` intersects the FST
with a query-owned streaming Unicode OSA state. Each FST arc consumes a UTF-8
byte; the DP advances only after a complete Unicode scalar. Invalid or partial
UTF-8 query terms are rejected, and invalid dictionary paths cannot match.
The empty-key envelope is checked separately, preserving the dictionary's
ordinal contract.

The state keeps the previous two DP rows and the preceding candidate code
point. Values saturate at `max_edits + 1`; this preserves all decisions at or
below the supported threshold. Transposition uses the row from two positions
back. Prefix pruning checks the current and previous minima conservatively,
and the maximum possible candidate length. Incomplete UTF-8 sequences do not
invent characters. Rows now store only columns `|candidate_depth - query_column| <= k`. Insertion
and deletion establish this length lower bound; transposition consumes two
characters on each side and does not invalidate it. Every out-of-band value
reads as `k+1`. For k<=2 each row holds at most five uint8 cells; both rows have
fixed capacity, including long queries. UTF-8 continuation bytes never advance
the row or its band. There is no silent maximum-token-length cutoff.

The adapter analyzes and deduplicates query tokens, collects matching term
ordinals, deduplicates those ordinals, then unions their adaptive posting lists.
Thus overlapping fuzzy expansions decode each posting at most once per query.
TF/positions are never read. Distance zero uses the existing exact MatchQuery.
FST and postings remain flat immutable storage; no new retained term strings,
per-term vectors, or auxiliary fuzzy index are introduced.

The initial full-width DP implementation is preserved in `/tmp/poc-fuzzy-initial`
for baseline profiling. Initial missing-term profiles attributed 92.17% sampled
self cycles to FST traversal (including inlined DP); no posting decode was present.
The compact-band change targets this path. The synchronous traversal now passes
its visitor by reference wrapper, avoiding per-arc copies of std::function while
preserving the caller-owned callback lifetime. Correctness and new performance
must be verified before claiming a speedup. Timing/profiles must not run
concurrently with compilation.

## Correctness and benchmark plan

`KnowhereFuzzyTest.cpp` contains:

1. FST enumeration checked against an independent full-matrix UTF-32 OSA oracle,
   with all strings of length <=5 over a/b/c, empty terms, Unicode including
   supplementary planes, transposition cases and >63-code-point queries.
   Distances 0/1/2, invalid 3/256/UINT32_MAX, malformed/truncated UTF-8 are covered.
2. Direct Tantivy/Knowhere/raw-token-oracle parity with standard/lowercase,
   stop filtering and Jieba. Unicode scalar conversion in the test uses the
   standard codecvt UTF-8 converter, independently of the production decoder.
3. Real resident build/holder attach, full segcore bitmap including NOT/null,
   and Search offsets/distances parity. Raw tokens are independently aggregated
   in a test-only word-to-row map; no encoded dictionary/postings are used by
   the oracle. There is no raw scan fallback in the production adapter.
4. Gated Release benchmark: AG News titles and descriptions, each 120,000 rows,
   plus a separately labelled synthetic Unicode corpus of 120,000 rows. The
   synthetic corpus is not pooled with real-news performance.

Each dataset runs 19 cases: empty/punctuation; highest-DF term at distances
0/1/2; lowest-DF ASCII term of length >=4 exact and replacement/insertion/deletion/
transposition/two-insert edits; long absent token; high+low OR; duplicate query;
short `a` distance2; NOT; Latin accented/CJK/Cyrillic queries. The actual query,
exact-term DF, final hits and both backend timings are printed. Query mutation
can coincide with another existing word; the independent oracle defines the
expected expansion, not assumptions about a particular typo. AG News Unicode
cases may be absent; the separate Unicode corpus supplies positive coverage.

Null every257, same analyzer, 4D exact L2 NQ1 topk10, seed20260920, one query
thread, OMP1 and expression cache off. All correctness checks run before warmup
or timing. There are 31 shuffled batches of four Search calls; median index15
and nearest-rank p95 index29. Process CPU includes workers/background threads;
caller-thread CPU and optional perf counters exclude workers. These are batch
means, not online latency percentiles. Holder attach is not disk reload.

Reproduction (root integrates the new source in the existing PoC CMake target):

```sh
# Existing LD_LIBRARY_PATH, LOCAL_STORAGE_PATH and MILVUS_TEST_ROOT_DIR required.
OMP_NUM_THREADS=1 POC_TEXT_CODEC=adaptive POC_BENCHMARK=1 \
POC_TEXT_DATA_DIR=/tmp/milvus-poc-text-data \
  cmake_build/unittest/scalar_poc_tests --gtest_filter=KnowhereFuzzy.AGNewsBuildLoadSearchBenchmark
```

Perf gate protocol matches the existing runner: POC_PROFILE_DATASET selects one
of `ag_news_titles`, `ag_news_descriptions`, `fuzzy_unicode_120k`;
POC_PROFILE_CASE is the emitted case name, POC_PROFILE_BACKEND is `knowhere` or
`tantivy`, and POC_PROFILE_GATE publishes `.ready` after correctness/setup,
waits at most60 seconds for `.go`, then runs isolated Search for8 seconds and
writes `.done`. Profile attribution runs are separate from normal measurements.

## Error and review boundaries

Distance outside [0,2] is request-content validation, including empty queries;
the new direct adapter and FST API throw the existing InvalidParameter code.
No error mapping, retry layer, cgo projection, or blanket catch is introduced.
The existing exec/parser distance checks remain unchanged. Analyzer/build
failures propagate through existing behavior. This document does not claim
production wire-category audits, fault recovery, or malformed-index support.

Tests/benchmarks have been implemented but not yet executed by this subtask;
root performs the unified build, test, measurement, and independent review.

## Executed validation and measured results

The unified resident PoC build, correctness runs, two benchmark rounds, independent
reviews and gated perf evidence are recorded in the
[expanded-feature report](../../../knowhere-scalar-poc/results/2026-09-21-expanded/README.md).
That report distinguishes index-only timings, native raw fallbacks, known
Tantivy/raw compatibility differences and accounted payload sizes. Earlier
baseline/proposal sections describe the optimization process, not additional
claims of production coverage.
