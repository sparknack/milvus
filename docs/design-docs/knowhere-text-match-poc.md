# Knowhere-derived resident TEXT_MATCH PoC

## Scope

Extend the existing StreamVByte docID-only PoC to tokenized `TEXT_MATCH` over
sealed resident documents. Retain Milvus's Tantivy analyzer; replace only the
term dictionary, postings and matching execution. This is not a production
IndexFactory option, growing implementation, persisted format, BM25 scorer,
phrase matcher or fuzzy matcher.

The scalar `InvertedIndexKnowhere<T>` remains separate. Text uses
`TextMatchIndexKnowhere` and the same `InvertedIndexKnowhereCore` posting codec.
`InvertedIndexTantivy.h/.cpp` and `Meta.h` remain unchanged.

## Integration

`TextMatchIndexBase` abstracts the text operations consumed by segcore, validity,
row count and accounted bytes. Existing `TextMatchIndex` continues to implement
Tantivy behavior; optional growing methods default to explicit Unsupported errors
for the sealed-only candidate. The existing Tantivy cache translator and builder
remain concrete and unchanged. `GetTextIndex` returns the common interface while
retaining its original pin owner, shared holder and published-snapshot lifetime.

Tests build either implementation and use the same resident text-index holder
and segcore runtime publication hooks to attach it. `TEXT_MATCH` uses
`runtime.text_indexes`; it does not use scalar `LoadIndex`/`scalar_indexings`.
This is a resident load/attach test, not Serialize -> file reload. Neither segment
loads the raw text column or an alternate scalar index. Search executes the real
UnaryExpr text-index path, validity handling and vector search for both backends.

## Build and query invariants

- Analyze documents with the configured Milvus/Tantivy tokenizer and filters.
  Insert each `(term, document ID)` once regardless of within-document frequency.
- Preserve total row count independently of terms so null, empty and tokenless
  documents have correct row domains. Null rows emit no postings.
- Build a sorted term dictionary and increasing unique postings. Use the existing
  256-entry docID blocks, with SIMD padding. The core supports StreamVByte0124
  and the dependency's AdaptiveBlockCodec; text now defaults to adaptive, while
  scalar retains StreamVByte. The selected format stays with the core through
  transactional rebuilds and is passed explicitly to each block reader. Adaptive
  calls encode_doc_ids, selecting bitpacking, all-equal or StreamVByte, without
  adding TF or the upstream singleton short form. No positions, weights
  or per-document term frequencies are stored.
- Build into temporary structures and publish only after successful analysis and
  encoding; failed builds leave the prior index intact.
- Clone the analyzer per query; all published posting state is read-only.
- Analyze the query with the same configuration. For minimum <= 1, OR the posting
  lists, deduplicating repeated terms. No analyzed tokens means no hits.
- For minimum > 1, first resolve every query clause through the dictionary. If
  fewer clauses can match than the minimum, return an empty bitmap without
  allocating per-row counters or decoding postings. Count duplicate clauses
  separately in this feasibility check, consistent with Tantivy.
- If the number of present clauses equals the minimum, every present clause
  must match. Deduplicate posting IDs only after this decision, order them by
  increasing document frequency, and use a density-selected intersection. Sparse candidates use the
  shortest-list-driven cursor path; if the shortest list exceeds roughly one
  candidate per 64 rows, sequential decode plus bitmap AND avoids repeated
  per-doc seeks. Cursors binary-search block maxima on forward seeks and decode only selected
  blocks. When another cursor advances past the candidate, seek the lead cursor
  to that document and restart alignment. Exhaustion ends the intersection.
  This also applies when missing clauses reduce a general threshold to AND.
- Otherwise decode present clauses and increment saturating per-row counts.
  Repeated query terms count as repeated clauses; repeated occurrences within a
  document do not. The original implementation omitted the dictionary preflight;
  the resulting missing-term slowdown was an implementation omission, not a codec
  or Knowhere limitation. Before/after measurements are retained in the report.
- Null validity remains separate from hits so `NOT TEXT_MATCH` excludes nulls.
- Embedded NUL is explicitly unsupported by this PoC's C-string analyzer bridge.
  Phrase/fuzzy queries and growing mutation return typed Unsupported errors.

## Correctness gate and performance method

Correctness precedes measurement. Compare the candidate with both Tantivy and an
independent document-token-set scan on small inputs: punctuation, case folding,
stop words, Chinese/Jieba, empty queries/documents, nulls, repeated terms,
thresholds above available clauses, block boundaries, failed rebuild and concurrent
queries. Existing Tantivy growing, sealed, null, phrase, fuzzy, cache, load and
array regressions exercise the common-interface change.

For each real AG News title/description dataset (120k original documents each),
compare full direct-query bitmaps, full segcore expression bitmaps (including NOT),
then Search offsets and distances before warmup. Stop before timing on mismatch.
Disable expression-result caching and raw-text fallback. Use identical vectors,
plans, nulls, single query thread, NQ=1, top-k=10, exact 4D L2. Randomize execution
order across 31 batches of four searches and report median/p95 of batch means.
Run twice; include build/attach time and accounted index bytes. Separate strace
runs audit query file accesses and do not supply reported performance numbers.

The default Tantivy text index also stores positions for phrase match, while this
candidate implements only TEXT_MATCH. Accounted bytes are not total heap/RSS or
peak build memory. Results compare these implementations and capabilities, not
isolated compression codecs or every possible optimization of either engine.

[Results and reproduction](../../../knowhere-scalar-poc/results/2026-09-20-text-match/README.md).
[Prior Tantivy source investigation](../../../knowhere-scalar-poc/tantivy-phrase-review.md).

[Knowhere (adaptive) versus Tantivy and full workload descriptions](../../../knowhere-scalar-poc/results/2026-09-20-block-adaptive/README.md).


## FST dictionary extension

The text adapter now compiles its sorted terms into a pinned cpp-fstlib dictionary
and passes uint32 ordinals to the posting core. The core no longer retains text
strings; scalar dictionaries are unchanged. Dictionary bytes can be serialized
and mapped read-only with owned lifetime and checksum verification. Postings,
metadata and nulls are still resident containers; this is dictionary-only mmap,
not production text-index persistence. See [selection, layout and ownership](../../../knowhere-scalar-poc/fst-review.md).


The current benchmark defaults to owned memory for the FST. Dictionary mmap is
an explicit `POC_DICTIONARY_MMAP=1` variant. Expanded low-frequency and mixed-DF
queries validate full results before collecting wall, process-CPU and caller-CPU
time; optional Linux perf counters measure only calling-thread user-space events.
[Results and known mixed-AND limitation](../../../knowhere-scalar-poc/results/2026-09-20-memory-lowfreq-cpu/README.md).


## Intersection cursor invariants

The cursor is query-local and borrows immutable bytes pinned by the text-index
holder. Its 256-doc buffer is independent for each term. Seek(target) is monotone
and idempotent for target <= current doc; Next and Seek stay exhausted once ended.
UINT32_MAX is reserved for exhaustion; builders already restrict IDs to INT32_MAX.
A fresh cursor performs no decode until Seek/Next. Block maxima locate candidate
blocks without touching skipped payloads; decoding restores the base from the
previous block maximum, including partial final blocks. Sparse-query work buffers are
O(distinct required terms * block size), plus the existing result bitmap. Dense
conjunctions use two temporary bitmaps and stop if their intersection is empty.
Neither AND path allocates the per-row uint32 threshold array. The density cutoff
is a heuristic, not a guarantee of optimal planning for all distributions.

Repeated clauses must retain their weight when choosing the query path. For
example `hot hot rare` with minimum 2 stays on the threshold path; minimum 3
uses intersection of `hot` and `rare`. `hot rare missing` with minimum 2 also
uses intersection. Missing-term infeasibility and OR behavior are unchanged.

Correctness tests compare cursor seeks with lower_bound, exercise block edges,
empty/exhausted cursors and the maximum signed docID in both codecs, and assert
a seek to doc 90000 skips more than 350 blocks while decoding exactly one.
Core intersections are checked against scan; text dispatch is checked against
both Tantivy and a scan, with duplicate/missing clauses, nulls and both codecs.

[Intersection correctness and performance results](../../../knowhere-scalar-poc/results/2026-09-20-intersection/README.md):
mixed-DF AND improves, while the current dense eight-term bitmap strategy still
regresses against the earlier counting implementation; the cutoff needs further evaluation.
