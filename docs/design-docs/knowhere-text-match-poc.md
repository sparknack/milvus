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
