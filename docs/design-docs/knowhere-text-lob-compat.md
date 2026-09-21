# Sealed TEXT LOB compatibility

## Source contract

`ChunkedSegmentSealedImpl::CreateTextIndexWithSchema` reads TEXT column values as
encoded inline/LOB references, not analyzer input. It resolves those references
through `TextColumnCache`, retaining global row offsets and NULL slots, before
calling Tantivy `AddTextSealed`. VARCHAR input bypasses this decoding. TEXT
references use the complete local or remote LOB base recorded by the runtime;
this change does not invent a storage layout, rewrite manifests or migrate data.
See [storage path contract](../agent_guides/storage/path_contract.md).

The existing sealed LOB test supplied a relative local path although the current
local filesystem is rooted at `/`. Its fixture now uses TestLocalPath plus its
private directory, following the existing complete-key contract. Growing remains
outside this PoC's implementation work.

## Shared decoding and lifetime fix

`TextLobIndexInput.h::VisitTextLobIndexInput` extracts the existing 1024-row
batching into one synchronous input visitor. The production Tantivy builder and
Knowhere compatibility test use the same visitor on real loaded columns. Query
implementations, analyzer semantics and production factory selection are unchanged.
Knowhere consumes decoded text into its existing failure-atomic resident Build,
then attaches the existing text holder and executes native segcore expressions.
This verifies the sealed input boundary; it does not register Knowhere as a
production CreateTextIndex factory option.

Independent review found a lifetime defect in the previous batching: pending
EncodedRef pointers borrowed column callback views across callbacks. A proxy
column may release its previous chunk pin, and the tail flush occurs after the
column's PinAllCells guard is destroyed. The shared visitor now owns pending
encoded bytes in a stable deque until the batch read returns. Decoded text views
remain callback-local. NULL rows never become decoder inputs and retain offsets.

## Validation

The integration fixture writes actual inline and external LOB values, including
72 KiB documents, through LobColumnWriter, persists column references into insert
binlogs and loads them into two sealed segments. It has 2061 rows, crosses both
1024-row batching boundaries, includes empty/NULL rows at the boundaries, and
checks every decoded plaintext and delivered row offset. Tantivy uses the real
CreateTextIndex entry. Knowhere uses the same shared visitor over its real column,
then resident Build/holder attach. TEXT threshold, exact/sloppy phrase, fuzzy,
missing terms and NOT compare direct query results, every segcore filter bit and
Search offsets/distances. The plaintext oracle uses Knowhere built directly from
original strings; therefore it is an independent *input decoding* oracle, not an
independent query algorithm. Tantivy supplies the separate query implementation.

A reusable/synchronously destroyed scratch-string enumerator catches borrowed
reference lifetime regressions. NULL-only batches, reader failure passthrough and
batch cardinality failure are additional visitor contracts. Holder publication
in the existing production builder occurs after decode/build/finalization; this
source trace does not by itself prove a production failure/retry integration test.

## Error boundary

Reader exceptions propagate unchanged through the visitor; it adds no blanket
catch or error classification. Batch cardinality mismatch is an internal contract
violation (AssertInfo), not a user query validation failure. Cache-level short
return guards and their tests are documented with the executed result report.
A guard at the cache's reader interface does not prove detection of short data
silently synthesized inside the upstream storage reader.

Executed results will be linked after the unified build and correctness run.

## Executed results

See the [compatibility report](../../../knowhere-scalar-poc/results/2026-09-21-compat/README.md) for passing correctness runs, scope limits, cast benchmark/perf and unified regression (326 passed, 3 existing skips).
