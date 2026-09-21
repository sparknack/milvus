# Knowhere resident PoC space compaction

This revision follows `c4a091894f`. It changes the resident representation, not
query semantics or the persisted format. See the [paired results](../../../knowhere-scalar-poc/results/2026-09-21-space/README.md).

## Sources of excess space and changes

- Scalar strings previously retained a `std::string` object per unique term,
  plus a separately built FST that scalar queries no longer consulted. Remove
  that unused FST. Store sorted strings as one byte pool plus 64-bit end offsets;
  binary search and pattern iteration use views into the pool. TEXT and NGRAM
  still retain their actively used FSTs.
- Resident posting metadata previously used 24 bytes/term: 64-bit offset and
  length, 32-bit DF and version. Retain only 64-bit offsets and 32-bit DFs in
  separate flat arrays (12 bytes/term). Derive length from adjacent offsets and
  encode the existing constant version when serializing. DF remains necessary
  for candidate selection and intersection ordering, independently of scoring.
- Appending the required SIMD padding to an exactly sized loaded byte vector
  caused capacity growth. Reserve the final padded size before assigning bytes;
  retain padding for safe decoder reads. Compact core/position arrays once at
  publication, rather than retaining construction growth slack.
- JSON flat numeric range costing previously duplicated a full 8-byte DF prefix
  sum per term. Keep one checkpoint per 64 terms and a total. Each boundary
  sums at most 63 existing DF entries. The selected postings and result semantics
  are unchanged; paired range timings check the small extra boundary work.

## Lifetimes, persistence and failure behavior

Builders and loaders construct a staged core before publication. String views
refer to its owned pool and stay valid until rebuild/destruction. `Term()` keeps
its existing value-returning API. Empty-only dictionaries use a valid empty
string pointer. Numeric ordering continues to use `KnowhereTermOrder`, including
signed/payload NaNs and signed zero.

The versioned snapshot bytes remain unchanged. Loading still validates all old
24-byte wire metadata, contiguous spans, docIDs and checksums before publishing;
wire metadata is temporary, not retained. This does not promise reduced peak
load memory. Old-format snapshots remain readable without recompressing postings.

## Verification

Tests cover pooled string lifetime after input destruction, empty/Unicode/prefix
cases, rebuilds and metadata allocation; DF checkpoints at partial/exact block
boundaries; existing malformed snapshot, NULL/validity, NaN and segcore contracts.
A separate glibc allocation probe compares identical before/after core inputs and
checks byte-identical snapshots. Its heap delta is not RSS.

Space measurements report Knowhere's allocated payload/capacity. Tantivy's
existing accounting reports index file bytes, so the two columns must not be
called equivalent resident memory. Query benchmarks use the same sealed segcore
Build → snapshot reload → attach → expression / Search flow, complete bitmaps
and Search results checked against Tantivy. Growing/mmap/production factory
integration remain outside this resident PoC.

## Follow-up: compressed layouts

The subsequent [compact layout revision](knowhere-compact-layout-poc.md) also
compresses resident and persisted metadata/keys/position directories and removes
short-list skip metadata. The unchanged-wire statements above describe only the
initial capacity-compaction revision.
