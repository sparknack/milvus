# Knowhere sealed index persistence PoC

## Scope and source contracts

This adds a versioned local snapshot and the existing ScalarIndex BinarySet
Serialize/Load interface for Knowhere's sealed adapters. It is a resident PoC,
not registration in the production index factory, remote Upload/LoadUnified,
mmap, Growing, or replacement of Tantivy analyzers.

Existing `InvertedIndexTantivy::Serialize` serializes NULL-offset metadata;
`Upload` separately finishes the writer and sends the Tantivy directory files
through DiskFileManager. Its storage Load caches those files and recreates the
reader. Tantivy owns segment files and Milvus owns their transport. `TextMatchIndex` also carries a separate runtime holder.
Knowhere must preserve its own immutable term/posting/position state before that
same segcore attachment; serializing documents and rerunning Build would not test
index persistence. This PoC does not copy Tantivy's file layout or introduce a
production storage namespace. Local test paths are complete private temporary
files as required by [storage paths](../agent_guides/storage/path_contract.md).

## Format

`KnowherePoCIO.h` writes explicit little-endian fields. Each object has an 8-byte
magic, version, type, codec, payload length, payload and FNV-1a checksum. Native
C++ struct layouts/pointers/capacities are not serialized. FNV detects accidental
corruption; it is not authentication. Unknown versions/types/codecs and trailing
bytes are rejected. This v1 format currently requires a little-endian host for
its compressed streams; it is not a cross-endian production specification.

- Core: sorted typed terms, flat posting metadata, NULL offsets, original
  compressed adaptive or streamvbyte bytes. SIMD padding is recreated on load.
- ARRAY: core plus row/element-domain marker; no new parent offset ownership.
- Text: analyzer configuration binding, ordinal core, ordered term strings,
  flat frequency and position streams and their directories. No original text.
- NGRAM: min/max gram configuration, ordinal core, ordered gram strings and
  row-size heuristic metadata. Full text remains in the segment for exact Phase2.
- JSON path/ARRAY/NGRAM: configured path/schema/cast, base index, typed validity
  and EXISTS; DOUBLE cast includes signed/payload-sensitive NaN side postings.
- JSON flat: configured root, row validity, canonical path table, scalar/element
  cores for BOOL/I64/U64/F64/string and five shape/validity masks per path. Numeric
  DF prefixes are reconstructed from validated term metadata.

TEXT/NGRAM rebuild their memory FST from persisted ordered terms; scalar
strings use an owned contiguous pool after the [space update](knowhere-space-poc.md).
Numeric DF prefixes now use 64-term checkpoints. These resident changes preserve
the snapshot format. Loading never analyzes
source documents or rebuilds compressed postings. Direct loading/mapping of FST
bytecode needs a separately validated format; this PoC does not treat a checksum
as proof that arbitrary FST bytecode is safe. A single snapshot file is adequate
for this resident lifecycle test; future multi-file/mmap design remains open.

## Loading and publication

Every load constructs staged state, bounds counts by available bytes before
allocation, validates streams, reconstructs derived metadata, then publishes.
An exception leaves the previously loaded index usable. JSON flat executors keep
shared ownership of their previous snapshot when their owner loads new state.

The core loader verifies sorted typed terms, posting extents/DF/block directories,
strictly increasing docIDs in range, NULL exclusion and block maxima. The bounded
codec reader checks every encoding tag/control/patch/byte count before invoking
the existing pointer-based decoder on padded memory. Positions validate expected
DF, document blocks, total frequencies, byte/position boundaries and cumulative
per-document positions. JSON wrappers additionally validate masks against actual
postings and schema/configuration; nested ARRAY refuses row-domain snapshots.

Malformed snapshots originate DataFormatBroken at the parser/validation sites;
there is no catch that stringifies/reclassifies them. File I/O in the test helper
is separate. No new cgo error projection, retry behavior or remote I/O classification
is claimed. OOM/cancellation/remote storage failures have not been fault-injected
for a production factory integration.

## Verification

Dedicated tests mutate lengths, type/codec/configuration, directories, NULL and
JSON masks, term order, frequencies and positions. Structural mutations also
recompute checksums: a checksum failure alone is not the safety test. Failure
checks compare the old serialized state and query results. BinarySet tests call
through ScalarIndex pointers, including derived JSON virtual dispatch.

`POC_SNAPSHOT_ROUNDTRIP=1` extends the existing fixtures:

1. Build the actual index; write its snapshot.
2. Destroy the serialization buffer and original index.
3. Create a fresh index with the expected schema/analyzer configuration.
4. Read the file, load, destroy the read buffer.
5. Attach to segcore and compare full filter bitmaps and vector Search with
   Tantivy and the existing independent oracles.

This covers scalar, ordinary/nested ARRAY, JSON path/cast/ARRAY/flat, VARCHAR and
JSON NGRAM including Phase2, TEXT/PHRASE/FUZZY, analyzer configurations and LOB
input. Test diagnostics report serialize/write/read/load times separately; the
old benchmark `build_ms` includes roundtrip setup when the flag is enabled and
must not be interpreted as pure Build time. Query timings exclude that setup.

Executed results and limitations are recorded in the separate persistence report.

## Issues exposed by the end-to-end test

The TEXT LOB roundtrip hit a real stack overflow while enumerating a 72 KiB token
through upstream recursive FST traversal. Enumeration, prefix, LIKE and fuzzy
walking now use a heap-backed iterative sibling stack and one reusable word
buffer, without a token length cap. Tests include 72/96 KiB terms, branching and
jump tables, mapping, and text snapshot reload with phrase/fuzzy queries.

Independent review also found the four outer adapter envelopes did not reject
unknown codec values when their checksum was recomputed. They now validate codec
zero explicitly; dedicated tests verify failure atomicity. DecodeChecked's padded
scratch storage is fixed-size stack memory (derived maximum codec block size),
avoiding per-block heap allocation during validation. NULL lookup and full token
validation remain work proportional to input size; no constant-time load claim.
