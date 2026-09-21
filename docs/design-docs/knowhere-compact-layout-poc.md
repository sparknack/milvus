# Compact resident and snapshot layouts for the Knowhere PoC

This follows the [Tantivy layout audit](../../../knowhere-scalar-poc/results/2026-09-21-layout/README.md)
and resident capacity fix at `7d0e6e09fd`. It changes real representations, including
snapshots, while preserving existing query and NULL semantics. No lossy numeric
quantization is used: FLOAT/DOUBLE bits, signed zero and NaN payloads retain the
existing `KnowhereTermOrder` behavior.

## Packed integer columns

`KnowherePackedVector` groups 128 integers per block. Each block stores its minimum
as an 8-byte reference, a one-byte bit width, and packed unsigned differences.
Block start offsets support direct random access to one value; loading does not
expand the column. The read path uses up to two machine-word fragments and an
8-byte private padding region. Padding and the reconstructible block-offset table
are not serialized. Values can use all 64 bits, including extrema.

The core's posting offsets and DFs use separate packed columns. Posting ends are
inferred from the next start, and the last end from the blob size. The repeated
per-term format version is removed. Numeric term values use the same lossless
representation over their original bit patterns. Publication releases all build
vectors. DF remains necessary for planning intersections and is not a scoring
payload.

The block size is 128 rather than Tantivy's 256: this is a local PoC choice that
bounds width inflation across varied distributions. It is not a compatibility
requirement and can be evaluated separately after the current measurements.

## String dictionaries

The scalar dictionary is sorted and divided into groups of 32 keys. A group stores
its common prefix and suffix once, followed by each key's middle. End offsets are
packed integer columns. Lookup/range binary search reconstructs only the probed
key; sequential pattern enumeration reuses query-local string scratch. Visitor
views are now valid only during the callback, not for the entire core lifetime.
All callers consume the views synchronously; no mutable shared cache is added.

This is block prefix/suffix compression, not a new scalar FST. It preserves direct
ordinal access and broad pattern iteration. It cannot match a minimal FST's
extreme compression for all regular key sets; measured sizes must make that
tradeoff visible. Full duplicate key vectors are not retained after publication.

TEXT and NGRAM continue using their active FST for resident queries. Their
snapshots now use the compressed string representation for saved keys; reload
validates ordering, ordinals and NUL restrictions before rebuilding the FST.
Temporary reconstruction buffers are discarded after load. No claim is made
that peak build/load memory equals final resident memory.

## Short postings and cursor behavior

New Adaptive postings with at most 256 documents retain their DF prefix and
adaptive integer payload, but omit block maxima and inter-block directories.
Long postings keep their existing directories and lazy block seeking. Short-list
seek decodes the only block and explicitly checks lower_bound against the end;
it cannot depend on a directory maximum to guarantee an in-block result.
The public MaxDoc accessor still returns the actual maximum when requested.

This benefits all short lists, not only singletons. Singleton inlining is not
part of this change. DocIDs and adaptive integer payload semantics are unchanged.

## Positions

The four term position-directory fields, two document-block fields and position
block offsets also use packed columns. Frequencies and position deltas keep the
existing adaptive encoding. A reader reconstructs term metadata once, and reads
doc-block metadata only when its cached frequency block changes. This avoids
repeated packed accesses for documents in the same block.

No positions, term frequencies, multi-slot/slop semantics or phrase validation
are discarded to obtain the savings.

## Versioning, validation and legacy reads

The existing checksum envelope remains. Supported layout discriminators are:

| Component | Old layout | New layout |
|---|---|---|
| Scalar core | codec 0 SVB / 1 legacy adaptive, full keys and 24-byte metadata | bit 1: packed keys/metadata; bit 0: adaptive; bit 2: directory-free short lists. Accepted new values: 2, 3, 7 |
| Positions | codec 1, fixed directories | codec 3, packed directories |
| TEXT/NGRAM keys | codec 0, full strings | codec 1, compressed key blocks |

Reserved/unknown combinations are rejected. These are private PoC formats, not
production protocols. New readers accept old snapshots; old readers reject new
layout discriminators. Legacy core loads retain the original posting stream
and its legacy codec flag rather than silently recompressing it. New builds use
the compact short-list format. Loading legacy positions compacts directories.

All loads stage state before publication. Packed columns validate block widths,
bounds, exact byte consumption and arithmetic overflow before exposing random
access. String blocks validate spans and shared-prefix/suffix lengths. Core
validation still checks ordering, DF agreement, contiguous posting extents,
non-NULL/in-domain docIDs, and long-block maxima. Positions still verify ranges,
TF sums, document position overflow and the actual bounded codec payloads.
Failures originate `DataFormatBroken` through the existing PoC checks; no string
exception conversion or new error-code translation is introduced.

## Scope and evidence

JSON's additional array-offset indexes, scalar/element separation, masks and
NaN side handling are preserved. Production factory, remote persistence, mmap
and Growing remain outside this work.

See [measured results](../../../knowhere-scalar-poc/results/2026-09-21-compact/README.md)
for old/new file and resident space, same-run Tantivy timings, full segcore parity,
sanitizers, and legacy snapshot allocation/query probes. Do not mix the historical
Tantivy wrapper's file-plus-NULL accounting with process RSS.
