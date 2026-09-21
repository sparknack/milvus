# Async loading for legacy scalar and vector indexes

## Scope

Scalar V3 keeps its packed `IndexEntryReader` / `IndexEntryWriter` path.
Legacy scalar V1/V2 and vector artifacts retain their existing FileSource /
FileSink formats. The pinned `use_async_load` setting selects the transport for
an entire cache cell, including metadata inspection and resource estimation.
This change does not add an asynchronous writer or change the persisted format.
BSON/JSON stats and the legacy TextMatch translator are outside this rollout.

## Read path

`SealedIndexTranslator` opens the logical legacy directory with
`V1RemoteSource::OpenAsync`, including asynchronous slice-metadata loading.
HYBRID's selector is read asynchronously before choosing the concrete loader.
`LoaderEntry::open_async` runs each family's shared decoding implementation;
`Open` selects the same implementation with synchronous source operations.
There is no intermediate V3 file or preload-all adapter.

`LegacyIndexLoader` validates each immutable physical object's envelope. Raw
payloads stream in bounded ranges. Parquet and encrypted envelopes retain one
complete decoding unit because their existing decoders require it. Each issued
range/unit acquires global admission asynchronously and holds the lease through
read, decoding and consumption. FileSource assembles physical slices in their
existing order into logical entries, memory targets or positioned file targets.

Remote reads and admission waits suspend. Local size/read/write calls run on
LocalFileIOPool. The family coroutine also runs there so local staging, mmap,
native initialization and failure cleanup retain their existing synchronous
semantics without blocking the remote read executor. When the local pool is
disabled, the configured async executor remains the existing fallback.
The cache's synchronous interface waits once at the coroutine boundary.

## Vector backends

Memory vector loading asynchronously fills BinarySet entries or the existing
combined mmap file. Embedding-list metadata/raw files remain separate. Validity
is restored before native deserialization; all-null and empty embedding-list
artifacts retain their metadata-only handling.

Disk vector loading probes `LoadIndexWithStream` before preparing engine files.
A native stream backend prepares only validity/empty-list sidecars; neither the
resource inspector nor FileSource eagerly inspects its engine objects. Other
backends stream engine files into their existing local directory. Native
Knowhere deserialization and its internal remote reads remain synchronous;
this change does not make Knowhere's internals coroutine-based.

## Lifetime and failure

Input buffers, targets and admission leases outlive every issued operation.
Cancellation stops new work but drains issued reads/writes before freeing their
storage. The factory merges the operation token with coroutine cancellation.
The loader checks cancellation again before returning a reader.

File output uses same-directory staging and the existing multi-file publication
rollback. Cancellation is checked inside the queued publication task. Successful
file preparation is still owned by the loading generation; initialization
failure destroys native readers before their backing directories.

Unaligned concatenation explicitly requests BUFFERED positioned writes, with
no padding between legacy payloads. Other writes retain the existing priority
policy, including HIGH using BUFFERED. Error carriers are preserved across
executor hops; corrupt envelopes produce DataFormatBroken and admission
cancellation produces FollyCancel. No Go error mapping or retry policy changes
are part of this implementation.

## Resources and validation

Legacy async estimates inspect decoded envelope sizes and count read/decode
scratch independently of refreshable admission limits. Family-specific resident
state includes bitmap expansion, rebuilt Marisa CSR and retained null-offset
sidecars. Native disk-stream engine files remain covered by Knowhere's existing
resource estimate. Request-owned buffers are not assigned to shared scratch
reservations.

Regression tests cover raw/Parquet/encrypted envelopes, exact slice assembly,
real legacy scalar/vector loading, mmap, NULL and empty-list state, typed I/O
and corruption errors, admission/read cancellation and local executor placement.
Local validation on 2026-09-21: `index_tests` passed all 12,696 tests from
52 suites, including 23 new legacy async tests and 64 FileWriter tests.
Changed C++ files passed clang-format 15; the segcore error-boundary guard and
`git diff --check` passed. The build used at most 16 outer jobs and one job per
nested dependency builder.

The tests exercise injected native asynchronous reads over real serialized
legacy objects. They do not establish real-S3 behavior, complete DISKANN native
deserialization, or end-to-end Go retry behavior. The disk test checks backend
selection and resource inspection; it is not a full disk-index load test.
