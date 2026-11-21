# Async load FIFO benchmark baseline

This branch is a benchmark control for `async-load-sequence` at
`b42112d91cdc19a9d9e1c6e8be8d921e4299b339`.

The async loader sources, call sites, configuration, and tests modified by the
sequence feature are restored byte-for-byte from its base,
`20746425a1e56f7b12709dc64a4b871cb8be338d`.
The new scheduling microbenchmark is removed because it requires the new API.

The original executor uses Folly's two-priority FIFO queue. Admission uses the
original per-priority FIFO lists. Async loading remains enabled. This compares
the full scheduling implementation, including admission ordering; it does not
isolate CPU queue ordering from admission ordering. The sequence image keeps
its default scheduling weights of 32:8:1; weights are not an experiment axis.

The updated milvus-common dependency and the build fixes in Makefile and
storage/CMakeLists.txt are retained from the sequence image to minimize unrelated
differences. The common sequence allocator is therefore present but unused by
the restored loader. All other application sources remain at b42112d91.

Validation before image build: exact restoration against the base was checked,
remaining production references to the removed sequence API were searched, and
git diff --check passed. Compilation and runtime validation require the remote
build and UAT experiment; no performance benefit is claimed here.
