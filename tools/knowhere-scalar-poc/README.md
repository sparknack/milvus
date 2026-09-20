# External Knowhere PoC tests

PoC test sources, benchmark scripts, datasets, results and HTML reports are maintained
in the independent sibling repository `../knowhere-scalar-poc`, outside Milvus.
This checkout uses `/home/ubuntu/knowhere-scalar-poc` locally; no remote is configured.

Enable its optional C++ test target with:

```sh
cmake -S internal/core -B cmake_build -DMILVUS_SCALAR_POC_DIR=/absolute/path/to/knowhere-scalar-poc
cmake --build cmake_build --target scalar_poc_tests -j8
```

See the external repository's README for corpus preparation, full benchmark runs,
sanitisers and report generation. The Milvus implementation remains under
`internal/core/src/index/`; design documents remain under `docs/design-docs/`.
PoC test artifacts are excluded from each implementation commit. The original
history is retained on a local backup branch; the external repository records
the old-to-new commit mapping.
