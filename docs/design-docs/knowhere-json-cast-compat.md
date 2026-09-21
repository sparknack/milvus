# Resident Knowhere JSON cast compatibility

Scope: sealed, resident JSON path scalar index; Tantivy remains the analyzer and
reference backend. No production routing, Growing, mmap or durable format change.

## Reference audit

Milvus exposes exactly one non-default cast function: `STRING_TO_DOUBLE`, only
with scalar `DOUBLE`. The API whitelist and type restriction are in
`internal/util/indexparamcheck/inverted_checker.go`. `ARRAY_DOUBLE` and JSON flat
are not additional supported cast-function combinations.

`common/JsonCastFunction.cpp` implements this with `std::stod`, rather than a
strict JSON-number parser. Leading whitespace, exponent and hexadecimal forms,
and trailing nonnumeric suffixes can succeed. Empty/nonnumeric strings and
out-of-range/underflow conversions fail. Numeric JSON values pass through;
booleans and containers are not converted. `JsonIndexBuilder.cpp` supplies this
same extraction to both adapters. A failed conversion is typed-invalid but can
still satisfy EXISTS; missing paths, JSON null, SQL NULL, and recursively empty
containers do not satisfy EXISTS.

Tantivy's cached source revision `96f3335`, `common/src/lib.rs::f64_to_u64`,
normalizes signed zero and encodes positive IEEE bits by flipping the sign bit,
negative bits by inversion. Its term/range queries use this ordering, including
signed NaNs and their payloads. Thus `STRING_TO_DOUBLE("nan")` cannot simply be
marked invalid or removed: the row remains typed-valid, positive NaN sorts above
positive infinity, and negative NaN sorts below negative infinity. This is
Tantivy index compatibility, not IEEE arithmetic comparison semantics.

## Adapter change

The ordinary Knowhere scalar core continues to reject NaN. The JSON path adapter
stores only NaN postings in an optional adaptive `uint64_t` core using Tantivy's
sortable bits. Normal doubles keep the existing core. Membership and range
queries merge the side postings; NOT IN uses the complete typed validity bitmap.
EXISTS remains independent of conversion validity. Build publishes the ordinary
and side cores transactionally after allocation. Indices without NaNs allocate
no side core and use the original range path. Both streams remain compressed,
flattened postings rather than resident per-term vectors.

## Verification and measurement

`KnowhereJsonCastTest.cpp` provides:

- Explicit fixtures independent of the shared conversion implementation,
  including conversion successes/failures, SQL NULL with retained JSON payload,
  wrong shapes, recursively empty containers, and signed nonfinite values.
- Multiple FieldData batches, escaped JSON pointers, direct validity/EXISTS
  checks, segcore full bitmap and vector Search parity, and actual backend query
  counters with expression result caching disabled.
- An independently normalized raw oracle for finite-bound predicates. Nonfinite
  values use finite extreme sentinels only in this oracle: all tested finite
  bounds are far inside those extremes. Direct backend comparisons separately
  check NaN and infinity query bounds, inclusion flags, equality and NOT IN.
- An opt-in 120,000-row benchmark through the same resident build/load/Search
  path. Raw oracle timing is diagnostic, not an uncast production scan claim.

Validation results and performance evidence are recorded by the parent task
once the shared binary is built and tests are run. This document does not claim
verification from compilation alone.

## Executed results

See the [compatibility report](../../../knowhere-scalar-poc/results/2026-09-21-compat/README.md) for passing correctness runs, scope limits, cast benchmark/perf and unified regression (326 passed, 3 existing skips).
