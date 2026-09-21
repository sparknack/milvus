# Resident Knowhere: multiple STRUCT ARRAY children

This compatibility extension keeps the existing sealed resident index representation. It adds a real segcore fixture with two scalar children (`items[number]`, `items[label]`), an aligned vector-array child (`items[embedding]`), and an ordinary row vector. It does not add recursive ARRAY indexing, Growing, element-nullable values, or a new join operator.

## Native implementation evidence

`InvertedIndexTantivy.cpp::build_index_for_array_nested` uses one monotonically increasing element document ID across all input FieldData blocks. NULL rows are skipped before reading values; empty arrays contribute no documents. Every duplicate element remains a separate document. The string overload follows the same rule. Independent child indexes therefore align when their valid row lengths agree, regardless of file/block boundaries.

`Schema.cpp::GetFirstArrayFieldInStruct`, `common/ArrayOffsets.h`, and `exec/operator/ElementFilterBitsNode.cpp` establish the other half of the contract: segcore owns row-to-element offsets and selects the shared struct domain before evaluating the element expression. Logical AND between child predicates happens in that element domain. The operator selects offset evaluation for very sparse parent filters and directly returns an all-filtered bitmap for zero parent hits. The index must not turn element matches into row IDs inside this path.

By contrast, ordinary ARRAY membership in `JsonContainsExpr.cpp` projects each child/term's element matches into row space. Two ARRAY_CONTAINS conjuncts may consequently match different elements in the same row. This is intentionally different from `element_filter(items, number == 7 AND label == "red")`.

[Lucene ToParentBlockJoinQuery](https://github.com/apache/lucene/blob/main/lucene/join/src/java/org/apache/lucene/search/join/ToParentBlockJoinQuery.java) similarly keeps child matching separate from projection into the parent document space. Its explicit child/parent document blocks are not copied here: Milvus already owns the offsets, so the existing native join/projection remains authoritative.

No codec or cursor change is needed for this distinction. The adaptive posting docID is a global element ordinal; the shared segcore expression chooses when to intersect and when to project.

## Added verification

`internal/core/src/index/KnowhereNestedArrayTest.cpp` builds Tantivy, Knowhere and raw sealed segments over 1,027 rows. Array lengths vary from zero to seven. It verifies every row boundary and total element count for all three children. Index inputs use different block boundaries per child, retain nonempty poison payload on NULL rows, and include repeated matching elements.

The six element expressions cover AND, OR, NOT, range plus inequality, multi-term IN, and an absent term. Each is tested with broad, one-row and zero-row parent predicates. Expression batches are 37 elements, deliberately crossing row and bitmap-word boundaries. Each execution compares every element bitmap bit with an independent row/element oracle and compares full vector Search offsets, element indices and distances across all three backends. Query counters prove that both scalar indexes are used during broad evaluation; sparse evaluation retains the native fallback.

An explicit discriminator row has `number=[7,0]`, `label=["blue","red"]`: row-level contains AND must match while same-element AND must not. A second row has two `(7,"red")` elements and must preserve both identities. The row-domain counterpart also compares the full bitmap and ordinary vector Search.

Compilation and execution are performed by the parent task; adding the source alone is not a passing-test claim. There is no new performance claim: the fixture verifies previously untested composition of existing nested operations. Inputs with mismatched sibling lengths are outside the supported schema contract and are not silently repaired by the index.

## Executed results

See the [compatibility report](../../../knowhere-scalar-poc/results/2026-09-21-compat/README.md) for passing correctness runs, scope limits, cast benchmark/perf and unified regression (326 passed, 3 existing skips).
