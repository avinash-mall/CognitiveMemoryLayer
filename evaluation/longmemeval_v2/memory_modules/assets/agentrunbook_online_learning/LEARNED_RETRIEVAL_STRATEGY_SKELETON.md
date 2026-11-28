# Learned Retrieval Strategy

This file contains online retrieval hints learned from previous memory queries
across task types in this run. Treat every note as retrieval guidance, not as a
final answer. A note is only a lead to inspect evidence; it never replaces exact
verification against the current question.

Use exactly the two retrieval sections below. Every row must include one of the
four evidence statuses: `directly_supported`, `contradicts_premise`,
`near_match_only`, or `insufficient`.

The row's evidence status describes how the previous completed task's cited
evidence fit that previous task. It is provenance for the old note, not a claim
that the note directly supports a future question.

Keep this file as a compact working set. Most queries should not add a row.
Prefer no edit, merge, or pruning over appending weak notes. Usually add at most
one strong row per completed query. Multiple rows are allowed only when they are
independent, crucial, reusable lessons that cannot be merged safely. Do not
create one row per cited span.
Use `directly_supported` only when the previous evidence was very high
confidence. Many useful retrieval observations should stay only in the query
trace and should not be written here.

## Past Queries

Use this hot-path section only for reusable prior-query evidence locations that
future retrieval should inspect first: `directly_supported` positive evidence or
exact-scope `contradicts_premise` closed-set absence. Keep entries short,
scoped, and guarded. Do not store answer text or option letters as reusable
lessons. Do not put `near_match_only` or `insufficient` rows here.

| Looking for | Evidence status | Evidence found | Applicability and guard | Fast path |
|---|---|---|---|---|

## Strategies

Use this section for reusable search tactics, shortcuts, exactness guards, and
gotchas that make future memory search faster or more exact. Strategies should
improve retrieval process, not memorize final answers. Tie each row to the
status of the evidence that produced it. `near_match_only` and `insufficient`
lessons belong here only when they prevent a concrete repeated mistake; they
should tell future retrieval where to continue searching or what false transfer
to avoid, not provide a reusable answer shortcut. A `near_match_only` row is a
reference-only hint for a similar workflow: use it only when helpful for
navigation or contrast, then verify exact current-target evidence.

| When | Evidence status | Try first | Guard |
|---|---|---|---|
