# Post-Remediation Deep Assessment & Findings (`Findings.md`)

> **Status:** Complete Deep Audit across all 64 canonical findings in `SystemAudit.md`, `ARCHITECTURE.md`, and the remediation commits (`93e65eb..da74a75`). All identified correctness, soundness, performance, and tooling issues have been resolved and verified.
> **Companion Document:** `ArchitectureProposals.md` (progressive architectural redesign proposals for major correctness, simplicity, and performance gains).

---

## 1. Executive Summary

### 1.1 Assessment of the Initial Remediation Wave (`93e65eb..fb8424e`)
Our four-subsystem deep audit evaluated every single one of the 64 canonical findings in `SystemAudit.md` (`SAFE-01..06`, `OCC-01..06`, `INS-01..04`, `REM-01..03`, `GC-01..05`, `ALLOC-01..03`, `PRIM-01..02`, `RNG-01..10`, `SUF-01..05`, `BATCH-01..06`, `TOOL-01..14`, `PERF-TOOL-01..04`, `SIM-01..16`, `PERF-01..25`) and every claim in `ARCHITECTURE.md`.

While the initial remediation wave (`93e65eb..fb8424e`) fixed the majority of the 64 findings, our deep audit uncovered **12 residual or newly introduced correctness, soundness, performance, and tooling issues**, all of which have now been fixed and committed in `9ffacd5..da74a75`:

| # | Area / Audit IDs | Issue Uncovered in Initial Remediation | Resolution Commit(s) |
|---|---|---|---|
| **1** | `INS-01`, `INS-02` (`src/leaf15.rs`) | `calculate_split_point` chose `split_pos` using only the existing 15 keys, and `split_and_insert` routed `insert_pos == split_pos` to the right leaf (`insert_goes_right = insert_pos >= split_pos`). When the inserted key shared `old[split_pos - 1].ikey`, `split_ikey` became equal to the left leaf's ikey, stranding the entire left equal-ikey run (`equal_run2.rs`). | `9ffacd5`, `8c4021c` |
| **2** | `INS-04` (`src/leaf15.rs`, `src/tree/generic.rs`, `insert.rs`, `layer.rs`, `batch.rs`) | Left branch of `split_and_insert`, `assign_slot_generic`, `insert_into_empty_leaf`, and `create_layer_concurrent_generic` still invoked `InlineBits::to_bits()` *after* mutating leaf state or allocating twig/sublayer leaves. | `9147069` |
| **3** | `OCC-03` (`src/nodeversion.rs`) | `mark_insert`, `mark_split`, and `mark_deleted` used `fence(Ordering::SeqCst)` (emitting a full serializing barrier on x86_64 on every write) instead of a `Release` store-store fence (`fence(Ordering::Release)`). | `cfa1188` |
| **4** | `PRIM-02` (`src/permuter.rs`) | `Permuter<WIDTH>::insert_from_back_immutable` lacked `& Self::WIDTH_MASK` for `WIDTH < 15`. | `6cfe353` |
| **5** | `SAFE-04` (`src/leaf15.rs`, `src/leaf_trait.rs`, `src/internode.rs`) | Ownership-bearing and raw-pointer mutators (`store_value*`, `update_value_in_place*`, `take_value*`, `store_layer`) were safe `pub fn`s, and `InternodeNode::split_into` did not assert `insert_pos <= WIDTH`. | `a1bc057`, `e0bd906` |
| **6** | `RNG-01..04` (`src/tree/range/cursor_key.rs`, `helper.rs`, `reverse_ctx.rs`) | `UNSHIFT_SENTINEL_LEN = 9` collided with all real keys of current-layer length 9 (`9`-byte keys at layer 0, `17`-byte keys at layer 1, etc.), causing forward scans to skip 9-byte keys and reverse scans to emit out-of-range/duplicate 9-byte keys; `lower_bound_full` skipped layer slots on `RangeBound::Excluded(>8B)`. | `e7e42f4` |
| **7** | `RNG-05`, `RNG-10` (`src/tree/range/batch_common.rs`, `range_bound.rs`, `forward_ctx.rs`, `reverse_ctx.rs`) | `process_batch_values` materialized full keys on every interior slot; `process_batch_keyed`/`process_batch_values` overwrote `ctx.cursor_key` *before* checking `has_changed()` (skipping keys `4..=15` in `batch_callback.rs`); `extract_ikey_at` ignored sublayer prefix divergence; batch retries did not track `cursor_emitted`. | `e3eed9f`, `0d45a5f` |
| **8** | `TOOL-14` (`src/tree/generic/print.rs`) | `print_leaf_slots` formatted `P::Output` (`ValuePtr(0x...)`) instead of `P::output_to_value_ref` (`P::Value: Debug`), breaking `print_promoted.rs`, and validated OCC *before* reading the slot table. | `f3e957a` |
| **9** | `GC-03`, `GC-05` (`src/tree/remove.rs`, `src/tree/coalesce.rs`) | `redirect_ikey_bounds_generic` abandoned ancestor separator updates on `RetryExhausted` after nulling `child[0]`; `try_remove_one` popped `deferred_chains` first, exhausting a locked leaf's 10 retries in one call and starving the queue. | `68c4ea9` |
| **10** | `REM-01..03`, `RNG-09`, `OCC-02` (`src/tree/remove.rs`, `traversal.rs`, `optimistic_reads.rs`, `get_guarded.rs`) | `remove_concurrent_generic` did not climb promoted sublayer roots, checked too-far-right outside `'forward`, charged `retry_count` on normal layer descents, lacked a null check in `is_sublayer_valid`, and ran redundant post-lock leaf searches; `get_with_guard_multi_layer` and `get_impl_multi_layer` did not retry when a value slot became empty or a sublayer was deleted while `ver.is_dirty()`. | `7339d70`, `e6c8bd9`, `d34a5fd` |
| **11** | `SAFE-06` (`src/suffix.rs`, `src/policy/suffix_ops.rs`, `src/suffix/compact.rs`, `inline.rs`, `sidecar.rs`) | `SuffixBag` exposed safe `&self` reallocating/compacting methods (`reserve`, `assign`, `try_assign`, `compact*`) that allowed safe single-threaded UAF, and `try_append_suffix` created `&mut Vec<u8>` while concurrent readers held `&[u8]` slices. | `0c1a71e` |
| **12** | `PERF-03`, `PERF-04`, `PERF-06`, `PERF-16`, `PERF-24`, `TOOL-04` (`src/tree/generic.rs`, `insert.rs`, `batch.rs`, `entry.rs`, `benches/memory_*.rs`) | `insert_batch_entries` and `Entry` bypassed `P::can_write_through()`; `insert_batch_into_locked_leaf_inner` broke out of full leaves (`size == 15`) before checking for in-place updates; `BatchInsertResult` pre-allocated `batch.len() / 4` upfront; `maybe_pre_allocate_suffix` ran before checking `NotFound`; `maintenance` flushed before `Coalesce::process_all`; `memory_dhat`/`memory_total` failed Clippy without `mimalloc`. | `4960f8d`, `da74a75` |

---

## 2. Full Audit Matrix Across All 64 `SystemAudit.md` Findings

| Category | Finding IDs | Final Status | Notes |
|---|---|---|---|
| **Soundness & Memory Safety** | `SAFE-01` .. `SAFE-06` | **ALL FIXED** | Guard-scoped `ValueRef<'g, V>`; `ValuePtr::as_ref` & leaf/internode ownership mutators `unsafe`; `is_atomic_word` gate; `SuffixBag` `&mut self` for reallocating APIs + raw-pointer append in `try_append_suffix`. |
| **OCC & Concurrency** | `OCC-01` .. `OCC-06` | **ALL FIXED** | Unconditional `mark_insert()`; `LOCK_BIT \| INSERTING_BIT` in `has_changed`; `fence(Ordering::Release)` store-store barrier; `child_version.is_deleted()` restart; dirty retry on empty slot / deleted sublayer. |
| **Insert, Split & Layer** | `INS-01` .. `INS-04` | **ALL FIXED** | Post-insert strict ikey split boundary & left routing on equal-ikey boundary; rightmost deleted leaf restart; `PreparedValue` pre-encoding across all insert/split/layer/batch paths. |
| **Remove** | `REM-01` .. `REM-03` | **ALL FIXED** | Unchanged-leaf fast path + under-lock full-key fallback; B-link advance & inside-loop too-far-right check; retry budget charged only on contention edges. |
| **Coalesce & GC** | `GC-01` .. `GC-05` | **ALL FIXED** | `ikey_bound`-only chain entries; whole-sublayer top-down validated GC; unlink before retire + loop on `RetryExhausted` in both cascade and `redirect_ikey_bounds_generic`; live unary levels preserved; both deferred queues drained once per call. |
| **Allocator & Reclamation** | `ALLOC-01` .. `ALLOC-03` | **ALL FIXED** | `try_with` TLS pool fallback; `catch_unwind` in `reclaim_box_value`; documented no-unwind `TreeAllocator` contract. |
| **Primitives** | `PRIM-01` .. `PRIM-02` | **ALL FIXED** | `ShardedCounter::load` `.max(0)` clamp; `Permuter<WIDTH>::WIDTH_MASK` on all shifts including `insert_from_back_immutable`. |
| **Range Scans** | `RNG-01` .. `RNG-10` | **ALL FIXED** | Explicit `unshift_sentinel: bool` on `CursorKey`; layer descent on `Excluded` bounds; sublayer-aware `CursorBound`; two-phase validate-before-commit batch scan protocol; interior fast path for values-only scans; fused `DoubleEndedIterator`. |
| **Suffix Storage** | `SUF-01` .. `SUF-05` | **ALL FIXED** | Drained slot cleared in `InlineSuffixBag`; zero-length suffixes skipped in compaction; unconditional `assert!` in public suffix APIs; `assign_init` contract documented. |
| **Batch & Entry API** | `BATCH-01` .. `BATCH-06` | **ALL FIXED** | Private `BatchEntry` fields; write-through enabled in both `insert_batch_with_guard` and `insert_batch_entries`; pre-mutation key validation & `Drop` cleanup; sequence-stable duplicate sort; library-level `#[global_allocator]` removed. |
| **Tooling, Tests & Docs** | `TOOL-01` .. `TOOL-14`, `PERF-TOOL-01..04` | **ALL FIXED** | `print_tree` gated & formats values via `P::output_to_value_ref` with post-slot OCC validation; strict Clippy clean with and without `mimalloc`; deterministic proptests; non-vacuous stress assertions; `ARCHITECTURE.md` & `SystemAudit.md` synchronized. |
| **Simplification & Perf** | `SIM-01..16`, `PERF-01..25` | **KEY WINS APPLIED; MAJOR REWRITES IN `ArchitectureProposals.md`** | Applied `SIM-01`, `SIM-04`, `SIM-05`, `SIM-14`, `PERF-03`, `PERF-04`, `PERF-06`, `PERF-16`, `PERF-24`; remaining multi-thousand-line structural consolidations (`SIM-02`, `SIM-08`, `SIM-09`) are specified in `ArchitectureProposals.md`. |
