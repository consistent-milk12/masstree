# Architecture Proposals (`ArchitectureProposals.md`)

> **Status:** Comprehensive Architectural & Subsystem Redesign Proposals for
> `masstree` **Goal:** Massively boost **correctness**, **simplicity**, and
> **performance** by eliminating the structural root causes behind the 64 audit
> findings in `SystemAudit.md` and the post-remediation findings in
> `Findings.md`.

---

## Executive Architectural Diagnosis

A root-cause analysis of all 64 findings in `SystemAudit.md` (`SAFE-01..06`,
`OCC-01..06`, `INS-01..04`, `REM-01..03`, `GC-01..05`, `RNG-01..10`,
`SUF-01..05`, `BATCH-01..06`, `SIM-01..16`, `PERF-01..25`) reveals that **over
80% of the bugs did not come from localized typos — they came from five
structural design choices**:

1. **Massive State-Machine Duplication in Range Scans (`src/tree/range/`, ~7,200 LOC across 14 files):**
   - Forward scans have 3 separate slot loops; reverse scans have 4 separate
     slot loops and 3 separate leaf-retreat functions; batch scans have 2 more
     loops.
   - Fixing a bug in one loop (e.g. `RNG-07` post-read OCC validation, `RNG-08`
     empty rightmost leaf retreat, `RNG-10` cursor state mutation before
     `has_changed`) repeatedly left the other copies broken.
2. **Untyped Sentinel Encoding in `CursorKey` (`UNSHIFT_SENTINEL_LEN = 9`):**
   - Representing the "just ascended from a sublayer (`unshift`)" state by
     setting `cursor_key.len = 9` collides with every real key whose
     current-layer suffix is 1 byte long (`len == 9`), breaking 9-byte keys in
     `compare_cursor_to_slot` and `ReverseScanCtx::process_slot_generic`.
3. **Permanent Layer-Slot Indirection After Sublayer Root Promotion:**
   - When a sublayer's initial root leaf `L0` splits and promotes to an
     `InternodeNode`, the parent leaf's layer slot is never updated — it
     continues pointing to `L0` forever. Every reader, range scanner, remover,
     sublayer GC worker, and tree printer must remember to climb `parent`
     pointers (`maybe_parent_generic`), causing `RNG-04`, `GC-02`, `TOOL-14`,
     `REM-02`, and `PERF-07`.
4. **`UnsafeCell<Vec<u8>>` Interior Mutability in `SuffixBag` + Eager Suffix
   Preallocation:**
   - Forming `&mut Vec<u8>` inside `SuffixBag::try_append_suffix` while
     lock-free readers hold `&[u8]` slices into the same buffer violates Rust's
     Stacked/Tree Borrows aliasing rules (`SAFE-06`), and pre-allocating a
     `15 * suffix_len` bag before checking whether the insert is an update or
     split wastes heap allocations on every suffix update (`PERF-01`,
     `PERF-06`).
5. **Three Parallel Entry/Insert Pipelines (`BatchEntry` vs `OutputEntry` vs
   `BatchValueEntry`, `insert` vs `insert_with_guard` vs `Entry`):**
   - Duplicated entry types and batch strategies (`SIM-08`) caused
     `insert_batch_entries`, `insert_with_guard`, and `Entry`
     (`OccupiedEntry::insert`, `and_modify`) to bypass write-through updates
     (`PERF-03`) and break out of full leaves (`size == 15`) even on in-place
     updates (`PERF-04`).

Below are **six concrete architectural proposals**, ordered by impact-to-effort
ratio, with exact data structures, invariants, and migration paths.

---

## Proposal 1: Unified Parametric Range Scan Engine & Typed `CursorKey` State (`src/tree/range/`)

### 1.1 Problem
`src/tree/range/` is currently ~7,200 lines of code spread across 14 files
containing:
- **3 forward slot loops:** `find_next_generic`,
  `find_next_with_duplicate_check_generic`, `find_next_single_layer`
- **4 reverse slot loops:** `find_prev_generic`/`process_slot_generic`,
  `find_prev_with_duplicate_check_generic`, `find_prev_single_layer`,
  `run_batch_reverse`
- **3 reverse leaf-retreat functions:** `advance_to_prev_leaf_generic`,
  `advance_prev_leaf_single_layer_snapshot`, `advance_prev_leaf`
- **2 batch slot processors:** `process_batch_keyed`, `process_batch_values`
- **Implicit length-9 sentinel (`UNSHIFT_SENTINEL_LEN = 9`):**
  `CursorKey::unshift()` sets `self.len = 9` to mean "ascended from sublayer;
  skip layer slot `== ikey`". Any real 9-byte key also has `current_len() == 9`,
  causing `compare_cursor_to_slot` (`helper.rs:105`) and `reverse_ctx.rs:605` to
  misclassify real 9-byte keys as post-unshift sentinels!

Because cursor state (`ctx.cursor_key` and `ctx.layer_stack`) is mutated
**in-place** while reading a slot *before* checking
`leaf.version().has_changed(cached_version)`, any concurrent modification (or
in-callback mutation in `scan_batch_ref` / `scan_intra_leaf_batch`) corrupts
`ctx.cursor_key` with a stale/reused slot's key before triggering
`ScanState::Retry` (`RNG-10`).

### 1.2 Proposed Design

#### A. Explicit `CursorPositionState` on `CursorKey` (Zero Collisions with 9-Byte Keys)
Replace `UNSHIFT_SENTINEL_LEN = 9` with an explicit `unshift_sentinel: bool`
flag on `CursorKey`:
```rust
pub struct CursorKey {
    buf: [u8; MAX_KEY_LENGTH],
    offset: usize,
    len: usize,
    full_len: usize,
    ikey: u64,
    /// True ONLY when positioned immediately after `unshift()` (ascending from a
    /// sublayer) or `shift_clear_reverse()`. Never true for a normal key of length 9.
    unshift_sentinel: bool,
}
```
- `is_unshift_sentinel(&self) -> bool` is `true` **only** after `unshift()` or
  `shift_clear_reverse()`.
- Every real key of length 9 has `unshift_sentinel == false`, eliminating the
  9-byte key collision in `compare_cursor_to_slot`, `lower_bound_full`,
  `upper_bound_reverse_full`, and `ReverseScanCtx::process_slot_generic`.
- In `lower_bound_full`, when `compare_cursor_to_slot` returns `Ordering::Equal`
  on a layer slot (`keylenx >= LAYER_KEYLENX`), only advance past the layer slot
  (`i + 1`) when `skip_equal && cursor.is_unshift_sentinel()`. When descending
  from an initial `RangeBound::Excluded(b"PREFIX00_A")`
  (`!cursor.is_unshift_sentinel()`), stop at `i` so the scan enters the
  `"PREFIX00"` sublayer (`RNG-04`).

#### B. Commit-After-Validate Slot Snapshot (`SlotSnapshot<P>`)
Enforce a strict two-phase protocol for every slot read across both iterator and
batch scans:
1. **Phase 1 — Speculative Read (No Cursor or Stack Mutation):** Read
   `slot_ikey`, `slot_keylenx`, and either `layer_ptr` (if
   `slot_keylenx >= LAYER_KEYLENX`) or `(suffix_opt, value_opt)` from `leaf`.
2. **Phase 2 — OCC Validation:** Check
   `if leaf.version().has_changed(cached_version) { return Retry; }`. If
   validation fails, `cursor_key` and `layer_stack` are **100% untouched** and
   still hold the last validly emitted key!
3. **Phase 3 — Bound Check & Cursor Commit:** Check the bound (`bound_ikey` /
   `key_in_bound`), commit `(slot_ikey, slot_keylenx, suffix_opt)` into
   `cursor_key`, mark `cursor_emitted`, and emit to the caller/callback.

#### C. Sublayer-Aware Bound Extraction (`extract_ikey_for_cursor`)
In multi-layer trees (`offset > 0`), a bound's 8-byte chunk at `offset` is only
relevant when the sublayer's prefix `cursor_key.full_key()[..offset]` equals
`bound_bytes[..offset]`:
- If `cursor_key.full_key()[..offset] != bound_bytes[..offset]`, the entire
  sublayer is strictly inside the bound, so `extract_ikey_for_cursor` returns
  `None` (zero bound checks needed inside that sublayer!).

#### D. Consolidating the 10 Loops into 1 Parametric Engine
Unify forward and reverse leaf retreat (`advance_to_prev_leaf_generic` handles
all reverse paths; delete `advance_prev_leaf_single_layer_snapshot` and
`advance_prev_leaf`) and cache the climbed sublayer root in
`ScanStackElement::set_root` (`PERF-07`).

### 1.3 Expected Impact
- **Correctness:** Eliminates `RNG-01..10` and makes mid-leaf `cursor_key`
  corruption (`RNG-10`) and 9-byte sentinel collisions impossible by
  construction.
- **Code Size:** Reduces `src/tree/range/` by ~3,500–4,500 LOC.
- **Performance:** Restores zero-allocation, zero-memcpy values-only scans
  (`scan_values` / `scan_values_rev`) on interior slots and sublayers, and
  eliminates redundant sublayer root climbs (`PERF-07`).

---

## Proposal 2: Direct Parent-Slot Update on Sublayer Root Promotion

### 2.1 Problem
When a sublayer's initial root leaf `L0` splits
(`Propagation::create_new_layer_root`), a new `InternodeNode` `R` is allocated
with `L0` and `L1` as its children and `R.mark_root()`. However,
**the parent leaf's layer slot is not updated** — it continues pointing to `L0`
(`L0.parent == R`). Consequently:
- Every point lookup (`get`), range scan (`reach_leaf_for_scan`), `insert`,
  `remove`, sublayer GC (`try_remove_sublayer`), and `print_tree` must climb
  `parent` pointers from `L0` to `R` on every traversal.
- Forgetting to climb caused `RNG-04`, `GC-02`, `TOOL-14`, and `REM-02`.

### 2.2 Proposed Design
Pass the sublayer's origin
`(parent_leaf_ptr: *mut LeafNode15<P>, parent_slot: usize)` down the layer
descent stack (already tracked in `LayerContext` during scans and easily passed
as `Option<ParentLayerSlot>` in `insert_concurrent`):
1. When `create_new_layer_root` promotes a sublayer root leaf `L0` to a new root
   internode `R`:
   - Keep `maybe_parent_generic` as a fallback for any concurrent reader in
     flight.
   - Try-lock `parent_leaf` (or check that
     `parent_leaf.keylenx(parent_slot) >= LAYER_KEYLENX` and
     `parent_leaf.load_layer_raw(parent_slot) == L0`) and atomically `CAS` /
     `Release`-store `R` into `parent_leaf.leaf_values[parent_slot]`.
2. Because `R` and `L0` are both EBR-protected and both have `NodeVersion` at
   offset 0 (`R` has `!is_leaf() && is_root()`, `L0` has
   `is_leaf() && parent == R`), readers that load the old pointer `L0` climb 1
   step to `R`, while all subsequent readers load `R` **directly in 0 hops**!

### 2.3 Expected Impact
- **Performance:** Eliminates $O(\text{sublayer height})$ pointer-chasing climbs
  (`parent` loads + version checks) on every access to a promoted sublayer
  (`PERF-07` and all point lookups/inserts/removes on deep keys).
- **Correctness:** Eliminates the entire class of "forgot to climb from `L0` to
  `R`" bugs.

---

## Proposal 3: Raw-Allocation `SuffixBag` (Zero-Aliasing-UB) & Lazy Suffix Allocation

### 3.1 Problem
1. **Stacked/Tree Borrows Hazard (`SAFE-06`):** `SuffixBag` stores
   `data: UnsafeCell<Vec<u8>>` alongside `data_ptr: AtomicPtr<u8>`. In
   `SuffixBag::try_append_suffix` (`src/suffix.rs:521, 530`), the writer calls
   `self.data_mut()` (`&mut *self.data.get()`) and
   `data.extend_from_slice(suffix)` while concurrent OCC readers hold `&[u8]`
   slices created from `self.data_ptr.load(Acquire)`. Under Rust's Stacked
   Borrows / Tree Borrows memory model, creating `&mut Vec<u8>` asserts unique
   access over the `Vec`'s heap buffer and invalidates concurrent reader shared
   borrows.
2. **Public `&self` Reallocation UAF (`SAFE-04`):** Public methods
   `SuffixBag::reserve(&self)` and `SuffixBag::assign(&self)` reallocate the
   backing `Vec<u8>` through `&self`, allowing safe single-threaded callers to
   hold `let s = bag.get(0).unwrap(); bag.reserve(10000);` and read freed
   memory.
3. **Wasted Pre-Allocations on Updates & Splits (`PERF-01`, `PERF-02`,
   `PERF-06`):**
   - `insert_concurrent` (`src/tree/generic/insert.rs:485`) calls
     `maybe_pre_allocate_suffix` (`15 * suffix.len()` bytes) *before* checking
     if `optimistic_search` is `Found` (an update!), `Layer`, `Conflict`, or a
     full leaf (`size == 15`).
   - `InlineSuffixBag::drain_to_external` (`src/suffix/inline.rs:342`) allocates
     an exact-fit `SuffixBag` with zero slack capacity, forcing the very next
     suffix insert to allocate and copy a second `SuffixBag`.

### 3.2 Proposed Design
1. **Append via Raw Pointer Without `&mut Vec<u8>` (`SAFE-06`):**
   - In `SuffixBag::try_append_suffix`, never call `self.data_mut()` or
     `Vec::extend_from_slice`!
   - Instead, check `new_used <= self.capacity.load(Ordering::Relaxed)`, copy
     bytes directly into the spare capacity via
     `unsafe { std::ptr::copy_nonoverlapping(suffix.as_ptr(), self.data_ptr.load(Ordering::Relaxed).add(offset), len) }`,
     update `self.used.store(new_used, Ordering::Relaxed)`, and publish
     `self.slots[slot].store(meta.bits(), Ordering::Release)`.
   - Only synchronize `Vec::set_len` in `Drop` or `&mut self` methods when the
     bag is exclusively owned!
2. **Require `&mut self` on Reallocating `SuffixBag` Methods (`SAFE-04`):**
   - Make `reserve`, `assign`, `try_assign`, `compact`, `compact_in_place`, and
     `compact_with_permuter` take `&mut self` (all production callers in
     `src/policy/suffix_ops.rs` and `src/suffix/inline.rs` operate on a freshly
     allocated local `mut new_bag: SuffixBag` before `Box::into_raw`
     publication!).
3. **Defer `maybe_pre_allocate_suffix` & Add 2× Drain Slack (`PERF-01`,
   `PERF-02`, `PERF-06`):**
   - In `src/tree/generic/insert.rs:485`, only call `maybe_pre_allocate_suffix`
     when
     `matches!(optimistic_search, InsertSearchResultGeneric::NotFound { .. }) && pre_lock_perm.size() < LeafNode15::<P>::WIDTH`.
   - In `InlineSuffixBag::drain_to_external_init` (`src/suffix/inline.rs:437`),
     allocate `(total + new_suffix.len()) * 2` capacity (clamped to
     `INITIAL_EXTERNAL_CAP`), matching `drain_and_rebuild`.

---

## Proposal 4: Unified `BatchEntry<P>` & Write-Through Value Pipeline (`src/tree/generic/batch.rs`, `entry.rs`)

### 4.1 Problem
1. `src/tree/generic/batch.rs` defines **three** nearly identical entry structs
   (`BatchEntry<P>`, `OutputEntry<P>`, `BatchValueEntry<P>`) and **two** batch
   strategies (`GenericBatch`, `ValueBatch`) (`SIM-08`).
2. `insert_batch_entries_with_guard` (`batch.rs:914-926`) converts
   `Vec<BatchEntry<P>>` into a newly allocated `Vec<OutputEntry<P>>` via
   `P::into_output` before traversal, bypassing `P::can_write_through()`
   (`PERF-03`).
3. `Entry::and_modify`, `Entry::insert_entry`, and `OccupiedEntry::insert`
   (`src/tree/generic/entry.rs`) also bypass `P::can_write_through()`,
   allocating a new `Box<V>` + EBR retirement on every update!
4. `process_leaf_batch` (`batch.rs:1118`) breaks out of the leaf loop whenever
   `perm.size() >= WIDTH` *before* checking if the current entry is an in-place
   update (`Found`) (`PERF-04`).
5. `run_clustered_batch` (`batch.rs:965`) pre-allocates
   `BatchInsertResult::with_capacity(batch.len() / 4)` even on 100%-insert
   workloads (`PERF-24`).

### 4.2 Proposed Design
1. **Single `BatchEntry<P>` + `ValueBatch` Handling Both Write-Through and Boxed
   Updates:**
   - Add `seq: usize` to `BatchEntry<P>` and use `BatchEntry<P>` directly in
     `ValueBatch`, deleting `BatchValueEntry<P>`, `OutputEntry<P>`, and
     `GenericBatch` (~200 LOC removed).
   - Inside `ValueBatch::try_insert_at` (`Found` arm), branch on
     `if P::can_write_through()` (in-place atomic write-through returning
     `ValueRef::Owned`) vs `else` (`P::into_output` + `update_existing_value`
     returning `P::output_to_value_ref`).
2. **Allow Full-Leaf Updates in `process_leaf_batch` (`PERF-04`):**
   - Remove the early `if perm.size() >= LeafNode15::<P>::WIDTH { break; }` at
     `batch.rs:1118`; `try_insert_at` already returns
     `BatchEntryResult::NeedsSplit` on `NotFound` when the leaf is full.
3. **Zero Upfront `old_values` Allocation (`PERF-24`):**
   - Initialize `BatchInsertResult::new()` (`Vec::new()` does not allocate until
     the first update occurs).
4. **Enable Write-Through in `Entry` (`PERF-03`):**
   - In `OccupiedEntry::insert`, `Entry::and_modify`, and `Entry::insert_entry`,
     dispatch to `insert_value_with_guard` when `P::can_write_through()`.

---

## Proposal 5: Coalesce Queue Two-Buffer Discipline & Stack-Allocated `Route` (`src/tree/coalesce.rs`, `src/tree/remove.rs`)

### 5.1 Problem
1. **Single-Call Retry Exhaustion & Starvation (`GC-05`):**
   `Coalesce::try_remove_one` (`src/tree/coalesce.rs:287`) pops from
   `queue.deferred_chains` **first**, while `requeue_chain` (`coalesce.rs:483`)
   pushes failed entries into `queue.deferred_chains`. When `queue.len() > 1` in
   `process_all` (or in `process_batch`), a single locked leaf is immediately
   popped again from `deferred_chains` up to `MAX_REQUEUE_COUNT` (10) times in
   the same maintenance call, permanently abandoning it and starving the
   remaining entries in `queue.chains`!
2. **Ancestor Separator Abandonment on `RetryExhausted` (`GC-03`):**
   `redirect_ikey_bounds_generic` (`src/tree/remove.rs:1013`) returns early on
   `LockedParentResult::RetryExhausted` after `child[0]` was already set to
   `null`, leaving the ancestor separator stale so keys in
   `[old_ikey, new_ikey)` route to `child[0] == null` and spin forever.
3. **Maintenance Flush Order (`PERF-16`) & Per-Remove `Route` Heap Allocation
   (`PERF-17`):**
   - `MassTreeGeneric::maintenance` (`src/tree/generic.rs:131-136`) calls
     `guard.flush(); guard.refresh();` *before*
     `Coalesce::process_all(self, guard)`, so nodes retired by `process_all` are
     not flushed.
   - `remove_concurrent_generic` (`src/tree/remove.rs:558`) pushes to a
     heap-allocated `Route = Vec<u64>` on every sublayer descent, even though 14
     out of 15 removes in a full leaf do not empty the leaf.

### 5.2 Proposed Design
1. **Strict Two-Queue Generation Discipline (`GC-05`, `SIM-04`):**
   - `drain_deferred()` moves both `deferred_chains -> chains` and
     `deferred_sublayers -> sublayers` at the start of `process_all` and
     `process_batch`.
   - `try_remove_one` **only** pops from `sublayers` and `chains` (never from
     `deferred_*`). Every failed entry is pushed to `deferred_*` and guaranteed
     to wait until the next maintenance call.
2. **Loop on `RetryExhausted` in `redirect_ikey_bounds_generic` (`GC-03`):**
   - Never abandon an in-flight ancestor separator redirect after `child[0]` has
     been nulled; loop with `StdHint::spin_loop()` on
     `LockedParentResult::RetryExhausted` until the parent lock is acquired or
     `NoParent` (root) is reached.
3. **Stack-Buffer `Route` During Remove (`PERF-17`) & Post-Coalesce Flush
   (`PERF-16`):**
   - Track the sublayer route in a stack array
     `route_buf: [u64; MAX_KEY_LENGTH / IKEY_SIZE]` (`[u64; 32]`) +
     `route_len: usize` during `remove_concurrent_generic`, and only allocate
     `Vec<u64>` inside `schedule_coalesce_cold` (`#[cold]`) when
     `new_perm.size() == 0` and `route_len > 0`.
   - Call `guard.flush(); guard.refresh();` *after* `Coalesce::process_all` in
     `maintenance`.

---

## Proposal 6: Fused `NodeVersion` Lock-with-Intent (`lock_for_insert`) & Unified Leaf Search (`SIM-02`, `PERF-05`)

### 6.1 Problem
1. Every leaf insert/remove currently performs two separate version mutations on
   entry: `lock_bounded()` (`CAS(v, v | LOCK_BIT)`) followed immediately by
   `lock.mark_insert()` (`store(v | LOCK_BIT | INSERTING_BIT, Release)` +
   `fence(Release)`).
2. `src/tree/generic/search.rs` duplicates insert search across
   `search_for_insert_generic` (3-way unrolled) and
   `search_for_insert_single_layer` (1-at-a-time loop), and retains dead
   `BINARY_SEARCH_THRESHOLD = 16` code (`SIM-02`, `PERF-05`).

### 6.2 Proposed Design
1. Add `NodeVersion::lock_for_insert(&self) -> LockGuard<'_>` which CASes
   `v -> v | LOCK_BIT | INSERTING_BIT` in a single atomic RMW (with
   `fence(Ordering::Release)`), setting `dirty_mask = INSERTING_BIT` on the
   returned `LockGuard` so `Drop` increments the insert version counter.
   Subsequent `lock.mark_insert()` calls see `INSERTING_BIT` already set and
   become a no-op!
2. Collapse `search_for_insert_generic` and `search_for_insert_single_layer`
   into a single `search_for_insert<const SINGLE_LAYER: bool>` using the 3-way
   unrolled ikey loop, and delete the dead `BINARY_SEARCH_THRESHOLD = 16`
   functions.

---

## Summary Matrix of Proposals

| Proposal | Target Subsystem | Bugs / Audit IDs Eliminated | LOC Delta | Performance Impact |
|---|---|---|---|---|
| **1. Unified Range Scan Engine & Typed `CursorKey`** | `src/tree/range/` | `RNG-01..10`, `SIM-09`, `SIM-11`, `PERF-07`, `PERF-08` | **-3,500 LOC** | **2–4× faster** `scan_values` / sublayer scans |
| **2. Direct Parent-Slot Update on Layer Promotion** | `src/tree/generic/split.rs`, `layer.rs` | `RNG-04`, `GC-02`, `TOOL-14`, `PERF-07` | +45 LOC | **Eliminates $O(h)$ parent climbs** on deep keys |
| **3. Raw-Pointer `SuffixBag` & Lazy Suffix Alloc** | `src/suffix.rs`, `src/suffix/inline.rs`, `insert.rs` | `SAFE-04`, `SAFE-06`, `PERF-01`, `PERF-02`, `PERF-06` | -40 LOC | **Eliminates wasted heap allocs** on suffix updates & first drain |
| **4. Unified `BatchEntry` & Write-Through Pipeline** | `src/tree/generic/batch.rs`, `entry.rs` | `SIM-08`, `PERF-03`, `PERF-04`, `PERF-24` | **-220 LOC** | **Zero-alloc updates** in `Entry`/`insert_batch_entries` & full-leaf batches |
| **5. Two-Queue Coalesce & Stack `Route`** | `src/tree/coalesce.rs`, `src/tree/remove.rs` | `GC-03`, `GC-05`, `SIM-04`, `PERF-16`, `PERF-17` | -30 LOC | **Zero heap alloc** on multi-layer `remove`; immediate EBR flush |
| **6. Fused `lock_for_insert` & Unified Search** | `src/nodeversion.rs`, `src/tree/generic/search.rs` | `OCC-01..03`, `SIM-02`, `PERF-05` | **-150 LOC** | **1 fewer atomic store** per insert/remove; faster single-layer search |
