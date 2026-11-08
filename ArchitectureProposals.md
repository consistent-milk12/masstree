# Architecture Proposals (`ArchitectureProposals.md`)

> **Status:** Initial progressive draft — major architectural and subsystem redesign proposals to simultaneously boost **correctness**, **simplicity**, and **performance** in `masstree`.

---

## Overview

The recent 39-commit remediation wave demonstrated a recurring theme across the codebase: **most high-severity concurrency, range-scan, and reclamation bugs came from duplicated state machines and split representations of the same logical concept.**

Specifically:
1. **Four separate B-link / layer-root descent engines** (`reach_leaf_concurrent_generic` + `advance_to_key_by_bound_generic` in `generic.rs`, `optimistic_reads.rs`, `remove.rs`, and `range/traversal.rs` + `helper.rs`), each handling `ROOT_BIT` promotion climbs, `DELETED_BIT`, "too far right" (`prev != null && ikey < ikey_bound`), and marked `next` pointers slightly differently.
2. **Promoted Layer Slot Indirection**: When a sublayer's root leaf splits and is promoted to an `InternodeNode`, the parent leaf's layer slot is *not* updated; it continues to point to the original **leftmost leaf** (`L0`), forcing every reader, writer, remover, range scanner, printer, teardown walker, and sublayer GC pass to climb `parent` pointers to discover the true sublayer root (`RNG-04`, `GC-02`, `TOOL-14`, `PERF-07`).
3. **Two-Step Lock + Dirty-Bit Publication (`NodeVersion`)**: Locking a node (`LOCK_BIT`) and marking it dirty (`INSERTING_BIT` / `SPLITTING_BIT`) are separate atomic operations with separate fences, leading to `OCC-01`, `OCC-02`, and `OCC-03` (and an extra `SeqCst` fence per write).
4. **Duplicated Single-Layer vs Multi-Layer Search & Scan State Machines**: Separate code paths for single-layer and multi-layer reads/inserts/scans (`SIM-02`, `SIM-08`, `SIM-09`, `RNG-07`) that drift in OCC validation and bound handling.
5. **Suffix Storage Allocation & Indirection Overhead**: Preallocating `15 * suffix_len` on every suffix insert before checking if the key is an update (`PERF-01`), zero-slack external bag drain rebuilds (`PERF-02`), and duplicated `EmbeddedSuffix` vs `SidecarSuffix` orchestration (`SIM-07`).

Below are five progressive, high-impact architectural proposals ordered by ROI (performance + correctness per line of code changed).

---

## Proposal 1: Fused `NodeVersion` Lock-with-Intent & Zero-Fence x86 OCC Protocol

### Problem
Currently, every leaf write performs **three** atomic operations on `NodeVersion` plus a permutation store:
1. `lock_bounded()`: `CAS(v, v | LOCK_BIT)` (`Acquire`)
2. `mark_insert()`: `store(v | LOCK_BIT | INSERTING_BIT, Release)` + `fence(SeqCst)`
3. `set_permutation(perm)`: `store(perm, Release)`
4. `drop(lock)`: `store((v + VINSERT_LOWBIT) & UNLOCK_MASK, Release)`

Because `OCC-01` made `mark_insert()` unconditional on slot assignments and `OCC-03` added `fence(Ordering::SeqCst)`, every write executes a full serializing `mfence`/`lock` barrier after acquiring the spinlock.

### Proposed Design
1. **Replace `fence(Ordering::SeqCst)` with `fence(Ordering::Release)` in `mark_insert` / `mark_split` / `mark_deleted`:**
   - In the C11/Rust memory model, `store(Relaxed/Release)` followed by `fence(Ordering::Release)` is a true store-store barrier: every write before the fence is ordered before every write after the fence.
   - On x86_64 (TSO), `fence(Ordering::Release)` compiles to a **zero-instruction compiler barrier**. On aarch64, it compiles to `dmb ish`.
2. **Introduce `lock_for_insert()` (Fused Lock + Dirty Bit CAS):**
   - On the insert/remove path, over 95% of lock acquisitions immediately set `INSERTING_BIT`.
   - Acquiring the lock via a single CAS from `v` to `v | LOCK_BIT | INSERTING_BIT` with `Ordering::AcqRel` (or `Acquire` + `fence(Release)`) atomically acquires the lock *and* publishes `INSERTING_BIT` in **one** atomic RMW instruction instead of two writes + a fence!
   - If the lock is aborted without mutating the node (e.g., `validate_post_lock` fails), a helper `lock.Abandon()` clears `LOCK_BIT | INSERTING_BIT` without incrementing `vinsert`.
3. **Reader `stable()` Single-Load Fast Path (`PERF-18`):**
   - Replace `load(Relaxed)` + `fence(Acquire)` in `NodeVersion::stable()` with `load(Ordering::Acquire)` on the clean fast path (matching `try_stable()`), eliminating the standalone `dmb ishld` fence on aarch64.

### Impact
- **Performance:** Removes 1 atomic store and 1 `SeqCst` hardware fence (`~20–30 cycles` on x86_64) from every insert, update, and remove critical section; reduces aarch64 reader descent barrier cost.
- **Correctness:** Eliminates the window where a node is locked for mutation without `INSERTING_BIT` set (`OCC-01`/`OCC-02`).

---

## Proposal 2: Direct Layer-Root Slot Publication (Eliminating Parent-Climb Indirection)

### Problem
When a sublayer leaf `L0` splits and is promoted to a new `InternodeNode` `I_root` (`promote_layer_root` in `src/tree/split/root_creation.rs`), the parent leaf's layer slot is **not** updated: it continues to point at `L0`. Every operation that enters a sublayer must therefore call `maybe_parent_generic` / `climb_layer_root` to walk `L0 -> parent -> ... -> I_root`.
- Forgetting this climb caused `RNG-04` (scans skipping keys in promoted layers), `GC-02` (promoted sublayers leaking forever), `TOOL-14` (`print_tree` omitting promoted layers), and `PERF-07`.
- Even when present, climbing from `L0` up to `I_root` touches `L0`'s cache line (and every intermediate internode up to the sublayer root) on *every* descent into that sublayer, polluting L1/L2 cache with the cold leftmost leaf `L0` even when searching for keys at the far right of the sublayer!

### Proposed Design
 Store the **parent leaf pointer and slot index** (or climb lazily and opportunistically CAS-update the parent leaf's layer slot, OR pass `(parent_leaf_ptr, layer_slot)` down when descending a layer) so `promote_layer_root` can update the parent leaf's layer pointer directly to `I_root` (or readers/writers can opportunistically swing the layer slot pointer to `I_root` via CAS)!
- **Opportunistic + Eager Slot Swing:**
  - When `promote_layer_root` creates `I_root`, `I_root` has `ROOT_BIT` set and `parent == null`.
  - If the parent leaf's layer slot is updated to point directly to `I_root` (using an atomic `Release` store or CAS on `values.ptrs[slot]`/`tags[slot]` while `keylenx == LAYER_KEYLENX`), readers descending into the sublayer immediately land on `I_root` in **zero climbs**!
  - Because `L0.parent` still points to `I_root`, any concurrent reader that loaded the old slot pointer `L0` *still* reaches `I_root` via the existing `maybe_parent_generic` climb — making the slot pointer update 100% lock-free and backward-compatible with concurrent readers!

### Impact
- **Performance:** Turns $O(\text{sublayer height})$ pointer-chasing climbs through cold leftmost leaves into an $O(1)$ direct jump to the true sublayer root `I_root`, eliminating 1–4 cache misses per sublayer descent.
- **Correctness:** Makes the parent slot pointer match the true sublayer root in steady state.

---

## Proposal 3: Unified B-Link & Layer Traversal Engine (`SublayerCursor`)

### Problem
Currently, four subsystems implement their own descent and B-link recovery loops:
1. Point reads (`optimistic_reads.rs` / `get_guarded.rs`)
2. Point writes (`generic.rs` / `insert.rs`)
3. Point deletes (`remove.rs`)
4. Range scans (`range/traversal.rs` / `forward_ctx.rs` / `reverse_ctx.rs`)

Each one has had bugs where one of the five B-link invariants was missing (`OCC-04`, `INS-03`, `REM-02`, `RNG-04`, `RNG-08`, `RNG-09`).

### Proposed Design
Extract a single, zero-cost generic traversal primitive in `src/tree/generic/traversal.rs`:
- `reach_stable_leaf_for_key<P>(start_root, target_ikey, guard) -> ReachResult<P>`
- Enforces all 5 invariants in one place:
  1. Climbs `parent` pointers to the current layer root (re-climbing on retry if the root itself was promoted).
  2. Restarts from root if an internode child is null or `is_deleted()`.
  3. On reaching a leaf, checks `is_deleted()` / `deleted_layer()`.
  4. Checks "too far right" (`!prev.is_null() && target_ikey < ikey_bound()`) and restarts from root.
  5. Advances along unmarked `next` while `target_ikey >= next.ikey_bound()`, waiting on marked splits, with a bounded hop count.

### Impact
- **Simplicity:** Deletes ~250 lines of duplicated, subtly diverging traversal code across `generic.rs`, `remove.rs`, and `range/traversal.rs`.
- **Correctness:** Guarantees that `get`, `insert`, `remove`, and `scan` obey the exact same B-link and layer-root invariants.

---

## Proposal 4: Lazy Key Materialization & Zero-Allocation Leaf Snapshots in Range Scans

### Problem
1. To fix `RNG-05` and `RNG-10`, commit `ee3364a` changed `process_batch_values` to call `build_slot_key` and `D::key_in_bound(bound, key)` on **every slot of every leaf**, destroying the key-materialization-free fast path of `scan_values` / `scan_values_rev`.
2. In `ForwardScanCtx` and `ReverseScanCtx`, single-layer and multi-layer scans have ~500 lines of near-duplicate logic (`SIM-09`), and every slot visit performs multiple redundant atomic loads (`is_value_empty_relaxed` followed by `load_value`, `PERF-08`).

### Proposed Design
1. **Boundary-Only Full-Key Check in `process_batch_values`:**
   - For any slot where `bound_ikey` is `None` (`RangeBound::Unbounded`) OR `slot_ikey != b_ikey`:
     - If `slot_ikey.cmp(&b_ikey) == D::exceeded_ordering()`, return `D::bound_exceeded()`.
     - Otherwise (`slot_ikey` is strictly inside the bound!), **skip `build_slot_key` completely**!
   - Only call `build_slot_key` + `D::key_in_bound(bound, key)` when `bound_ikey == Some(slot_ikey)` (at most 1–10 slots at the exact boundary of the entire scan!).
   - For `RNG-10` (mid-leaf `version_changed()` retry after a callback mutates the tree): only reconstruct the last emitted slot's key into `cursor_key` **on the cold `has_changed` retry path** (using the previous slot index `prev_slot` before returning `D::version_changed()`, or keeping `(last_slot_ikey, last_slot_keylenx)` in registers and materializing `cursor_key` only if a version change is detected!).
   - Wait: what if the callback mutated `prev_slot` itself? If we save `(slot_ikey, slot_keylenx)` (16 bytes in registers) or only update `cursor_key.assign_store_ikey(slot_ikey)` + `assign_store_length` (and copy suffix only if `slot_keylenx == KSUF_KEYLENX`), then for all inline keys (`<= 8` bytes per layer), recording the cursor is just two register stores (`cursor_key.assign_store_ikey(slot_ikey)` and `cursor_key.mark_key_complete()`), without calling `to_be_bytes` + buffer slice comparisons on unbounded scans! Even better, if `has_changed` is checked before `visitor(output)` and after `visitor(output)`, we know the exact moment the callback mutated the leaf!
2. **Eliminate Redundant `is_value_empty_relaxed` Precheck (`PERF-08`):**
   - `leaf.load_value(slot)` already returns `None` when the slot is empty. Removing the redundant `is_value_empty_relaxed` atomic load before `load_value` saves 1 atomic load per visited slot across all batch scans.

### Impact
- **Performance:** Restores the 2–3× throughput advantage of `scan_values` / `scan_values_rev` while preserving 100% of the `RNG-05` and `RNG-10` correctness guarantees.

---

## Proposal 5: Late Suffix Allocation & Slack-Preserving Drain (`PERF-01`, `PERF-02`, `SIM-07`)

### Problem
1. **`PERF-01` (Allocation on every suffix update):** `insert_concurrent` calls `maybe_pre_allocate_suffix` *before* locking the leaf and *before* checking whether the insert is an update (`Found`), a conflict (`Conflict`), or a split (`NeedsSplit`). Every update to an existing long key allocates and immediately drops a `WIDTH * suffix_len` heap buffer!
2. **`PERF-02` (Zero-slack external bag rebuilds):** When `InlineSuffixBag` drains to an external `SuffixBag` and merges existing external suffixes, it sizes the new bag to the exact byte count, leaving `0` bytes of free capacity so the very next suffix insert must reallocate and copy the entire bag again.
3. **`SIM-07` (`EmbeddedSuffix` vs `SidecarSuffix` duplication):** Both types duplicate the same inline-then-external lookup and drain orchestration.

### Proposed Design
1. **Move Suffix Preallocation to the `NotFound` Slot-Assignment Path:**
   - Remove `maybe_pre_allocate_suffix` from the pre-lock loop in `insert_concurrent`.
   - In `insert_new_value` (when `NotFound` has a valid `slot`), check if the suffix fits in the inline bag or existing external bag capacity first; only if it does not fit, allocate the new `SuffixBagCell` with $2\times$ slack capacity (`max(used * 2 + new_suffix.len(), 256)`).
2. **Unify `SuffixStore` via `SuffixCore` (`SIM-07`):**
   - Factor the `(InlineSuffixBag, AtomicPtr<SuffixBagCell>)` pair into a single `SuffixCore` struct embedded directly in `EmbeddedSuffix` and behind the pointer in `SidecarSuffix`.

### Impact
- **Performance:** Eliminates **100% of heap allocations on long-key updates** (`Found` path) and reduces external suffix bag allocations under suffix churn from $O(N)$ reallocations to amortized $O(1)$.
