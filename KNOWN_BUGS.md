# Known Bugs and Issues

This document tracks known bugs, fixed issues, and current limitations in the MassTree implementation.

Last updated: 2025-12-23

---

## Table of Contents

1. [Active Issues](#active-issues)
2. [Fixed Issues](#fixed-issues)
3. [Architectural Limitations](#architectural-limitations)
4. [Testing Notes](#testing-notes)

---

## Active Issues

### 1. Intermittent Stack Overflow in High-Volume Tests

**Status:** Under investigation
**Affected tests:** `large_volume_20k_keys_8_threads` (intermittent)
**Symptom:** Thread stack overflow with signal SIGABRT

```text
thread '<unknown>' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

**Analysis:**

The recursive `propagate_internode_split_concurrent()` function holds parent locks across recursive calls. While tree depth for 20k keys with WIDTH=15 is only ~4 levels (well within normal stack limits), the failure may occur under specific conditions:

- High contention causing deep retry chains
- System resource exhaustion
- Nextest process isolation edge cases

The thread name `<unknown>` suggests this may be a system-level issue rather than a code bug. The failure is not consistently reproducible.

**Workaround:** The issue appears to be transient. If encountered, re-run the test. Consider increasing thread stack size for stress tests if failures persist.

**Files:** `src/tree/locked.rs` (propagate_internode_split_concurrent)

---

### 2. Small WIDTH with Layers Has Known Issues

**Status:** Known limitation
**Affected configuration:** WIDTH=4 combined with multi-layer keys

The combination of small node width (WIDTH=4) and trie layering (keys longer than 8 bytes with shared prefixes) has known bugs that are not fully resolved.

**Workaround:** Use default WIDTH=15 for multi-layer workloads. Tests requiring layers explicitly use WIDTH=15.

**Reference:** `src/tree/locked.rs:2022` - Test comment documents this limitation

---

### 3. Node Retirement Not Implemented (Memory Growth)

**Status:** Planned for Phase 3 completion
**Impact:** Memory usage grows monotonically under split-heavy workloads

When leaf or internode splits occur, the old node structures are not retired via seize. Nodes remain allocated until the entire tree is dropped. This can cause unbounded memory growth in long-running services with many updates.

**Workaround:** For long-running workloads, periodically recreate the tree or ensure workload has bounded key space (updates rather than new inserts).

**Files:** `src/alloc.rs` (SeizeAllocator)

---

## Fixed Issues

### Concurrency Race Conditions

#### CAS Insert Slot Stealing

**Fixed in:** Permutation freezing implementation
**Root cause:** CAS insert used value-based CAS instead of NULL-claim semantics, allowing two threads to race for the same slot.

**Fix:** Enforce NULL-claim semantics in `cas_slot_value()`. Only the first thread to claim an empty slot succeeds.

```rust
// Now enforces: only claim slots that are NULL
leaf.cas_slot_value(slot, std::ptr::null_mut(), new_ptr)
```

**File:** `src/tree/cas_insert.rs:216`

---

#### Split Clobbers CAS-Published Permutation

**Fixed in:** Permutation freezing implementation
**Root cause:** Split's `set_permutation()` could overwrite a permutation that a CAS insert had just published, orphaning entries.

**Fix:** Splits now freeze the permutation before modifying, preventing concurrent CAS publishes.

**Files:** `src/leaf.rs:1670`, `src/freeze.rs`

---

#### Parent Propagation Race Condition

**Fixed in:** Revalidation and retry loop implementation
**Root cause:** Grandparent could be accessed without holding its lock, leading to "parent must be in grandparent" panics when concurrent splits moved nodes.

**Fix:** Added `locked_parent_internode()` with revalidation loop and bounded retries.

**Files:** `src/tree/locked.rs:1116-1203`

---

### Split and Traversal Issues

#### Null-Next Pointer Marking

**Fixed in:** Link split implementation
**Root cause:** `link_split()` could attempt to mark a null pointer, causing undefined behavior.

**Fix:** Added null check before marking.

```rust
// FIXED: Never mark a null pointer
if old_next.is_null() { ... }
```

**File:** `src/leaf/link.rs:71`

---

#### Suffix Migration During Splits

**Fixed in:** Split implementation
**Root cause:** Suffix data was not migrated when keys with `keylenx == KSUF_KEYLENX` were moved to the new leaf during splits, causing lookup failures for long keys.

**Fix:** Explicitly migrate suffix data during `split_into()` and `split_all_to_right()`.

**Files:** `src/leaf.rs:1658`, `src/leaf.rs:1708`, `src/leaf.rs:1772`, `src/leaf.rs:1808`

---

#### B-Link Navigation in Read Path

**Fixed in:** Optimistic read implementation
**Root cause:** Read path did not properly follow B-links when a split occurred after version check but before key search.

**Fix:** Added split detection and B-link following in `search_leaf_concurrent()`.

**Files:** `src/tree/optimistic.rs:742`, `src/tree/optimistic.rs:752`

---

#### Writer Membership Revalidation

**Fixed in:** Locked insert implementation
**Root cause:** Writer could lock a stale leaf after a split moved the target key to a sibling.

**Fix:** Added post-lock membership check comparing key against next leaf's bound.

**File:** `src/tree/locked.rs:723`

---

### Memory and Pointer Issues

#### Arc Memory Leak in Layer Creation

**Fixed in:** Layer creation implementation
**Root cause:** When converting a value slot to a layer pointer, the existing Arc was not properly dropped, causing memory leaks.

**Fix:** Explicit drop of Arc before slot overwrite.

**File:** `src/tree/layer.rs`

---

#### Root Aliasing Violations

**Fixed in:** Root pointer redesign
**Root cause:** `RootNode` enum with pointer aliasing caused Miri violations.

**Fix:** Removed `RootNode` enum; tree uses only `root_ptr: AtomicPtr<u8>` with explicit type casting.

**File:** `src/tree.rs`

---

### Ordering and Comparison Issues

#### Layer Key Ordering

**Fixed in:** Layer creation implementation
**Root cause:** Key comparison used ikey-only comparison, incorrectly handling prefix-of-other cases.

**Fix:** Use length-aware comparison that correctly handles exhausted keys.

**File:** `src/tree/layer.rs:60`, `src/tree/layer.rs:112`

---

## Architectural Limitations

### 1. WIDTH <= 15 Is Structural

The `Permuter` uses 4-bit slots packed into a u64. This fundamentally limits WIDTH to at most 15.

**File:** `src/permuter.rs`

---

### 2. Slot-0 / ikey_bound Invariant

Leaf slot 0 stores `ikey_bound()` for B-link navigation. Insert logic must avoid reusing slot 0 for a different ikey unless specific conditions are met.

**File:** `src/leaf.rs`

---

### 3. Recursive Split Propagation

Split propagation is recursive, not iterative. Each level of the tree adds a stack frame. While bounded by tree height (typically 4-5 for millions of keys), this could theoretically cause stack issues for extremely deep trees.

**File:** `src/tree/locked.rs`

---

### 4. No Range Scans or Deletion

Phase 4 features (range iteration, key deletion) are not implemented.

---

## Testing Notes

### Orphan Detection

The codebase includes diagnostic tools for detecting orphaned slots (slots with non-NULL values not referenced by the permutation). These can indicate bugs in split or CAS insert logic.

**File:** `src/leaf/orphan.rs`

Usage:

```rust
if leaf.has_orphaned_slots() {
    let orphans = leaf.find_orphaned_slots();
    // Handle orphans...
}
```

---

### Test Hooks for Concurrency Testing

Test hooks allow injecting barriers at specific points to force thread interleavings:

- `set_before_cas_publish_hook()` - Before CAS permutation publish
- `set_after_freeze_hook()` - After freeze succeeds in split

**File:** `src/tree/test_hooks.rs`

---

### Debug Counters

Lightweight counters track key operations for diagnosis:

- `CAS_INSERT_SUCCESS_COUNT`
- `CAS_INSERT_RETRY_COUNT`
- `LOCKED_INSERT_COUNT`
- `SPLIT_COUNT`
- `SEARCH_NOT_FOUND_COUNT`

**File:** `src/tree/optimistic.rs`

---

## Reporting New Bugs

When reporting bugs:

1. Include the exact error message or panic backtrace
2. Note the key pattern (length, distribution)
3. Note thread count and any relevant configuration
4. If possible, provide a minimal reproducing test case
5. Check tracing logs if available (`--features tracing`)
