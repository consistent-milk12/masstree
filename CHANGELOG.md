# Changelog

## 0.9.4

Write-through soundness fixes and deferred-allocation batch insert.

### Soundness Fixes

- `CAN_WRITE_THROUGH` now requires `!needs_drop::<V>()`, preventing write-through for types with `Drop` impls. Without this, atomic load would create a bitwise copy of an ownership-bearing value, risking double-free with concurrent updates.
- Range scan value reads for write-through types now use `output_as_ref_sound` / `resolve_value_ref` (atomic read into scratch buffer) instead of plain pointer dereference, closing a data race with concurrent write-through stores.

### Write-Through Batch Insert

- New `BatchValueEntry<P>` defers `P::Value -> P::Output` conversion: updates use `value_ref()` (borrow, no allocation), inserts use `take_value()` (move, single allocation). For write-through types, batch updates are fully allocation-free.
- `insert_batch` and `insert_batch_with_guard` dispatch to the value path when `CAN_WRITE_THROUGH` is true, skipping `clone_value_from_output` entirely.

### Cleanup

- Batch suffix pre-allocation now calls shared `maybe_pre_allocate_suffix` from insert.rs instead of inlining the logic (13 lines removed)
- Removed `P::Value: Clone` bound from `insert_concurrent_value` impl block (write-through reads the old value atomically, no clone needed)
- Empty leaf insert in `insert_concurrent_value` simplified: returns `Ok(None)` directly instead of mapping over a guaranteed-`None` result

## 0.9.3

Write-through updates, remove path robustness, and performance improvements.

### Write-Through Update Optimization

- New `CAN_WRITE_THROUGH` trait const on `LeafPolicy`: for `BoxPolicy<V>` where V is a naturally-aligned primitive <= 8 bytes, updates modify the value in place through the Box pointer, bypassing Box allocation and EBR retirement (~45ns savings per update)
- Deferred allocation insert path (`insert_concurrent_value`): takes `P::Value` directly, only allocates `P::Output` on the new-key path. Hot update path is allocation-free.
- Alignment guard: `CAN_WRITE_THROUGH` requires `size_of::<V>().is_power_of_two() && align_of::<V>() >= size_of::<V>()`, excluding packed structs that lack hardware atomicity guarantees
- Atomic value access: write-through read/write now uses size-dispatched `AtomicU8/U16/U32/U64` load/store instead of `ptr::read`/`ptr::copy_nonoverlapping`, closing the formal data race UB with concurrent OCC readers. `clone_value_from_output` also uses atomic read for write-through types.

### New Public APIs

- `MassTree15::with_batch_size` / `MassTree15Inline::with_batch_size`: create trees with custom retirement batch size
- `insert_value_with_guard`: returns `Option<P::Value>` directly, avoiding `ValuePtr` unwrapping for callers that only need the value
- Default retirement batch size increased to 256 (from seize default of 32), reducing `sys_membarrier` syscall frequency under write-heavy workloads

### Remove Path

- Single-layer search fast path: keys <= 8 bytes skip layer/suffix branches, using Relaxed loads (mirrors insert path)
- Forward-loop re-scan after lock contention: `Retry` re-reads version and permutation on the same leaf instead of a full top-down internode traversal
- Value prefetch in remove search: `prefetch_value` after ikey match and `prefetch_suffix` before suffix comparison, hiding cache misses during `finish_remove`
- Pre-lock prefetch: `prefetch_for_search` called before `lock_bounded` in verify path, warming cache lines during the spin loop
- Cold path extraction: empty-leaf coalesce scheduling moved to `#[cold] #[inline(never)]` helper, keeping `finish_remove` compact for the common case
- Added Release ordering comment for permutation store in `finish_remove`: Release is required because concurrent insert threads read the permutation without the lock

### Bug Fixes

- `merge_old_external_perm`: faulty early-exit optimization silently dropped suffix data when rebuilding the external suffix bag. Under concurrent insert/remove/coalesce workloads, this caused a panic in `create_layer_concurrent_generic`: "conflict slot N should have a suffix"
- Fixed 5 flaky concurrent remove tests: Release/Acquire on done-flags, unconditional read batch before spin loop guarantees progress regardless of scheduling
- Removed redundant clone in `OccupiedEntry::insert`

### Test Coverage

- New remove+get edge case tests: multi-layer keys, multi-reader multi-writer, suffix keys, remove-reinsert cycles, single-leaf boundary, split boundary, empty key, guarded API, churn stress

### Relaxed ordering optimization, value prefetch pipeline, and insert fast path

- Relaxed ordering for OCC read paths: keylenx, value emptiness, and value loads downgraded from Acquire to Relaxed (permutation Acquire + `has_changed()` already provides synchronization)
- Relaxed ordering for under-lock writes: slot clears, moves, and value updates downgraded from Release to Relaxed (lock Acquire/Release provides synchronization)
- Insert fast path: BoxPolicy `update_existing_value` reconstructs old output from retire handle pointer, eliminating a redundant atomic load
- Value prefetch pipeline: forward scan prefetches next slot's value, reverse scan prefetches before emit, leaf prefetch covers second cache line of values array
- Batch insert: replaced `Vec<*mut u8>` with `[*mut u8; 16]` stack array for deferred suffix bag retires
- Redundant `is_value_empty` check removed from `get_guarded` (load_value already returns None for empty slots)
- New `maintenance()` API: flush retired memory, refresh EBR epoch, and process coalesce in one call
- New `debug-print` feature: OCC-safe tree visualization with box-drawing characters and sublayer recursion

## 0.9.2

SuffixBagCell interior mutability, suffix prefetch, and concurrent correctness fixes.

- `SuffixBagCell` wrapper eliminates `&mut SuffixBag` / `&SuffixBag` aliasing UB in concurrent paths
- Append-only suffix assignment prevents overwriting existing suffix bytes during concurrent reads
- Suffix prefetch warms sidecar data before OCC read for cache-friendly lookups
- Remove: parent child pointer cleared before retire (was after, risking write to reclaimed memory)
- Remove: `load_layer_raw` moved inside critical section
- Optimistic reads: value pointer loaded before OCC validation (was after, TOCTOU gap)
- Coalesce: `clear_queued()` added for leftmost empty leaf
- Batch insert: suffix buffer pre-allocation outside lock

## 0.9.1

Route-based re-traversal for sublayer GC.

- Fixes UAF from duplicate queue entries and stale pointers in deeply nested chains.
- The correctness fixes for sublayer GC leads to substantial performance improvements.
