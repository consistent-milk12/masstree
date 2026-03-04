# Changelog

## 0.9.3

Relaxed ordering optimization, value prefetch pipeline, and insert fast path.

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
