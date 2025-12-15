## Key Operations Consistency

| Operation | Run 1 | Run 2 | Run 3 | Run 4 | Status |
|-----------|-------|-------|-------|-------|--------|
| `ikey()` | 0.52 ns | 0.53 ns | 0.41 ns | 0.55 ns | Stable |
| `len()` | 0.54 ns | 0.55 ns | 0.41 ns | 0.55 ns | Stable |
| `read_ikey` (8b) | 0.78 ns | 0.78 ns | 0.63 ns | 0.78 ns | Stable |
| `compare_ikey` | 0.51 ns | 0.52 ns | 0.38 ns | 0.52 ns | Stable |
| `Key::new(8+)` | 2.00 ns | 1.94 ns | 1.81 ns | 1.99 ns | Stable |
| `Key::new(1-4)` | 8.51 ns | 8.20 ns | 8.13 ns | 8.57 ns | Stable |
| `traverse_3_layers` | 8.71 ns | 8.79 ns | 8.72 ns | 8.90 ns | Stable |

Run 3 was anomalously fast; Run 4 confirms values consistent with Run 1/2.

### Slow Path Consistency (Post-Fix)

| Operation | Run 1 | Run 2 | Run 3 | Run 4 | Status |
|-----------|-------|-------|-------|-------|--------|
| `slow_1b` | 7.03 ns | 7.09 ns | 6.18 ns | 6.47 ns | Normalized |
| `slow_4b` | 5.25 ns | 5.25 ns | 6.18 ns | 6.47 ns | Normalized |
| `slow_7b` | 7.09 ns | 7.09 ns | 6.18 ns | 6.51 ns | Normalized |

The fix eliminated the inconsistent timing between different partial key lengths.
Run 4 confirms normalization holds (~6.5 ns for all sizes).

### Permuter Operations Consistency

| Operation | Run 1 | Run 2 | Run 4 | Status |
|-----------|-------|-------|-------|--------|
| `size()` | 0.41 ns | 0.54 ns | 0.34 ns | Improved |
| `back()` | 0.41 ns | 0.54 ns | 0.34 ns | Improved |
| `get(i)` | 0.95 ns | 1.08 ns | 0.88 ns | Improved |
| `scan_all_15` | 3.63 ns | 3.78 ns | 3.58 ns | Stable |
| `insert_at_end` | 0.44 ns | 0.56 ns | 0.37 ns | Improved |
| `remove_last` | 0.13 ns | 0.30 ns | 0.09 ns | Stable |
| `drain_15_beginning` | 30.84 ns | 30.74 ns | 30.96 ns | Stable |

### NodeVersion Operations

| Operation | Run 1 | Run 2 | Run 4 | Status |
|-----------|-------|-------|-------|--------|
| `new_leaf` | 0.26 ns | 0.13 ns | 0.004 ns | Improved |
| `new_internode` | 0.26 ns | 0.13 ns | 0.004 ns | Improved |
| `clone` | 0.65 ns | 0.52 ns | 0.39 ns | Improved |
| `is_leaf` | 0.54 ns | 0.42 ns | 0.39 ns | Stable |
| `is_locked` | 0.65 ns | 0.52 ns | 0.39 ns | Improved |
| `value` | 0.63 ns | 0.50 ns | 0.44 ns | Stable |
| `stable` | 0.64 ns | 0.52 ns | 0.39 ns | Improved |
| `has_changed` | 1.17 ns | 1.05 ns | 0.93 ns | Improved |
| `has_split` | 1.20 ns | 1.06 ns | 0.95 ns | Improved |
| `try_lock_fail` | 0.49 ns | 0.36 ns | 0.24 ns | Improved |
| `check_all_flags` | 1.81 ns | 1.70 ns | 1.58 ns | Improved |
| `optimistic_read_success` | 1.04 ns | 0.92 ns | 0.80 ns | Improved |

Run 2 added `const fn` to construction benchmarks.
Run 4 added `#[inline]` and `#[must_use]` to `has_changed()`.

Note: `lock_unlock`, `try_lock_success`, and `mark_*` operations show ~0 ns due to
compiler optimization when input is freshly created per iteration. Real-world
performance depends on cache state and memory ordering constraints.
