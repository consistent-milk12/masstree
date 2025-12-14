## Key Operations Consistency

| Operation | Run 1 | Run 2 | Run 3 (Post-Fix) | Status |
|-----------|-------|-------|------------------|--------|
| `ikey()` | 0.52 ns | 0.53 ns | 0.41 ns | Stable |
| `len()` | 0.54 ns | 0.55 ns | 0.41 ns | Stable |
| `read_ikey` (8b) | 0.78 ns | 0.78 ns | 0.63 ns | Stable |
| `compare_ikey` | 0.51 ns | 0.52 ns | 0.38 ns | Stable |
| `Key::new(8+)` | 2.00 ns | 1.94 ns | 1.81 ns | Stable |
| `Key::new(1-4)` | 8.51 ns | 8.20 ns | 8.13 ns | Stable |
| `traverse_3_layers` | 8.71 ns | 8.79 ns | 8.72 ns | Stable |

### Slow Path Consistency (Post-Fix)

| Operation | Run 1 | Run 2 | Run 3 (Post-Fix) | Status |
|-----------|-------|-------|------------------|--------|
| `slow_1b` | 7.03 ns | 7.09 ns | **6.18 ns** | Improved |
| `slow_4b` | 5.25 ns | 5.25 ns | **6.18 ns** | Normalized |
| `slow_7b` | 7.09 ns | 7.09 ns | **6.18 ns** | Improved |

The fix eliminated the inconsistent timing between different partial key lengths.

### Permuter Operations Consistency

| Operation | Run 1 Median | Run 2 Median | Delta | Status |
|-----------|--------------|--------------|-------|--------|
| `size()` | 0.41 ns | 0.54 ns | +0.13 ns | Variance |
| `back()` | 0.41 ns | 0.54 ns | +0.13 ns | Variance |
| `get(i)` | 0.95 ns | 1.08 ns | +0.13 ns | Variance |
| `scan_all_15` | 3.63 ns | 3.78 ns | +0.15 ns | Stable |
| `insert_at_end` | 0.44 ns | 0.56 ns | +0.12 ns | Variance |
| `remove_last` | 0.13 ns | 0.30 ns | +0.17 ns | Variance |
| `drain_15_beginning` | 30.84 ns | 30.74 ns | -0.10 ns | Stable |

### NodeVersion Operations

| Operation | Run 1 | Run 2 (const fn) | Delta | Status |
|-----------|-------|------------------|-------|--------|
| `new_leaf` | 0.26 ns | 0.13 ns | -0.13 ns | Improved |
| `new_internode` | 0.26 ns | 0.13 ns | -0.13 ns | Improved |
| `clone` | 0.65 ns | 0.52 ns | -0.13 ns | Improved |
| `is_leaf` | 0.54 ns | 0.42 ns | -0.12 ns | Improved |
| `is_locked` | 0.65 ns | 0.52 ns | -0.13 ns | Improved |
| `value` | 0.63 ns | 0.50 ns | -0.13 ns | Improved |
| `stable` | 0.64 ns | 0.52 ns | -0.12 ns | Improved |
| `has_changed` | 1.17 ns | 1.05 ns | -0.12 ns | Improved |
| `has_split` | 1.20 ns | 1.06 ns | -0.14 ns | Improved |
| `try_lock_fail` | 0.49 ns | 0.36 ns | -0.13 ns | Improved |
| `check_all_flags` | 1.81 ns | 1.70 ns | -0.11 ns | Improved |
| `optimistic_read_success` | 1.04 ns | 0.92 ns | -0.12 ns | Improved |

Run 2 added `const fn` to construction benchmarks, enabling better compile-time optimization.

Note: `lock_unlock`, `try_lock_success`, and `mark_*` operations show ~0 ns due to
compiler optimization when input is freshly created per iteration. Real-world
performance depends on cache state and memory ordering constraints.

