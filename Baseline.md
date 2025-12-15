# Benchmark Baseline

**Last Updated:** 2025-12-15 (Run 5)
**Benchmark Framework:** [divan](https://github.com/nvzqz/divan)
**Timer Precision:** 20-30 ns
**Rust Toolchain:** 1.92.0 (stable)

## Key Operations

| Operation | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|-----------|-------|-------|-------|-------|-------|--------|
| `ikey()` | 0.52 ns | 0.53 ns | 0.41 ns | 0.55 ns | **0.41 ns** | Stable |
| `len()` | 0.54 ns | 0.55 ns | 0.41 ns | 0.55 ns | **0.41 ns** | Stable |
| `read_ikey` (8b) | 0.78 ns | 0.78 ns | 0.63 ns | 0.78 ns | **0.63 ns** | Stable |
| `compare_ikey` | 0.51 ns | 0.52 ns | 0.38 ns | 0.52 ns | **0.38 ns** | Stable |
| `Key::new(8+)` | 2.00 ns | 1.94 ns | 1.81 ns | 1.99 ns | **1.78 ns** | Stable |
| `Key::new(1-4)` | 8.51 ns | 8.20 ns | 8.13 ns | 8.57 ns | **7.97 ns** | Stable |
| `traverse_3_layers` | 8.71 ns | 8.79 ns | 8.72 ns | 8.90 ns | **8.66 ns** | Stable |

Run 5 confirms Run 3 values were not anomalous - fast path operations consistently achieve ~0.4 ns.

### Slow Path (Partial Keys)

| Operation | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|-----------|-------|-------|-------|-------|-------|--------|
| `slow_1b` | 7.03 ns | 7.09 ns | 6.18 ns | 6.47 ns | **6.21 ns** | Normalized |
| `slow_4b` | 5.25 ns | 5.25 ns | 6.18 ns | 6.47 ns | **6.21 ns** | Normalized |
| `slow_7b` | 7.09 ns | 7.09 ns | 6.18 ns | 6.51 ns | **6.23 ns** | Normalized |

All partial key sizes now have consistent timing (~6.2 ns).

---

## Permuter Operations

| Operation | Run 1 | Run 2 | Run 4 | Run 5 | Status |
|-----------|-------|-------|-------|-------|--------|
| `size()` | 0.41 ns | 0.54 ns | 0.34 ns | **0.55 ns** | Stable |
| `back()` | 0.41 ns | 0.54 ns | 0.34 ns | **0.55 ns** | Stable |
| `get(i)` | 0.95 ns | 1.08 ns | 0.88 ns | **1.09 ns** | Stable |
| `scan_all_15` | 3.63 ns | 3.78 ns | 3.58 ns | **3.81 ns** | Stable |
| `insert_at_end` | 0.44 ns | 0.56 ns | 0.37 ns | **0.57 ns** | Stable |
| `remove_last` | 0.13 ns | 0.30 ns | 0.09 ns | **0.34 ns** | Stable |
| `drain_15_beginning` | 30.84 ns | 30.74 ns | 30.96 ns | **31.31 ns** | Stable |
| `fill_15_at_end` | — | — | 6.55 ns | **6.04 ns** | Stable |

Sub-nanosecond variations are within timer precision noise.

---

## NodeVersion Operations

| Operation | Run 1 | Run 2 | Run 4 | Run 5 | Status |
|-----------|-------|-------|-------|-------|--------|
| `new_leaf` | 0.26 ns | 0.13 ns | 0.004 ns | **0.26 ns** | Stable |
| `new_internode` | 0.26 ns | 0.13 ns | 0.004 ns | **0.26 ns** | Stable |
| `clone` | 0.65 ns | 0.52 ns | 0.39 ns | **0.70 ns** | Stable |
| `is_leaf` | 0.54 ns | 0.42 ns | 0.39 ns | **0.55 ns** | Stable |
| `is_locked` | 0.65 ns | 0.52 ns | 0.39 ns | **0.66 ns** | Stable |
| `value` | 0.63 ns | 0.50 ns | 0.44 ns | **0.55 ns** | Stable |
| `stable` | 0.64 ns | 0.52 ns | 0.39 ns | **0.70 ns** | Stable |
| `has_changed` | 1.17 ns | 1.05 ns | 0.93 ns | **1.18 ns** | Stable |
| `has_split` | 1.20 ns | 1.06 ns | 0.95 ns | **1.22 ns** | Stable |
| `try_lock_fail` | 0.49 ns | 0.36 ns | 0.24 ns | **0.50 ns** | Stable |
| `check_all_flags` | 1.81 ns | 1.70 ns | 1.58 ns | **1.87 ns** | Stable |
| `optimistic_read_success` | 1.04 ns | 0.92 ns | 0.80 ns | **1.06 ns** | Stable |

Run 4 had unrealistically low values (0.004 ns) due to compiler optimizations on freshly created inputs. Run 5 shows real-world costs (~0.26 ns for construction).

---

## Notes

- **Run 2:** Added `const fn` to construction benchmarks
- **Run 4:** Added `#[inline]` and `#[must_use]` to `has_changed()`
- **Run 5:** Clean baseline with 20-30 ns timer precision

`lock_unlock`, `try_lock_success`, and `mark_*` operations show ~0 ns due to compiler optimization when input is freshly created per iteration. Real-world performance depends on cache state and memory ordering constraints.

---

## Reference Values (Run 5)

Use these as the canonical baseline for future comparisons:

| Module | Operation | Target |
|--------|-----------|--------|
| Key | `ikey()` | ≤0.5 ns |
| Key | `compare_ikey` | ≤0.5 ns |
| Key | `Key::new(8+)` | ≤2.0 ns |
| Key | `traverse_3_layers` | ≤9.0 ns |
| Permuter | `scan_all_15` | ≤4.0 ns |
| Permuter | `drain_15_beginning` | ≤32 ns |
| NodeVersion | `optimistic_read_success` | ≤1.2 ns |
| NodeVersion | `has_changed` | ≤1.3 ns |
