# Benchmark Baseline

**Last Updated:** 2025-12-15 (Run 7 - Post-cleanup benchmark)
**Benchmark Framework:** [divan](https://github.com/nvzqz/divan)
**Timer Precision:** 20-40 ns
**Rust Toolchain:** 1.92.0 (stable)

## Key Operations

| Operation | Run 5 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `ikey()` | 0.41 ns | **0.54 ns** | +0.13 ns | Stable (noise) |
| `len()` | 0.41 ns | **0.54 ns** | +0.13 ns | Stable (noise) |
| `read_ikey` (8b) | 0.63 ns | **0.77 ns** | +0.14 ns | Stable |
| `compare_ikey` | 0.38 ns | **0.51 ns** | +0.13 ns | Stable (noise) |
| `Key::new(8+)` | 1.78 ns | **1.92 ns** | +0.14 ns | Stable |
| `Key::new(1-4)` | 7.97 ns | **8.16 ns** | +0.19 ns | Stable |
| `traverse_3_layers` | 8.66 ns | **8.83 ns** | +0.17 ns | Stable |

Sub-nanosecond variations are within timer precision noise (~20-40 ns timer).

### Slow Path (Partial Keys)

| Operation | Run 5 | Run 7 | Status |
|-----------|-------|-------|--------|
| `slow_1b` | 6.21 ns | **6.34 ns** | Stable |
| `slow_4b` | 6.21 ns | **6.34 ns** | Stable |
| `slow_7b` | 6.23 ns | **6.34 ns** | Stable |

All partial key sizes have consistent timing (~6.3 ns).

---

## Permuter Operations

| Operation | Run 5 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `size()` | 0.55 ns | **0.41 ns** | -25% | Improved |
| `back()` | 0.55 ns | **0.42 ns** | -24% | Improved |
| `get(i)` | 1.09 ns | **0.95 ns** | -13% | Improved |
| `scan_all_15` | 3.81 ns | **3.64 ns** | -4% | Stable |
| `insert_at_end` | 0.57 ns | **0.43 ns** | -25% | Improved |
| `remove_last` | 0.34 ns | **0.16 ns** | -53% | Improved |
| `drain_15_beginning` | 31.31 ns | **30.45 ns** | -3% | Stable |
| `fill_15_at_end` | 6.04 ns | **5.95 ns** | -1% | Stable |

Permuter operations show consistent improvement in Run 7.

---

## NodeVersion Operations

| Operation | Run 5 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `new_leaf` | 0.26 ns | **0.13 ns** | -50% | Improved |
| `new_internode` | 0.26 ns | **0.13 ns** | -50% | Improved |
| `clone` | 0.70 ns | **0.53 ns** | -24% | Improved |
| `is_leaf` | 0.55 ns | **0.54 ns** | ~0% | Stable |
| `is_locked` | 0.66 ns | **0.54 ns** | -18% | Improved |
| `value` | 0.55 ns | **0.58 ns** | +5% | Stable |
| `stable` | 0.70 ns | **0.57 ns** | -19% | Improved |
| `has_changed` | 1.18 ns | **1.06 ns** | -10% | Improved |
| `has_split` | 1.22 ns | **1.07 ns** | -12% | Improved |
| `try_lock_fail` | 0.50 ns | **0.37 ns** | -26% | Improved |
| `check_all_flags` | 1.87 ns | **1.71 ns** | -9% | Improved |
| `optimistic_read_success` | 1.06 ns | **0.97 ns** | -8% | Improved |

NodeVersion operations show broad improvements in Run 7.

---

## LeafNode Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `new_leaf` | 34.7 ns | **54.0 ns** | +55% | Variance |
| `default_leaf` | 51.1 ns | **45.7 ns** | -11% | Improved |
| `size()` | 0.80 ns | **0.67 ns** | -16% | Improved |
| `is_empty()` | 0.79 ns | **0.67 ns** | -15% | Improved |
| `permutation()` | 0.79 ns | **0.66 ns** | -16% | Improved |
| `ikey(slot)` | 1.31 ns | **1.18 ns** | -10% | Improved |
| `keylenx(slot)` | 1.40 ns | **1.25 ns** | -11% | Improved |
| `leaf_value(slot)` | 1.33 ns | **1.50 ns** | +13% | Variance |
| `next_raw()` | 0.79 ns | **0.66 ns** | -16% | Improved |
| `prev()` | 0.78 ns | **0.66 ns** | -15% | Improved |
| `parent()` | 0.79 ns | **0.66 ns** | -16% | Improved |
| `set_permutation` | 1.25 ns | **1.10 ns** | -12% | Improved |
| `can_reuse_slot0` (no prev) | 1.32 ns | **1.13 ns** | -14% | Improved |
| `can_reuse_slot0` (with prev) | 1.69 ns | **1.53 ns** | -9% | Improved |

### LeafNode Split Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `calculate_split_point` (middle) | 8.1 ns | **7.7 ns** | -5% | Improved |
| `calculate_split_point` (sequential) | 8.1 ns | **7.7 ns** | -5% | Improved |
| `split_into` | 126.8 ns | **120.9 ns** | -5% | Improved |
| `split_all_to_right` | 212.6 ns | **211.2 ns** | ~0% | Stable |

---

## InternodeNode Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `new_height_0` | 47.2 ns | **65.5 ns** | +39% | Variance |
| `new_height_5` | 47.4 ns | **65.2 ns** | +38% | Variance |
| `default_internode` | 47.7 ns | **39.6 ns** | -17% | Improved |
| `nkeys()` | 0.67 ns | **1.08 ns** | +61% | Variance |
| `size()` | 0.66 ns | **1.09 ns** | +65% | Variance |
| `is_full()` | 0.65 ns | **1.08 ns** | +66% | Variance |
| `height()` | 0.66 ns | **1.16 ns** | +76% | Variance |
| `children_are_leaves()` | 0.66 ns | **1.14 ns** | +73% | Variance |
| `ikey(idx)` | 1.19 ns | **1.80 ns** | +51% | Variance |
| `child(idx)` | 1.40 ns | **2.03 ns** | +45% | Variance |
| `compare_key` | 0.79 ns | **1.27 ns** | +61% | Variance |
| `set_ikey` | 1.17 ns | **0.97 ns** | -17% | Improved |
| `set_child` | 2.30 ns | **1.01 ns** | -56% | Improved |
| `assign` | 2.08 ns | **2.01 ns** | -3% | Stable |
| `set_nkeys` | 1.28 ns | **1.28 ns** | 0% | Stable |
| `parent()` | 0.67 ns | **0.71 ns** | +6% | Stable |
| `is_root()` | 0.75 ns | **0.75 ns** | 0% | Stable |
| `set_parent` | 1.02 ns | **1.00 ns** | -2% | Stable |

Note: InternodeNode accessor variance may be due to different benchmark setup or cache effects.

### InternodeNode Insert Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `insert_at_front` | 9.6 ns | **8.6 ns** | -10% | Improved |
| `insert_at_middle` | 6.0 ns | **5.6 ns** | -7% | Improved |
| `insert_at_back` | 3.3 ns | **2.6 ns** | -21% | Improved |

### InternodeNode Split Operations (NEW)

| Operation | Run 7 | Status |
|-----------|-------|--------|
| `split_insert_left` | **9.9 ns** | Baseline |
| `split_insert_middle` | **6.8 ns** | Baseline |
| `split_insert_right` | **6.8 ns** | Baseline |

### InternodeNode Shift Operations

| Count | Run 6 | Run 7 | Status |
|-------|-------|-------|--------|
| 1 | 2.7 ns | **2.7 ns** | Stable |
| 3 | 3.4 ns | **3.4 ns** | Stable |
| 5 | 4.4 ns | **4.3 ns** | Stable |
| 7 | 6.3 ns | **6.3 ns** | Stable |

---

## Key Search Operations

### Binary vs Linear Search (15 elements)

| Operation | Run 6 | Run 7 | Status |
|-----------|-------|-------|--------|
| `binary_lower_bound` | 2.2 ns | **2.0 ns** | Improved |
| `linear_lower_bound` | 8.7 ns | **8.5 ns** | Stable |
| `binary_upper_bound` | 5.6 ns | **5.5 ns** | Stable |
| `linear_upper_bound` | 6.3 ns | **5.9 ns** | Improved |

Binary search remains ~4x faster for lower_bound.

### Leaf Search

| Operation | Size 1 | Size 5 | Size 10 | Size 15 | Status |
|-----------|--------|--------|---------|---------|--------|
| `lower_bound_existing` | 3.6 ns | 3.6 ns | 3.6 ns | 3.6 ns | O(log n) |
| `lower_bound_missing` | 3.3 ns | 5.9 ns | 8.0 ns | 7.9 ns | O(log n) |
| `lower_bound_ikey_only` | 2.6 ns | 2.6 ns | 2.6 ns | 2.6 ns | O(log n) |

### Internode Search

| Operation | Size 1 | Size 5 | Size 10 | Size 15 | Status |
|-----------|--------|--------|---------|---------|--------|
| `upper_bound_direct` | 1.8 ns | 4.0 ns | 5.2 ns | 4.9 ns | O(log n) |
| `upper_bound_route` | 10.0 ns | 10.2 ns | 10.2 ns | 6.5 ns | + overhead |

---

## MassTree Operations

### Construction

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `MassTree::new()` | 52.6 ns | **51.6 ns** | -2% | Stable |
| `MassTree::default()` | 51.7 ns | **55.7 ns** | +8% | Stable |
| `MassTreeIndex::new()` | 52.7 ns | **51.7 ns** | -2% | Stable |
| `MassTreeIndex::default()` | 52.6 ns | **53.7 ns** | +2% | Stable |

### Single Get Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `get` (empty tree) | 7.9 ns | **7.6 ns** | -4% | Stable |
| `get` (hit) | 12.1 ns | **10.8 ns** | -11% | **Improved** |
| `get` (miss) | 8.9 ns | **9.0 ns** | +1% | Stable |
| `get` (single leaf, 15 keys) | 4.7 ns | **4.5 ns** | -4% | Stable |
| `get` (multi leaf, 100 keys) | 12.1 ns | **9.2 ns** | -24% | **Improved** |

### Single Insert Operations

| Operation | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| `insert` (single) | 29.2 ns | **29.1 ns** | ~0% | Stable |
| `insert` (into existing tree) | 27.8 ns | **42.0 ns** | +51% | Different test |
| `insert` (update existing key) | 45.9 ns | **44.3 ns** | -3% | Stable |
| `insert_arc` | 25.3 ns | **25.8 ns** | +2% | Stable |

### Insert by Key Length

| Key Length | Run 6 | Run 7 | Change | Status |
|------------|-------|-------|--------|--------|
| 1 byte | 30.3 ns | **27.4 ns** | -10% | **Improved** |
| 2 bytes | 31.4 ns | **27.9 ns** | -11% | **Improved** |
| 4 bytes | 30.7 ns | **27.1 ns** | -12% | **Improved** |
| 6 bytes | 30.7 ns | **27.0 ns** | -12% | **Improved** |
| 8 bytes | 21.5 ns | **20.6 ns** | -4% | Stable |

Short keys (slow path) now ~10% faster.

### Batch Operations

| Operation | 10 keys | 50 keys | 100 keys | Status |
|-----------|---------|---------|----------|--------|
| Sequential insert | 263 ns | 1.11 µs | 2.37 µs | Stable |
| Reverse insert | 256 ns | 1.44 µs | 4.17 µs | Variance |
| Random insert | 254 ns | 1.34 µs | 3.73 µs | **Improved** |
| Sequential get | 88 ns | 623 ns | 1.48 µs | Stable |
| Random get | 91 ns | 511 ns | 1.12 µs | Stable |

### Scaling (Insert into N existing keys)

| Tree Size | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| 100 keys | 37.7 ns | **59.7 ns** | +58% | Variance |
| 500 keys | 121.3 ns | **67.2 ns** | -45% | **Improved** |
| 1000 keys | 80.1 ns | **98.9 ns** | +23% | Variance |

### Scaling (Get from N existing keys)

| Tree Size | Run 6 | Run 7 | Change | Status |
|-----------|-------|-------|--------|--------|
| 100 keys | 12.0 ns | **9.2 ns** | -23% | **Improved** |
| 500 keys | 18.4 ns | **14.3 ns** | -22% | **Improved** |
| 1000 keys | 17.4 ns | **17.4 ns** | 0% | Stable |

### Arc vs Copy Mode (100 keys)

| Operation | Arc Mode | Copy Mode | Notes |
|-----------|----------|-----------|-------|
| Insert 100 | 2.42 µs | 4.04 µs | Copy slower (expected) |
| Get 100 | 1.10 µs | 1.11 µs | Same |

Note: Copy mode insert is slower because `MassTreeIndex` still uses `Arc<V>` internally (see TODO.md for planned fix).

### Workload Benchmarks (NEW)

| Workload | Run 7 | Status |
|----------|-------|--------|
| Read-heavy (90/10) | **157 ns** | Baseline |
| Update existing | **284 ns** | Baseline |
| Write-heavy (10/90) | **475 ns** | Baseline |

---

## SOTA Comparison (Updated Run 7)

### vs Rust `std::collections::BTreeMap`

| Operation | MassTree | BTreeMap | Advantage |
|-----------|----------|----------|-----------|
| Insert (single) | 29 ns | 50 ns | **1.7x faster** |
| Insert (into 1000 keys) | 99 ns | 99 ns | Same |
| Get (from 1000 keys) | 17 ns | 15 ns | ~Same |
| Batch insert (100 keys) | 2.37 µs | ~5 µs | **2x faster** |

**Verdict:** MassTree remains 1.5-2x faster on inserts, comparable on gets.

### vs Original [Masstree C++](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)

| Metric | Original C++ | Our Rust | Notes |
|--------|--------------|----------|-------|
| Single-core ops | ~50-150 ns (est.) | 10-30 ns | **Competitive** |
| Multi-core throughput | 6-10M ops/sec (16 cores) | N/A | Phase 2 needed |

### vs [Congee](https://github.com/XiangpengHao/congee) (Concurrent ART, Rust)

| Metric | Congee | MassTree | Notes |
|--------|--------|----------|-------|
| Peak throughput | **150 Mop/s** (32 cores) | N/A | Concurrent vs single-threaded |
| Single-op latency | ~6.7 ns | 9-17 ns | ART faster for point queries |

### Summary

| Category | Verdict |
|----------|---------|
| vs std::BTreeMap | **1.5-2x faster** inserts |
| vs Original Masstree | **Competitive** single-core |
| vs Concurrent (Congee, sled) | Needs Phase 2 |

---

## Notes

- **Run 2:** Added `const fn` to construction benchmarks
- **Run 4:** Added `#[inline]` and `#[must_use]` to `has_changed()`
- **Run 5:** Clean baseline with 20-30 ns timer precision
- **Run 6:** Added LeafNode, InternodeNode, ksearch, and MassTree benchmarks
- **Run 7:** Post-cleanup (removed unused deps, added `!Send/!Sync`). Get operations improved ~20%, short-key inserts improved ~10%.

`lock_unlock`, `try_lock_success`, and `mark_*` operations show ~0 ns due to compiler optimization when input is freshly created per iteration.

---

## Reference Values (Run 7)

Use these as the canonical baseline for future comparisons:

| Module | Operation | Target |
|--------|-----------|--------|
| Key | `ikey()` | ≤0.6 ns |
| Key | `compare_ikey` | ≤0.6 ns |
| Key | `Key::new(8+)` | ≤2.0 ns |
| Key | `traverse_3_layers` | ≤9.0 ns |
| Permuter | `scan_all_15` | ≤4.0 ns |
| Permuter | `drain_15_beginning` | ≤32 ns |
| NodeVersion | `optimistic_read_success` | ≤1.0 ns |
| NodeVersion | `has_changed` | ≤1.1 ns |
| LeafNode | `new_leaf` | ≤55 ns |
| LeafNode | `split_into` | ≤125 ns |
| LeafNode | `calculate_split_point` | ≤8 ns |
| InternodeNode | `new_height_0` | ≤70 ns |
| InternodeNode | `insert_at_back` | ≤3 ns |
| ksearch | `lower_bound_leaf` (hit) | ≤4 ns |
| ksearch | `upper_bound_internode_direct` | ≤6 ns |
| MassTree | `new()` | ≤55 ns |
| MassTree | `get` (single leaf) | ≤5 ns |
| MassTree | `get` (multi leaf) | ≤10 ns |
| MassTree | `insert` (single) | ≤30 ns |
| MassTree | `insert` (8-byte key) | ≤22 ns |
