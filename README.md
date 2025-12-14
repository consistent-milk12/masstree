# Section 1: Introduction

After studying the CPP implementation, I thought naming the repository as 'madtree' was appropriate. As actually implementing this by myself is indeed an insane thing to do. I still wanted to try and see how far I can go.

Masstree is a high-performance concurrent trie of B+ trees designed for in-memory key-value storage. It combines the cache efficiency of B+ trees with the no-rebalancing property of tries by slicing keys into 8-byte chunks, where each chunk navigates a separate B+ tree layer.

**Source:** <https://github.com/kohler/masstree-beta>

Three benchmark runs were performed: two before the fix, one after.

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

