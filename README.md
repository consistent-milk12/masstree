# masstree

A high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable-length keys by building a trie of B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)

**Disclaimer:** This is an independent implementation. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions.

## Status

Current Version: v0.9.0

Significantly reduced duplicate code through trait based abstractions, while improving performance further. All standard data structure features are implemented and streamlined, on top the specialized features like range scans and prefix queries. From latency analysis, it seems to have a surprisingly strong tail latency profile for a concurrent ordered map. The cache-line aligned layout design and aggressive prefetching seems to have paid off, on top of masstree's original excellent architecture.

The next issues to work on are the memory profile and memcomparable mappings for general key types, before 1.0.

## vs Rust Concurrent Maps (12T SMT)

> Source: `runs/run194_read_write.txt` (masstree), `runs/run184_read_write_reworked.txt` (competitors)
> **Config:** 12 threads on 6 physical cores (SMT/hyperthreading), 200 samples.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | 48.14 | 14.49 | 10.80 | 14.94 | 3.22x |
| 02_zipfian | 41.79 | 18.86 | 12.92 | 3.28 | 2.22x |
| 03_shared_prefix | 29.48 | 14.61 | 11.08 | 14.36 | 2.02x |
| 04_high_contention | 62.40 | 21.64 | 16.90 | 1.97 | 2.88x |
| 05_large_dataset | 17.00 | 11.35 | 7.50 | 9.17 | 1.50x |
| 06_single_hot_key | 15.42 | 6.82 | 7.46 | 2.52 | 2.07x |
| 07_mixed_50_50 | 33.89 | 8.73 | 6.46 | 14.86 | 2.28x |
| 08_8byte_keys | 52.92 | 24.97 | 14.06 | 19.69 | 2.12x |
| 09_pure_read | 50.77 | 24.12 | 14.40 | 17.40 | 2.11x |
| 10_remove_heavy | 20.90 | 11.18 | 7.18 | 4.26 | 1.87x |
| 13_insert_only_fair | 35.46 | 24.29 | 15.13 | 6.30 | 1.46x |
| 14_pure_insert | 14.08 | 11.96 | 9.86 | 2.30 | 1.18x |

## High-Impact Workloads (12T SMT)

> Source: `runs/run195_high_impact.txt` (masstree), `runs/run188_high_impact.txt` (competitors)
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

Benchmarks targeting Masstree's architectural advantages: long keys, variable-length keys,
hot key patterns, mixed operations, prefix queries, deep trie traversal, and mixed prefix workloads.

| Benchmark | masstree15 | indexset | tree_index | skipmap | MT vs Best |
|-----------|------------|----------|------------|---------|------------|
| 01_long_keys_128b | 39.21 | 14.60 | 12.83 | 10.84 | 2.69x |
| 02_multiple_hot_keys | 40.31 | 13.99 | 13.31 | 13.36 | 2.88x |
| 03_mixed_get_insert_remove | 27.12 | 5.923 | 10.02 | 8.488 | 2.71x |
| 04_variable_long_keys | 27.54 | 9.427 | 8.393 | 7.651 | 2.92x |
| 05_prefix_queries (Kitem/s) | 1040 | n/a | 495.9 | 144.9 | 2.10x |
| 06_deep_trie_traversal | 25.53 | 13.49 | 8.102 | 9.308 | 1.89x |
| 07_deep_trie_read_only | 30.99 | 14.81 | 14.54 | 13.37 | 2.09x |
| 08_variable_keys_arc | 27.74 | 11.67 | 10.45 | 8.875 | 2.38x |
| 09_prefix_realistic_mixed | 4.862 | n/a | 2.968 | 1.045 | 1.64x |

## Range Scans (6T Physical)

> Source: `runs/run191_range_scan.txt`
> **Config:** Physical cores only, 200 samples, performance governor

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | 26.73 | 13.07 | 2.04x |
| 02_reverse_scan | 27.87 | 13.27 | 2.10x |
| 03_clustered_scan | 25.16 | 13.08 | 1.92x |
| 04_sparse_scan | 22.28 | 12.78 | 1.74x |
| 05_shared_prefix_scan | 22.93 | 15.33 | 1.50x |
| 06_suffix_differ_scan | 21.28 | 14.55 | 1.46x |
| 07_hierarchical_scan | 24.47 | 14.60 | 1.68x |
| 08_adversarial_splits | 26.83 | 8.807 | 3.05x |
| 09_interleaved_scan | 22.41 | 12.86 | 1.74x |
| 10_blink_stress_scan | 24.48 | 12.81 | 1.91x |
| 11_random_keys_scan | 26.02 | 12.36 | 2.10x |
| 12_long_keys_64b_scan | 26.06 | 14.24 | 1.83x |
| 15_full_scan_aggregate | 183.1 | 84.59 | 2.16x |
| 16_insert_heavy | 23.18 | 17.38 | 1.33x |
| 17_hot_spot | 11.24 | 5.318 | 2.11x |

## Range Scans (12T SMT)

> Source: `runs/run191_range_scan.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | 31.77 | 14.49 | 2.19x |
| 02_reverse_scan | 29.96 | 14.66 | 2.04x |
| 03_clustered_scan | 31.85 | 14.54 | 2.19x |
| 04_sparse_scan | 30.95 | 14.58 | 2.12x |
| 05_shared_prefix_scan | 24.88 | 18.62 | 1.34x |
| 06_suffix_differ_scan | 23.41 | 18.16 | 1.29x |
| 07_hierarchical_scan | 26.99 | 18.46 | 1.46x |
| 08_adversarial_splits | 29.10 | 11.32 | 2.57x |
| 09_interleaved_scan | 24.82 | 15.34 | 1.62x |
| 10_blink_stress_scan | 30.05 | 15.12 | 1.99x |
| 11_random_keys_scan | 25.62 | 14.70 | 1.74x |
| 12_long_keys_64b_scan | 28.29 | 17.77 | 1.59x |
| 15_full_scan_aggregate | 230.8 | 111.9 | 2.06x |
| 16_insert_heavy | 25.09 | 21.88 | 1.15x |
| 17_hot_spot | 10.61 | 6.458 | 1.64x |

## Tail Latency (Exact Percentiles)

> Source: `runs/run198_tail_latency.txt`
> **Config:** 50k samples per benchmark (exact sorted-array backend), mimalloc

### Point Lookups

| Benchmark | masstree | dashmap | treeindex | skipmap | indexset |
|-----------|----------|---------|-----------|---------|----------|
| get 1T p50 | 117 ns | 57 ns | 228 ns | 340 ns | 260 ns |
| get 1T p99.99 | 739 ns | 340 ns | 1.68 us | 4.81 us | 4.06 us |
| get 8T p50 | 302 ns | 437 ns | 436 ns | 581 ns | 591 ns |
| get 8T p99.99 | 1.02 us | 712 ns | 4.83 us | 3.18 us | 12.1 us |
| get 1M 1T p50 | 370 ns | 165 ns | 471 ns | 661 ns | 531 ns |
| get zipf 8T p50 | 310 ns | 205 ns | 295 ns | 541 ns | 521 ns |

### Inserts

| Benchmark | masstree | dashmap | treeindex | skipmap | indexset |
|-----------|----------|---------|-----------|---------|----------|
| insert 1T p50 | 110 ns | 80 ns | 210 ns | 180 ns | 541 ns |
| insert 1T p99.99 | 6.10 us | 32.9 us | 4.33 us | 3.82 us | 5.16 us |
| insert 8T p50 | 681 ns | 315 ns | 751 ns | 952 ns | 6.65 us |
| insert 8T p99.99 | 140 us | 15.4 us | 86.4 us | 63.7 us | 158 us |

### Mixed Read/Write (8T)

| Benchmark | masstree | dashmap | treeindex | skipmap | indexset |
|-----------|----------|---------|-----------|---------|----------|
| 90/10 p50 | 491 ns | 411 ns | 511 ns | 711 ns | 606 ns |
| 90/10 p99.99 | 6.22 us | 821 ns | 9.91 us | 10.7 us | 3.77 us |
| 50/50 p50 | 571 ns | 519 ns | 1.08 us | 1.10 us | 781 ns |
| 50/50 p99.99 | 5.88 us | 1.51 us | 92.6 us | 45.7 us | 5.81 us |

### Range Scans (8T, 50-key scan)

| Benchmark | masstree | treeindex | skipmap |
|-----------|----------|-----------|---------|
| scan p50 | 461 ns | 821 ns | 3.80 us |
| scan p99.99 | 5.24 us | 5.92 us | 12.5 us |

## Install

```toml
[dependencies]
masstree = { version = "0.9.0", features = ["mimalloc"] }
```

MSRV is Rust 1.92+ (Edition 2024).

The `mimalloc` feature sets the global allocator. If your project already uses a custom allocator, omit this feature.

## Quick Start

```rust
use masstree::{MassTree, RangeBound};

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

// Insert (returns None for new keys, Some(old) for existing)
tree.insert_with_guard(b"hello", 123, &guard);
tree.insert_with_guard(b"world", 456, &guard);

// Point lookup (returns copy for inline storage)
assert_eq!(tree.get_with_guard(b"hello", &guard), Some(123));

// Remove
tree.remove_with_guard(b"hello", &guard).unwrap();
assert_eq!(tree.get_with_guard(b"hello", &guard), None);

// Range scan
tree.scan(
    RangeBound::Included(b"a"),
    RangeBound::Excluded(b"z"),
    |key, value| {
        println!("{:?} -> {}", key, value);
        true // continue scanning
    },
    &guard,
);

// Prefix scan
tree.scan_prefix(b"wor", |key, value| {
    println!("{:?} -> {}", key, value);
    true
}, &guard);
```

## Ergonomic APIs

For simpler use cases, auto-guard versions create guards internally:

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();

// Auto-guard versions (simpler but slightly more overhead per call)
tree.insert(b"key1", 100);
tree.insert(b"key2", 200);

assert_eq!(tree.get(b"key1"), Some(100));
assert_eq!(tree.len(), 2);
assert!(!tree.is_empty());

tree.remove(b"key1").unwrap();
```

### Range Iteration

```rust
use masstree::{MassTree, RangeBound};

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

// Populate
for i in 0..100u64 {
    tree.insert_with_guard(&i.to_be_bytes(), i, &guard);
}

// Iterator-based range scan
for entry in tree.range(RangeBound::Included(b""), RangeBound::Unbounded, &guard) {
    println!("{:?} -> {:?}", entry.key(), entry.value());
}

// Full iteration
for entry in tree.iter(&guard) {
    println!("{:?}", entry.key());
}
```

## When to Use

**May work well for:**

- Long keys with shared prefixes (URLs, file paths, UUIDs)
- Range scans over ordered data
- Mixed read/write workloads
- High-contention scenarios (the trie structure helps here)

**Consider alternatives for:**

- Unordered point lookups → `dashmap`
- Integer keys only → `congee` (ART-based)
- Read-heavy with rare writes → `RwLock<BTreeMap>`

## Which Tree Type Should I Use?

| Type | Value Requirement | `get_ref()` | Best For |
|------|-------------------|-------------|----------|
| `MassTree<V>` | `V: InlineBits` (Copy + ≤64 bits) | Nope | `u64`, `i32`, `f64`, small tuples |
| `MassTree15<V>` | `V: Send + Sync + 'static` | Yes | `String`, `Vec`, custom structs |
| `MassTree15Inline<V>` | `V: InlineBits` | Nope | Same as `MassTree<V>` (explicit alias) |

### `MassTree<V>` (Default, True-Inline)

- Values stored directly in leaf nodes (zero allocation per insert)
- Returns `V` by copy: `get_with_guard() → Option<V>`
- Use `scan()` for range iteration

### `MassTree15<V>` (Box-Based)

- Values stored as `Box<V>` raw pointers (heap allocation per insert)
- Returns `ValuePtr<V>`: a zero-cost `Copy` pointer with `Deref` to `&V`
- Supports `get_ref() → Option<&V>` for zero-copy access
- Use `scan_ref()` for zero-copy range iteration

`MassTree<V>` is the recommended default for `Copy` types like `u64`, `i32`, `f64`, pointers, etc.
Use `MassTree15<V>` explicitly when you need to store non-Copy types like `String`.

```rust
use masstree::{MassTree, MassTree15, MassTree15Inline};

// Default: inline storage for Copy types (recommended)
let tree: MassTree<u64> = MassTree::new();

// Box-based storage for non-Copy types
let tree_box: MassTree15<String> = MassTree15::new();

// Explicit inline storage (same as MassTree)
let inline: MassTree15Inline<u64> = MassTree15Inline::new();
```

## How It Works

Masstree splits keys into 8-byte chunks, creating a trie where each node is a B+tree:

```text
Key: "users/alice/profile" (19 bytes)
     └─ Layer 0: "users/al" (8 bytes)
        └─ Layer 1: "ice/prof" (8 bytes)
           └─ Layer 2: "ile" (3 bytes)
```

Keys with shared prefixes share upper layers, making lookups efficient for hierarchical data.

## Examples

The `examples/` directory contains comprehensive usage examples:

```bash
cargo run --example basic_usage --features mimalloc --release      # Core API walkthrough
cargo run --example rayon_parallel --features mimalloc --release   # Parallel processing with Rayon
cargo run --example tokio_async --features mimalloc --release      # Async integration with Tokio
cargo run --example url_cache --features mimalloc --release        # Real-world URL cache
cargo run --example session_store --features mimalloc --release    # Concurrent session store
```

### Rayon Integration

MassTree works seamlessly with Rayon for parallel bulk operations:

```rust
use masstree::MassTree15Inline;
use rayon::prelude::*;
use std::sync::Arc;

let tree: Arc<MassTree15Inline<u64>> = Arc::new(MassTree15Inline::new());

// Parallel bulk insert (~10M ops/sec)
(0..1_000_000).into_par_iter().for_each(|i| {
    let key = format!("key/{i:08}");
    let guard = tree.guard();
    let _ = tree.insert_with_guard(key.as_bytes(), i, &guard);
});

// Parallel lookups (~45M ops/sec)
let sum: u64 = (0..1_000_000).into_par_iter()
    .map(|i| {
        let key = format!("key/{i:08}");
        let guard = tree.guard();
        tree.get_with_guard(key.as_bytes(), &guard).unwrap_or(0)
    })
    .sum();
```

### Tokio Integration

MassTree is thread-safe but guards cannot be held across `.await` points:

```rust
use masstree::MassTree15;
use std::sync::Arc;

let tree: Arc<MassTree15<String>> = Arc::new(MassTree15::new());

// Spawn async tasks that share the tree
let handle = tokio::spawn({
    let tree = Arc::clone(&tree);
    async move {
        // Guard must be scoped - cannot be held across await!
        {
            let guard = tree.guard();
            let _ = tree.insert_with_guard(b"key", "value".to_string(), &guard);
        } // guard dropped here

        tokio::time::sleep(Duration::from_millis(10)).await;

        // Create new guard after await
        let guard = tree.guard();
        tree.get_with_guard(b"key", &guard)
    }
});

// For CPU-intensive operations, use spawn_blocking
let tree_clone = Arc::clone(&tree);
tokio::task::spawn_blocking(move || {
    let guard = tree_clone.guard();
    for entry in tree_clone.iter(&guard) {
        // Process entries...
    }
}).await;
```

## Crate Features

- `mimalloc` — Use mimalloc as global allocator (recommended)
- `tracing` — Enable structured logging to `logs/masstree.jsonl`

## License

MIT. See `LICENSE`.

## References

- [Masstree Paper (EuroSys 2012)](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)
- [C++ Reference Implementation](https://github.com/kohler/masstree-beta)
