# masstree

A high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable-length keys by building a trie of B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)

**Disclaimer:** This is an independent implementation. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions.

## Features

- Ordered map for byte keys (lexicographic ordering)
- Lock-free reads with version validation
- Concurrent inserts and deletes with fine-grained leaf locking
- Zero-copy range scans with `scan_ref` and `scan_prefix`
- High-throughput value-only scans with `scan_values` (skips key materialization)
- Memory reclamation via hyaline scheme (`seize` crate)
- Lazy leaf coalescing for deleted entries
- High-performance inline variant `MassTree15Inline`, this is only usable on
Copy types.

## Status

**v0.7.4** — Core feature complete.

| Feature | Status |
|---------|--------|
| `get`, `get_ref` | Lock-free with version validation |
| `insert` | Fine-grained leaf locking |
| `remove` | Concurrent deletion with memory reclamation |
| `scan`, `scan_ref`, `scan_prefix` | Zero-copy range iteration |
| `scan_values`, `scan_values_rev` | High-throughput value-only scans |
| `DoubleEndedIterator` | Reverse iteration support |
| Leaf coalescing | Lazy queue-based cleanup |
| Memory reclamation | Hyaline scheme via `seize` crate |

## vs C++ Masstree (12T, 10s)

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| **rw3** (forward-seq) | 70.50 | 45.34 | **156%** |
| **highcontention** (500 keys, 64B) | 78.04 | 58.62 | **133%** |
| **rw4** (reverse-seq) | 56.14 | 43.24 | **130%** |
| **rw1** (random insert+read) | 10.11 | 8.16 | **124%** |
| **same** (10 hot keys) | 2.57 | 2.07 | **124%** |
| **uscale** (random 140M) | 10.77 | 8.81 | **122%** |
| **rw2g98** (98% reads) | 24.71 | 20.70 | **119%** |
| **wscale** (wide random) | 9.00 | 8.12 | **111%** |

## vs Rust Concurrent Maps (12T SMT)

> Source: `runs/run158_read_write.txt`
> **Config:** 12 threads on 6 physical cores (SMT/hyperthreading), 200 samples.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | **49.85** | 20.64 | 14.14 | 17.73 | **2.42x** |
| 02_zipfian | **43.39** | 18.38 | 14.98 | 3.03 | **2.36x** |
| 03_shared_prefix | **24.86** | 15.17 | 12.98 | 16.37 | **1.52x** |
| 04_high_contention | **76.28** | 16.44 | 17.85 | 1.97 | **4.27x** |
| 05_large_dataset | **20.79** | 12.44 | 10.10 | 11.22 | **1.67x** |
| 06_single_hot_key | **10.66** | 4.29 | 6.51 | 2.39 | **1.64x** |
| 07_mixed_50_50 | **36.60** | 9.89 | 7.20 | 16.89 | **2.17x** |
| 08_8byte_keys | **59.86** | 32.17 | 17.39 | 20.90 | **1.86x** |
| 09_pure_read | **56.25** | 29.26 | 20.64 | 19.09 | **1.92x** |
| 10_remove_heavy | **21.52** | 18.01 | 8.22 | 4.54 | **1.19x** |
| 13_insert_only_fair | **37.25** | 24.25 | 16.72 | 6.12 | **1.54x** |
| 14_pure_insert | **14.33** | 13.67 | 10.52 | 2.50 | **1.05x** |

## High-Impact Workloads (12T SMT)

> Source: `runs/run154_high_impact_twig_optimization.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

Benchmarks targeting Masstree's architectural advantages: long keys, variable-length keys,
hot key patterns, mixed operations, prefix queries, and deep trie traversal.

| Benchmark | masstree15 | indexset | tree_index | skipmap | MT vs Best |
|-----------|------------|----------|------------|---------|------------|
| 01_long_keys_128b | **34.95** | 14.58 | 14.98 | 11.15 | **2.33x** |
| 02_multiple_hot_keys | **40.97** | 14.24 | 12.43 | 13.26 | **2.88x** |
| 03_mixed_get_insert_remove | **27.24** | 6.00 | 11.93 | 8.85 | **2.28x** |
| 04_variable_long_keys | **28.17** | 9.30 | 8.29 | 7.51 | **3.03x** |
| 05_prefix_queries (Kitem/s) | **426.3** | n/a | 14.56 | 140.7 | **3.02x** |
| 06_deep_trie_traversal | **18.16** | 13.77 | 11.16 | 8.84 | **1.32x** |
| 07_deep_trie_read_only | **27.90** | 15.05 | 17.35 | 15.28 | **1.61x** |
| 08_variable_keys_arc | **29.56** | 11.13 | 11.55 | 8.46 | **2.56x** |

## Range Scans (6T Physical)

> Source: `runs/run161_range_scan.txt`
> **Config:** Physical cores only, 100 samples, performance governor

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **28.42** | 13.47 | **2.11x** |
| 02_reverse_scan | **27.09** | 13.59 | **1.99x** |
| 03_clustered_scan | **29.54** | 13.40 | **2.20x** |
| 04_sparse_scan | **29.43** | 13.39 | **2.20x** |
| 05_shared_prefix_scan | **25.49** | 15.17 | **1.68x** |
| 06_suffix_differ_scan | **22.21** | 15.66 | **1.42x** |
| 07_hierarchical_scan | **27.12** | 15.62 | **1.74x** |
| 08_adversarial_splits | **28.71** | 8.40 | **3.42x** |
| 09_interleaved_scan | **25.42** | 13.55 | **1.88x** |
| 10_blink_stress_scan | **29.31** | 13.58 | **2.16x** |
| 11_random_keys_scan | **29.44** | 13.54 | **2.17x** |
| 12_long_keys_64b_scan | **27.68** | 15.63 | **1.77x** |
| 15_full_scan_aggregate | **178.1** | 83.36 | **2.14x** |
| 16_insert_heavy | **26.93** | 19.02 | **1.42x** |
| 17_hot_spot | **9.51** | 2.99 | **3.18x** |

## Range Scans (12T SMT)

> Source: `runs/run161_range_scan.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 100 samples

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **30.38** | 15.27 | **1.99x** |
| 02_reverse_scan | **28.51** | 15.14 | **1.88x** |
| 03_clustered_scan | **30.50** | 15.18 | **2.01x** |
| 04_sparse_scan | **30.37** | 15.11 | **2.01x** |
| 05_shared_prefix_scan | **26.12** | 21.48 | **1.22x** |
| 06_suffix_differ_scan | **24.00** | 21.08 | **1.14x** |
| 07_hierarchical_scan | **27.98** | 21.17 | **1.32x** |
| 08_adversarial_splits | **29.23** | 10.04 | **2.91x** |
| 09_interleaved_scan | **26.24** | 15.30 | **1.72x** |
| 10_blink_stress_scan | **30.14** | 15.23 | **1.98x** |
| 11_random_keys_scan | **29.70** | 15.17 | **1.96x** |
| 12_long_keys_64b_scan | **28.25** | 21.33 | **1.32x** |
| 15_full_scan_aggregate | **176.8** | 113.5 | **1.56x** |
| 16_insert_heavy | **30.06** | 25.26 | **1.19x** |
| 17_hot_spot | **9.69** | 4.04 | **2.40x** |

## Install

```toml
[dependencies]
masstree = { version = "0.7.4", features = ["mimalloc"] }
```

MSRV is Rust 1.92+ (Edition 2024).

The `mimalloc` feature sets the global allocator. If your project already uses a custom allocator, omit this feature.

## Quick Start

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

// Insert
tree.insert_with_guard(b"hello", 123, &guard).unwrap();
tree.insert_with_guard(b"world", 456, &guard).unwrap();

// Point lookup (returns copy for inline storage)
assert_eq!(tree.get_with_guard(b"hello", &guard), Some(123));

// Remove
tree.remove_with_guard(b"hello", &guard).unwrap();
assert_eq!(tree.get_with_guard(b"hello", &guard), None);

// Range scan
tree.scan(b"a"..b"z", |key, value| {
    println!("{:?} -> {}", key, value);
    true // continue scanning
}, &guard);

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
tree.insert(b"key1", 100).unwrap();
tree.insert(b"key2", 200).unwrap();

assert_eq!(tree.get(b"key1"), Some(std::sync::Arc::new(100)));
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
    tree.insert_with_guard(&i.to_be_bytes(), i, &guard).unwrap();
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

## Type Aliases

| Type | Storage | Value Requirement |
|------|---------|-------------------|
| `MassTree<V>` | Inline (default) | `V: InlineBits` (Copy + fits in 64 bits) |
| `MassTree15<V>` | Arc-based | `V: Send + Sync + 'static` |
| `MassTree15Inline<V>` | True inline | `V: InlineBits` |

`MassTree<V>` is the recommended default for `Copy` types like `u64`, `i32`, `f64`, pointers, etc.
Use `MassTree15<V>` explicitly when you need to store non-Copy types like `String`.

```rust
use masstree::{MassTree, MassTree15, MassTree15Inline};

// Default: inline storage for Copy types (recommended)
let tree: MassTree<u64> = MassTree::new();

// Arc-based storage for non-Copy types
let tree_arc: MassTree15<String> = MassTree15::new();

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
