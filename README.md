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

**v0.8.2** — Core feature complete.

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
| **rw3** (forward-seq) | 70.62 | 45.34 | **156%** |
| **same** (10 hot keys) | 2.82 | 2.07 | **136%** |
| **rw4** (reverse-seq) | 58.52 | 43.24 | **135%** |
| **highcontention** (500 keys, 64B) | 78.04 | 58.62 | **133%** |
| **rw1** (random insert+read) | 9.70 | 8.16 | **119%** |
| **rw2g98** (98% reads) | 24.10 | 20.70 | **116%** |
| **uscale** (random 140M) | 10.11 | 8.81 | **115%** |
| **wscale** (wide random) | 8.93 | 8.12 | **110%** |

## vs Rust Concurrent Maps (12T SMT)

> Source: `runs/run184_read_write_reworked.txt`
> **Config:** 12 threads on 6 physical cores (SMT/hyperthreading), 200 samples.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | **34.80** | 14.49 | 10.80 | 14.94 | **2.33x** |
| 02_zipfian | **37.63** | 18.86 | 12.92 | 3.28 | **2.00x** |
| 03_shared_prefix | **24.39** | 14.61 | 11.08 | 14.36 | **1.67x** |
| 04_high_contention | **58.85** | 21.64 | 16.90 | 1.97 | **2.72x** |
| 05_large_dataset | **15.15** | 11.35 | 7.50 | 9.17 | **1.34x** |
| 06_single_hot_key | **14.88** | 6.82 | 7.46 | 2.52 | **2.00x** |
| 07_mixed_50_50 | **33.27** | 8.73 | 6.46 | 14.86 | **2.24x** |
| 08_8byte_keys | **54.04** | 24.97 | 14.06 | 19.69 | **2.16x** |
| 09_pure_read | **49.32** | 24.12 | 14.40 | 17.40 | **2.05x** |
| 10_remove_heavy | **18.75** | 11.18 | 7.18 | 4.26 | **1.68x** |
| 13_insert_only_fair | **35.05** | 24.29 | 15.13 | 6.30 | **1.44x** |
| 14_pure_insert | **13.90** | 11.96 | 9.86 | 2.30 | **1.16x** |

## High-Impact Workloads (12T SMT)

> Source: `runs/run180_high_impact.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

Benchmarks targeting Masstree's architectural advantages: long keys, variable-length keys,
hot key patterns, mixed operations, prefix queries, deep trie traversal, and mixed prefix workloads.

| Benchmark | masstree15 | indexset | tree_index | skipmap | MT vs Best |
|-----------|------------|----------|------------|---------|------------|
| 01_long_keys_128b | **34.08** | 14.66 | 15.32 | 10.25 | **2.22x** |
| 02_multiple_hot_keys | **42.11** | 13.85 | 15.47 | 14.47 | **2.72x** |
| 03_mixed_get_insert_remove | **22.52** | 5.92 | 11.86 | 8.64 | **1.90x** |
| 04_variable_long_keys | **24.34** | 8.57 | 9.01 | 7.67 | **2.70x** |
| 05_prefix_queries (Kitem/s) | **849.8** | n/a | 702.7 | 143.1 | **1.21x** |
| 05_prefix_values (Mitem/s) | **2.193** | n/a | n/a | n/a | — |
| 06_deep_trie_traversal | **17.91** | 13.87 | 10.97 | 8.56 | **1.29x** |
| 07_deep_trie_read_only | **26.22** | 14.90 | 16.23 | 11.97 | **1.62x** |
| 08_variable_keys_arc | **26.17** | 11.14 | 11.14 | 7.94 | **2.35x** |
| 09_prefix_realistic_mixed | **4.551** | n/a | 3.964 | 0.995 | **1.15x** |

## Range Scans (6T Physical)

> Source: `runs/run182_range_scan.txt`
> **Config:** Physical cores only, 100 samples, performance governor

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **29.29** | 13.88 | **2.11x** |
| 02_reverse_scan | **28.30** | 13.53 | **2.09x** |
| 03_clustered_scan | **28.93** | 13.85 | **2.09x** |
| 04_sparse_scan | **29.30** | 13.83 | **2.12x** |
| 05_shared_prefix_scan | **24.69** | 17.11 | **1.44x** |
| 06_suffix_differ_scan | **22.04** | 16.97 | **1.30x** |
| 07_hierarchical_scan | **26.14** | 16.96 | **1.54x** |
| 08_adversarial_splits | **28.58** | 9.00 | **3.18x** |
| 09_interleaved_scan | **24.85** | 14.67 | **1.69x** |
| 10_blink_stress_scan | **28.44** | 14.37 | **1.98x** |
| 11_random_keys_scan | **28.44** | 14.22 | **2.00x** |
| 12_long_keys_64b_scan | **27.72** | 17.03 | **1.63x** |
| 15_full_scan_aggregate | **174.6** | 106.8 | **1.63x** |
| 16_insert_heavy | **25.41** | 19.39 | **1.31x** |
| 17_hot_spot | **11.15** | 4.60 | **2.42x** |

## Range Scans (12T SMT)

> Source: `runs/run182_range_scan.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 100 samples

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **29.53** | 16.49 | **1.79x** |
| 02_reverse_scan | **28.52** | 16.53 | **1.73x** |
| 03_clustered_scan | **28.26** | 16.51 | **1.71x** |
| 04_sparse_scan | **28.80** | 16.61 | **1.73x** |
| 05_shared_prefix_scan | **25.09** | 20.39 | **1.23x** |
| 06_suffix_differ_scan | **23.12** | 19.46 | **1.19x** |
| 07_hierarchical_scan | **27.04** | 20.29 | **1.33x** |
| 08_adversarial_splits | **28.55** | 11.54 | **2.47x** |
| 09_interleaved_scan | **24.39** | 15.54 | **1.57x** |
| 10_blink_stress_scan | **29.64** | 15.93 | **1.86x** |
| 11_random_keys_scan | **29.25** | 15.27 | **1.92x** |
| 12_long_keys_64b_scan | **27.75** | 19.30 | **1.44x** |
| 15_full_scan_aggregate | **226.7** | 124.8 | **1.82x** |
| 16_insert_heavy | **25.15** | 23.16 | **1.09x** |
| 17_hot_spot | **10.39** | 5.34 | **1.95x** |

## Install

```toml
[dependencies]
masstree = { version = "0.8.2", features = ["mimalloc"] }
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
