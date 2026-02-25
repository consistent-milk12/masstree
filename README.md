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

**v0.9.0** — Core feature complete.

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

## vs Rust Concurrent Maps (12T SMT)

> Source: `runs/run193_read_write.txt` (masstree), `runs/run184_read_write_reworked.txt` (competitors)
> **Config:** 12 threads on 6 physical cores (SMT/hyperthreading), 200 samples.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | **40.85** | 14.49 | 10.80 | 14.94 | **2.74x** |
| 02_zipfian | **36.64** | 18.86 | 12.92 | 3.28 | **1.94x** |
| 03_shared_prefix | **26.97** | 14.61 | 11.08 | 14.36 | **1.85x** |
| 04_high_contention | **66.13** | 21.64 | 16.90 | 1.97 | **3.06x** |
| 05_large_dataset | **15.93** | 11.35 | 7.50 | 9.17 | **1.40x** |
| 06_single_hot_key | **15.21** | 6.82 | 7.46 | 2.52 | **2.04x** |
| 07_mixed_50_50 | **32.76** | 8.73 | 6.46 | 14.86 | **2.20x** |
| 08_8byte_keys | **52.67** | 24.97 | 14.06 | 19.69 | **2.11x** |
| 09_pure_read | **53.36** | 24.12 | 14.40 | 17.40 | **2.21x** |
| 10_remove_heavy | **21.58** | 11.18 | 7.18 | 4.26 | **1.93x** |
| 13_insert_only_fair | **34.03** | 24.29 | 15.13 | 6.30 | **1.40x** |
| 14_pure_insert | **13.06** | 11.96 | 9.86 | 2.30 | **1.09x** |

## High-Impact Workloads (12T SMT)

> Source: `runs/run188_high_impact.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

Benchmarks targeting Masstree's architectural advantages: long keys, variable-length keys,
hot key patterns, mixed operations, prefix queries, deep trie traversal, and mixed prefix workloads.

| Benchmark | masstree15 | indexset | tree_index | skipmap | MT vs Best |
|-----------|------------|----------|------------|---------|------------|
| 01_long_keys_128b | **34.28** | 14.60 | 12.83 | 10.84 | **2.35x** |
| 02_multiple_hot_keys | **39.25** | 13.99 | 13.31 | 13.36 | **2.81x** |
| 03_mixed_get_insert_remove | **21.99** | 5.923 | 10.02 | 8.488 | **2.20x** |
| 04_variable_long_keys | **25.04** | 9.427 | 8.393 | 7.651 | **2.66x** |
| 05_prefix_queries (Kitem/s) | **846.9** | n/a | 495.9 | 144.9 | **1.71x** |
| 06_deep_trie_traversal | **17.66** | 13.49 | 8.102 | 9.308 | **1.31x** |
| 07_deep_trie_read_only | **27.55** | 14.81 | 14.54 | 13.37 | **1.86x** |
| 08_variable_keys_arc | **26.26** | 11.67 | 10.45 | 8.875 | **2.25x** |
| 09_prefix_realistic_mixed | **4.986** | n/a | 2.968 | 1.045 | **1.68x** |

## Range Scans (6T Physical)

> Source: `runs/run191_range_scan.txt`
> **Config:** Physical cores only, 200 samples, performance governor

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **26.73** | 13.07 | **2.04x** |
| 02_reverse_scan | **27.87** | 13.27 | **2.10x** |
| 03_clustered_scan | **25.16** | 13.08 | **1.92x** |
| 04_sparse_scan | **22.28** | 12.78 | **1.74x** |
| 05_shared_prefix_scan | **22.93** | 15.33 | **1.50x** |
| 06_suffix_differ_scan | **21.28** | 14.55 | **1.46x** |
| 07_hierarchical_scan | **24.47** | 14.60 | **1.68x** |
| 08_adversarial_splits | **26.83** | 8.807 | **3.05x** |
| 09_interleaved_scan | **22.41** | 12.86 | **1.74x** |
| 10_blink_stress_scan | **24.48** | 12.81 | **1.91x** |
| 11_random_keys_scan | **26.02** | 12.36 | **2.10x** |
| 12_long_keys_64b_scan | **26.06** | 14.24 | **1.83x** |
| 15_full_scan_aggregate | **183.1** | 84.59 | **2.16x** |
| 16_insert_heavy | **23.18** | 17.38 | **1.33x** |
| 17_hot_spot | **11.24** | 5.318 | **2.11x** |

## Range Scans (12T SMT)

> Source: `runs/run191_range_scan.txt`
> **Config:** 12 threads on 6 physical cores (SMT), 200 samples

| Benchmark | masstree15_inline | tree_index | MT vs TI |
|-----------|-------------------|------------|----------|
| 01_sequential_full_scan | **31.77** | 14.49 | **2.19x** |
| 02_reverse_scan | **29.96** | 14.66 | **2.04x** |
| 03_clustered_scan | **31.85** | 14.54 | **2.19x** |
| 04_sparse_scan | **30.95** | 14.58 | **2.12x** |
| 05_shared_prefix_scan | **24.88** | 18.62 | **1.34x** |
| 06_suffix_differ_scan | **23.41** | 18.16 | **1.29x** |
| 07_hierarchical_scan | **26.99** | 18.46 | **1.46x** |
| 08_adversarial_splits | **29.10** | 11.32 | **2.57x** |
| 09_interleaved_scan | **24.82** | 15.34 | **1.62x** |
| 10_blink_stress_scan | **30.05** | 15.12 | **1.99x** |
| 11_random_keys_scan | **25.62** | 14.70 | **1.74x** |
| 12_long_keys_64b_scan | **28.29** | 17.77 | **1.59x** |
| 15_full_scan_aggregate | **230.8** | 111.9 | **2.06x** |
| 16_insert_heavy | **25.09** | 21.88 | **1.15x** |
| 17_hot_spot | **10.61** | 6.458 | **1.64x** |

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
