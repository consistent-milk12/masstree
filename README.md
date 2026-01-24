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
- Two node widths: `MassTree` (WIDTH=24) and `MassTree15` (WIDTH=15)
- Extremely high-performance inline variant `MassTree15Inline`, this is only usable on
Copy types.

## Status

**v0.6.6** — Major performance enhancements. Core feature complete. Beats C++ Masstree on 5/7 benchmarks and
Rust alternatives on 11/12 workloads. Passes Miri with strict-provenance flag. Concurrent data structures
require extensive stress testing, the test suite is comprehensive (998 tests total) but edge cases may remain.

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

**Not yet implemented:** `Entry` API, `Extend`/`FromIterator`.

## vs C++ Masstree (12T, 10s)

Results vary between runs and hardware configurations. The `same` benchmark
(10 hot keys, 12 threads) consistently shows the largest improvement over C++,
achieving **1.7x higher throughput** under extreme contention. Possible factors:

- **Hyaline memory reclamation** — Unlike the C++ epoch-based reclamation (EBR),
  Hyaline (via `seize` crate) allows readers to avoid quiescent state registration
- **Lazy coalescing** — Empty leaves are queued for deferred cleanup rather than
  removed inline, avoiding lock-coupling issues during removes
- **Sharded length counter** — 16 cache-line-aligned shards for `len()` tracking
  (C++ doesn't track global count)

Note: The optimistic read protocol (version-based OCC) is the original Masstree design,
not a divergence. One minor divergence: `has_changed()` uses `> (LOCK_BIT | INSERTING_BIT)`
instead of C++'s `> lock_bit`, ignoring both bits 0-1. This is safe because version
*counters* (VINSERT/VSPLIT) are the source of truth, INSERTING_BIT is only set while
modifications are in-flight and not yet visible to readers. See `src/nodeversion.rs:643-673`
for the full safety argument.

The forward-sequential gap (rw3) narrowed from 57% to 81% but remains under investigation.

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| **rw4** (reverse-seq) | 59.00 | 48.14 | **123%** |
| **same** (10 hot keys) | 3.56 | 2.09 | **170%** |
| **rw2g98** (98% reads) | 25.81 | 23.04 | **112%** |
| **uscale** (random 140M) | 11.05 | 10.58 | **104%** |
| **wscale** (wide random) | 9.56 | 9.03 | **106%** |
| **rw1** (random insert+read) | 11.01 | 11.23 | 98% |
| **rw3** (forward-seq) | 40.54 | 50.34 | 81% |

## vs Rust Concurrent Maps (6T Physical, Rigorous)

> Source: `runs/run150_read_write_correctness.txt`
> **Config:** Physical cores only, 200 samples, performance governor.

This can be considered the current baseline.

Note: MassTree's `insert()` has upsert semantics, it updates existing keys and returns the old
value. TreeIndex's `insert()` fails on existing keys, requiring a `remove()+insert()` fallback.
Pure insert benchmarks (13, 14) use fresh keys only, providing a fairer comparison for
insert-heavy workloads where TreeIndex performs better.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | **28.03** | 13.93 | 8.78 | 12.23 | **2.01x** |
| 02_zipfian | **30.89** | 11.63 | 9.90 | 4.20 | **2.66x** |
| 03_shared_prefix | **15.57** | 8.48 | 7.66 | 11.80 | **1.32x** |
| 04_high_contention | **59.10** | 14.78 | 12.94 | 3.47 | **4.00x** |
| 05_large_dataset | **13.76** | 8.98 | 6.71 | 7.68 | **1.53x** |
| 06_single_hot_key | **18.02** | 4.50 | 5.94 | 4.04 | **3.03x** |
| 07_mixed_50_50 | **25.99** | 5.67 | 5.13 | 12.12 | **2.14x** |
| 08_8byte_keys | **43.67** | 21.52 | 11.86 | 16.95 | **2.03x** |
| 09_pure_read | **42.10** | 22.88 | 13.70 | 13.31 | **1.84x** |
| 10_remove_heavy | **15.02** | 11.62 | 5.07 | 3.93 | **1.29x** |
| 13_insert_only_fair | **22.49** | 17.77 | 10.37 | 5.42 | **1.27x** |
| 14_pure_insert | 9.93 | **11.42** | 8.13 | 2.17 | 0.87x |

**Single-thread latency:** masstree15 achieves **836 µs** median read latency vs tree_index 1.35 ms (**1.61x faster**).

**Build time:** masstree15 builds at **8.46 Mitem/s** vs skipmap 6.17, tree_index 4.35, indexset 1.86 (**1.37–4.6x faster**).

Note: The `14_pure_insert` numbers above are from the updated `benches/concurrent_read_write.rs` benchmark which uses unique keys per insert (true insert-only). Older `runs/run150_read_write_correctness.txt` results for `14_pure_insert` were affected by key reuse.

## Range Scans (6T Physical, Rigorous)

> Source: `runs/run149_range_scan_correctness.txt` (inline-optimized)
> **Config:** Physical cores only, 100 samples, performance governor

| Benchmark | masstree15_inline | tree_index | MT vs TI | vs run139 |
|-----------|-------------------|------------|----------|-----------|
| 01_sequential_full_scan | **30.73** | 15.34 | **2.00x** | -5% |
| 02_reverse_scan | **23.35** | 15.17 | **1.54x** | -1% |
| 03_clustered_scan | **30.84** | 15.16 | **2.03x** | -2% |
| 04_sparse_scan | **30.83** | 15.36 | **2.01x** | -4% |
| 05_shared_prefix_scan | **26.75** | 15.40 | **1.74x** | +1% |
| 06_suffix_differ_scan | **23.69** | 16.44 | **1.44x** | +49% |
| 07_hierarchical_scan | **28.93** | 16.67 | **1.74x** | +66% |
| 08_adversarial_splits | **30.23** | 9.23 | **3.28x** | +61% |
| 09_interleaved_scan | **26.98** | 15.07 | **1.79x** | +66% |
| 10_blink_stress_scan | **29.94** | 15.14 | **1.98x** | +44% |
| 11_random_keys_scan | **30.53** | 15.35 | **1.99x** | +46% |
| 12_long_keys_64b_scan | **29.78** | 16.45 | **1.81x** | +55% |
| 15_full_scan_aggregate | **1.93 G** | 1.10 G | **1.75x** | +13% |
| 16_insert_heavy | **22.89** | 16.32 | **1.40x** | +1% |
| 17_hot_spot | **10.26** | 3.05 | **3.37x** | n/a |

Note: `17_hot_spot` is sensitive to update semantics. MassTree overwrites existing keys on `insert()`. For `scc::TreeIndex`, the benchmark now emulates overwrite updates via `remove()+insert()` to match semantics (see `benches/range_masstree15_inline.rs`). The `runs/run149_range_scan_correctness.txt` hot-spot result predates this semantic fix.

## Install

```toml
[dependencies]
masstree = { version = "0.6.6", features = ["mimalloc"] }
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

// Point lookup
assert_eq!(tree.get_ref(b"hello", &guard), Some(&123));

// Remove
tree.remove_with_guard(b"hello", &guard).unwrap();
assert_eq!(tree.get_ref(b"hello", &guard), None);

// Range scan (zero-copy)
tree.scan_ref(b"a"..b"z", |key, value| {
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
- Pure insert-only workloads → `scc::TreeIndex`
- Integer keys only → `congee` (ART-based)
- Read-heavy with rare writes → `RwLock<BTreeMap>`

## Variant Selection

Two variants are provided with different performance characteristics:

| Variant | Best For |
|---------|----------|
| `MassTree15` | Range scans, writes, shared-prefix keys, contention |
| `MassTree` (WIDTH=24) | Random-access reads, single-threaded point ops |

`MassTree15` tends to perform better in our benchmarks due to cheaper u64 atomics and better cache utilization. Consider it for most workloads unless you have uniform random-access patterns.

```rust
use masstree::{MassTree, MassTree15, MassTree24Inline, MassTree15Inline};

// Default: WIDTH=24, Arc-based storage
let tree: MassTree<u64> = MassTree::new();

// WIDTH=15, Arc-based storage (recommended for most workloads)
let tree15: MassTree15<u64> = MassTree15::new();

// Inline storage for Copy types (no Arc overhead)
let inline: MassTree24Inline<u64> = MassTree24Inline::new();
let inline15: MassTree15Inline<u64> = MassTree15Inline::new();
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
cargo run --example basic_usage --release      # Core API walkthrough
cargo run --example rayon_parallel --release   # Parallel processing with Rayon
cargo run --example tokio_async --release      # Async integration with Tokio
cargo run --example url_cache --release        # Real-world URL cache
cargo run --example session_store --release    # Concurrent session store
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
