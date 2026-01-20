# masstree

A high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable-length keys by building a trie of B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)

**Disclaimer:** This is an independent implementation. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions.

## Features

- Ordered map for byte keys (lexicographic ordering)
- Lock-free reads with version validation
- Concurrent inserts and deletes with fine-grained leaf locking
- Zero-copy range scans with `scan_ref` and `scan_prefix`
- Memory reclamation via hyaline scheme (`seize` crate)
- Lazy leaf coalescing for deleted entries
- Two node widths: `MassTree` (WIDTH=24) and `MassTree15` (WIDTH=15)

## Status

**v0.5.11** — Core feature complete. Heavily tested but concurrent data
structures require extensive stress testing beyond what tests (even though still
quite comprehensive) provide. The unsafe code passes Miri with strict-provenance
flag. It should be noted that the C++ implementation is a research project, not
a library specifically developed for production usage. So it is quite possible
that there are still various rare edge cases and soundness issues that still
haven't been addressed. Fixed several memory leaks and one UB caught by miri
in new targeted tests.

| Feature | Status |
|---------|--------|
| `get`, `get_ref` | Lock-free with version validation |
| `insert` | Fine-grained leaf locking |
| `remove` | Concurrent deletion with memory reclamation |
| `scan`, `scan_ref`, `scan_prefix` | Zero-copy range iteration |
| `DoubleEndedIterator` | Reverse iteration support |
| Leaf coalescing | Lazy queue-based cleanup |
| Memory reclamation | Hyaline scheme via `seize` crate |

**Tests:** 977 total (including doc tests), 584 lib tests. Ignored long-running stress tests pass with `cargo test -- --ignored`. Miri strict provenance clean.

**Not yet implemented:** `Entry` API, `Extend`/`FromIterator`.

## Performance

In benchmarks, MassTree is typically **1.5-4x faster** than comparable ordered maps with support for variable length keys (`indexset`, `crossbeam-skiplist`, `scc::TreeIndex`, `RwLock<BTreeMap>`) for mixed read/write workloads, and **up to 13x faster** under write contention where `RwLock` collapses. It only loses in pure insert workloads (~10% to `scc::TreeIndex`) and single-threaded bulk construction (~30% to `skipmap`), these aren't realistic workloads, but still noteworthy.

See `runs/` for detailed benchmark data.

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

The forward-sequential gap (rw3) remains under investigation.

| Benchmark | Rust | C++ | Ratio | Winner |
|-----------|------|-----|-------|--------|
| **rw4** (reverse-seq) | 59.00 | 48.14 | **123%** | **Rust** |
| **same** (10 hot keys) | 3.56 | 2.09 | **170%** | **Rust** |
| **rw2g98** (98% reads) | 25.81 | 23.04 | **112%** | **Rust** |
| **uscale** (random 140M) | 11.05 | 10.58 | **104%** | **Rust** |
| **wscale** (wide random) | 9.56 | 9.03 | **106%** | **Rust** |
| **rw1** (random insert+read) | 11.01 | 11.23 | 98% | Tie |
| **rw3** (forward-seq) | 26.49 | 50.34 | 53% | C++ |

## vs Rust Concurrent Maps (6T Physical, Rigorous)

> Source: `runs/run135_read_write_rigorous.txt`
> **Config:** Physical cores only, 200 samples, performance governor.

This can be considered the current baseline.

Note: MassTree's `insert()` has upsert semantics, it updates existing keys and returns the old
value. TreeIndex's `insert()` fails on existing keys, requiring a `remove()+insert()` fallback.
Pure insert benchmarks (13, 14) use fresh keys only, providing a fairer comparison for
insert-heavy workloads where TreeIndex performs better.

| Benchmark | masstree15 | tree_index | skipmap | indexset | MT vs Best |
|-----------|-----------|------------|---------|----------|------------|
| 01_uniform | **22.44** | 15.16 | 9.36 | 12.71 | **1.48x** |
| 02_zipfian | **26.20** | 11.81 | 10.63 | 5.57 | **2.22x** |
| 03_shared_prefix | **18.25** | 8.52 | 8.81 | 13.07 | **1.40x** |
| 04_high_contention | **67.49** | 15.81 | 13.04 | 3.53 | **4.27x** |
| 05_large_dataset | **12.02** | 9.41 | 6.87 | 7.92 | **1.28x** |
| 06_single_hot_key | **15.21** | 4.49 | 6.07 | 3.92 | **2.51x** |
| 07_mixed_50_50 | **18.44** | 5.68 | 5.20 | 12.26 | **1.50x** |
| 08_8byte_keys | **40.78** | 23.30 | 12.00 | 16.33 | **1.75x** |
| 09_pure_read | **33.16** | 24.03 | 14.14 | 13.92 | **1.38x** |
| 10_remove_heavy | **18.71** | 11.75 | 5.51 | 3.71 | **1.59x** |
| 13_insert_only_fair | 18.16 | **19.78** | 11.53 | 5.71 | 0.92x |
| 14_pure_insert | 8.71 | **12.78** | 6.39 | 2.35 | **0.68x** |

## Install

```toml
[dependencies]
masstree = { version = "0.5.11", features = ["mimalloc"] }
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
