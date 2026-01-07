# masstree

A high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable-length keys by building a trie of B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)

**Disclaimer:** This is an independent implementation. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions.

## Features

- Ordered map for byte keys (lexicographic ordering)
- Lock-free reads with version validation
- Concurrent inserts and deletes with fine-grained leaf locking
- Zero-copy range scans with `scan_ref` and `scan_prefix`
- Memory reclamation via epoch-based deferred cleanup
- Lazy leaf coalescing for deleted entries
- Two node widths: `MassTree` (WIDTH=24) and `MassTree15` (WIDTH=15)

## Status

**v0.3.0** — Core feature complete. It has been heavily tested but I am not sure about whether it should be usd in actual projects. Such low-level cncurrent data structures usually need a lot of stress testing and have a lot of edge cases that are not easily noticeable. The unsafe code passes miri with strict-provenance flag, but that doesn't really ensure correctness.

| Feature | Status |
|---------|--------|
| `get`, `get_ref` | Lock-free with version validation |
| `insert` | Fine-grained leaf locking |
| `remove` | Concurrent deletion with memory reclamation |
| `scan`, `scan_ref`, `scan_prefix` | Zero-copy range iteration |
| Leaf coalescing | Lazy queue-based cleanup |
| Memory reclamation | Seize-based epoch reclamation |

**Tests:** 755 tests (466 unit + 88 ported from C++ reference + integration). Miri strict provenance clean.

**Not yet implemented:** `Entry` API, `DoubleEndedIterator`, `Extend`/`FromIterator`.

## Install

```toml
[dependencies]
masstree = { version = "0.3", features = ["mimalloc"] }
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

## Benchmarks

6 physical cores, `mimalloc` allocator, 200 samples per benchmark. Your mileage may vary.

### Mixed Read/Write (90% read, 10% write, 6 threads)

| Workload | MassTree15 | IndexSet | TreeIndex | SkipMap |
|----------|------------|----------|-----------|---------|
| Uniform | 19.3 M/s | 10.3 M/s | 11.0 M/s | 7.8 M/s |
| Zipfian | 21.9 M/s | 5.0 M/s | 9.5 M/s | 8.5 M/s |
| High contention | 51.7 M/s | 3.7 M/s | 12.0 M/s | 11.4 M/s |
| Single hot key | 16.7 M/s | 3.3 M/s | 3.5 M/s | 5.4 M/s |

The high-contention result likely reflects the per-node versioning design. Pure insert workloads favor TreeIndex.

### Pure Read (6 threads)

| Workload | MassTree15 | IndexSet | TreeIndex | SkipMap |
|----------|------------|----------|-----------|---------|
| Uniform | 27.5 M/s | 12.8 M/s | 19.2 M/s | 12.0 M/s |
| 8-byte keys | 35.2 M/s | 13.8 M/s | 15.9 M/s | 9.6 M/s |

### Single-Thread Latency

| Structure | Read Latency |
|-----------|--------------|
| MassTree15 | 771 µs |
| TreeIndex | 1,310 µs |
| IndexSet | 1,377 µs |
| SkipMap | 1,864 µs |

### vs C++ Reference (6 threads)

| Workload | Rust | C++ | Ratio |
|----------|------|-----|-------|
| 98% reads (rw2g98) | 37.68 M/s | 19.1 M/s | 197% |
| 90% reads (rw2g90) | 29.58 M/s | 13.5 M/s | 219% |
| Hotspot contention (same) | 6.67 M/s | 2.57 M/s | 259% |
| Updates (uscale) | 17.68 M/s | 9.3 M/s | 190% |
| Sequential keys (rw3) | 33.91 M/s | 39.3 M/s | 86% |
| Reverse sequential (rw4) | 27.22 M/s | 35.9 M/s | 76% |

Mixed results. Performs well on contention-heavy workloads but trails on sequential key patterns (76-86%). The C++ implementation has optimizations for sequential access that aren't yet ported.

Note: This implementation diverges from C++ in several ways (notably hyaline-based memory reclamation via `seize`). Direct comparison is imperfect.

### vs RwLock\<BTreeMap\> (6 threads)

The main question: when does a complex lock-free structure beat a simple `RwLock<BTreeMap>`?

**Mixed read/write workloads (where MassTree is designed to help):**

| Workload | MassTree15 | std::RwLock | parking_lot |
|----------|------------|-------------|-------------|
| 90/10 uniform | 13.6 M/s | 3.2 M/s | 5.2 M/s |
| 95/5 zipfian (hot keys) | 36.5 M/s | 6.8 M/s | 10.8 M/s |

MassTree's lock-free reads and per-node versioning help when writers need to make progress. The 2-5x advantage shows up when there's actual contention.

**Pure read workloads (where RwLock naturally wins):**

| Workload | MassTree15 | RwLock (batched) |
|----------|------------|------------------|
| Point reads | 13.2 M/s | 13.0 M/s |
| Range scans | 125 M/s | 1.2 G/s |

For read-only workloads, RwLock has minimal overhead (one atomic to acquire) while MassTree pays for version validation on every access. Range scans are particularly lopsided because RwLock holds the lock for the entire scan. This is expected - lock-free structures pay complexity costs that only matter under contention.

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

## AI Assist Disclaimer

It should be obvious that such a high number of tests, benchmarks and docs could not be written by hand this fast. Even though the full design and implementation was written by hand, there's still a a significant amount of AI generated code. I have gone through most of the docs,tests and benches to ensure correctness and also added the 'prompt' for the agent I used to analyze C++ codebase.

Apart from writing the above mentioned things, it was also used to write prototype ideas to optimize the implementation (most of which (like 80-90%) didn't work out well, and I had to just revert or remove them entirely, this can be seen if you go through the commits). The CAS insert fast path and a direct port of leaf coalescing was something that the model (Opus 4.5) was pushing aggressively, even though it was fundamentally unsound for masstree and leads to EXTREMELY subtle data races and synchronization issues and transient stress test failures.
