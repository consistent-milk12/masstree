# masstree

A high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable-length keys by building a trie of B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EuroSys 2012).

**Disclaimer:** This is an independent implementation. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions (MIT PDOS, Harvard).

## Features

- Ordered map for byte keys (lexicographic ordering)
- Lock-free reads with version validation
- Concurrent inserts and deletes with fine-grained leaf locking
- Zero-copy range scans with `scan_ref` and `scan_prefix`
- Memory reclamation via epoch-based deferred cleanup
- Lazy leaf coalescing for deleted entries
- Two node widths: `MassTree` (WIDTH=24) and `MassTree15` (WIDTH=15)

## Status

**v0.3.0** — Core feature complete. Not production-ready.

| Feature | Status |
|---------|--------|
| `get`, `get_ref` | Lock-free with version validation |
| `insert` | Fine-grained leaf locking |
| `remove` | Concurrent deletion with memory reclamation |
| `scan`, `scan_ref`, `scan_prefix` | Zero-copy range iteration |
| Leaf coalescing | Lazy queue-based cleanup |
| Memory reclamation | Seize-based epoch reclamation |

**Tests:** 731+ tests (466 unit + 88 ported from C++ reference + integration). Miri strict provenance clean.

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

## When to Use

**Best for:**
- Long keys with shared prefixes (URLs, file paths, UUIDs)
- Range scans over ordered data
- Mixed read/write workloads
- High-contention scenarios

**Consider alternatives for:**
- Unordered point lookups → `dashmap` (1.6-1.9x faster)
- Write-heavy at 6+ threads → `indexset` (better write scaling)
- Integer keys only → `congee` (ART-based)

## Variant Selection

Two variants are provided with different performance characteristics:

| Variant | Best For |
|---------|----------|
| `MassTree15` | Range scans, writes, shared-prefix keys, contention |
| `MassTree` (WIDTH=24) | Random-access reads, single-threaded point ops |

`MassTree15` wins **79% of benchmarks** due to cheaper u64 atomics and better cache utilization. Use it unless you have uniform random-access patterns.

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

6 physical cores, `mimalloc` allocator.

### Read Throughput (6 threads)

| Key Size | MassTree | SkipMap | IndexSet | TreeIndex |
|----------|----------|---------|----------|-----------|
| 8B | **64 Mitem/s** | 44 Mitem/s | 35 Mitem/s | 33 Mitem/s |
| 32B | **54 Mitem/s** | 18 Mitem/s | 17 Mitem/s | 17 Mitem/s |

### Write Performance (6 threads)

| Workload | MassTree | SkipMap | IndexSet | TreeIndex |
|----------|----------|---------|----------|-----------|
| Disjoint keys | **18 ms** | 28 ms | 87 ms | 18 ms |
| High contention | **8 ms** | 15 ms | 23 ms | 24 ms |

### Range Scans (6 threads)

| Scan Type | MassTree | IndexSet | TreeIndex |
|-----------|----------|----------|-----------|
| Sequential | **4.7 Mitem/s** | 0.6 Mitem/s | 3.2 Mitem/s |
| Long keys (64B) | **4.6 Mitem/s** | 0.6 Mitem/s | 3.1 Mitem/s |

MassTree is **31-55% faster** than TreeIndex and **6-8x faster** than IndexSet on range scans.

### vs C++ Reference

| Workload | Rust | C++ | Ratio |
|----------|------|-----|-------|
| 90% reads | 12.8 Mop/s | 13.8 Mop/s | 93% |
| 98% reads | 17.8 Mop/s | 18.9 Mop/s | 94% |
| Contention | 3.5 Mop/s | 2.6 Mop/s | **132%** |

Read path at parity with C++. Rust wins on contention handling.

## How It Works

Masstree splits keys into 8-byte chunks, creating a trie where each node is a B+tree:

```text
Key: "users/alice/profile" (19 bytes)
     └─ Layer 0: "users/al" (8 bytes)
        └─ Layer 1: "ice/prof" (8 bytes)
           └─ Layer 2: "ile" (3 bytes)
```

Keys with shared prefixes share upper layers, making lookups efficient for hierarchical data.

## Crate Features

- `mimalloc` — Use mimalloc as global allocator (recommended)
- `tracing` — Enable structured logging to `logs/masstree.jsonl`

## License

MIT. See `LICENSE`.

## References

- [Masstree Paper (EuroSys 2012)](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)
- [C++ Reference Implementation](https://github.com/kohler/masstree-beta)
