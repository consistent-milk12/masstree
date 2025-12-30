# masstree

`masstree` is a beta high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable length keys by building a trie of small B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EuroSys 2012).

This release is published as `0.2.2`. The crate is feature-complete for core operations (get, insert, range scans) but still being validated for correctness and performance under high contention.

This crate does a lot of allocation. In my testing, the default global allocator can be much slower than `mimalloc` for these patterns. The C++ Masstree codebase uses a custom allocator, and this Rust port does not have an equivalent yet.

**Disclaimer:** This is an independent learning project. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions (MIT PDOS, Harvard).

## What it is

- Ordered map for byte keys, ordered by lexicographic byte order
- Concurrent reads with version validation, no read locks
- Concurrent inserts with fine-grained leaf locking
- Zero-copy range scans with weakly consistent iteration
- Variable length keys (default limit: 256 bytes, configurable)

If you only need `u64` keys, an ART like `congee` can be faster. If you do not need ordering, a hash map like `dashmap` can be simpler.

## Status

This crate is in active development and still changing.

Implemented:

- `get`, `get_with_guard`, and `get_ref` — lock-free reads with version validation
- `insert` and `insert_with_guard` — fine-grained leaf locking
- `scan`, `scan_ref`, and `scan_prefix` — zero-copy range iteration
- Leaf and internode splits with proper B-link tree semantics

Not implemented yet:

- Deletion (planned for 0.3.0)
- Keys longer than 256 bytes (configurable limit, currently panics)

### Version roadmap

| Version | Features |
|---------|----------|
| 0.1.x | Initial implementation (get, insert, splits) |
| 0.2.0 | Range scans (`scan`, `scan_ref`, `scan_prefix`) |
| **0.2.2** | Box elimination, optimized range scans |
| 0.3.0 | Deletion (planned) |

## Install

Add this to your `Cargo.toml`:

```toml
[dependencies]
masstree = { version = "0.2.2", features = ["mimalloc"] }
```

MSRV is Rust `1.92`.

The `mimalloc` feature sets the global allocator for your whole program. If your project already selects a global allocator, leave this feature off and configure `mimalloc` at the binary level instead.

## Quick start

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

// Insert
tree.insert_with_guard(b"hello", 123, &guard).unwrap();
tree.insert_with_guard(b"world", 456, &guard).unwrap();

// Point lookup
assert_eq!(tree.get_ref(b"hello", &guard), Some(&123));

// Range scan (zero-copy)
let mut sum = 0u64;
tree.scan_ref(b"h".., |_key, value| {
    sum += *value;
    true  // continue scanning
});
assert_eq!(sum, 123 + 456);
```

Notes:

- `get()` returns an `Arc<V>` for `MassTree<V>`. For read-heavy workloads, prefer `get_ref()` which avoids the Arc clone overhead.
- `scan_ref()` provides zero-copy access to keys and values. Use `scan()` if you need owned values.

## Benchmarks

These numbers are from `runs/run29_point_ops_optimized.md` (point operations, 6 physical cores) and `runs/run30_range_scans_optimized.md` (range scans). The tables show median results.

### Point Operations

Read throughput at 6 threads:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| Read scaling (8B keys) | **64.2 Mitem/s** | 44.0 Mitem/s | 34.9 Mitem/s | 33.2 Mitem/s |
| Read scaling (32B keys) | **53.9 Mitem/s** | 18.0 Mitem/s | 17.1 Mitem/s | 16.7 Mitem/s |

Write benchmarks at 6 threads, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| Concurrent writes (disjoint) | **18.3 ms** | 28.4 ms | 86.6 ms | 18.3 ms |
| Concurrent writes (contention) | **8.08 ms** | 14.9 ms | 23.1 ms | 23.5 ms |

Single threaded insert, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| Single-threaded insert | **11.5 ms** | 13.3 ms | 41.7 ms | 17.7 ms |

### Range Scans

Scan throughput at 6 threads:

| Benchmark | `MassTree` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- |
| Sequential full scan | **4.69 Mitem/s** | 0.63 Mitem/s | 3.22 Mitem/s |
| Reverse scan | **4.57 Mitem/s** | 0.60 Mitem/s | 3.13 Mitem/s |
| Long keys (64B) scan | **4.60 Mitem/s** | 0.57 Mitem/s | 3.11 Mitem/s |

MassTree outperforms TreeIndex by **31-55%** on range scans and is **6-8x faster** than IndexSet.

### Similar structures used in benchmarks

- `MassTree` from this crate
- `SkipMap` from `crossbeam-skiplist`
- `IndexSet` from `indexset`
- `TreeIndex` from `scc`

To reproduce the benchmark suite in this repo:

```bash
cargo bench --bench concurrent_maps24 --features mimalloc  # Point operations
cargo bench --bench range_concurrent --features mimalloc   # Range scans
```

## How keys work

Masstree splits each key into 8 byte chunks. Each chunk is handled by a B+tree layer. When keys share prefixes, they share the earlier layers.

This crate currently uses 24 slot leaf nodes. That reduces split frequency, but it requires a `u128` permutation (via `portable-atomic`) and it is still being tuned.

## Features

- `tracing`: enables structured tracing to `logs/masstree.jsonl`
- `mimalloc`: uses `mimalloc` as the global allocator, recommended for performance in this crate

## License

MIT. See `LICENSE`.
