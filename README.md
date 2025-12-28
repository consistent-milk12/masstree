# masstree

`masstree` is a beta high-performance concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable length keys by building a trie of small B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EuroSys 2012).

This release is published as `0.2.0`. The crate is feature-complete for core operations (get, insert, range scans) but still being validated for correctness and performance under high contention.

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
| **0.2.0** | Range scans (`scan`, `scan_ref`, `scan_prefix`) |
| 0.3.0 | Deletion (planned) |

## Install

Add this to your `Cargo.toml`:

```toml
[dependencies]
masstree = { version = "0.2.0", features = ["mimalloc"] }
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

These numbers are from `runs/run23_point_ops.md` (point operations, 6 physical cores) and `runs/run20_range.md` (range scans). The tables show median results.

### Point Operations

Read throughput at 6 threads:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `10a_read_scaling_8B` | **86.7 Mitem/s** | 42.6 Mitem/s | 31.9 Mitem/s | 30.7 Mitem/s |
| `10b_read_scaling_32B` | **45.0 Mitem/s** | 17.6 Mitem/s | 17.1 Mitem/s | 16.7 Mitem/s |

Write benchmarks at 6 threads, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `01_concurrent_writes_disjoint` | **17.5 ms** | 28.0 ms | 80.3 ms | 18.1 ms |
| `02_concurrent_writes_contention` | **7.6 ms** | 14.6 ms | 21.1 ms | 22.7 ms |

Single threaded insert, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `03_single_threaded_insert` | **8.5 ms** | 12.5 ms | 42.0 ms | 17.9 ms |

### Range Scans

Scan throughput at 6 threads (10K entries scanned per operation):

| Benchmark | `MassTree` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- |
| `01_sequential_full_scan` | **4.46 Mitem/s** | 0.61 Mitem/s | 3.56 Mitem/s |
| `02_reverse_scan` | **4.46 Mitem/s** | 0.60 Mitem/s | 3.41 Mitem/s |
| `12_long_keys_64b_scan` | **4.29 Mitem/s** | 0.66 Mitem/s | 3.41 Mitem/s |

MassTree outperforms TreeIndex by 25-30% on range scans for short keys (≤8 bytes) and matches it for long keys.

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
