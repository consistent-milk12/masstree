# masstree

`masstree` is an alpha concurrent ordered map for Rust. It stores keys as `&[u8]` and supports variable length keys by building a trie of small B+trees, based on the [Masstree paper](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EuroSys 2012).

This release is published as `0.1.7`. It is not production ready yet. I am still validating correctness and performance under high contention.

This crate does a lot of allocation. In my testing, the default global allocator can be much slower than `mimalloc` for these patterns. The C++ Masstree codebase uses a custom allocator, and this Rust port does not have an equivalent yet.

**Disclaimer:** This is an independent learning project. It is not endorsed by, affiliated with, or connected to the original Masstree authors or their institutions (MIT PDOS, Harvard).

## What it is

- Ordered map for byte keys, ordered by lexicographic byte order
- Concurrent reads with version validation, no read locks
- Concurrent inserts with fine grained leaf locking
- Variable length keys up to 256 bytes

If you only need `u64` keys, an ART like `congee` can be faster. If you do not need ordering, a hash map like `dashmap` can be simpler.

## Status

This crate is in active development and still changing.

Implemented:

- `get`, `get_with_guard`, and `get_ref`
- `insert` and `insert_with_guard` for updates and new keys
- Leaf and internode splits

Not implemented yet:

- Range scans
- Deletion
- Keys longer than 256 bytes (currently panics)

## Install

Add this to your `Cargo.toml`:

```toml
[dependencies]
masstree = { version = "0.1.7", features = ["mimalloc"] }
```

MSRV is Rust `1.92`.

The `mimalloc` feature sets the global allocator for your whole program. If your project already selects a global allocator, leave this feature off and configure `mimalloc` at the binary level instead.

## Quick start

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

tree.insert_with_guard(b"hello", 123, &guard).unwrap();
assert_eq!(tree.get_ref(b"hello", &guard), Some(&123));
```

Notes:

- `get()` returns an `Arc<V>` for `MassTree<V>`. For read-heavy workloads, prefer `get_ref()` which avoids the Arc clone overhead.

## Benchmarks

These numbers are only here as early context. They are from `runs/run13_atomic.md` using the `concurrent_maps24` benchmark suite. The tables below show median results from that run.

Read throughput at 32 threads:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `10a_read_scaling_8B` | 82.51 Mitem/s | 70.66 Mitem/s | 53.89 Mitem/s | 52.89 Mitem/s |
| `10b_read_scaling_32B` | 73.78 Mitem/s | 30.73 Mitem/s | 33.92 Mitem/s | 26.50 Mitem/s |

Write benchmarks at 32 threads, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `01_concurrent_writes_disjoint` | 59.83 ms | 110.7 ms | 174 ms | 60.21 ms |
| `02_concurrent_writes_contention` | 57.38 ms | 55.92 ms | 293.6 ms | 85.95 ms |

Single threaded insert, median time per run:

| Benchmark | `MassTree` | `SkipMap` | `IndexSet` | `TreeIndex` |
| --- | --- | --- | --- | --- |
| `03_single_threaded_insert` | 8.966 ms | 12.66 ms | 42.03 ms | 17.88 ms |

The comparison set in that benchmark file uses:

- `MassTree` from this crate
- `SkipMap` from `crossbeam-skiplist`
- `IndexSet` from `indexset`
- `TreeIndex` from `scc`

To reproduce the benchmark suite in this repo:

```bash
cargo bench --bench concurrent_maps24 --features mimalloc
```

## How keys work

Masstree splits each key into 8 byte chunks. Each chunk is handled by a B+tree layer. When keys share prefixes, they share the earlier layers.

This crate currently uses 24 slot leaf nodes. That reduces split frequency, but it requires a `u128` permutation (via `portable-atomic`) and it is still being tuned.

## Features

- `tracing`: enables structured tracing to `logs/masstree.jsonl`
- `mimalloc`: uses `mimalloc` as the global allocator, recommended for performance in this crate

## License

MIT. See `LICENSE`.
