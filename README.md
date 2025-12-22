# masstree

A high-performance concurrent ordered map for Rust.

[![Crates.io](https://img.shields.io/crates/v/masstree.svg)](https://crates.io/crates/masstree)
[![Documentation](https://docs.rs/masstree/badge.svg)](https://docs.rs/masstree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Rust has `dashmap` for concurrent hash maps (millions of downloads/month), but no high-perm concurrent ordered map crates. The current options are (to my knowledge):

| Approach | Problem |
|----------|---------|
| `Mutex<BTreeMap>` | Serializes all access |
| `RwLock<BTreeMap>` | Reader contention on lock acquisition |
| `crossbeam-skiplist` | Poor cache locality, slower than B+trees |

MassTree fills this gap with a cache-efficient B+tree that scales.

## Benchmark Highlights

Two benchmark suites: `lock_comparison` (vs `Mutex`/`RwLock<BTreeMap>`) and `concurrent_maps` (vs `crossbeam-skiplist`).

### Where MassTree Does Well

**Concurrent read throughput (32 threads):**

| Structure | Throughput | vs MassTree |
|-----------|------------|-------------|
| **MassTree** | **140.7 Mitem/s** | — |
| RwLock<BTreeMap> | 36.1 Mitem/s | 3.9x slower |
| Mutex<BTreeMap> | 4.0 Mitem/s | 35x slower |

**Long keys with deep shared prefixes (16 common bytes), concurrent reads at 16 threads:**

| Structure | Median | vs MassTree |
|-----------|--------|-------------|
| **MassTree** | **1.46 ms** | — |
| RwLock<BTreeMap> | 3.13 ms | 2.1x slower |
| Mutex<BTreeMap> | 34.8 ms | 24x slower |

**Single-threaded lookups scale better with key length:**

| Key Size | MassTree | RwLock | Winner |
|----------|----------|--------|--------|
| 32B (no prefix) | **88.5 µs** | 105.0 µs | MassTree +19% |
| 24B (deep prefix) | **87.6 µs** | 138.6 µs | MassTree +58% |
| 32B (deep prefix) | **97.5 µs** | 116.5 µs | MassTree +19% |

### Where MassTree Is Slower

**Short keys (≤16B), single-threaded lookups:**

| Key Size | MassTree | RwLock | Difference |
|----------|----------|--------|------------|
| 8B | 70.8 µs | **60.5 µs** | MassTree 17% slower |
| 16B | 82.2 µs | **65.2 µs** | MassTree 26% slower |

The trie structure adds overhead for keys that fit in a single 8-byte slice. This is inherent to the algorithm—each lookup must traverse the trie even for short keys.

**Mixed read/write workloads with Zipfian distribution:**

| Threads | MassTree | SkipMap | Difference |
|---------|----------|---------|------------|
| 1 | 9.75 ms | **1.91 ms** | MassTree 5x slower |
| 16 | 20.85 ms | **6.00 ms** | MassTree 3.5x slower |

Write-heavy hot spots expose layer creation overhead for long keys.

### Concurrent Scaling Comparison

Read scaling from 1 to 32 threads:

| Threads | MassTree | RwLock | Mutex |
|---------|----------|--------|-------|
| 1 | 1.26 ms | 1.20 ms | 1.59 ms |
| 8 | **2.19 ms** | 3.43 ms | 26.0 ms |
| 16 | **2.94 ms** | 5.60 ms | 53.0 ms |
| 32 | **5.21 ms** | 9.77 ms | 109 ms |

MassTree's optimistic reads avoid lock acquisition entirely—readers never block each other.

### Summary

This should be possible, after it is properly implemented with all features and rigorously stress tested for correctness and subtle concurrency issues (there's currently one that's getting actively worked on and is blocking concurrent writes, but I have concrete plan in mind to fix it). I WOULDN'T RECOMMEND USING THIS CRATE FOR ANYTHING CURRENTLY.

| Use Case | Recommendation |
|----------|----------------|
| Short keys (≤16B), single-threaded | Use `BTreeMap` |
| Long keys (24B+), any thread count | **Use MassTree** |
| High read concurrency (8+ threads) | **Use MassTree** |
| Write-heavy Zipfian workloads | Use `crossbeam-skiplist` |
| Mixed workloads, shared key prefixes | **Use MassTree** |

## Status: Alpha

**Not production ready at all.** Core functionality works; APIs may change.

| Feature | Status |
|---------|--------|
| Concurrent get |  Lock-free, version-validated |
| Concurrent insert |  CAS fast path + locked fallback |
| Split propagation |  Leaf and internode splits |
| Memory safety |  Miri strict provenance |
| Range scans | Planned |
| Deletion | Planned |
| Full memory reclamation | Planned (using `seize`) |

**305 tests passing** (unit, integration, property, loom, shuttle)

## Quick Start

```rust
use masstree::MassTree;
use std::sync::Arc;
use std::thread;

let tree = Arc::new(MassTree::<u64>::new());

let handles: Vec<_> = (0..8).map(|i| {
    let tree = Arc::clone(&tree);
    thread::spawn(move || {
        let guard = tree.guard();

        // Insert (fine-grained locking, not serialized)
        tree.insert_with_guard(b"user:1000", i, &guard).unwrap();

        // Read (lock-free, returns &V)
        if let Some(value) = tree.get_ref(b"user:1000", &guard) {
            println!("Got: {}", *value);
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }
```

### Use mimalloc for Best Performance

MassTree's allocation pattern benefits significantly from mimalloc:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

Single-threaded improvement: **17% faster**. The gap widens under contention.

## How It Works

MassTree implements the [Masstree algorithm](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EUROSYS 2012):

**Trie of B+trees**: Keys are split into 8-byte slices. Each slice navigates a B+tree layer. Longer keys chain through layers.

```text
Key: "hello world!"

Layer 0: "hello wo" → B+tree lookup
                ↓
Layer 1: "rld!\0\0\0\0" → B+tree lookup → Value
```

**Optimistic concurrency**: Readers check version numbers before and after reading. If a concurrent write occurred, they retry. No locks acquired for reads.

**Fine-grained writes**: Writers lock only the target leaf node. CAS fast-path for simple inserts avoids locking entirely when possible.

**B-link trees**: Split operations use sibling pointers for lock-free traversal, allowing reads to proceed during structural modifications.

**Memory Reclamation**: The `NodeAllocator` trait abstracts memory management, designed for integration with [seize](https://github.com/ibraheemdev/seize)'s hyaline-based reclamation scheme.

## Limitations

1. **No deletion** — keys cannot be removed (planned)
2. **No range scans** — ordered iteration not yet implemented (planned)
3. **Arena reclamation** — nodes freed only when tree drops
4. **Key length** — max 256 bytes, longer keys panic
5. **Long key insert cost** — multi-layer creation adds overhead vs skip lists

## Running Benchmarks

```bash
# Recommended: use mimalloc
cargo bench --bench concurrent_maps --features mimalloc
cargo bench --bench lock_comparison --features mimalloc

# Compare against standard allocator
cargo bench --bench concurrent_maps

# Run tests
cargo test

# Miri (requires nightly)
cargo +nightly miri test
```

## Contributing

This project needs help with:

- **Permutation Freeze** - Currently working this. Please check the open issues.
- **Deletion + memory reclamation** — integrate seize for safe concurrent deletes
- **Range scans** — ordered iteration like `BTreeMap::range()`
- **Benchmarking** — more workloads, different hardware, methodology validation
- **Stress testing** — adversarial concurrent patterns

## References

- [Masstree: A Cache-Friendly Mashup of Tries and B-Trees](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (EUROSYS 2012)
- [C++ Reference Implementation](https://github.com/kohler/masstree-beta)
- [seize — Hyaline Memory Reclamation](https://github.com/ibraheemdev/seize)

## Disclaimer

Independent Rust implementation of the Masstree algorithm. Not affiliated with or endorsed by the original authors.

## License

MIT
