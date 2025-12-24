# masstree

A high-performance concurrent ordered map for Rust, supporting variable-length keys.

[![Crates.io](https://img.shields.io/crates/v/masstree.svg)](https://crates.io/crates/masstree)
[![Documentation](https://docs.rs/masstree/badge.svg)](https://docs.rs/masstree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What This Is

A Rust implementation of the [Masstree algorithm](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf) (Mao, Kohler, Morris — EUROSYS 2012). MassTree is a trie of B+trees designed for high-throughput concurrent access with variable-length keys.

## When to Use MassTree

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Integer keys (u64) | **Congee (ART)** | 2x faster, specialized for integers |
| Variable-length keys | **MassTree** | Congee can't do this |
| String keys | **MassTree** | Congee can't do this |
| Unordered, any keys | **DashMap** | Simpler, no ordering overhead |
| Single-threaded ordered | **BTreeMap** | No concurrency overhead |

**MassTree's niche**: When you need a concurrent ordered map with arbitrary byte-sequence keys and high read throughput.

## Status: Alpha

**Not production ready.** Core functionality works; APIs may change.

| Feature | Status |
|---------|--------|
| Concurrent get | Lock-free, version-validated |
| Concurrent insert | Works (CAS + locked fallback) |
| Split propagation | Leaf and internode splits |
| Memory safety | Miri strict provenance clean |
| Range scans | Planned |
| Deletion | Planned |

**341 tests passing** (unit, integration, property, loom, shuttle)

## Benchmarks

All benchmarks run on the same hardware with `--features mimalloc`. Median throughput at 32 threads.

### The Honest Comparison: MassTree vs Congee (ART)

For 8-byte integer keys, Congee wins decisively:

| Structure | 32 threads | Scaling (1→32) |
|-----------|------------|----------------|
| **Congee (ART)** | **291.8 Mitem/s** | 7.8x |
| MassTree | 143.1 Mitem/s | 6.6x |
| skiplist_guarded | 88.0 Mitem/s | 7.2x |
| DashMap | 75.2 Mitem/s | 2.7x |

**Why ART is faster**: Adaptive Radix Trees index directly by key bytes (no comparisons). For fixed-size integer keys, this is optimal. MassTree's B+tree layers add overhead that only pays off with variable-length keys.

### Where MassTree Wins

**32-byte keys** (Congee cannot participate — only supports usize):

| Structure | 32 threads | vs MassTree |
|-----------|------------|-------------|
| **MassTree** | **108 Mitem/s** | — |
| DashMap | 70.5 Mitem/s | 1.5x slower |
| skiplist_guarded | 40.5 Mitem/s | 2.7x slower |
| skipmap | 37.1 Mitem/s | 2.9x slower |
| indexset | 37.2 Mitem/s | 2.9x slower |

**vs Lock-wrapped BTreeMap** (8-byte keys, 32 threads):

| Structure | Throughput | vs MassTree |
|-----------|------------|-------------|
| **MassTree** | **143 Mitem/s** | — |
| RwLock\<BTreeMap\> | 36 Mitem/s | 4x slower |
| Mutex\<BTreeMap\> | 4 Mitem/s | 35x slower |

### vs Original C++ Masstree

Direct comparison using the same methodology as the [C++ reference implementation](https://github.com/kohler/masstree-beta)'s `rw1` test: 10M random i32 keys, shuffled read access, value verification.

| Threads | C++ Masstree | Rust MassTree | Difference |
|---------|--------------|---------------|------------|
| 1 | 1.83 Mops/s | 2.12 Mitem/s | +16% |
| 2 | 3.38 Mops/s | 4.15 Mitem/s | +23% |
| 4 | 5.71 Mops/s | 8.02 Mitem/s | +40% |
| 8 | 9.00 Mops/s | 14.05 Mitem/s | +56% |
| 16 | 11.79 Mops/s | 15.29 Mitem/s | +30% |
| 32 | 13.42 Mops/s | 18.04 Mitem/s | +34% |

Scaling: C++ 7.3x vs Rust 8.5x (1→32 threads).

**Why we're faster**: We're simpler, not more optimized. The C++ implementation supports features we lack (deletion, range scans, variable-length key suffixes), and each feature adds overhead. We also benefit from [seize](https://github.com/ibraheemdev/seize)'s Hyaline memory reclamation (simpler than C++'s epoch-based RCU) and mimalloc.

**Caveats**: This is one specific workload (read-heavy, shuffled access). The C++ implementation has been battle-tested for over a decade. We haven't done serious optimization work yet (no SIMD in hot path). These numbers may not generalize to your use case.

To reproduce:

```bash
# Rust
just apples

# C++ (from reference/ directory)
# Just clone the original repo: https://github.com/kohler/masstree-beta
./configure && make
./mttest -j32 -l 10000000 rw1
```

### Benchmark Caveats

These numbers are best-case for MassTree's design:

- Read benchmarks use `get_ref(&key, &guard)` with a **guard pinned once per thread**
- DashMap's `get()` creates a guard per call (no batch API available)
- `crossbeam-skiplist::SkipMap` pins/unpins internally per call
- Write-heavy benchmarks not shown — MassTree has known overhead for layer creation

### Summary

| Scenario | Winner |
|----------|--------|
| Integer keys, any thread count | Congee (2x faster) |
| Variable-length keys, high concurrency | MassTree |
| String keys, high concurrency | MassTree |
| Unordered hash map needs | DashMap |
| Single-threaded ordered map | std::collections::BTreeMap |

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

        // Insert with fine-grained locking
        tree.insert_with_guard(b"user:1000", i, &guard).unwrap();

        // Lock-free read
        if let Some(value) = tree.get_ref(b"user:1000", &guard) {
            println!("Got: {}", *value);
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }
```

### Use mimalloc for Best Performance

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

## How It Works

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

## Divergences from C++ Masstree

This is not a direct port. Key differences from the [original C++ implementation](https://github.com/kohler/masstree-beta):

| Aspect | C++ Masstree | Rust MassTree |
|--------|--------------|---------------|
| **Memory Reclamation** | Epoch-Based RCU | Hyaline via [seize](https://github.com/ibraheemdev/seize) |
| **Node Width** | 15 slots (u64 permuter) | 15 slots (planning 24 via u128) |

### Why Hyaline over Epoch-Based RCU?

The C++ implementation uses epoch-based reclamation where readers announce entry/exit from critical sections and memory is freed after all readers from an epoch have departed.

We use [seize](https://github.com/ibraheemdev/seize)'s Hyaline scheme which:

- Has simpler API (just `Guard` and `retire`)
- Provides better worst-case latency (no epoch advancement delays)
- Handles nested critical sections naturally
- Is the state-of-the-art for Rust concurrent data structures

### Planned: WIDTH=24 (Novel Optimization)

The original C++ and our current implementation use WIDTH=15 because the permutation encoding fits in a `u64` (15 slots × 4 bits + 4 bits size = 64 bits).

We're implementing WIDTH=24 using `u128` storage via the [`portable-atomic`](https://crates.io/crates/portable-atomic) crate:

- 24 slots × 5 bits + 5 bits size = 125 bits (fits in u128)
- 60% more capacity per node = ~40% fewer splits
- Reduces lock contention under high thread counts

This optimization wasn't practical for the 2012 C++ implementation but is feasible now with modern 128-bit atomic support.

## Limitations

1. **No deletion** — keys cannot be removed (planned)
2. **No range scans** — ordered iteration not yet implemented (planned)
3. **Integer keys** — use Congee instead, it's 2x faster
4. **Max key length** — 256 bytes
5. **Memory** — nodes freed only when tree drops (full reclamation planned)

## Running Benchmarks

```bash
# Full comparison (recommended)
cargo bench --bench concurrent_maps --features mimalloc

# Just the throughput scaling benchmarks
cargo bench --bench concurrent_maps --features mimalloc -- 08a_read_scaling
cargo bench --bench concurrent_maps --features mimalloc -- 08b_read_scaling

# vs lock-wrapped BTreeMap
cargo bench --bench lock_comparison --features mimalloc
```

## References

- [Masstree Paper (EUROSYS 2012)](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)
- [C++ Reference Implementation](https://github.com/kohler/masstree-beta)
- [Congee — Concurrent ART](https://github.com/XiangpengHao/congee)
- [seize — Hyaline Memory Reclamation](https://github.com/ibraheemdev/seize)

## License

MIT

## Disclaimer

Independent Rust implementation. Not affiliated with or endorsed by the original Masstree authors.
