# masstree

A concurrent ordered map for Rust, based on the Masstree algorithm (trie of B+trees).

[![Crates.io](https://img.shields.io/crates/v/masstree.svg)](https://crates.io/crates/masstree)
[![Documentation](https://docs.rs/masstree/badge.svg)](https://docs.rs/masstree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Status: Alpha

**Not production ready.** Use for evaluation and feedback only.

| Feature | Status | Notes |
|---------|--------|-------|
| Concurrent get | Works | Lock-free, version-validated |
| Concurrent insert | Works | CAS fast path + locked fallback |
| Split propagation | Works | Leaf and internode splits at any level |
| Concurrency patterns | Tested | Loom tests core patterns (not full tree) |
| Linearizability | Tested | Shuttle tests simplified model |
| Memory safety | Verified | Miri strict provenance |
| Memory reclamation | Partial | Nodes not reclaimed until tree drop |
| Range scans | Missing | Planned for v0.2 |
| Deletion | Missing | Planned for v0.2 |
| Stress testing | Limited | Tested up to 8 threads |

**305 tests passing** (unit + integration + property + loom + shuttle)

## Why This Crate?

Rust lacks a popular concurrent ordered map. The ecosystem:

| Crate | Type | Downloads/month |
|-------|------|-----------------|
| `dashmap` | Unordered HashMap | 10.7 million |
| `crossbeam-skiplist` | Ordered SkipList | 220k |
| `masstree` | Ordered B+tree | *new* |

Skip lists have poor cache locality. B+trees allocate nodes in blocks, improving cache utilization. MassTree brings cache-efficient concurrent ordered maps to Rust.

## Quick Start

### Concurrent API (recommended)

```rust
use masstree::MassTree;
use std::sync::Arc;
use std::thread;

// MassTree is Send + Sync when V: Send + Sync
let tree = Arc::new(MassTree::<u64>::new());

// Spawn concurrent readers and writers
let handles: Vec<_> = (0..4).map(|i| {
    let tree = Arc::clone(&tree);
    thread::spawn(move || {
        let guard = tree.guard();

        // Insert with guard (fine-grained locking)
        let _ = tree.insert_with_guard(b"key", i, &guard);

        // get_ref returns &V (zero-copy, best for read-heavy workloads)
        if let Some(value) = tree.get_ref(b"key", &guard) {
            println!("Got: {}", *value);
        }

        // get_with_guard returns Arc<V> (if you need ownership beyond guard)
        let owned = tree.get_with_guard(b"key", &guard);
    })
}).collect();

for h in handles {
    let _ = h.join().unwrap();
}
```

### Single-threaded API

```rust
use masstree::MassTree;

let mut tree: MassTree<u64> = MassTree::new();

// insert() requires &mut self - not for concurrent use
let _ = tree.insert(b"alice", 100)?;
let _ = tree.insert(b"bob", 200)?;

// get() returns Arc<V>
assert_eq!(*tree.get(b"alice").unwrap(), 100);
# Ok::<(), masstree::tree::InsertError>(())
```

**Note:** Keys must be 0-256 bytes. Longer keys will panic.

## Performance

### Single-Threaded Reads (vs other concurrent maps)

Benchmarks from `cargo bench --bench concurrent_maps` (1000 lookups per iteration):

| Key Size | MassTree | IndexSet | SkipMap | MassTree Advantage |
|----------|----------|----------|---------|-------------------|
| 8 bytes  | 81 µs    | 192 µs   | 278 µs  | **2.4x / 3.4x faster** |
| 16 bytes | 95 µs    | 194 µs   | 281 µs  | **2.0x / 3.0x faster** |
| 32 bytes | 96 µs    | 186 µs   | 263 µs  | **1.9x / 2.7x faster** |

MassTree's multi-layer design has minimal overhead for longer keys (~18% slowdown from 8B to 32B).

### Single-Threaded Inserts

| Key Size | MassTree | SkipMap | IndexSet | Notes |
|----------|----------|---------|----------|-------|
| 8 bytes  | 71 µs    | 170 µs  | 536 µs   | **MassTree 2.4x faster** |
| 32 bytes | 226 µs   | 158 µs  | 523 µs   | SkipMap 1.4x faster (multi-layer cost) |

MassTree excels with short keys; longer keys require layer creation overhead.

## Concurrent Scaling

### Allocator Matters

MassTree's performance depends heavily on the memory allocator. **Use mimalloc for best results.**

```rust
// Add to your binary's main.rs
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

### Read Scaling with mimalloc (recommended)

| Threads | MassTree | SkipMap | IndexSet | Notes |
|---------|----------|---------|----------|-------|
| 1       | 5.3 ms   | 8.1 ms  | 6.9 ms   | **MassTree 1.5x faster** |
| 8       | 15.5 ms  | 16.1 ms | 16.3 ms  | **MassTree fastest** |
| 16      | 28.6 ms  | 28.6 ms | 28.8 ms  | Tie |
| 32      | 50.3 ms  | 56.0 ms | 55.0 ms  | **MassTree 10% faster** |

### Read Scaling with Standard Allocator

| Threads | MassTree | SkipMap | IndexSet | Notes |
|---------|----------|---------|----------|-------|
| 1       | 9.4 ms   | 13.1 ms | 11.6 ms  | MassTree fastest |
| 8       | 83.4 ms  | 47.7 ms | 53.6 ms  | SkipMap faster |
| 16      | 179 ms   | 106 ms  | 112 ms   | SkipMap 1.7x faster |

### Allocator Comparison (32 threads)

| Allocator | MassTree | vs Standard |
|-----------|----------|-------------|
| **mimalloc** | **50 ms** | **7.5x faster** |
| Standard  | 376 ms   | baseline |
| jemalloc  | 1001 ms  | 2.7x slower (avoid!) |

### Why Allocator Matters

The standard glibc allocator has lock contention under MassTree's allocation pattern. mimalloc uses segment-based allocation that scales much better. jemalloc has a pathological interaction—avoid it with MassTree.

## Architecture

MassTree implements the [Masstree algorithm](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf):

- **Trie of B+trees**: Keys split into 8-byte chunks, each navigating a B+tree layer
- **Optimistic reads**: Version validation, retry on concurrent modification
- **Fine-grained locking**: Per-leaf locks for writes, CAS for simple inserts
- **B-link structure**: Lock-free traversal during splits

```text
Key: "hello world!"
      ↓
Layer 0: "hello wo" → B+tree lookup
      ↓
Layer 1: "rld!\0\0\0\0" → B+tree lookup → Value
```

## Value Storage

Two modes:

```rust
// Default: Arc<V> for any value type
let tree: MassTree<MyStruct> = MassTree::new();

// Index mode: Copy types (planned optimization)
let index: MassTreeIndex<u64> = MassTreeIndex::new();
```

`Arc<V>` enables safe concurrent reads without guard lifetimes. The trade-off is atomic refcount overhead on access.

## Known Limitations

1. **Allocator-sensitive** - Requires mimalloc for best concurrent performance (see above)
2. **Memory grows monotonically** - Split nodes stay allocated until tree drop
3. **No range scans** - Ordered iteration not yet implemented
4. **No deletion** - Keys cannot be removed
5. **Insert overhead for long keys** - Multi-layer creation slower than SkipMap
6. **Key length limit** - Keys > 256 bytes will panic (not a Result error)

## Contributing

Feedback and bug reports welcome! This is an alpha release—expect rough edges.

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench --bench comparison

# Run with Miri (requires nightly)
cargo +nightly miri test
```

## References

- [Masstree Paper (EUROSYS 2012)](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)
- [C++ Implementation](https://github.com/kohler/masstree-beta)
- [seize (memory reclamation)](https://github.com/ibraheemdev/seize)

## Disclaimer

This is an **independent Rust implementation** of the Masstree data structure. It is **not affiliated with, endorsed by, or connected to** the original authors (Eddie Kohler, Yandong Mao, Robert Morris) or their institutions.

## License

MIT
