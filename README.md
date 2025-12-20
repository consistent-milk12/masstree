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

        // Get with guard (lock-free, version-validated)
        tree.get_with_guard(b"key", &guard)
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

Benchmarks vs `BTreeMap` (single-threaded, `cargo bench --bench comparison`):

| Operation | MassTree | BTreeMap | Ratio |
|-----------|----------|----------|-------|
| Get (hit, n=1000) | 21 ns | 54 ns | **2.6x faster** |
| Get (miss, n=1000) | 9 ns | 91 ns | **10x faster** |
| Insert (populated) | 128 ns | 181 ns | **1.4x faster** |
| Mixed 90/10 r/w | 755 ns | 1.87 µs | **2.5x faster** |

**Where BTreeMap wins:**

| Operation | MassTree | BTreeMap | Notes |
|-----------|----------|----------|-------|
| Insert (empty) | 91 ns | 15 ns | Higher fixed overhead |
| Update existing | 478 ns | 141 ns | Arc swap cost |

## Concurrent Scaling

Tested against DashMap (note: different data structure category):

| Threads | MassTree | DashMap | Notes |
|---------|----------|---------|-------|
| 2 (reads) | 1.24ms | 1.90ms | MassTree 35% faster |
| 8 (reads) | 5.17ms | 5.49ms | MassTree 6% faster |
| 8 (high contention) | 4.34ms | 2.44ms | DashMap 44% faster |

MassTree scales well for reads. High write contention favors DashMap's sharded design.

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

1. **Memory grows monotonically** - Split nodes stay allocated until tree drop
2. **No range scans** - Ordered iteration not yet implemented
3. **No deletion** - Keys cannot be removed
4. **Update overhead** - `Arc` swap is slower than in-place mutation
5. **Limited stress testing** - Verified up to 8 threads
6. **Key length limit** - Keys > 256 bytes will panic (not a Result error)

## Roadmap

- **v0.1.0** (current): Core concurrent get/insert
- **v0.2.0**: Range scans, deletion, seize retirement
- **v0.3.0**: Performance optimization, stress testing
- **v1.0.0**: Production ready

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
