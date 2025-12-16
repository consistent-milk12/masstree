# masstree

An experimental Rust implementation of the Masstree algorithm, a high-performance concurrent key-value store based on a cache-friendly trie of B+trees.

This project **attempts to reimplement** the [original C++ Masstree](https://github.com/kohler/masstree-beta) developed at MIT in Rust, with planned divergences for safety, ergonomics, and Rust idioms. While the core algorithm remains the same, this implementation introduces value lifetime management via `Arc<V>`, type-state locking, and modern concurrency primitives.

**Status:** Phase 1 complete - single-threaded core with `get`/`insert`/split operations working. Multi-layer keys and concurrency planned for Phase 2-3.

## Overview

Masstree is a high-performance concurrent trie of B+trees designed for in-memory key-value storage. It combines the cache efficiency of B+trees with the no-rebalancing property of tries by slicing keys into 8-byte chunks, where each chunk navigates a separate B+tree layer.

## Disclaimer

This is an **independent Rust implementation** of the Masstree data structure. It is **not affiliated with, endorsed by, or connected to** the original authors (Eddie Kohler, Yandong Mao, Robert Morris) or their institutions (Harvard College, MIT, University of California). This project is a study and reimplementation of the published algorithm for educational and practical use in Rust projects.

## Why Rust?

The original Masstree is a very interesting and performant concurrent data structure, but its C++ implementation relies on manual memory management, platform-specific atomics, and subtle pointer tricks that make it difficult to extend or verify. Rust's ownership model and type system offer an opportunity to express similar algorithms with compile-time safety guarantees, eliminate entire classes of concurrency bugs, and produce a codebase that's easier to audit and maintain.

**This is not a faithful port.** The implementation diverges in meaningful ways to leverage Rust's strengths and work within its constraints. Performance targets are aspirational—initial focus is on correctness, safety, and learning the algorithm deeply.

## Value Storage Strategy

### How C++ Handles This

The original C++ Masstree is fully generic via templates, not limited to index-style values. The `leafvalue` class uses a union that can store any `value_type` inline (if pointer-sized) or as a pointer to external data. The codebase includes implementations for `value_bag` (database rows), `value_string`, `value_array`, and raw `uint64_t`.

C++ can return raw pointers or references directly because:

- Callers hold a "critical section" via `threadinfo` passed to every operation
- Using a reference after releasing is undefined behavior, but C++ allows it
- Memory safety is the caller's responsibility, enforced by convention

### Why Rust Needs a Different Approach

In Rust, we can't return `&V` from an optimistic read:

- The borrow checker requires proof that references outlive their use
- Nodes can be reclaimed by epoch-based GC while the caller still holds `&V`
- Guard-tied lifetimes are possible but create a complex API

### Proposed Solution: Dual Storage Modes

**Default Mode: `MassTree<V>` with `Arc<V>`**

Values are stored as `Arc<V>` (atomic reference-counted pointers). On read, the `Arc` is cloned (a cheap atomic increment), giving the caller an owned handle that survives node reclamation. This decouples value lifetime from the epoch-based memory reclamation used for nodes.

- Works with any `V: Send + Sync + 'static`
- No `Clone` requirement on `V` for reads
- Small overhead from atomic refcount operations

**Index Mode: `MassTreeIndex<V: Copy>`**

For performance-critical use cases with small, copyable values (`u64` handles, pointers, fixed-size structs), an index variant stores values inline. Reads copy the value directly from the slot, avoiding the `Arc` indirection entirely.

- Maximum throughput for index-style workloads
- Requires `V: Copy`
- Best suited for database indexes, handle maps, and similar patterns

This dual-mode approach provides equivalent flexibility to the C++ implementation, expressed through Rust's type system rather than raw pointers.

## Divergences from the C++ Implementation

Both implementations use epoch-based reclamation (EBR) for node memory safety—C++ via `threadinfo`, this implementation via `crossbeam-epoch` (if implemented as planned). The difference lies in **value lifetime management**:

| Aspect | C++ Masstree | Rust Masstree |
|--------|--------------|---------------|
| Node safety | EBR (manual `threadinfo&` passing) | EBR (epoch pinning is internal) |
| Value safety | User's responsibility | `Arc<V>` handles automatically |
| Misuse | Silent undefined behavior | Compile error or safe behavior |
| API burden | Must pass `threadinfo&` to every call | Clean API, no manual tracking |

**Key differences:**

1. **Values can't outlive their storage** — In C++, storing a `char*` to heap data, deleting the entry, then accessing the pointer is silent UB. With `Arc<V>`, the data lives until the last reference drops.

2. **No manual coordination** — C++ users must ensure value cleanup happens after all readers finish. `Arc`'s refcount handles this automatically.

3. **Composable safety** — `MassTree<Vec<String>>` would just work. In C++, complex value types need careful RCU-aware destructor implementations.

4. **Zero-cost when not needed** — `MassTreeIndex<u64>` would provide C++ equivalent performance for simple cases where `Arc` overhead matters.

## Current Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| `key` | Complete | 8-byte ikey extraction, layer traversal, suffix handling |
| `permuter` | Complete | Const-generic WIDTH, u64-encoded slot permutation |
| `nodeversion` | Complete | Versioned lock with type-state guard pattern (single-threaded) |
| `leaf` | Complete | LeafNode struct, split operations, B-link pointers, slot-0 rule |
| `internode` | Complete | Routing nodes, split-with-insert, height-based child typing |
| `ksearch` | Complete | Binary search for leaves and internodes |
| `tree` | Complete | `MassTree` with `get`/`insert`, split propagation, arena allocation |
| Scan/Remove | Planned | Range scans and key deletion not yet implemented |

**Phase 1 Constraints:**
- Keys limited to 0-8 bytes (single-layer only)
- Single-threaded (no concurrent access)
- Arena-based allocation (no node reclamation)

**Note on `MassTreeIndex`:** The current `MassTreeIndex<V: Copy>` is a convenience wrapper that still uses `Arc<V>` internally. True inline storage for `V: Copy` values is planned but not yet implemented.
