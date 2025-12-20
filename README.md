# masstree

An experimental Rust implementation of the Masstree algorithm, a high-performance concurrent key-value store based on a cache-friendly trie of B+trees.

This project **attempts to reimplement** the [original C++ Masstree](https://github.com/kohler/masstree-beta)

**Status:** Phase 3 in progress - concurrent `get` and `insert` with version-validated optimistic reads and locked writes. Split propagation works under contention. Loom and Shuttle tests verify linearizability. Memory reclamation and lock-free linking still TODO.

## Overview

Masstree is a high-performance concurrent trie of B+trees designed for in-memory key-value storage. It combines the cache efficiency of B+trees with the no-rebalancing property of tries by slicing keys into 8-byte chunks, where each chunk navigates a separate B+tree layer.

**Same algorithm, different implementation:** This project implements the same core algorithm as the C++ original: trie of B+trees, optimistic concurrency with version validation, B-link structure for lock-free traversal, permuter-based slot ordering. The divergences (Hyaline reclamation, `Arc<V>` values, type-state locking) are implementation choices that leverage Rust's strengths while preserving the algorithmic design.

## Disclaimer

This is an **independent Rust implementation** of the Masstree data structure. It is **not affiliated with, endorsed by, or connected to** the original authors (Eddie Kohler, Yandong Mao, Robert Morris) or their institutions (Harvard College, MIT, University of California). This project is a study and reimplementation of the published algorithm for educational and practical use in Rust projects.

## Why Rust?

The original Masstree is a very interesting and performant concurrent data structure, but its C++ implementation relies on manual memory management, platform-specific atomics, and subtle pointer tricks that make it difficult to extend or verify. Rust's ownership model and type system offer an opportunity to express similar algorithms with compile-time safety guarantees, eliminate entire classes of concurrency bugs, and produce a codebase that's easier to audit and maintain.

**This is not a faithful port.** The implementation diverges in meaningful ways to leverage Rust's strengths and work within its constraints. The focus of this project is on correctness, safety, and learning the algorithm deeply.

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
- Nodes can be reclaimed by deferred reclamation while the caller still holds `&V`
- Guard-tied lifetimes are possible but create a complex API

### Proposed Solution: Dual Storage Modes

**Default Mode: `MassTree<V>` with `Arc<V>`**

Values are stored as `Arc<V>` (atomic reference-counted pointers). On read, the `Arc` is cloned (a cheap atomic increment), giving the caller an owned handle that survives node reclamation. This decouples value lifetime from the deferred memory reclamation used for nodes.

- Works with any `V: Send + Sync + 'static`
- No `Clone` requirement on `V` for reads
- Small overhead from atomic refcount operations

**Index Mode: `MassTreeIndex<V: Copy>`**

For performance-critical use cases with small, copyable values (`u64` handles, pointers, fixed-size structs), an index variant stores values inline. Reads copy the value directly from the slot, avoiding the `Arc` indirection entirely.

- Maximum throughput for index-style workloads
- Requires `V: Copy`
- Best suited for database indexes, handle maps, and similar patterns

This dual-mode approach provides equivalent flexibility to the C++ implementation, expressed through Rust's type system rather than raw pointers.

## Memory Reclamation: Hyaline vs Classic Epoch-Based

The original C++ Masstree uses classic epoch-based reclamation (EBR) via `threadinfo` and a global epoch counter (`globalepoch`). Every operation takes a `threadinfo&` parameter, and deferred nodes are stored in per-thread "limbo lists" tagged with the current epoch. Periodically, `rcu_quiesce()` scans all threads to find the minimum active epoch and frees everything older.

This Rust implementation will use **hyaline reclamation** via the [`seize`](https://github.com/ibraheemdev/seize) crate instead (see the [Hyaline paper](https://arxiv.org/pdf/2108.02763.pdf)). The key differences:

| Aspect | C++ EBR (`kvthread.hh`) | Rust Hyaline (`seize`) |
|--------|-------------------------|------------------------|
| **Epoch coordination** | Global counter, all threads sync | Per-thread, no global barrier |
| **Stalled threads** | Block reclamation for everyone | Filtered out via epoch tracking |
| **Reclamation timing** | Batch (every 128 frees) | Balanced across threads |
| **Latency distribution** | Unpredictable spikes | Predictable, smoother |
| **API burden** | Pass `threadinfo&` everywhere | Internal guard management |

**Why this matters for Masstree:**

Masstree is read-heavy with optimistic readers that rarely block. Classic EBR's batch reclamation (triggered every N operations) causes latency spikes that disrupt the otherwise smooth read path. Hyaline's per-thread balancing keeps reclamation work distributed, avoiding these spikes.

Additionally, seize handles "stalled" threads gracefully: a thread blocked on I/O won't prevent other threads from reclaiming memory, unlike the C++ implementation where one slow thread blocks `min_active_epoch()` for everyone.

## Divergences from the C++ Implementation

Both implementations use deferred reclamation for node memory safety (C++ via `threadinfo`, Rust via `seize`). The difference lies in **value lifetime management**:

| Aspect | C++ Masstree | Rust Masstree |
|--------|--------------|---------------|
| Node safety | EBR (manual `threadinfo&` passing) | Hyaline via `seize` (internal) |
| Value safety | User's responsibility | `Arc<V>` handles automatically |
| Misuse | Silent undefined behavior | Compile error or safe behavior |
| API burden | Must pass `threadinfo&` to every call | Clean API, no manual tracking |

**Key differences:**

1. **Values can't outlive their storage**: In C++, storing a `char*` to heap data, deleting the entry, then accessing the pointer is silent UB. With `Arc<V>`, the data lives until the last reference drops.

2. **No manual coordination**: C++ users must ensure value cleanup happens after all readers finish. `Arc`'s refcount handles this automatically.

3. **Composable safety**: `MassTree<Vec<String>>` just works. In C++, complex value types need careful RCU-aware destructor implementations.

4. **Zero-cost when not needed**: `MassTreeIndex<u64>` provides C++-equivalent performance for simple cases where `Arc` overhead matters.

## Performance

Benchmarks compare MassTree against `std::collections::BTreeMap` using identical key generation and access patterns. Run `cargo bench --bench comparison` to reproduce.

### Where MassTree Performs Well

| Operation | Tree Size | MassTree | BTreeMap | Ratio |
|-----------|-----------|----------|----------|-------|
| Insert (populated) | 1000 | 104 ns | 156 ns | 1.5x faster |
| Get (hit) | 1000 | 22 ns | 60 ns | 2.7x faster |
| Get (miss) | 1000 | 17 ns | 104 ns | 6.1x faster |
| Batch get (500 keys) | 500 | 10 µs | 36 µs | 3.6x faster |
| Mixed 90/10 read/write | 1000 | 0.6 µs | 1.9 µs | 3.2x faster |

### Where BTreeMap Performs Well

| Operation | Tree Size | MassTree | BTreeMap | Ratio |
|-----------|-----------|----------|----------|-------|
| Insert (empty tree) | 1 | 52 ns | 23 ns | BTreeMap 2.3x faster |

MassTree has higher fixed overhead for initialization, which amortizes quickly with use.

**NOTE**: Interesting results but these benches are expected to severely degrade and BTreeMap should end up winning in most categories after I start working on adding concurrency.

### Caveats

- **Concurrency in progress**: Basic concurrent `get`/`insert` work, but memory reclamation and some edge cases still need work
- **No range scans yet**: Ordered iteration not implemented
- **Different trade-offs**: MassTree is optimized for byte-slice keys; BTreeMap is more general

See `Baseline.md` for detailed benchmark history.

## Current Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| `key` | Complete | 8-byte ikey extraction, layer traversal, suffix handling |
| `permuter` | Complete | Const-generic WIDTH, u64-encoded slot permutation |
| `nodeversion` | Complete | Atomic versioned lock with CAS-based acquisition, backoff |
| `leaf` | Complete | LeafNode struct, split operations, B-link pointers, layer support |
| `internode` | Complete | Routing nodes, split-with-insert, height-based child typing |
| `ksearch` | Complete | Binary search for leaves and internodes |
| `tree` | In Progress | Concurrent `get`/`insert` with version validation, split propagation |
| `suffix` | Complete | SuffixBag for keys > 8 bytes with per-slot metadata |
| `alloc` | Complete | `NodeAllocator` trait, `SeizeAllocator` (Miri-compliant) |
| Scan/Remove | Planned | Range scans and key deletion not yet implemented |

**Current Capabilities:**

- Keys from 0-256 bytes (full trie layering for long keys)
- Concurrent `get` (optimistic, lock-free) and `insert` (locked, with split propagation)
- Pluggable allocation via `NodeAllocator` trait
- Miri-verified for strict pointer provenance
- Loom + Shuttle tested for linearizability

**Note on `MassTreeIndex`:** The current `MassTreeIndex<V: Copy>` is a convenience wrapper that still uses `Arc<V>` internally. True inline storage for `V: Copy` values is planned but not yet implemented.
