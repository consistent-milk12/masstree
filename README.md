# madtree (Rust)

After studying the C++ implementation, I decided to name this repository `madtree`, because actually implementing it by myself is, indeed, an insane thing to do. I still wanted to try and see how far I can go.

Masstree is a high-performance concurrent trie of B+ trees designed for in-memory key-value storage. It combines the cache efficiency of B+ trees with the no-rebalancing property of tries by slicing keys into 8-byte chunks, where each chunk navigates a separate B+ tree layer.

**Source:** <https://github.com/kohler/masstree-beta>

## Why Rust?

The original Masstree is a very interesting and performant concurrent data structure, but its C++ implementation relies on manual memory management, platform-specific atomics, and subtle pointer tricks that make it difficult to extend or verify. Rust's ownership model and type system offer an opportunity to express the same algorithms with compile-time safety guarantees, eliminate entire classes of concurrency bugs, and produce a codebase that's easier to audit and maintain. This reimplementation aims to match/get close to the original's performance while leveraging Rust's strengths: safe abstractions over unsafe primitives, fearless™ concurrency, and a modern toolchain for testing and benchmarking support.

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

## Improvements Over The C++ Implementation

Both implementations use epoch-based reclamation (EBR) for node memory safety—C++ via `threadinfo`, this implementation via `crossbeam-epoch` (if implemented as planned). The difference lies in **value lifetime management**:

| Aspect | C++ Masstree | Rust MassTree |
|--------|--------------|---------------|
| Node safety | EBR (manual `threadinfo&` passing) | EBR (epoch pinning is internal) |
| Value safety | User's responsibility | `Arc<V>` handles automatically |
| Misuse | Silent undefined behavior | Compile error or safe behavior |
| API burden | Must pass `threadinfo&` to every call | Clean API, no manual tracking |

**Concrete improvements:**

1. **Values can't outlive their storage** — In C++, storing a `char*` to heap data, deleting the entry, then accessing the pointer is silent UB. With `Arc<V>`, the data lives until the last reference drops.

2. **No manual coordination** — C++ users must ensure value cleanup happens after all readers finish. `Arc`'s refcount handles this automatically.

3. **Composable safety** — `MassTree<Vec<String>>` would just work. In C++, complex value types need careful RCU-aware destructor implementations.

4. **Zero-cost when not needed** — `MassTreeIndex<u64>` would provide C++ equivalent performance for simple cases where `Arc` overhead matters.

## Current Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| `key` | Implemented | 8-byte ikey extraction, layer traversal, suffix handling |
| `permuter` | Implemented | Const-generic WIDTH, u64-encoded slot permutation |
| `nodeversion` | In progress | Versioned lock with type-state guard pattern |
| Leaf nodes | Design complete | Pending implementation |
| Tree operations | Planned | Get, insert, scan, remove |
