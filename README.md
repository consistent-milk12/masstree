# masstree

[![Crates.io](https://img.shields.io/crates/v/masstree.svg)](https://crates.io/crates/masstree)
[![Docs.rs](https://docs.rs/masstree/badge.svg)](https://docs.rs/masstree)
[![CI](https://github.com/consistent-milk12/masstree/actions/workflows/ci.yml/badge.svg)](https://github.com/consistent-milk12/masstree/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/consistent-milk12/masstree/blob/main/LICENSE)

A concurrent ordered map for Rust. `masstree` stores byte-slice keys of up to
256 bytes in a trie of B+trees, combining ordered iteration with optimistic
point reads. It is designed for variable-length keys and shared-prefix keys;
key lookup uses byte ordering without hashing the keys.

This is an independent Rust implementation of the
[Masstree design](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf). It
is not affiliated with or endorsed by the original Masstree authors.

[API documentation](https://docs.rs/masstree) ·
[Examples](https://github.com/consistent-milk12/masstree/tree/main/examples)

## Features

- **Byte keys, 0–256 bytes** — keys are `&[u8]`; ordering is lexicographic.
- **Concurrent reads and writes** — optimistic point reads with version
  validation, per-node writer locks, and B-link leaf sibling chains for
  traversal.
- **Hyaline memory reclamation** — guard-protected access and reference-counted
  retirement batches through `seize`.
- **Ordered scans** — forward and reverse ranges, prefix scans, iterators, and
  visitor-style scans with early termination.
- **Inline or boxed values** — store `InlineBits` encodings directly in the
  leaf, or arbitrary `Send + Sync + 'static` values in a `Box`.
- **Batch inserts** — sort a group of entries once and insert them with a shared
  guard.
- **Entry API** — `or_insert`, `and_modify`, and `try_remove`.
- **Application-selected allocator** — the library does not install a global
  allocator. The benchmark helpers opt into mimalloc's v2 backend when the
  `mimalloc` feature is enabled.

## Installation

```toml
[dependencies]
masstree = "1.0"
```

Use a current stable Rust toolchain. The crate uses Rust 2024; see
[Toolchains and platforms](#toolchains-and-platforms) for the tested setup.

## Quick start

Values stored inline must implement `InlineBits`, which encodes a
`Copy + Send + Sync + 'static` value as a `u64`. Built-in implementations include
`u64`, `i32`, `f64`, `bool`, and `()`. Use the default `MassTree` alias for them:

```rust
use masstree::{MassTree, RangeBound};

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

assert!(tree.insert_with_guard(b"alice", 1, &guard).is_none());
assert!(tree.insert_with_guard(b"bob", 2, &guard).is_none());

assert_eq!(tree.get_with_guard(b"alice", &guard).map(|v| *v), Some(1));
assert_eq!(tree.len(), 2);

// A visitor returns false to stop the scan early.
let mut total = 0;
tree.scan(
    RangeBound::Included(b"a"),
    RangeBound::Excluded(b"z"),
    |_key, value| {
        total += value;
        true
    },
    &guard,
);
assert_eq!(total, 3);

assert_eq!(
    tree.remove_with_guard(b"alice", &guard).unwrap().map(|v| *v),
    Some(1),
);
```

For heap-backed values, use `MassTree15`. Its `get_ref` returns a `ValueRef`:
values such as `String` are borrowed under the guard, while primitive types
that support in-place atomic updates return an owned copy.

```rust
use masstree::MassTree15;

let sessions: MassTree15<String> = MassTree15::new();
let guard = sessions.guard();

assert!(
    sessions
        .insert_with_guard(b"session:42", "active".to_owned(), &guard)
        .is_none()
);

// Owned, auto-guard lookup.
assert_eq!(sessions.get(b"session:42").as_deref(), Some("active"));

// Zero-copy lookup tied to the guard lifetime.
if let Some(value) = sessions.get_ref(b"session:42", &guard) {
    assert_eq!(value.as_str(), "active");
}
```

### Type aliases

| Type                  | Value storage          | Typical values                    |
| --------------------- | ---------------------- | --------------------------------- |
| `MassTree<V>`         | Inline 64-bit encoding | `u64`, `i32`, `f64`, `bool`, `()` |
| `MassTree15Inline<V>` | Inline 64-bit encoding | Explicit name for `MassTree<V>`   |
| `MassTree15<V>`       | Heap-allocated `Box<V>` | `String`, `Vec<u8>`, structs, enums |

`MassTreeGeneric<P, A>` is the underlying implementation, parameterized by a
leaf policy and an allocator. Application code normally uses one of the aliases
above.

## Iteration and prefix scans

```rust
use masstree::{MassTree, RangeBound};

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

for (key, value) in [(b"user:1", 1u64), (b"user:2", 2), (b"post:1", 3)] {
    let _ = tree.insert_with_guard(key, value, &guard);
}

// Ordered iteration over the whole tree.
for entry in tree.iter(&guard) {
    println!("{} -> {}", String::from_utf8_lossy(entry.key()), entry.value());
}

// Prefix scan.
let mut users = 0;
tree.scan_prefix(
    b"user:",
    |_key, _value| {
        users += 1;
        true
    },
    &guard,
);
assert_eq!(users, 2);

// Range iterator. Bounds can be Included, Excluded, or Unbounded.
for entry in tree.range(
    RangeBound::Included(b"user:"),
    RangeBound::Unbounded,
    &guard,
) {
    println!("{}", String::from_utf8_lossy(entry.key()));
}
```

`scan_prefix` and `scan` deliver values through the callback without allocating
a key buffer per entry. The iterator API returns owned keys as `Vec<u8>`.

For boxed values, use `scan_ref` or `scan_batch_ref` to receive `&V` references
valid for the callback duration. These borrow values such as `String` directly;
for primitive types updated atomically in place, they reference an atomic copy.
The policy-output scan and iterator APIs yield opaque `ValuePtr<V>` handles for
`MassTree15`, with no safe value accessor. Read boxed values through `get`,
`get_ref`, or the reference-based scan methods.

Scans and iterators preserve key order but do not provide a snapshot of the
whole tree. Their results can reflect concurrent changes made during traversal.

## Entry API and batch inserts

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

let visits = tree
    .entry_with_guard(b"visits", &guard)
    .and_modify(|value| *value + 1)
    .or_insert(1);
assert_eq!(*visits, 1);

let result = tree.insert_batch(vec![
    (b"alpha".to_vec(), 10u64),
    (b"beta".to_vec(), 20u64),
]);
assert_eq!(result.inserted, 2);
assert_eq!(result.updated, 0);
assert_eq!(result.total(), 2);
```

The entry API does not hold a lock between classification and mutation. Under
concurrent writers, an `and_modify` computation may use a stale snapshot, and
`VacantEntry::insert` can overwrite a concurrently inserted value. Use an
external lock shared by all competing writers when the read-modify-write must
be atomic.

Batch inserts preserve input order for duplicate keys, so the last input value
for a key wins in the absence of competing writes. A batch is not a transaction:
other threads can observe partial progress, and a panic during insertion can
leave earlier entries applied.

## Concurrency model

Point lookups, inserts, removals, and scans use a `seize::LocalGuard` to protect
nodes and values from reclamation. The guard marks the current thread active
in the tree's collector. Convenience methods such as `get`, `insert`, and
`remove` create their own guard; explicit-guard methods let you reuse one across
many operations:

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

for (i, key) in [b"a", b"b", b"c"].into_iter().enumerate() {
    let _ = tree.insert_with_guard(key, i as u64, &guard);
}
```

Guard-scoped methods such as `get_with_guard` and `remove_with_guard` return a
`ValueRef` whose lifetime is tied to the guard. Inline values and boxed primitive
types that support atomic updates return owned copies; other boxed values return
protected borrows. The convenience methods `get`, `insert`, and `remove`
require `V: Clone` and return owned values, cloning borrowed values when needed.

`MassTree<V>` is `Send + Sync` for `V: InlineBits`; `MassTree15<V>` is
`Send + Sync` for `V: Send + Sync + 'static`. A tree can be shared through `Arc`;
each thread creates its own guard. Use a guard only with the tree that created
it; passing another tree's guard panics.

Point reads validate optimistic node snapshots and retry on conflicts. They
can spin or yield while a writer has marked a node dirty, and a lookup through
a promoted layer can briefly try-lock the parent leaf to refresh its root
pointer. Point reads therefore have no lock-free progress guarantee.

### Reclamation model

`masstree` uses [`seize`](https://crates.io/crates/seize), which implements the
[Hyaline reclamation scheme](https://arxiv.org/abs/1905.07903) (specifically the
Hyaline-1 variant). Retired nodes, boxed values, and replaced suffix storage are
appended to a thread-local batch. When the batch reaches its threshold, or when
`tree.flush(&guard)` is called, the collector attempts to publish the batch to
active threads' reservation lists. Publication requires at least as many retired
objects as active threads; smaller batches can remain pending even after a
flush. Participating threads release their batch references when they leave or
refresh their protected region, and reclamation runs once all references have
been released. This distributes reclamation work without a global epoch or
periodic progress checks. Each tree owns its collector; dropping the tree drains
pending retirements and destroys the remaining tree.

`LocalGuard` is neither `Send` nor `Sync`. Reusing it avoids repeated guard
entry and exit, but holding it for a long time can delay reclamation. The
default retirement threshold is 256 objects; the `with_batch_size` constructors
can tune it. `maintenance` attempts queued leaf/sublayer cleanup, then flushes
the current thread's batch and refreshes the guard. Refreshing can invalidate
older guard-scoped references, which is why `maintenance` takes `&mut` access
to the guard. Refresh has no effect while another guard from the same tree is
alive on the same thread; other threads' guards can also delay reclamation.

The default allocator recycles reclaimed node blocks through thread-local
pools, so reclamation does not necessarily return their memory to the system.

`len()` is a sharded count. Under concurrent mutation it is approximate,
not a linearizable snapshot.

Deletion is lazy: removing the last entry from a leaf queues the empty leaf for
coalescing. In long-running, delete-heavy workloads, call `maintenance`
periodically between batches of operations, without another guard for this
tree on the calling thread:

```rust
use masstree::MassTree;

let tree: MassTree<u64> = MassTree::new();
let mut guard = tree.guard();
// ... operations ...
tree.maintenance(&mut guard);
```

## Feature flags

| Feature                 | Default | Description                                                                                  |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------- |
| `debug-print`           | yes     | Enables `print_tree` and `print_tree_to` for inspecting the tree structure.                   |
| `tracing`               | no      | Structured tracing events and the `init_tracing` helper.                                      |
| `no-prefetch`           | no      | Disables the library's explicit software prefetch hints.                                     |
| `small-suffix-capacity` | no      | Halves inline suffix data capacity (512 to 256 bytes); shrinks embedded leaves or heap sidecars. |
| `sidecar-suffix`        | no      | Stores inline-policy suffixes in a lazily allocated heap sidecar instead of the embedded bag.  |
| `debug-routing`         | no      | Prints routing and split diagnostics to stderr. Intended for debugging only.                   |
| `mimalloc`              | no      | Optional mimalloc v2 backend; benchmark helpers install it as their global allocator.          |

To omit the default debug-printing support, use this dependency declaration:

```toml
[dependencies]
masstree = { version = "1.0", default-features = false }
```

## Design

The tree routes keys by 8-byte chunks. The top-level B+tree indexes the
first chunk. When two keys share a chunk and both continue with different
suffixes, the suffix slot becomes a layer pointer to a subtree for the next
chunk. A key ending at that chunk can coexist in a separate terminal slot.
A leaf can store the remaining bytes as a suffix, so a long key does not require
a layer for every chunk. Trie depth is bounded at 32 layers by the 256-byte key
limit; each visited layer may require its own B+tree search.

Nodes use fixed-width layouts: 15 slots per leaf, up to 16 children per
internode, separate 64-byte regions for the leaf version word and
permutation, and compile-time size/offset assertions on 64-bit targets. Leaf
splits publish a right sibling through the B-link chain before parent
propagation completes. Readers and range scans use version checks, retries,
and these links to find moved keys.

Reads use optimistic concurrency control: load a node version, read node fields,
then validate the relevant version bits. Writers take a per-node spin lock,
mark structural mutations dirty, and publish them with a version increment on
unlock. The [`seize`](https://crates.io/crates/seize) Hyaline collector defers
reclamation; the default allocator recycles node blocks through thread-local
pools.

## Constraints

- Keys must be 0–256 bytes. Point operations reject longer keys with a panic;
  explicit batch inserts validate all input keys before applying entries.
  `Extend` validates one chunk at a time, so earlier chunks can already be
  applied when a later key is rejected.
- Keys are ordered lexicographically by byte value. Fixed-width unsigned
  integers encoded in big-endian order preserve numeric ordering. Signed
  integers need an order-preserving transformation, such as flipping the sign
  bit before big-endian encoding; floating-point keys need a separate encoding.
- `MassTree<V>` accepts only types implementing `InlineBits`; use
  `MassTree15<V>` for `String`, `Vec<u8>`, and other heap-backed values.
- Values must be `Send + Sync + 'static`.
- The tree does not support custom comparators.
- `len()` is approximate during concurrent mutation.
- Coalescing removes eligible empty leaves and empty sublayers; it does not
  rebalance partially filled nodes. Call `maintenance` during delete-heavy
  workloads. Batching and active guards can delay reclamation as described above.

## Toolchains and platforms

The crate uses Rust 2024. CI runs stable Rust on Ubuntu x86_64, plus nightly
Rust for Miri. `clippy.toml` sets a 1.92 lint baseline, but `Cargo.toml` does not
declare a `rust-version` and CI does not test that compiler version, so a Rust
1.92 minimum is not verified by the current checks.

The source includes size/offset assertions for 64-bit node layouts and software
prefetch implementations for x86_64 and aarch64. Other architectures use no-op
prefetch helpers. Build and runtime compatibility on aarch64 and other
platforms are not checked by the current CI configuration.

## Examples

Run these commands from a checkout of this repository. Examples live in
[`examples/`](https://github.com/consistent-milk12/masstree/tree/main/examples):

```bash
cargo run --example basic_usage
cargo run --example url_cache --release
cargo run --example session_store --release
cargo run --example rayon_parallel --release
cargo run --example tokio_async --release
```

## Development

```bash
cargo fmt --all -- --check
cargo test --lib
cargo test --tests
cargo test --doc
cargo check --all-targets --no-default-features
cargo check --all-targets --all-features
```

CI also runs simplified Loom models and the Miri leak-test target. Loom does
not model the full tree, and this Miri command covers only `tests/leak_tests.rs`.
The latter requires a nightly toolchain with the Miri component installed:

```bash
RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
    cargo +nightly miri test --test leak_tests
```

Architecture and audit notes are in
[`notes/`](https://github.com/consistent-milk12/masstree/tree/main/notes). The
[`failures/`](https://github.com/consistent-milk12/masstree/tree/main/failures)
tree contains standalone audit reproducers and regression tests, including
intentional compile-fail cases; it is not part of the normal test suite. Follow
each reproducer's instructions for its expected result. The audit includes
historical findings and a remediation-status section. Documentation can lag
the implementation; check the current source and tests when resolving a
discrepancy.

## License

MIT. See
[`LICENSE`](https://github.com/consistent-milk12/masstree/blob/main/LICENSE).

## References

- [Cache Craftiness for Fast Multicore Key-Value Storage (EuroSys 2012)](https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf)
- [Snapshot-Free, Transparent, and Robust Memory Reclamation for Lock-Free Data Structures (PLDI 2021)](https://arxiv.org/abs/1905.07903)
- [`seize`](https://crates.io/crates/seize), the Hyaline reclamation
  implementation used by this crate
- [C++ reference implementation](https://github.com/kohler/masstree-beta)
