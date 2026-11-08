# Masstree (Rust) — Architecture

> **Scope.** This document describes the architecture of the Masstree
> implementation in this repository (crate `masstree`, version **1.0.0**, Rust
> edition 2024). It is an independent Rust implementation of the Masstree design
> — a trie of B+ trees — not a line-by-line port of the original C++ code. The
> document covers the source layout, data representation, concurrency protocol,
> every major operation, storage policies, reclamation, the public API,
> invariants, test architecture, performance design, and known limitations. File
> references are relative to the repository root.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository and Build Structure](#2-repository-and-build-structure)
3. [Architectural Model](#3-architectural-model)
4. [Data Representation](#4-data-representation)
5. [Storage Policies](#5-storage-policies)
6. [Suffix Storage](#6-suffix-storage)
7. [Concurrency Architecture](#7-concurrency-architecture)
8. [Operations](#8-operations)
9. [Public API](#9-public-api)
10. [Invariants](#10-invariants)
11. [Testing and Validation](#11-testing-and-validation)
12. [Performance Architecture](#12-performance-architecture)
13. [Feature Matrix and Configuration](#13-feature-matrix-and-configuration)
14. [Design Notes, Divergences, and Audit Points](#14-design-notes-divergences-and-audit-points)
15. [Glossary](#15-glossary)

- [Appendix A — Constants](#appendix-a--constants)
- [Appendix B — File Index](#appendix-b--file-index)
- [Appendix C — References](#appendix-c--references)

---

## 1. Executive Summary

`masstree` is a concurrent ordered map from arbitrary byte keys (`&[u8]`, 0–256
bytes) to values. It combines two ideas:

1. **A trie of B+ trees.** A long key is split into 8-byte chunks ("ikeys"). The
   top-level B+ tree indexes the first chunk. When two keys share the same chunk
   _and both continue with different suffixes_, the shared prefix is represented
   by a _layer pointer_: the leaf slot points to a whole sub-B+tree that indexes
   the next chunk. This continues for up to 32 layers (256 bytes). A short key
   and a suffix key with the same ikey can coexist without a layer.
2. **Optimistic concurrency control (OCC) with per-node locks.** Point reads are
   lock-free: a reader snapshots a per-node version word, reads the node, and
   validates that the version did not change. Writers take a per-node spin lock
   and publish a new version on unlock. Splits are B+-tree-style with a leaf
   `next` chain (B-link), so readers can follow moved keys instead of blocking.

The crate exposes three type aliases over one generic implementation:

| Alias                 | Policy            | Value storage                        | Typical use                       |
| --------------------- | ----------------- | ------------------------------------ | --------------------------------- |
| `MassTree<V>`         | `InlinePolicy<V>` | Value bits stored inside the leaf    | `Copy` values ≤ 8 bytes (default) |
| `MassTree15Inline<V>` | `InlinePolicy<V>` | Same as above                        | Explicit inline variant           |
| `MassTree15<V>`       | `BoxPolicy<V>`    | Heap `Box<V>` pointed to by the leaf | `String`, `Vec`, large structs    |

`MassTreeGeneric<P, A>` is the underlying type, parameterized by a
**leaf policy** `P` (value + suffix storage) and an **allocator** `A`. The
in-tree production allocator is `SeizeAllocator<P>`, built on the `seize`
epoch-based reclamation (EBR) crate; `TreeAllocator` is public, so external
allocators can be supplied through `with_allocator`.

Key properties:

- **Lock-free reads**, validated per leaf/internode with `NodeVersion`.
- **Fine-grained writes**: one spin lock per node; splits use hand-over-hand
  parent locking.
- **B-link leaves** with a split-in-progress mark bit on `next`.
- **Epoch-based reclamation** via `seize`; nodes are recycled through
  thread-local size-class pools.
- **Lazy coalescing**: deletion only empties a leaf and enqueues it; a later
  `maintenance` call unlinks empty leaves and collapses empty internodes. There
  is no leaf merge/rebalance.
- **Range scans** forward and reverse, with callback, iterator, and batch
  variants; sublayers are traversed depth-first in key order.
- **Two suffix strategies**: an in-leaf 512-byte (or 256-byte) suffix bag plus
  spill-to-heap, or a lazily allocated suffix sidecar. `BoxPolicy` always uses
  the sidecar; `InlinePolicy` uses the embedded bag by default.

The implementation is heavily optimized for cache behavior: fixed-width 15-slot
nodes, a 64-bit permutation array, cache-line isolation of the
version/permutation, software prefetching, deferred allocation ("write-through")
for small boxed values, a sharded length counter, and per-thread node pools.

---

## 2. Repository and Build Structure

### 2.1 Source tree

```
src/                                   ~29.6k non-test LOC, ~42.7k including colocated tests
├── lib.rs                             crate docs, module declarations, public re-exports
├── key.rs                             Key: 8-byte ikey chunks, shift/unshift, ordering
├── ksearch/                            internode routing search (scalar upper_bound)
├── nodeversion.rs                     32-bit versioned lock: OCC + writer lock
│   ├── bounded.rs                     bounded spin lock (16 spins, then yield)
│   └── {unit,concurrent,loom}_tests.rs
├── ordering.rs                        canonical Acquire/Release/AcqRel constants
├── link.rs                            LSB pointer marking for split-in-progress
├── permuter.rs                        64-bit permutation of 15 leaf slots
├── hints.rs                           likely/unlikely branch hints
├── prefetch.rs                        x86_64/aarch64 read/write prefetch
├── shard_counter.rs                   16-way cache-line-padded length counter
├── inline/
│   ├── bits.rs                        InlineBits: encode Copy values into u64
│   └── sentinel.rs                    real-static-address sentinel for inline slots
├── internode.rs                       InternodeNode (15 keys, 16 children, height)
│   ├── accessors.rs                   key/child/parent accessors and prefetch
│   └── layout.rs                      compile-time 320-byte / 5-cache-line assertions
├── leaf15.rs                          LeafNode15<P>: WIDTH=15 B-link leaf
│   ├── layout.rs                      compile-time size/offset assertions
│   └── unit_tests.rs
├── leaf_trait.rs                      TreePermutation / TreeLeafNode / TreeInternode traits
├── policy.rs                          LeafPolicy / ValueArray / SuffixStore / ValuePtr / RefPolicy
│   ├── box_values.rs                  BoxValueArray (heap pointers)
│   ├── inline_values.rs               InlineValueArray (sentinel + AtomicU64 bits)
│   ├── embedded_suffix.rs             in-leaf suffix store
│   ├── sidecar_suffix.rs              lazy external suffix store
│   ├── suffix_ops.rs                  shared embedded/sidecar algorithms
│   └── tests.rs
├── suffix.rs                          SuffixBag, SuffixBagCell, SlotMeta, PermutationProvider
│   ├── inline.rs                      InlineSuffixBag (fixed 512/256-byte buffer)
│   └── sidecar.rs                     SuffixSidecar
├── alloc_trait.rs                     TreeAllocator<P> trait
├── alloc15.rs                         SeizeAllocator<P>, iterative tree teardown
├── node_pool.rs                       thread-local size-class freelists
├── retirement.rs                      BatchedRetire::flush wrapper
├── tree.rs                            MassTreeGeneric, aliases, errors, Drop
└── tree/
    ├── generic.rs                     guard/maintenance/reach_leaf/count/entry hooks
    │   ├── search.rs                  optimistic leaf search
    │   ├── insert.rs                  insert algorithm + strategies
    │   ├── insert_strategy.rs         GenericInsert vs WriteThroughInsert
    │   ├── layer.rs                   sublayer ("layer") creation, twig chains
    │   ├── optimistic_reads.rs        point reads (single/multi layer)
    │   │   └── get_guarded.rs         guarded get implementation
    │   ├── split.rs                   leaf split preparation + no-abandon protocol
    │   ├── batch.rs                   batch insert
    │   ├── entry.rs                   Entry/OccupiedEntry/VacantEntry
    │   └── print.rs                   debug tree printer (debug-print feature)
    ├── split/                         hand-over-hand split propagation
    │   ├── propagation.rs             parent propagation loop
    │   ├── propagation_context.rs     lifetime-unified RAII context
    │   ├── parent_locking.rs          membership validation
    │   └── root_creation.rs           main-root and layer-root creation
    ├── remove.rs                      deletion, parent cleanup, internode cascade
    ├── coalesce.rs                    lazy empty-leaf queue + sublayer GC
    ├── range/                         ordered scans
    │   ├── api.rs / api_ref.rs        callback/iterator/ref APIs
    │   ├── iterator.rs                RangeIter, KeysIter, ValuesIter
    │   ├── iterator/                  bounds, entries, batch adapters
    │   ├── cursor_key.rs              fixed 256-byte scan cursor
    │   ├── scan_state.rs              forward/back state machine + stacks
    │   ├── find.rs / find_rev.rs      per-leaf position algorithms
    │   ├── forward_ctx.rs             forward context/driver
    │   ├── reverse_ctx.rs             reverse context/driver
    │   ├── helper.rs / traversal.rs   leaf positioning and internode descent
    │   └── batch_common.rs            generic batch scan engine
    └── batch_utils.rs                 batch helpers/statistics
```

**Logical layering** (dependencies point downward):

```
tree/                 tree.rs, generic/, split/, remove, coalesce, range/, batch_utils
    ↓
allocation            alloc_trait, alloc15, node_pool, retirement
    ↓
nodes                 leaf15, internode
    ↓
storage contracts     leaf_trait, policy, suffix
    ↓
primitives            nodeversion, permuter, key, ksearch, inline, shard_counter
    ↓
core utilities        ordering, hints, prefetch, link
```

The layers are directional at the algorithm level: primitives never depend on
tree operations, the node types never depend on the tree facade, and the tree
facade depends on everything below it. At the Rust module level there are a few
intentional type-level cycles: `policy` names `SuffixBagCell` from `suffix`,
`suffix` names the `TreePermutation` re-export from `leaf_trait`, and
`leaf_trait` names `LeafPolicy` from `policy`. These are contract references,
not algorithmic cycles.

### 2.2 Test tree

```
tests/                                   ~14k LOC, 402 integration test functions
├── common/mod.rs                        shared tracing initialization
├── cpp_tests.rs + cpp/                   88 C++-mttest-style workload tests (28 modules)
├── tree_proptests.rs (+ .proptest-regressions)
├── key_proptests.rs
├── nodeversion_proptests.rs
├── permuter_proptests.rs
├── concurrent_correctness.rs
├── concurrent_regression.rs
├── stress_tests.rs
├── stress_adaptive.rs
├── adaptive_insert.rs
├── batch_insert_tests.rs
├── range_scan_tests.rs
├── deleted_leaf_handling.rs
├── phase2_regression.rs
├── split_suffix_regression.rs
├── leak_tests.rs
└── true_inline.rs
```

Module-level tests live next to their modules as `*_tests.rs` / `unit_tests.rs`.
In total the checkout contains roughly **1,027 `#[test]` functions**, about
**625** of them in `src/` and **402** in `tests/`. See
[§11](#11-testing-and-validation) for the full taxonomy.

### 2.3 Benchmarks, examples, tooling

```
benches/            12 harness-less benchmark targets (divan, pbench, custom harnesses)
examples/           basic_usage, print_tree, rayon_parallel, tokio_async, url_cache, session_store
cpp_benches/        C++ mttest result files + parser
rust_benches/       Rust mttest result files + parser
runs/               ~112 saved benchmark runs used by README tables
scripts/            bench comparison, C++ runner, version bump
justfile            build/test/lint/bench/Miri recipes
Cargo.toml          features, profiles, benchmark/example declarations
clippy.toml         MSRV 1.92, complexity/style limits
rustfmt.toml        edition 2024, width 100
rust-analyzer.toml  feature selection for IDE analysis
```

`Cargo.toml` defines twelve `[[bench]]` targets (`concurrent_read_write*`,
`high_impact*`, `mttest`, `range_masstree15`, `storage_gap`, `tail_latency`,
`memory_total`, `memory_dhat`) and six examples. Release builds use fat LTO, one
codegen unit, `panic = "abort"`, and symbol stripping; `release-with-debug` and
`profiling` profiles retain debug info.

---

## 3. Architectural Model

### 3.1 A trie of B+ trees

Masstree's core idea is to avoid storing arbitrarily long keys in B+ tree nodes
while preserving lexicographic order. Every 8-byte chunk of a key becomes an
unsigned big-endian integer (`ikey`), so comparing `u64`s reproduces bytewise
lexicographic order for that chunk.

- The **main tree** indexes `ikey_0` (bytes 0–7).
- A leaf slot normally stores a terminal value for a key that ends within this
  chunk, or a suffix for a key that continues past it.
- If two keys have the same `ikey_0` and _different suffixes_, the leaf cannot
  order them by the first chunk alone. The implementation replaces the slot with
  a **layer pointer**: a pointer to a sub-B+tree whose keys are the remaining
  bytes after the first 8.
- The same rule applies recursively inside sublayers. A key of 256 bytes can
  require 32 layers.

A layer pointer is tagged by `keylenx = 128` (`LAYER_KEYLENX`). A suffix key is
tagged by `keylenx = 64` (`KSUF_KEYLENX`); an inline key that fits entirely in
the current chunk is tagged by its remaining length (0–8).

### 3.2 Node types

There are exactly two node types:

| Node            |                                                                                     Size (64-bit) | Contents                                                                        | Links                    |
| --------------- | ------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------- | ------------------------ |
| `LeafNode15<P>` | 448 B (`BoxPolicy`), 1152 B (`InlinePolicy`, 512-B suffix), 896 B (256-B suffix), 576 B (sidecar) | 15 slots: ikey + keylenx + value/layer + optional suffix; permutation; modstate | `next`, `prev`, `parent` |
| `InternodeNode` |                                                                             320 B (5 cache lines) | 15 separator ikeys + 16 child pointers; height                                  | `parent` only            |

Both begin with a `NodeVersion` at offset 0. This is what enables type-erased
parent pointers (`*mut u8`) and type-erased traversal: the `ISLEAF_BIT` in the
version word tells the code how to interpret the pointer.

A tree is rooted at either a leaf or an internode. A root node has `ROOT_BIT`
set and a null parent. A _main root_ is the node stored in
`MassTreeGeneric::root_ptr`; a _layer root_ is a root node referenced by a leaf
slot but not stored in `root_ptr`.

### 3.3 High-level diagram

```
                         MassTreeGeneric<P, A>
        ┌───────────────────────────────────────────────────────────┐
        │ collector: seize::Collector        (EBR domain)           │
        │ allocator: A                       (node pool + teardown) │
        │ root_ptr: AtomicPtr<u8> ────────────────────────────┐     │
        │ count: ShardedCounter                              │     │
        │ coalesce_queue: CoalesceQueue                      │     │
        └────────────────────────────────────────────────────┼─────┘
                                                             ▼
                                            InternodeNode (height = 1)
                                  ┌──────────────┼───────────────┐
                                  ▼              ▼               ▼
                            InternodeNode(h=0)  ...             ...
                             ┌────┼────┐
                             ▼    ▼    ▼
                         Leaf ↔ Leaf ↔ Leaf      B-link chain (next/prev)
                             │
                             │ slot: keylenx = 128 (LAYER)
                             ▼
                     sublayer root (leaf or internode)
                             │
                             └── same node types, same allocator, same OCC
```

### 3.4 End-to-end operation map

| Operation              | Read/write path                                           | Key modules                                                                            |
| ---------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `get` / `contains_key` | OCC descent → leaf search → validate; no locks            | `tree/generic/optimistic_reads.rs`, `get_guarded.rs`, `tree/generic.rs::reach_leaf...` |
| `get_ref`              | Same, returns `ValueRef` for `BoxPolicy`                  | `policy.rs` (`RefPolicy`)                                                              |
| `insert`               | OCC descent → leaf lock → update / assign / split / layer | `tree/generic/insert.rs`, `tree/split/*`, `tree/generic/layer.rs`                      |
| `remove`               | OCC descent → leaf lock → clear slot → enqueue empty leaf | `tree/remove.rs`, `tree/coalesce.rs`                                                   |
| range scan             | OCC descent + leaf chain walk + layer stack               | `tree/range/*`                                                                         |
| batch insert           | Sort by ikey → per-leaf runs                              | `tree/generic/batch.rs`                                                                |
| maintenance            | flush EBR + process coalesce queue                        | `tree/generic.rs`, `tree/coalesce.rs`                                                  |

---

## 4. Data Representation

### 4.1 Keys and ikeys

`src/key.rs` defines:

```rust
pub const IKEY_SIZE: usize = 8;
pub const MAX_KEY_LENGTH: usize = 256; // 32 layers

pub struct Key<'a> {
    data: &'a [u8],        // original bytes, never modified
    ikey: u64,             // current 8-byte chunk, big-endian
    shift_count: usize,    // number of 8-byte chunks consumed
    suffix_start: usize,   // offset of the suffix in `data`
}
```

- `Key::new` asserts `len <= 256`.
- `read_ikey` reads eight bytes with `u64::from_be_bytes`; a short tail is
  zero-padded. Because the encoding is big-endian, unsigned `u64` comparison
  equals lexicographic byte comparison for that chunk.
- `current_len() = data.len() - shift_count * 8`.
- `has_suffix() = current_len() > 8`.
- `shift()` advances one chunk and sets `suffix_start = min(offset + 8, len)`.
- `unshift()` / `unshift_all()` go back; `shift_by(8 * n)` applies multiple
  shifts.
- `suffix()` returns `data[suffix_start..]`.

**Per-layer keylenx encoding** (`src/leaf15.rs`):

|                                               Value | Meaning                                                                              |
| --------------------------------------------------: | ------------------------------------------------------------------------------------ |
|                                             `0..=8` | Terminal key; the value is the remaining length at this layer. `0` is the empty key. |
|                               `64` (`KSUF_KEYLENX`) | Key continues past this chunk; the remainder is stored as a suffix.                  |
| `>= 128` (`LAYER_KEYLENX`, currently exactly `128`) | Slot holds a pointer to a sublayer, not a value.                                     |

Ordering within one ikey is: terminal lengths 1, 2, …, 8, then suffix (`64`),
then layer (`128`). `Key::compare` implements this; it deliberately does **not**
compare suffix bytes. Suffix bytes are compared separately by the leaf search,
because a leaf holds at most one suffix key per ikey.

Examples:

```
key = b"hello"                 ikey = 0x68656c6c6f000000, keylenx = 5
key = b"hello world"           ikey = 0x68656c6c6f20776f ("hello wo"), keylenx = 64,
                               suffix = b"rld"
key = b"prefix00abc"           ikey = 0x7072656669783030 ("prefix00"), suffix = b"abc"
key = b"prefix00xyz"           same ikey, different suffix  -> layer creation if both are
                               inserted and conflict in the same leaf
```

A key that ends exactly at an 8-byte boundary has no suffix at that layer; the
next layer begins on the next `shift()`. An empty key is representable
(`ikey = 0`, `keylenx = 0`) and is covered by tests.

### 4.2 Leaf node

`LeafNode15<P>` (`src/leaf15.rs`) is `#[repr(C, align(64))]`:

```rust
pub struct LeafNode15<P: LeafPolicy> {
    version: NodeVersion,          // offset 0, 4 bytes
    modstate: AtomicU8,            // offset 4, 1 byte
    _pad0: [u8; 55],               // pad to cache line 1

    permutation: AtomicPermuter15, // offset 64, 8 bytes
    _pad1: [u8; 56],               // pad to cache line 2

    ikey0: [AtomicU64; 15],        // offset 128, 120 bytes
    keylenx: [AtomicU8; 15],       // offset 248, 15 bytes

    values: P::Values,             // offset 264
    suffix: P::Suffix,             // offset 384 (Box) / 504 (Inline)
    next: AtomicPtr<Self>,         // B-link successor; LSB = split mark
    prev: AtomicPtr<Self>,         // B-link predecessor
    parent: AtomicPtr<u8>,         // type-erased parent internode
}
```

`src/leaf15/layout.rs` verifies these offsets and sizes at compile time for
64-bit targets:

| Policy / feature                                     |   Size | Cache lines |
| ---------------------------------------------------- | -----: | ----------: |
| `BoxPolicy<u64>` (sidecar suffix)                    |  448 B |           7 |
| `InlinePolicy<u64>`, embedded 512-B suffix           | 1152 B |          18 |
| `InlinePolicy<u64>`, `small-suffix-capacity` (256 B) |  896 B |          14 |
| `InlinePolicy<u64>`, `sidecar-suffix`                |  576 B |           9 |

Cache-line 0 holds only the version and modstate; cache-line 1 holds only the
permutation. This separates the read-mostly header from the CAS-heavy
permutation and avoids false sharing between readers and slot allocators.

`modstate` is a 1-byte lifecycle word:

| Bits                              | Meaning                                         |
| --------------------------------- | ----------------------------------------------- |
| `0b00` (`MODSTATE_INSERT`)        | Normal / insert state                           |
| `0b01` (`MODSTATE_REMOVE`)        | Legacy remove state (essentially unused)        |
| `0b10` (`MODSTATE_DELETED_LAYER`) | The sublayer this leaf belongs to has been GC'd |
| `0b11` (`MODSTATE_EMPTY`)         | Leaf is empty and may be coalesced              |
| `0b100` (`MODSTATE_QUEUED_BIT`)   | Leaf is already in the coalesce queue           |

The queued bit is set with atomic `fetch_or`; only the winner enqueues.
`mark_empty`, `mark_deleted_layer`, and `clear_empty_state` preserve or clear
the queued bit appropriately. `try_mark_queued` is the deduplication primitive.

**Slot 0 rule.** `ikey_bound()` is `ikey0[0]` (the smallest ikey of a
non-leftmost leaf). Physical slot 0 is special: if a leaf has a predecessor,
slot 0's ikey is the leaf's routing lower bound and must not be recycled for a
different ikey. The insert path detects this and either uses another free
physical slot or splits. Removal does not rewrite `ikey_bound`; it is a
conservative routing bound, not a strict minimum after deletions.

**Occupancy** is determined by the value array, not by `keylenx`: a cleared slot
and the empty key both have `keylenx = 0`. `is_value_empty`/`is_slot_empty` are
authoritative.

**Trait usage.** Despite the generic-looking `TreeLeafNode` trait, the tree
algorithms are monomorphized directly over `LeafNode15<P>`; the trait is a
specification/extension surface rather than a dynamic-dispatch boundary.
`TreeInternode` is genuinely used generically for routing and propagation, and
`TreePermutation` is used generically by suffix draining/compaction.

### 4.3 Internode node

`InternodeNode` (`src/internode.rs`) is `#[repr(C, align(64))]`:

```rust
pub struct InternodeNode {
    version: NodeVersion,       // offset 0
    nkeys: AtomicU8,            // offset 4
    height: u8,                 // offset 5
    _pad: [u8; 2],              // offset 6
    parent: AtomicPtr<u8>,      // offset 8
    ikey0: [AtomicU64; 15],     // offset 16, 120 bytes
    child: [AtomicPtr<u8>; 16], // offset 136, 128 bytes
}
```

The raw size is 264 bytes, padded to exactly 320 bytes = 5 cache lines
(`src/internode/layout.rs`).

Routing convention:

- `child[0]` covers keys `< ikey0[0]`.
- `child[i]` covers keys in `[ikey0[i-1], ikey0[i])`.
- `child[nkeys]` is the rightmost child and covers keys `>= ikey0[nkeys-1]`.

The search `upper_bound_internode` returns the first index `i` with
`search_ikey < ikey0[i]`, or `i+1` when equal, or `nkeys` at the end. This
routes equality to the right child, which is correct because separators are
lower bounds.

`height` is `0` when children are leaves, `1+` when children are internodes.
Internodes do **not** store keylenx or suffix data; routing uses only the 8-byte
separator ikey. Only leaves participate in the `next`/`prev` B-link chain;
internodes have parent pointers only.

### 4.4 The permutation array

`Permuter<WIDTH>` (`src/permuter.rs`) packs a permutation of `0..WIDTH` into one
`u64`:

```
bits 63..4 : 15 four-bit nibbles, one per logical position
bits  3..0 : size (number of active logical slots) in the low nibble

position p (0..14) occupies bits 4*(p+1) .. 4*(p+1)+3
```

`WIDTH = 15` is the maximum, because 15 nibbles plus the size nibble exactly
fill a `u64` (`MAX_WIDTH = 15`). The initial value places physical slots in
reverse order, so `back()` is the lowest unused slot:

```
INITIAL(15) = 0x0123_4567_89AB_CDE0   // slot 14 at position 0 ... slot 0 at position 14
SORTED(15)  = 0xEDCB_A987_6543_2100   // slot i at position i
```

Core operations:

- `insert_from_back(logical_pos)` claims the physical slot at the back, shifts
  positions `logical_pos..size` one nibble right, and increments size.
- `remove(logical_pos)` compacts later logical positions down and parks the
  removed physical slot at the first free position (`new_size`), not necessarily
  at the back.
- `remove_to_back` is a variant that parks at the back (used by some
  tests/helpers).
- `swap_free_slots`, `exchange`, and `rotate` support free-slot rearrangement.
- `make_sorted(n)` builds the identity permutation for the first `n` positions
  and fills the rest with descending unused slots.
- `debug_assert_valid` verifies that all 15 slot indices appear exactly once.

`AtomicPermuter15` is a `#[repr(transparent)]` wrapper around `AtomicU64` with
load/store/CAS. Insertion stores the permutation Relaxed under the leaf lock and
relies on the Release unlock for publication. Removal stores it with Release
because lock-free readers may still hold the old permutation. CAS support exists
but is not used by the current insert path. The permutation is the leaf's
logical ordering: physical slot contents can stay in place while logical
positions move, which makes splits and deletes cheap.

`TreePermutation` (`src/leaf_trait.rs`) is the generic trait surface used by
algorithms that only need the permutation (notably suffix draining).
`PermutationProvider` is a read-only `size()`/`get()` view used by suffix
compaction.

### 4.5 NodeVersion

`NodeVersion` (`src/nodeversion.rs`) is a single `AtomicU32` with this layout:

```
31      30     29      28  27........9  8.....3  2        1        0
ISLEAF  ROOT   DELETED UNUSED  vsplit     vinsert SPLITTING INSERTING LOCK
```

| Bit(s) | Constant         | Meaning                           |
| -----: | ---------------- | --------------------------------- |
|      0 | `LOCK_BIT`       | Node spin lock held               |
|      1 | `INSERTING_BIT`  | Insert/update in progress (dirty) |
|      2 | `SPLITTING_BIT`  | Split/delete in progress (dirty)  |
|    3–8 | `VINSERT_LOWBIT` | 6-bit insert version counter      |
|   9–27 | `VSPLIT_LOWBIT`  | 19-bit split version counter      |
|     28 | `UNUSED1_BIT`    | Reserved, cleared on unlock       |
|     29 | `DELETED_BIT`    | Node logically deleted            |
|     30 | `ROOT_BIT`       | Root of a main tree or layer      |
|     31 | `ISLEAF_BIT`     | Node is a leaf (vs internode)     |

`DIRTY_MASK = INSERTING_BIT | SPLITTING_BIT`.

**Unlock formulas** (`LockGuard::drop`):

- If `SPLITTING_BIT` was set: `(locked + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK`.
  This increments `vsplit`, clears lock/dirty and low version bits, clears
  `ROOT_BIT` and `UNUSED1_BIT`, and preserves `DELETED` and `ISLEAF`.
- Otherwise: `(locked + ((locked & INSERTING_BIT) << 2)) & UNLOCK_MASK`. The
  inserting bit (bit 1) is shifted to bit 3, incrementing `vinsert`; lock/dirty
  bits are cleared; `ROOT_BIT` is preserved.
- A pure lock/unlock with neither dirty bit set does not change the version.

`mark_insert()` sets `INSERTING_BIT` with a Release store followed by a Release
store-store fence; it is idempotent. `mark_split()` sets `SPLITTING_BIT`.
`mark_deleted()` sets `DELETED_BIT | SPLITTING_BIT`, deliberately forcing the
split-style unlock: `vsplit` increments and `ROOT_BIT` is cleared while
`DELETED_BIT` persists.

`new_for_split(source)` creates a split-locked version for a newly allocated
sibling: `(source & ISLEAF_BIT) | LOCK_BIT | SPLITTING_BIT`, with zero counters
and no root/deleted bits. `unlock_for_split()` increments `vsplit` and clears
the lock, yielding `vsplit == 1`, unlocked, non-root.

**OCC read protocol:**

1. `stable()`: Relaxed load; if clean, issue an Acquire fence and return the
   word. If dirty, spin with exponential backoff and yield after two spins.
2. Read node fields (permutation, keys, keylenx, value/suffix).
3. `has_changed(old)`: `(old ^ current) > LOCK_BIT`. Any change to a counter,
   `INSERTING_BIT`, `SPLITTING_BIT`, `DELETED`, `ROOT`, `UNUSED1`, or `ISLEAF`
   is detected. Changes to only the lock bit are ignored; an in-flight insert
   marks the insert bit and increments `vinsert` on unlock, so a reader that
   snapshotted before the insert sees a change.
4. `has_split(old)`: `(old ^ current) >= VSPLIT_LOWBIT`. This deliberately
   includes `ROOT_BIT` changes, so it is a superset that also catches root-state
   changes and can false-positive on `mark_root`.
5. `has_changed_or_locked(old)`: the conservative variant that also returns true
   if a dirty bit is currently set. Used by coalesce route lookups.

Counters are finite: `vinsert` wraps every 64 inserts and carries into `vsplit`;
`vsplit` wraps after ~524,288 splits. Detection compares the whole word, so any
higher-bit change is caught; a complete 25-bit wrap within one reader's window
is theoretically undetectable. This is tested in `nodeversion/unit_tests.rs`.

`bounded.rs` provides `lock_bounded()`: an immediate CAS, then at most 16 spin
attempts with backoff, then `yield_now` and reset. It is used by insert, batch
insert, and remove.

`LockGuard<'a>` is a `!Send + !Sync`, `#[must_use]` RAII token that stores a raw
`*const NodeVersion` and the locked word. It proves at the type level that the
lock is held and releases it on drop, even during unwinding.

### 4.6 B-link protocol and pointer tagging

`src/link.rs` defines `Linker` with a single low address bit (`MARK_BIT = 1`):

```rust
mark_ptr(p)   // p | 1
unmark_ptr(p) // p & !1
is_marked(p)  // p & 1 != 0
```

Leaves are 64-byte aligned, so bit 0 is free. The mark is applied to the `next`
pointer:

- During a split, the old leaf's `next` is marked while the new sibling is
  installed.
- During unlink, the predecessor's `next` is CAS-marked while the chain is
  repaired.
- Readers never dereference a marked pointer. They call
  `safe_next`/`unmark_ptr`; a marked pointer means "split or unlink in
  progress", and the reader waits (`wait_for_split`) or advances through the
  unmarked predecessor.

`prev` is used for the "too far right" check (if `prev != null` and
`target_ikey < ikey_bound`, the reader is in the wrong leaf) and for chain
unlinking. `parent` is a type-erased `AtomicPtr<u8>`; the `NodeVersion::is_leaf`
bit at offset 0 tells accessors whether to cast to `LeafNode15` or
`InternodeNode`.

---

## 5. Storage Policies

### 5.1 Trait hierarchy

`LeafPolicy` (`src/policy.rs`) is a sealed trait
(`Send + Sync + Sized + 'static`) that selects value and suffix storage:

```rust
pub trait LeafPolicy: sealed::LeafPolicySealed + Send + Sync + Sized + 'static {
    type Value: Send + Sync;                 // user-facing value type
    type Output: Clone + Send + Sync;        // handle returned by get/scan
    type Values: ValueArray<Self::Output>;
    type Suffix: SuffixStore;

    const NEEDS_RETIREMENT: bool;
    const SUPPORTS_REF: bool;
    const CAN_WRITE_THROUGH: bool = false;

    // emptiness, prefetch, raw pointer access, layer transfer, cloning,
    // output/value conversion, retirement, optional write-through update,
    // optional reference access
}
```

Implementations:

| Trait impl                    | `Value` | `Output`      | `Values`              | `Suffix`                            | `NEEDS_RETIREMENT` | `CAN_WRITE_THROUGH`           |
| ----------------------------- | ------- | ------------- | --------------------- | ----------------------------------- | ------------------ | ----------------------------- |
| `BoxPolicy<V>`                | `V`     | `ValuePtr<V>` | `BoxValueArray<V>`    | `SidecarSuffix`                     | `true`             | closed primitive list (`is_atomic_word`) |
| `InlinePolicy<V: InlineBits>` | `V`     | `V`           | `InlineValueArray<V>` | `EmbeddedSuffix` or `SidecarSuffix` | `false`            | `false`                       |

`RefPolicy` is a marker trait implemented only by `BoxPolicy`. It enables
`get_ref` and the `*_ref` scan APIs, returning `ValueRef<'g, V>` tied to the
guard's lifetime.

`ValuePtr<V>` is a `#[repr(transparent)] NonNull<V>` with `Copy` and an opaque,
address-only `Debug`. It has **no safe value accessor**: a handle can outlive
its EBR guard and the tree, so public read APIs return `ValueRef<'a, V>`
instead, either `Borrowed(&'a V)` or `Owned(V)` (an atomic copy for
write-through values). Internal code that has validated OCC and holds the guard
uses the unsafe `ValuePtr::as_ref` contract. `LeafPolicy::output_to_value_ref`
is the boundary conversion; `clone_value_from_output` and
`RefPolicy::output_as_ref` are unsafe for the same reason.

`RetireHandle` is either `Ptr(*mut u8)` or `Noop`; it must be consumed exactly
once and is used to defer destruction of replaced boxed values or suffix bags.

### 5.2 BoxPolicy

`BoxValueArray<V>` (`src/policy/box_values.rs`) is 15 `AtomicPtr<u8>` slots (120
bytes). A slot is:

- `null` — empty;
- non-null — a `Box<V>` pointer (or a layer pointer; `keylenx` is
  authoritative).

Reads use `Acquire`; writes use `Release`. `take` atomically swaps in null.
`update_in_place` reads the old pointer relaxed, installs the new pointer with
Release, and returns the old pointer as a `RetireHandle`. On replacement, the
old `Box` is retired through the epoch (`guard.defer_retire`).

**Write-through optimization.** Write-through is restricted to a closed list of
primitive integer/float/`bool`/`char` types (`is_atomic_word`), not a
size/alignment heuristic. Those types have no padding, no ownership, and every
bit pattern the store path can produce is valid, so the existing allocation's
contents can be atomically read/written through
`AtomicU8`/`AtomicU16`/`AtomicU32`/`AtomicU64` (`src/policy/box_values.rs`).
Readers use `atomic_read_value` on the write-through path and public reads
return `ValueRef::Owned`, so a concurrent update cannot produce a torn or
aliased read. The pointer itself never changes, so no reclamation is needed.

`insert_batch` uses a separate `BatchValueEntry` path for write-through policies
to defer allocation until a new slot is actually needed. Guard-scoped batch
variants return `ValueRef`s rather than manufacturing unfreeable Boxes for old
values.

### 5.3 InlinePolicy

`InlineValueArray<V>` (`src/policy/inline_values.rs`) is 15 tag pointers plus 15
`AtomicU64` payloads (240 bytes total). The encoding is:

| Tag                | Payload        | Meaning               |
| ------------------ | -------------- | --------------------- |
| `null`             | ignored        | Empty slot            |
| sentinel address   | `V::to_bits()` | Terminal inline value |
| any other non-null | ignored        | Layer pointer         |

The sentinel is the address of a real static byte (`INLINE_SENTINEL` in
`src/inline/sentinel.rs`), not a fabricated pointer, which keeps
strict-provenance and Miri happy. `store` writes the payload with Relaxed and
then publishes the sentinel tag with Release; `load` reads the tag with Acquire
and the payload Relaxed. `take` nulls the tag and leaves stale bits. `clear`
only nulls the tag. `move_slot` copies payload then tag. `update_in_place`
rewrites the 64-bit payload with Release and returns `Noop` (inline values are
never retired).

`InlineBits` (`src/inline/bits.rs`) defines `to_bits`/`from_bits` for:

- `u8`, `u16`, `u32`, `u64`, `usize` (64-bit targets)
- `i8`, `i16`, `i32`, `i64`, `isize` (64-bit targets)
- `f32`, `f64` (bit-preserving)
- `bool` (nonzero → true)
- `()`
- `(u32, u32)`
- `(u16, u16, u16, u16)`

There is no built-in implementation for `String`, `Vec`, `Arc`, `u128`, or
`Option`; larger/owned types belong in `BoxPolicy`. The trait is public, so user
`Copy` types can implement it.

### 5.4 Slot state views

`SlotKind<O>` (`Empty`, `Value(O)`, `Layer(*mut u8)`) and `SlotState` (`Empty`,
`Value`, `Layer`) are derived views, not stored state machines.
`LeafNode15::classify_slot` reads `keylenx` first:

- `keylenx >= 128` → layer;
- otherwise load the value array → value or empty.

`classify_slot_light` avoids materializing the output. The actual transition
protocol is: write the value/suffix, publish `keylenx` (suffix bytes first, then
`KSUF_KEYLENX`), then publish the permutation (Release or via unlock).

### 5.5 Policy selection in the tree

`src/tree.rs` defines the aliases:

```rust
pub type MassTree<V>          = MassTree15Inline<V>;
pub type MassTree15<V>        = MassTreeGeneric<BoxPolicy<V>,   SeizeAllocator<BoxPolicy<V>>>;
pub type MassTree15Inline<V>  = MassTreeGeneric<InlinePolicy<V>, SeizeAllocator<InlinePolicy<V>>>;
```

`MassTree::new()` therefore creates an inline tree, optimized for small `Copy`
values such as `u64`. `MassTree15::<String>::new()` creates a boxed tree. Both
share every algorithm; only the policy's value/suffix operations and node sizes
differ.

---

## 6. Suffix Storage

### 6.1 Where suffixes live

A leaf stores the first 8 bytes of each key in `ikey0`. If the key continues,
the remaining bytes are the **suffix** (1–248 bytes at layer 0, depending on key
length). `keylenx = 64` marks a suffix key. The suffix length is stored in the
suffix store's per-slot metadata, not in `keylenx`.

A leaf holds at most one suffix key per ikey. Two different suffixes for the
same ikey cannot be ordered within the leaf, so the insert path creates a
sublayer instead (§8.4). A short key (≤ 8 bytes) and a suffix key with the same
ikey _can_ coexist in one leaf: their keylenx values order them (short before
`64`). A layer pointer (`128`) sorts after both.

### 6.2 `InlineSuffixBag`

`src/suffix/inline.rs` defines the in-leaf bag:

```rust
#[repr(C)]
pub struct InlineSuffixBag {
    slots: [AtomicU32; 15],    // packed (offset: u16, len: u16); EMPTY offset = u16::MAX
    size: AtomicU16,           // bytes used in `data`
    suffix_count: AtomicU8,    // number of active suffixes
    pad: u8,
    data: UnsafeCell<[u8; CAPACITY]>, // 512 bytes default, 256 with small-suffix-capacity
}
```

Properties:

- Each `(offset, length)` pair is packed into one `AtomicU32`, so offset and
  length are read atomically together.
- `try_assign` overwrites an existing slot's allocation when the new suffix
  fits; otherwise appends at `size` if capacity allows. It returns `false` when
  the suffix does not fit, triggering spill.
- Metadata stores use `Acquire`/`Release`; `size` and `suffix_count` are
  Relaxed.
- `clear` marks the slot empty and decrements the count but does not reclaim
  bytes.
- `clone` copies the fixed data array and metadata without compaction.
- `drain_core`/`drain_to_external*` compute the space needed for active
  suffixes, copy them into a heap `SuffixBag`, and append the new suffix.
  Draining preserves the inline bytes until the external pointer is published,
  so a concurrent reader never observes neither store. The slot that was
  replaced is then cleared in the inline bag (SUF-01), so an old inline suffix
  cannot shadow the new external value. Migrated sibling slots can leave
  orphaned inline bytes, which is safe but can waste space.

The bag is `!Sync` because of its `UnsafeCell`; it relies on the leaf lock for
writes and OCC for reads. It is deliberately not re-exported from `lib.rs` as a
standalone concurrent type.

### 6.3 Heap `SuffixBag`

`SuffixBag` owns one raw allocation through atomic pointer, capacity, and used
length fields. Fifteen `AtomicU64` metadata words store
`(offset: u32, len: u16)`. The header is 144 bytes on 64-bit targets, checked at
compile time; there is no second `Vec` header or per-write suffix-count RMW.

`from_vec` takes over an empty buffer. `assign` and `reserve` require exclusive
ownership and grow geometrically. Internal initialization uses the explicit
unsafe `assign_locked` / `reserve_locked` contracts; holding a published leaf's
lock alone is insufficient because readers may retain slices.

`try_assign_append_only` copies only into spare capacity, publishes the used
length, then Release-stores the slot metadata. It never reallocates or
overwrites published bytes, including bytes whose slot was cleared or
reassigned. Readers Acquire-load metadata and `used` before creating a slice. A
`PhantomData<Cell<()>>` keeps the raw-buffer owner `!Sync`: atomic metadata does
not permit multiple writers. The tree serializes writers under the leaf lock and
keeps cells alive with EBR. Drop reconstructs the original allocation exactly
once.

The unused public compaction, clone, comparison, count, and in-place assignment
APIs, and their `PermutationProvider` trait, were removed (breaking low-level
API change). Live tree paths already use append or replacement bags, so no tree
operation loses functionality. SUF-02's empty-suffix regression now verifies
preservation through the remaining growth path.

### 6.4 Embedded vs sidecar stores

`SuffixStore` (`src/policy.rs`) abstracts two implementations:

**`EmbeddedSuffix`** (`src/policy/embedded_suffix.rs`):

```rust
#[repr(C)]
pub struct EmbeddedSuffix {
    inline_ksuf: UnsafeCell<InlineSuffixBag>,
    external_ksuf: AtomicPtr<SuffixBagCell>,
}
```

- `get` checks the inline bag first, then the external bag.
- `assign` tries inline, then append-only external, then drain-and-rebuild.
- `reserve_external` sizes an external cell on an unpublished split destination.
- `clear` clears both stores.
- `drop_storage` frees the external cell on leaf destruction.
- `init_at_zero` writes only the 60-byte slot-sentinel header into
  already-zeroed storage; count, size, and buffer remain zero from leaf
  initialization.
- `prefetch` is a no-op (the data is in the leaf).

**`SidecarSuffix`** (`src/policy/sidecar_suffix.rs`):

```rust
#[repr(C)]
pub struct SidecarSuffix {
    sidecar: AtomicPtr<SuffixSidecar>,
}
```

The sidecar is a heap-allocated `SuffixSidecar` containing an `InlineSuffixBag`
and an external `AtomicPtr<SuffixBagCell>`. It is allocated lazily on the first
non-empty suffix and never moved for the lifetime of the leaf. `BoxPolicy`
always uses this store; `InlinePolicy` uses it only with the `sidecar-suffix`
feature. The trade-off is one extra indirection per suffix access (and a lazy
allocation) in exchange for a much smaller leaf (448 B or 576 B instead of 1152
B).

`src/policy/suffix_ops.rs` contains the shared algorithms used by both stores:
inline lookup, external pointer publication, append-only assignment,
drain-and-rebuild, merging the old external bag's live suffixes using the
permutation, and EBR retirement of replaced cells. Rebuilds gather the full live
set from both stores before allocating, preferring inline data when both stores
contain a slot. Capacity includes at least 512 bytes or one full live set of
append slack. Inserts into leaves with existing external storage skip
speculative suffix-buffer allocation; a necessary rebuild sizes its buffer on
the cold path.

### 6.5 Split migration and feature effects

When a leaf splits, suffixes are **copied** to the new leaf, not moved:

- `prepare_split_leaf` initializes the new leaf with a split-locked version and
  totals the suffixes moving right, including a new right-side insertion. If
  that set exceeds inline capacity it reserves twice the required bytes before
  moving any entries. Destinations that fit inline do not allocate an external
  bag.
- `move_entry_to` copies the ikey/keylenx, moves the value, copies the suffix
  with `assign_ksuf_init` (which can drain inline suffixes on the new leaf),
  then clears the source suffix and value.
- The left permutation is truncated and published Release; the right permutation
  is `make_sorted` and stored Relaxed because the right leaf is not yet visible.

Feature effects:

| Feature                 | Effect                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------- |
| default                 | Embedded 512-byte suffix bag in `InlinePolicy`; leaf 1152 B                        |
| `small-suffix-capacity` | 256-byte inline bag; leaf 896 B; `INLINE_KSUF_CAPACITY` becomes 256                |
| `sidecar-suffix`        | `InlinePolicy` uses `SidecarSuffix`; leaf 576 B; `embedded_suffix` module excluded |

The external `SuffixBag` initial capacity remains 128 bytes in all
configurations.

---

## 7. Concurrency Architecture

### 7.1 The OCC protocol

Every node has one `NodeVersion`. Readers are lock-free:

```
v = node.version.stable()          // wait while DIRTY_MASK is set
perm = node.permutation.load(Acquire)
... read ikey/keylenx/value/suffix/child ...
if node.version.has_changed(v) { retry / follow B-link / restart }
```

`stable()` spins only if a writer is currently dirty. A clean version is read
with a Relaxed load followed by an Acquire fence; `try_stable()` uses a single
Acquire load and returns `None` if dirty. The fence/load pairs with the writer's
Release unlock.

Writers serialize on the same word:

```
let mut guard = node.version.lock_bounded(); // CAS LOCK_BIT
guard.mark_insert();                          // or mark_split / mark_deleted
... mutate slot data ...
drop(guard);                                  // Release: increment vinsert or vsplit
```

While the lock is held, data writes may be Relaxed because the lock provides
mutual exclusion; the Release unlock publishes them. `mark_insert`/`mark_split`
use a Release store plus Acquire fence so that the dirty bit is visible before
subsequent data stores.

`LockGuard` prevents accidental double-unlock and is `!Send + !Sync`; it is
released on drop during unwinding.

### 7.2 Linearization points

| Operation         | Linearization point                                                              |
| ----------------- | -------------------------------------------------------------------------------- |
| Point read        | successful `has_changed` validation after reading the slot                       |
| Insert (new slot) | permutation publication + Release unlock                                         |
| Insert (update)   | in-place value store + Release unlock; boxed replacement retires the old pointer |
| Remove            | Release store of the new permutation after clearing value/keylenx/suffix         |
| Leaf split        | `link_sibling`'s Release `next` store + parent insertion + `unlock_for_split`    |
| Root creation     | Release store to `root_ptr` (main root) or parent-link installation (layer root) |
| Delete/coalesce   | `DELETED_BIT` set + unlink + retirement                                          |

The effective publication point for a new insert is the permutation/unlock pair,
not the individual slot write.

### 7.3 Memory-ordering policy

`src/ordering.rs` centralizes the orderings:

| Constant      | Ordering  | Used for                                                                                    |
| ------------- | --------- | ------------------------------------------------------------------------------------------- |
| `READ_ORD`    | `Acquire` | concurrent loads of ikeys, permutation, children, values, next/prev/parent, suffix metadata |
| `WRITE_ORD`   | `Release` | corresponding stores and permutation publication                                            |
| `CAS_SUCCESS` | `AcqRel`  | linking CAS, root/link publication                                                          |
| `CAS_FAILURE` | `Acquire` | reading the current pointer after a failed CAS                                              |
| `RELAXED`     | `Relaxed` | data under the lock, unpublished nodes, prefetch, counters                                  |

The writer's Release unlock synchronizes with a reader's `stable()` fence/load.
Field-level Acquire loads pair with Release stores for pointers and suffix
metadata. `has_changed` itself uses an Acquire load, so all prior reader loads
are ordered before the validation check.

### 7.4 Split protocol (concurrency view)

A leaf split is a two-phase operation designed so readers never see a partially
split node:

1. **Prepare (before the point of no return).** The left leaf is locked. The
   split point is chosen, root flags are captured (because split unlock clears
   `ROOT_BIT`), and the right leaf is allocated but remains unpublished. After
   the mark, it is initialized split-locked with `new_for_split`.
2. **Mark.** `lock.mark_split()` sets `SPLITTING_BIT`. From here on, propagation
   must complete; all subsequent allocations are infallible (abort on OOM).
3. **Move + insert.** `split_and_insert` moves entries to the right leaf or
   inserts into either side, builds both permutations, and returns
   `SplitInsertResult { split_ikey, insert_target }`. The right leaf's version
   remains split-locked.
4. **Link.** `link_sibling` marks the left `next`, sets the right's
   `prev`/`next`, updates the old successor's `prev`, then publishes the new
   sibling with a Release store.
5. **Parent insertion.** The propagation loop locks the parent and validates
   membership. If there is room, it inserts `(split_ikey, right_ptr)`, sets the
   right leaf's parent, and unlocks the right leaf.
6. **Parent split if full.** The parent is marked split, a split-locked sibling
   is allocated, `split_into` moves keys/children (mid = 8), children are
   reparented, and the loop moves up to the grandparent with the popup
   separator.
7. **Root creation.** At the top, a new root internode is allocated. For the
   main root, it is published with a Release store to `root_ptr`, then child
   parent pointers are installed and the old left child's `ROOT_BIT` is cleared.
   For a layer root, a new layer internode is created and linked via parent
   pointers; the parent leaf's layer slot is not rewritten.

Readers detect the split through `has_split` (vsplit changed). If the key is
greater than the old leaf's last key, they follow `next`; a marked `next` means
the link is still being installed, so they wait for the version to become stable
and retry.

### 7.5 B-link and deleted-leaf handling

Leaves form a doubly linked chain. `next` has three states:

- `null` — end of chain;
- unmarked pointer — successor;
- marked pointer — split/unlink in progress.

Reader behavior:

- `check_blink_chain` follows an unmarked `next` if the target ikey is at or
  beyond the next leaf's `ikey_bound`.
- A marked `next` triggers `wait_for_split` (spin until stable or deleted) and a
  retry.
- A deleted leaf (`DELETED_BIT`) is skipped by following its unmarked `next`; if
  that is not possible, the reader re-reaches from the layer root. If the
  deleted leaf was the root of a sublayer, the reader calls `unshift_all` and
  restarts from the main root.
- The "too far right" check: if `prev != null` and `target_ikey < ikey_bound()`,
  the reader was routed to a leaf that is now to the right of the key and
  restarts from the root.

This is why readers can remain lock-free while leaves are split, unlinked, and
retired underneath them.

### 7.6 Epoch-based reclamation

Every tree owns a `seize::Collector`. `tree.guard()` enters the collector and
returns a `LocalGuard`. Guards can be reused for many operations; they must not
outlive the tree, and a guard from one tree must not be used with another
(`verify_guard` panics).

Retirement paths:

- Leaves: `guard.defer_retire(ptr, reclaim_leaf15::<P>)`; the reclaimer runs
  `drop_in_place` then returns the block to the thread-local pool.
- Internodes: `guard.defer_retire(ptr, reclaim_internode)`.
- Boxed values: `guard.defer_retire(ptr, drop(Box::from_raw(ptr)))`.
- Suffix bags/sidecars: dedicated reclaimers.
- Coalesced leaves/sublayers: retired after unlinking.

`guard.flush()` moves the local retirement batch into the collector's
reservation lists; `guard.refresh()` advances the epoch.
`MassTreeGeneric::maintenance` does flush + refresh + coalesce processing.
`Drop` clears the coalesce queue, calls `collector.reclaim_all()` under
`&mut self` (sound because no guards can remain), then calls
`allocator.teardown_tree(root)`.

The default retirement batch size is 256 (`with_batch_size` overrides), trading
syscall frequency against reclamation latency.

### 7.7 Lock ordering and backoff

- Insert and split propagation hold a child lock and then lock its parent (child
  → parent). A concurrent splitter of the same parent is detected by membership
  revalidation; the operation backs off and retries.
- Coalesce of an ordinary empty leaf holds the leaf lock and locks its parent
  (child → parent). The parent cleanup happens _before_ the leaf is marked
  deleted, so a failure leaves the leaf alive and requeues it.
- Sublayer GC locks the parent leaf first, then the sublayer root (parent →
  child), using `try_lock` and requeueing on failure rather than blocking.
- All blocking locks are spin locks. `lock_bounded` caps at 16 spins then
  yields; general `lock` uses exponential backoff (1, 2, 4, 8, 16, 16, … pause
  spins). Split contention uses a separate capped backoff (up to 64 spins, then
  yield).
- There is no writer fairness or priority inheritance; long critical sections
  (splits, suffix drains) can delay readers, which is why the implementation
  keeps them short and retries outside locks.

### 7.8 Allocation, node pool, and teardown

`TreeAllocator<P>` (`src/alloc_trait.rs`) is the allocation contract:

```rust
fn alloc_leaf_direct(is_root, is_layer_root) -> *mut LeafNode15<P>;
unsafe fn retire_leaf(ptr, guard);
fn alloc_internode_direct(height) -> *mut u8;
fn alloc_internode_direct_root(height) -> *mut u8;
fn alloc_internode_direct_for_split(parent_version, height) -> *mut u8;
unsafe fn retire_internode_erased(ptr, guard);
fn teardown_tree(root_ptr);
```

The only in-tree implementation is `SeizeAllocator<P>` (`src/alloc15.rs`), a
zero-sized type containing only `PhantomData<P>`. Allocation is in-place:

- leaf: `pool_alloc_leaf::<P>()`, then
  `LeafNode15::init_at(ptr, is_root || is_layer_root)`;
- ordinary internode: `pool_alloc_internode()`, then
  `InternodeNode::init_at(ptr, height)`;
- root internode: `InternodeNode::init_at_root(ptr, height)` (sets `ROOT_BIT`);
- split sibling:
  `InternodeNode::init_at_for_split(ptr, parent_version, height)`, which writes
  a split-locked `NodeVersion`.

`src/node_pool.rs` keeps a thread-local `ThreadPool` with 20 size-class buckets,
one per cache-line count up to 1280 bytes. A bucket is an intrusive freelist
(`head`, `count`) with capacity 512; a refill allocates 128 blocks at a time
from the global allocator. Blocks are cache-line aligned and carry no per-node
header; the first 8 bytes of a free block store the next pointer. Nodes may be
freed on a different thread than they were allocated (seize batch reclamation),
so they land in the reclaimer's pool and may be reused by that thread. Teardown
bypasses the pool and calls the global allocator directly so that no blocks
remain cached in TLS after tree destruction.

`teardown_tree` is iterative to avoid stack overflow: it uses an
`ArrayVec<*mut u8, 128>` with a heap `Vec` overflow, dispatches on
`NodeVersion::is_leaf`, and for a leaf pushes every layer slot's
`find_layer_root` (following parent pointers to the true layer root) before
deallocating the leaf. Internodes push children `0..=nkeys` in reverse and are
deallocated without `drop_in_place` (they own no heap data). `find_layer_root`
is essential because a layer slot may point at a non-root leaf after layer-root
promotion.

---

## 8. Operations

### 8.1 Concurrent traversal (`reach_leaf_concurrent_generic`)

`src/tree/generic.rs` implements the core descent used by reads, writes, remove,
and scans:

1. `target_ikey = key.ikey()`. Two alternating pointer/version slots (`n[2]`,
   `v[2]`, `sense`) implement a double buffer.
2. **Climb.** Starting from the supplied node (main root or layer root), follow
   parent pointers until a node with `ROOT_BIT` or `DELETED_BIT` is found. This
   normalizes a pointer to a promoted layer's original leaf into the current
   layer root.
3. **Descend.** Load the current node. `try_stable()` returning `None` restarts
   from the start. If the node is a leaf, return it.
4. For an internode, compute
   `child_idx = upper_bound_internode(target_ikey, node)`, load the child with a
   guard-protected Acquire load, and prefetch it. If the parent's children are
   internodes, also prefetch the child's header and first child.
5. Read the child's stable version, then validate the parent with
   `!parent.version.has_changed(v)`. If unchanged, switch `sense` and continue.
   If changed but not split, recompute the child from the parent's new stable
   version. If split and the target is greater than the parent's last key,
   restart from the start.

`maybe_parent_generic` repeatedly follows parent pointers until null; it is used
at layer boundaries and by insert/remove/scan to find the actual root of a
layer.

### 8.2 Point reads

Public entry points:

```rust
tree.get(key)                    // creates an internal guard; requires V: Clone
tree.get_with_guard(key, &guard) // returns Option<P::Output>
tree.contains_key(key)
tree.get_ref(key, &guard)        // BoxPolicy only; returns ValueRef<'g, V>
```

`get_with_guard` dispatches on `key.has_suffix()`:

- **Single-layer** (`current_len <= 8`): reach leaf, handle
  deleted/B-link/too-right, `try_stable`, scan the permutation for equal ikey +
  equal keylenx + non-empty value, prefetch the value, load the output, then
  validate. On a version change with split, `advance_to_key_generic` may move to
  the next leaf; otherwise retry the search.
- **Multi-layer** (`current_len > 8`): the same loop, but
  `search_leaf_multi_layer` scans equal-ikey slots and compares suffixes. If the
  slot is a layer pointer and the query still has a suffix, the reader descends.
  `descend_twig_chain` optimizes single-entry "twig" layers: it follows layer
  pointers without re-running a full internode descent, and falls back to a root
  re-descent if the twig was garbage-collected or promoted.

The value/suffix snapshot is validated after the complete payload read and
before it is returned. `get_with_guard` returns `ValueRef<'g, P::Value>`:
`Borrowed` for non-write-through boxed values and `Owned` (atomic copy) for
write-through values, so a concurrent writer cannot mutate the referenced
memory. `get()` clones the value while its internal guard is alive; `get_ref`
uses the same guard-scoped model. For `InlinePolicy`, an owned copy is returned.

### 8.3 Insert

Public entry points and their differences:

| API                                     | Value input | Output              | Allocation behavior                                                                 |
| --------------------------------------- | ----------- | ------------------- | ----------------------------------------------------------------------------------- |
| `insert(key, value)`                    | `P::Value`  | `Option<P::Value>`  | uses write-through when `CAN_WRITE_THROUGH`; otherwise `into_output` then clone old |
| `insert_with_guard(key, value, &guard)` | `P::Value`  | `Option<P::Output>` | always `into_output` up front (even for write-through policies)                     |
| `insert_value_with_guard`               | `P::Value`  | `Option<P::Value>`  | uses write-through when available                                                   |
| `insert_output_with_guard` (crate)      | `P::Output` | `Option<P::Output>` | used by the Entry API                                                               |

The core is `insert_concurrent<S: InsertStrategy>`
(`src/tree/generic/insert.rs`). `InsertStrategy` deduplicates the generic and
write-through paths:

- `GenericInsert`: `Input = Output = P::Output`, identity conversion; updates
  call `update_existing_value` (which may retire an old box).
- `WriteThroughInsert`: `Input = P::Value`, `Output` allocated lazily only when
  a new slot is needed; updates call `update_existing_value_write_through` (no
  allocation/retirement).

Full flow:

1. `single_layer_mode = !key.has_suffix()`. Load the main root;
   `in_sublayer = false`.
2. `'retry`: `layer_root = maybe_parent_generic(layer_root)`;
   `reach_leaf_concurrent_generic`.
3. `advance_to_key_by_bound_generic`: follow B-links forward while the key is
   at/above the next leaf's `ikey_bound`, with a 128-hop limit. On exhaustion,
   `unshift_all`, reload the root, and retry.
4. `'forward`: snapshot the leaf version and permutation; run the optimistic
   insert search (`search_for_insert_single_layer` or
   `search_for_insert_generic`). Search results are:
   - `Found { slot }` — exact key;
   - `NotFound { logical_pos }` — insertion position;
   - `Conflict { slot }` — same ikey, different suffix (layer needed);
   - `Layer { slot }` — existing layer pointer.
5. **Layer fast path:** if the result is `Layer` and the pointer is non-null,
   drop to the sublayer without locking the parent: `key.shift()`,
   `layer_root = ptr`, `in_sublayer = true`, `'retry`.
6. **Suffix preallocation:** if the key has a suffix and no buffer is available,
   reserve `WIDTH * suffix_len` bytes when the suffix exceeds inline capacity or
   `perm_size * suffix_len >= inline_capacity`. The buffer is passed to
   `assign_ksuf_prealloc`.
7. **Lock + validate:** `lock_bounded`. `validate_post_lock` requires both the
   version and the raw permutation to be unchanged (catching ABA).
   `validate_membership` rejects a marked `next`, a key below `ikey_bound`, or a
   key at/above the next leaf's bound; each outcome drops the lock and
   advances/retries. A `deleted_layer` leaf triggers an unshift/root retry.
8. **Empty leaf reuse:** if the permutation size is 0 and the leaf is reusable
   (leftmost, or `ikey_bound == key.ikey`), clear the empty state, assign
   physical slot 0, set `make_sorted(1)`, unlock, increment count, return `None`
   for the old value.
9. **Found:** if the value became empty (concurrent remove), continue forward;
   otherwise update in place, unlock, retire any old box, return the old output.
10. **NotFound:** `find_usable_slot` checks capacity (`size >= 15` → split),
    takes `perm.back()`, applies the slot-0 rule, and returns a physical slot.
    On success, `assign_slot_generic` writes ikey/value/keylenx/suffix, swaps
    the chosen free slot to the back if necessary, inserts the logical position,
    verifies the allocated slot, stores the permutation, unlocks, retires any
    deferred suffix bag, increments count, and returns `None`.
11. **NeedsSplit:** build `SplitInsertData` (ikey, keylenx, suffix, value) and
    call `handle_leaf_split_and_insert_generic` with the lock transferred by
    value. Count increments once; the return is `None`.
12. **Conflict:** `handle_suffix_conflict` marks insert, creates a sublayer,
    takes the old value for the layer, retires the old handle if needed, clears
    the old suffix, sets `keylenx = 128`, and stores the layer pointer. Count
    increments once; return `None`.

`assign_slot_generic` (`src/tree/generic.rs`) owns only the slot payload: it
marks the leaf inserting if needed, writes ikey/value, and writes either
`KSUF_KEYLENX` + suffix or `current_len`. It does not touch the permutation or
the count.

### 8.4 Layer creation

A layer is created when the insert search returns `Conflict`: the same ikey is
already present with a suffix, and the new key's suffix differs.
`create_layer_concurrent_generic` (`src/tree/generic/layer.rs`) builds the
sublayer:

1. Read the existing suffix and construct
   `existing_key = Key::from_suffix(existing_suffix)`. Clone the existing output
   (the conflict slot must be terminal).
2. Shift the new key once if it has a suffix, so both keys are positioned at the
   chunk after the shared ikey.
3. Compare the two keys.
4. While the keys are equal and both still have suffixes (i.e. they share
   another full 8-byte chunk), allocate a single-entry **twig** leaf with
   `ROOT_BIT` and a null parent, store the shared ikey at slot 0, set
   `make_sorted(1)`, and link the previous twig's slot 0 to it with
   `keylenx = 128`. Shift both keys and repeat.
5. Allocate the final leaf and assign the two keys in order: less first, greater
   second; if equal, shorter `current_len` first (prefix before extension). Each
   key is written with `assign_from_key`, which sets ikey, value, and either a
   suffix (`KSUF`) or a terminal length.
6. Wire the last twig (or return the final leaf directly) and give the parent
   leaf's slot the returned pointer with `keylenx = 128`.

A twig chain is an optimization for long common prefixes: it avoids creating a
full internode for each identical 8-byte chunk. The twig leaves are roots with
`ROOT_BIT`; the read path recognizes a single-entry twig and follows it
directly.

### 8.5 Split and root creation (operation view)

`handle_leaf_split_and_insert_generic` (`src/tree/generic/split.rs`):

1. **Prepare** (`prepare_split`): compute the split point with
   `calculate_split_point`, capture `is_main_root`/`is_layer_root` before
   `mark_split` (split unlock clears `ROOT_BIT`), and allocate the right leaf.
   The split path is infallible; `calculate_split_point` selects a strict
   post-insert ikey boundary (with a defensive midpoint fallback only for an
   impossible single-run leaf).
2. **Mark:** `lock.mark_split()`. Point of no return.
3. **Split+insert:** `left_leaf.split_and_insert(...)`:
   - `calculate_split_point` chooses a strict boundary after splicing in the new
     key; append, prepend, and midpoint-adjacent cases all walk to that
     boundary. `split_and_insert` routes the new key **left** when
     `insert_pos == split_pos` and its ikey equals the left leaf's last key, so
     the parent separator is strictly greater than every left key.
   - The right leaf is initialized split-locked. Entries are moved/copied, the
     new key is inserted on the correct side, and both permutations are built.
     The new value is stored from its pre-encoded `PreparedValue` token, so user
     encoding cannot run after `mark_split`.
   - `link_sibling` installs the B-link (marked while linking).
   - Returns the right leaf's first ikey as the separator and which side
     received the insert.
4. **Propagate** (`Propagation::propagation_loop`):
   - Get the left parent. If null and layer root → `promote_layer_root` (new
     layer internode, set children/parents, clear root bits, unlock right). If
     null and main root → `create_main_root` (new internode at height 0 or
     `left.height()+1`, Release store to `root_ptr`, set parents, clear left
     root bit).
   - Otherwise lock the parent, re-read the left pointer's parent, and validate
     membership by scanning `child[0..=nkeys]`. If the child moved, drop the
     lock, back off, retry.
   - If the parent has room: mark insert, `insert_key_and_child`, set the right
     child's parent, unlock the right leaf, drop locks, done.
   - If full: mark split, allocate a split-locked sibling, `split_into` at
     mid=8, reparent moved children, set the right sibling's parent, unlock it,
     then hand over: the full parent becomes the new left node and the sibling
     becomes the new right node, with the popup separator. Repeat at the
     grandparent.
5. **Unlock right:** `unlock_for_split` publishes `vsplit=1`, unlocked,
   non-root, only after the parent pointer is installed.

Root creation uses an unconditional Release store to `root_ptr`, not a CAS. This
is safe because only the holder of the old root's lock can reach root creation,
and the no-abandon protocol ensures propagation completes after `mark_split`. A
layer-root promotion does not update the parent leaf's layer slot: the slot
still points at the original left leaf, whose parent pointer now points at the
new layer internode. Readers climb parent pointers to find the true layer root.

### 8.6 Remove

Public API:

```rust
tree.remove(key)              // internal guard; returns Result<Option<V>, RemoveError>
tree.remove_with_guard(key, &guard) // returns Result<Option<P::Output>, RemoveError>
```

`RemoveError` has a single variant, `RetryLimitExceeded`, produced when the
outer layer loop exceeds 1,000 retries. Not-found is `Ok(None)`.

Algorithm (`src/tree/remove.rs`):

1. Build a `Key`; set `single_layer_mode = !key.has_suffix()`. Load the main
   root and an empty `Route` (`Vec<u64>` of per-layer ikeys, used for sublayer
   GC).
2. `'layer_loop`: `reach_leaf_concurrent_generic`, then `'forward`:
   - `stable()` + load permutation; `search_for_remove` scans logically,
     comparing ikey, keylenx, and suffix bytes. It returns `NotFound`,
     `Found { ki, kp }`, or `DescendLayer`.
   - If the version changed, retry the leaf search.
   - `NotFound` → `Ok(None)`.
   - `Found` → `lock_and_verify_for_remove`:
     - `lock_bounded`; reject `deleted_layer` with `RestartFromRoot`.
     - Re-read the permutation; require `size > ki`; re-run the full-key search
       (ikey, keylenx, suffix) under the lock, so a same-ikey insert/remove
       cannot shift the logical position onto another key.
     - If the slot is a layer, validate the sublayer and descend (push route,
       shift key).
     - Otherwise return a `RemoveCursor` holding the lock.
   - `DescendLayer` → push `key.ikey()` to the route, `key.shift()`,
     `layer_root = layer_ptr`, `'layer_loop`.
   - `Retry` → re-scan the same leaf. `RestartFromRoot` → `unshift_all`, reload
     root, clear route.
3. `RemoveCursor::finish_remove`:
   - `mark_insert`; `take_value` (null the slot); if `NEEDS_RETIREMENT`, retire
     the old box through EBR.
   - `set_keylenx(slot, 0)`; if the old keylenx was `KSUF`, `clear_ksuf`.
   - `perm.remove(ki)`; publish with a Release `set_permutation` (readers read
     it without the lock).
   - `dec_count`; if size is 0, call `schedule_coalesce_cold`.

Removal never touches internodes directly. It only empties the leaf and enqueues
it. Parent cleanup happens later in coalesce.

### 8.7 Lazy coalescing and GC

`src/tree/coalesce.rs` implements deferred cleanup. `CoalesceQueue` has four
lock-free `crossbeam::SegQueue`s plus an abandoned counter:

| Queue                | Entry                                                | Purpose                                                                   |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `chains`             | `ChainEntry { leaf_ptr, ikey_bound, requeue_count }` | Empty non-sublayer leaf, pointer-based, processed in the same guard scope |
| `sublayers`          | `SublayerEntry { route, requeue_count }`             | Empty leaf inside a sublayer; re-traverse from root                       |
| `deferred_chains`    | `DeferredChainEntry { ikey_bound, requeue_count }`   | Retry next maintenance; no pointer, re-traverse                           |
| `deferred_sublayers` | `SublayerEntry`                                      | Retry next maintenance                                                    |

Deduplication uses the leaf's queued bit (`try_mark_queued`). `schedule_chain`
is used when the route is empty (top-level leaf); `schedule_sublayer` when the
leaf is inside a layer.

Processing (`Coalesce::process_all` / `process_batch`) starts by draining
**both** deferred queues into their main queues, then repeatedly:

1. one sublayer entry (dead sublayer trees consume memory and serve no lookups);
2. one chain entry (re-traverse from root).

`process_all` limits a call to one attempt per entry that was present at entry,
so a failed entry cannot burn its retry budget in a single call. `process_batch`
uses its explicit limit.

**Chain processing:**

- `try_lock` the leaf; if size > 0 or `deleted_layer`, skip.
- If `prev == null`, the leaf is leftmost and cannot be unlinked; clear its
  queued bit.
- Otherwise `finish_chain_coalesce`:
  - **Parent cleanup before delete:** `remove_leaf_from_immediate_parent` locks
    the parent, finds the child slot, nulls it, shifts keys/children down, and
    redirects ancestor ikey bounds if the leftmost child was removed. If the
    parent is now empty and non-root, it returns the parent lock for cascade.
  - On success: `mark_deleted`, `unlink_from_chain`, `retire_leaf`. If the
    parent became empty, `try_cascade_internodes` collapses empty non-root
    internodes upward.
  - On failure: requeue with an incremented count; `MAX_REQUEUE_COUNT = 10`,
    then increment `abandoned`.

**Sublayer GC:**

- `find_sublayer_parent` re-traverses the `Route` from the root (OCC + B-link,
  max 64 hops).
- Lock the parent leaf, re-find the layer slot under the lock, load the sublayer
  pointer.
- Lock the sublayer root _parent-first_ (try_lock; failure requeues).
- If the sublayer is deleted or non-empty, clear its queued bit and stop.
- If it has `prev` or `next` siblings, requeue (not isolated).
- Otherwise `mark_insert` the parent, `clear_slot_and_permutation`, optionally
  mark the parent empty and queue it, capture the parent's `ikey_bound`,
  `mark_deleted_layer` the sublayer, drop locks, and retire the sublayer leaf.
  Cascade the parent if it became empty.

**Deferred entries:** requeues always target a deferred queue, so each entry
gets at most one attempt per maintenance call. Chain deferred entries store only
`ikey_bound` and re-traverse from the root, avoiding cross-call pointer lifetime
issues. Requeue exhaustion leaves the queued bit set; it is cleared when the
leaf is reused (`clear_empty_state`) or when the tree is dropped.

`maintenance(&mut guard)` runs `process_all` and then `guard.flush()` +
`guard.refresh()`, so nodes retired by coalescing are reclaimed in the same
call. The tree also exposes `process_coalesce`, `process_coalesce_batch`,
`pending_coalesce`, `coalesce_deferred`, and `coalesce_abandoned` for
diagnostics and manual maintenance.

### 8.8 Range scans

#### API

- Iterator: `range(start, end, &guard)`, `iter`, `keys`, `values`, `first`,
  `last`, `first_with_guard`, `last_with_guard`.
- Callback: `scan`, `scan_intra_leaf_batch`, `scan_rev_batch`, `scan_values`,
  `scan_values_rev`, `scan_prefix`, `scan_prefix_values`.
- Reference variants (`BoxPolicy`): `scan_ref`, `scan_batch_ref`,
  `scan_intra_leaf_batch_ref`, `scan_rev_batch_ref`.
- Collectors: `collect_entries`, `collect_keys`, `collect_values`.

`RangeBound` is `Unbounded`, `Included(&[u8])`, or `Excluded(&[u8])`; it
converts from `std::ops::Bound`. `ScanEntry<O>` owns a `Vec<u8>` key and an
output value. The callback APIs avoid the key allocation and value clone.

`RangeIter` implements `Iterator + DoubleEndedIterator + FusedIterator`;
`KeysIter` and `ValuesIter` implement `Iterator + FusedIterator` but not
`DoubleEndedIterator`. The iterator borrows the tree and guard; `ScanEntry`
values are valid only while the guard is held.

#### Cursor and stacks

`CursorKey` (`src/tree/range/cursor_key.rs`) is a fixed `[u8; 256]` buffer plus
`offset`, `len`, a cached big-endian `ikey`, and an explicit `unshift_sentinel`
flag. It describes the current scan position at the current layer; parent key
bytes remain in the buffer. `shift` consumes 8 bytes; `shift_clear` enters a
sublayer at its minimum (ikey 0); `shift_clear_reverse` enters at its maximum
(ikey `u64::MAX`). `unshift` ascends and marks the cursor as a sentinel, meaning
"treat the parent layer-pointer slot as equal". The flag, not the length,
distinguishes the sentinel from a real 9-byte key (`len == 9`); `full_key()` is
meaningful after the emitted slot's key is materialized.

`ScanStackElement` holds a layer root, current leaf, cached version and
permutation, logical `ki`, and `last_ikey` for monotonicity. `LayerStack` is an
`ArrayVec<LayerContext, 6>` with heap spill for deeper tries. Reverse uses
`BackStackElement` with a signed `ki` (`-1` = before first).

The scan state machine (`src/tree/range/scan_state.rs`) is:

```
Forward: Emit | FindNext | Down | Up | Retry
Reverse: Emit | FindPrev | Down | Up | Retry
```

- `FindNext`/`FindPrev`: find the next slot/leaf.
- `Down`: enter a sublayer.
- `Up`: return to the parent layer.
- `Retry`: re-descend from the current layer root.
- `Emit`: emit a slot.

Flags (`iter_flags.rs`) pack `EXHAUSTED`, `INITIALIZED`, `EMIT_EQUAL`,
`NEEDS_DUPLICATE_CHECK`, `CURSOR_EMITTED`, `SINGLE_LAYER_MODE` (forward),
`UPPER_BOUND` (reverse), and a pending descent mode (`DOWN_FOLLOW_SUFFIX` /
`DOWN_FRESH_SUBTREE`). `single_layer_mode` is enabled when the start key and any
bounded end key are at most 8 bytes (an unbounded end is always compatible); a
sublayer descent disables it.

`CURSOR_EMITTED` records whether the cursor holds a key this scan already
emitted. It lets a retry use a strict lower/upper bound so it cannot re-emit
that key. `DOWN_FOLLOW_SUFFIX` means a pending sublayer descent shares the
cursor's 8-byte prefix, so the real suffix must be carried into the sublayer;
`DOWN_FRESH_SUBTREE` means the layer has a different prefix, so every key in it
is beyond the last emitted key and the dedup state is cleared.

#### Forward algorithm

`find_initial` reaches a leaf with `reach_leaf_for_scan` (OCC internode
descent), rejects deleted leaves, caches the permutation, and finds the first
slot at/after the start bound with `lower_with_position`. A layer pointer
descends; a suffix key is compared bytewise; an inline key is emitted only if
`emit_equal` or it is past the bound.

`find_next_generic` checks the current leaf, enforces `slot_ikey >= last_ikey`,
optionally deduplicates against the cursor, and either descends a layer or
emits. When the leaf is exhausted, `advance_leaf_generic` follows `next`; a
marked pointer waits for the split, a null pointer ascends, and a deleted leaf
is retried. `find_retry` re-descends from the current layer root.

Slot reads follow a snapshot/validate/commit protocol: read the ikey, keylenx,
suffix, and value (or layer pointer) first; validate the cached leaf version;
only then write the cursor, push the layer stack, or publish a snapshot.
`find_initial` performs its position search and initial emission in a scratch
`CursorKey`, so a failed validation cannot overwrite the start bound.

`find_initial`/`find_next` are wrapped by `ForwardScanCtx::advance`, which
drives the state machine and handles `Down`/`Up` transitions. `handle_down`
consumes the recorded descent mode: a same-prefix real suffix is carried into
the sublayer with `shift()`; otherwise the cursor moves to the sublayer minimum
with `shift_clear()`, clearing the dedup state only for a different prefix.
`handle_up` pops the layer and repositions in the parent after the layer
pointer.

#### Reverse algorithm

Reverse starts from the end bound (`ReverseScanCtx::new_with_bound`) using
`CursorKey::for_reverse_scan` (unbounded → ikey `u64::MAX`, len 9).
`find_initial_reverse` reaches a leaf, uses `stable_reverse` to follow `next`
while the cursor is at/after the next leaf's bound (handling keys moved right by
splits), then `lower_reverse` picks the slot at or before the bound. Layer
pointers descend; suffix keys compare stored suffix bytes; inline keys emit only
when `emit_equal || upper_bound`. Slot payloads are read before validation, and
the cursor, flags, and stack are committed only after `has_changed(version)`
succeeds. Reverse descent follows a same-prefix suffix with `shift()`; a
different prefix starts the sublayer at its maximum with
`shift_clear_reverse()`. A retry uses a strict upper bound when the cursor
already holds an emitted key, so it is not re-emitted.

`find_prev_generic` walks slots downward, enforces non-increasing ikeys, and
descends layers. `advance_to_prev_leaf_generic` follows `prev` and starts at
`size-1`. `reposition_back_generic` retries up to 16 times from the current
layer root; on exhaustion it returns `Retry` so the outer state machine can
restart (this fixes a prior bug where exhaustion ended the scan early).
`handle_down_back` uses `shift_clear_reverse + upper_bound`; `handle_up_back`
anchors the parent position by pointer identity of the completed sublayer root.

Reverse has no final end-bound recheck at emission; this is why value-only
reverse scans document an approximate bound for suffixes. The iterator path
materializes keys and checks exact bounds.

#### Prefix scans

`scan_prefix_inner`:

1. Assert `prefix.len() <= 256`.
2. Compute an exclusive upper bound on the stack: increment the rightmost
   non-`0xFF` byte (`"abc"` → `"abd"`, `"ab\xFF"` → `"ac"`); an all-`0xFF`
   prefix has an unbounded end.
3. `descend_prefix_layers` walks complete 8-byte chunks through layer pointers
   using `find_layer_child_root` (OCC read, version/deletion checks, equal-ikey
   layer slot with non-empty value).
4. If descending:
   - **Aligned prefix** (`len == chunks*8`): the exact prefix key lives in the
     parent layer, so fetch it separately with `get_with_guard`, then scan the
     sublayer unbounded. This emits the exact value first.
   - **Unaligned prefix**: scan the descended layer with
     `Included(prefix)..Excluded(successor)`.
5. If no descent: a normal root scan `Included(prefix)..Excluded(successor)`.

`scan_prefix_values` uses the values-only batch path and documents that
non-aligned prefixes have an approximate end bound (it may over-include entries
sharing the boundary ikey).

#### Scan concurrency semantics

Scans are **not snapshot-isolated**. Each leaf is read with OCC and validated.
Inserts ahead of the cursor may appear; inserts behind it do not. Removed slots
are skipped. Emitted keys are strictly monotonic. Concurrent splits are handled
by B-links and marked pointers; deleted leaves are skipped or retried. Duplicate
suppression uses `NEEDS_DUPLICATE_CHECK` plus cursor comparison after retries,
layer transitions, and splits.

#### Batch scans

`batch_common.rs` is direction-parameterized (`Forward`/`Backward` zero-sized
markers, `BatchDirection` trait). `process_batch_keyed` loops over a leaf's
permutation, materializes keys only when needed, descends layers, and validates
OCC once per leaf. Both batch processors follow a strict snapshot protocol: read
the slot metadata/suffix/value first, validate the leaf version, and only then
commit the cursor or layer stack. A callback that mutates the current leaf
therefore cannot corrupt the position a retry uses. `RangeBound::cursor_bound`
classifies a bound inside the current sublayer once (`Inside`, `At(ikey)`, or
`Outside`) by comparing the consumed prefix, so a values-only scan never
descends a layer whose keys are all beyond the bound and never needs a per-slot
key comparison inside a wholly-in-range sublayer. `process_batch_values` still
skips interior key materialization, performing a full-key bound check only at
the bound's 8-byte ikey. The cursor records ikey/length (and the `KSUF` suffix)
for cold retries, and both directions track whether the cursor holds an
already-emitted key so a retry cannot re-emit it. `batch_forward.rs` and
`batch_reverse.rs` expose `for_each*` methods that drive the scan state machine
with per-leaf batch strategies (`IntraLeafRefStrategy`, `IntraLeafCopyStrategy`,
`ValuesOnlyStrategy`). These are the fastest scan APIs and avoid per-entry
allocation except for the key buffer.

### 8.9 Batch insert

`src/tree/generic/batch.rs` provides:

```rust
tree.insert_batch(entries)                 // entries: (Vec<u8>, P::Value)
tree.insert_batch_with_guard(entries, &guard)
tree.insert_batch_entries(entries)         // pre-built BatchEntry<P>
```

`BatchEntry<P>` owns a key, an optional user value, cached ikey, and an input-order
sort tiebreaker. There is one runner for both policies; the old `OutputEntry`,
`BatchValueEntry`, `GenericBatch`, and `BatchOps` variants are gone.

1. Collect owning entries and validate every key before tree mutation or boxed
   output conversion. A bad key or iterator panic drops all pending values.
2. Sort by `(ikey, full key, input position)`, so duplicate keys are deterministic
   and the last input value wins.
3. Reach a leaf, validate its version/membership, lock once, and process the run
   below its next sibling's bound. Full leaves still allow in-place updates.
4. Use write-through for primitive boxed updates; convert other values only when
   their destination is known. Split/layer cases fall back to the single-key
   algorithm. Empty-leaf reuse shares the single-key initializer.
5. Retire displaced suffix bags through an RAII buffer declared before the lock,
   so unlocking precedes retirement on success and unwind. A later encoding panic
   may leave earlier entries applied, but does not leak their retired bags.

Guarded results contain `ValueRef`s directly, with no intermediate result-vector
conversion. `insert_batch` clones old values before releasing its internal guard.
New batches start with an empty old-values vector. Batch preallocation no longer
allocates speculative suffix buffers for updates or split/layer fallbacks.

Helpers in `src/tree/batch_utils.rs` (re-exported as `masstree::batch`):
`zip_into_entries`, `from_iter`, `sequential_keys`, `sequential_u64_keys`,
`BatchStats::analyze` (counts, unique ikeys, average length, suffix/single-layer
counts, locality factor, estimated leaves).

### 8.10 Entry API

`entry_with_guard(key, &guard)` returns `Entry::{Occupied, Vacant}`.
Classification uses `get_with_guard`, so it is a **point-in-time snapshot**, not
an atomic read-modify-write. The docs warn that `and_modify` can lose updates
and `VacantEntry::insert` can overwrite a concurrent insertion.

Operations:

- `Entry::or_insert`, `or_insert_with`, `or_insert_with_key`, `or_default` —
  insert if vacant, return `P::Output`.
- `and_modify(f)` — compute a new value from the cached output and insert it
  (non-atomic).
- `insert_entry(value)` — insert/replace and return an `OccupiedEntry` with the
  new cached output.
- `OccupiedEntry::insert`, `remove`, `try_remove`, `remove_entry`,
  `try_remove_entry`, `into_value`.
- `VacantEntry::insert`, `into_key`.

Lifetimes tie the key and guard to `'e` and the tree to `'t`, preventing the
entry from outliving either.

### 8.11 Iteration traits

`FromIterator<(K, V)>` requires `A: Default`, creates the allocator, builds a
tree with `with_allocator`, and delegates to `Extend`. Both use one guard and
the same sorted, leaf-grouped batch insertion pipeline as the explicit batch
APIs. Collection starts with 1,024 entries for short or unknown-size inputs,
doubling up to 16,384; a lower size hint of at least 16,384 starts at that cap.
Chunks own the original keys and borrow each key's bytes once, avoiding per-key
copies while bounding temporary storage. Equal keys retain input order, including
across chunks, so the last value wins. The collection specialization discards
replaced values without allocating an old-value result vector.

---

## 9. Public API

### 9.1 Types and aliases

Re-exported from `lib.rs`:

```rust
pub use tree::{MassTree, MassTree15, MassTree15Inline, MassTreeGeneric};
pub use tree::{
    BatchEntry, BatchInsertResult, Entry, OccupiedEntry, RemoveEntryResult, VacantEntry,
};
pub use tree::{InsertError, RemoveError};
pub use tree::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
pub use tree::batch;
pub use policy::{BoxPolicy, LeafPolicy, RefPolicy as RefLeafPolicy, ValuePtr, ValueRef};
pub use key::{IKEY_SIZE, Key, MAX_KEY_LENGTH};
pub use alloc_trait::TreeAllocator;
pub use alloc15::SeizeAllocator;
pub use internode::InternodeNode;
pub use leaf_trait::{TreeInternode, TreeLeafNode, TreePermutation};
pub use leaf15::LeafNode15;
pub use nodeversion::NodeVersion;
pub use permuter::{AtomicPermuter, AtomicPermuter15, MAX_WIDTH, Permuter, Permuter15};
pub use suffix::SuffixBag;
pub use retirement::BatchedRetire;
```

`InlineBits` is re-exported from `inline::bits`. `InlineSuffixBag` is
intentionally **not** re-exported because it relies on the leaf's lock/OCC
protocol and is not a standalone concurrent type. `Linker` is crate-only.

### 9.2 Method surface

| Category         | Methods                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Construct        | `MassTree::new`, `MassTree15::new`, `MassTree15Inline::new`, `with_batch_size`, `with_allocator`, `with_allocator_batch_size`, `Default` |
| Guards           | `guard`, `flush`, `maintenance`                                                                                                          |
| Size             | `len`, `is_empty`                                                                                                                        |
| Point read       | `get`, `get_with_guard`, `get_ref`, `contains_key`, `contains_key_with_guard`                                                            |
| Point write      | `insert`, `insert_with_guard`, `insert_value_with_guard`                                                                                 |
| Delete           | `remove`, `remove_with_guard`                                                                                                            |
| Entry            | `entry_with_guard`                                                                                                                       |
| Batch            | `insert_batch`, `insert_batch_with_guard`, `insert_batch_entries`                                                                        |
| Iterate          | `range`, `iter`, `keys`, `values`, `first`, `last`, `first_with_guard`, `last_with_guard`                                                |
| Scan (callback)  | `scan`, `scan_intra_leaf_batch`, `scan_rev_batch`, `scan_values`, `scan_values_rev`, `scan_prefix`, `scan_prefix_values`                 |
| Scan (reference) | `scan_ref`, `scan_batch_ref`, `scan_intra_leaf_batch_ref`, `scan_rev_batch_ref`                                                          |
| Collect          | `collect_entries`, `collect_keys`, `collect_values`                                                                                      |
| Maintenance      | `pending_coalesce`, `process_coalesce`, `process_coalesce_batch`, `coalesce_abandoned`, `coalesce_deferred`                              |
| Debug            | `debug_root_ptr`, `print_tree`, `print_tree_to`, `Debug`, `Display`                                                                      |

`get`/`remove`/`first`/`last` require `P::Value: Clone` because they create an
internal guard and return an owned value after dropping it.
`get_with_guard`/`remove_with_guard` return `ValueRef<'g, P::Value>`, and
`insert_batch_with_guard`/`insert_batch_entries` return
`BatchInsertResult<ValueRef<'g, P::Value>>`; the value is valid only while the
guard is held.

### 9.3 Guard and value lifetime

```rust
let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();          // enters the EBR epoch

tree.insert_with_guard(b"k", 1, &guard);
let v = tree.get_with_guard(b"k", &guard);       // Option<ValueRef<'_, u64>>
let old = tree.remove_with_guard(b"k", &guard);  // Result<Option<ValueRef<'_, u64>>, _>
// value references are valid until `guard` is dropped
```

For `BoxPolicy`, `get_with_guard` returns a `ValueRef`; write-through values are
atomic copies, boxed values stay alive until the guard is dropped even if
retired. `get()` clones the value before dropping its internal guard. `get_ref`
uses the same guard-scoped model.

A guard can be reused across many operations, but only with the tree that
created it. Passing a guard from a different tree panics (`verify_guard`).
Guards are not `Send`; concurrency is expressed by each thread calling
`tree.guard()`.

### 9.4 Example: canonical usage

```rust
use masstree::{MassTree, RangeBound};

let tree: MassTree<u64> = MassTree::new();
let guard = tree.guard();

tree.insert_with_guard(b"hello", 123, &guard);
tree.insert_with_guard(b"world", 456, &guard);

assert_eq!(tree.get_with_guard(b"hello", &guard), Some(123));

tree.scan(
    RangeBound::Included(b"a"),
    RangeBound::Excluded(b"z"),
    |key, value| {
        println!("{key:?} -> {value}");
        true // continue
    },
    &guard,
);

let entries: Vec<_> = tree
    .range(RangeBound::Included(b"a"), RangeBound::Unbounded, &guard)
    .collect();

tree.remove_with_guard(b"hello", &guard).unwrap();
```

Examples in `examples/` show rayon chunked parallelism (one guard per chunk),
tokio `spawn_blocking`, a TTL URL cache with prefix invalidation, a dual-index
session store, and tree printing.

---

## 10. Invariants

The following invariants are central to correctness. They are enforced by
construction, debug assertions, and tests.

1. **Key length.** `0 <= key.len() <= 256`; at most 32 layers. `Key::new` panics
   above 256.
2. **ikey order.** Each `ikey` is the big-endian 8-byte chunk, zero-padded.
   Unsigned `u64` comparison equals lexicographic byte comparison for that
   chunk.
3. **keylenx.** `0..=8` = terminal remaining length; `64` = suffix; `>=128` =
   layer. Occupancy is determined by the value array, not `keylenx`.
4. **At most one suffix key per ikey per leaf.** Different suffixes with the
   same ikey become a sublayer. Short (≤8-byte) keys and one suffix key may
   coexist.
5. **Permutation bijection.** `size <= 15`; positions `0..size` are key-sorted;
   every physical slot `0..15` appears exactly once in the 64-bit permutation.
   `debug_assert_valid` checks this.
6. **Slot 0 bound.** For a non-leftmost leaf, `ikey0[0]` is the leaf's lower
   bound and slot 0 must not be reused for a different ikey. Inserts either pick
   another free slot or split.
7. **Internode separators.** `ikey0` is strictly increasing; `child[i]` covers
   keys below `ikey0[i]`; `child[nkeys]` is the rightmost child. Equality routes
   right.
8. **Height.** `height == 0` means children are leaves; `height h` means
   children are internodes of height `h-1`. Height is immutable after
   publication and debug-capped at 15.
9. **Roots.** A root has `ROOT_BIT` and null parent. The main root is the node
   in `root_ptr`; a layer root has `ROOT_BIT`, null parent, and is not
   `root_ptr`. Split unlock clears `ROOT_BIT`, so root status must be captured
   before `mark_split`.
10. **B-link chain.** Leaves are linked by `next`/`prev`; only leaves
    participate. `next` LSB is the split/unlink mark. A marked pointer must
    never be dereferenced. The leftmost leaf is never unlinked.
11. **Split publication.** A split-created sibling is split-locked and has a
    null parent until propagation installs the parent; only then is it unlocked.
    `link_sibling` marks the old `next` for the whole move-and-link window.
12. **Suffix publication.** Suffix bytes are written before `KSUF_KEYLENX` is
    published; external bags are append-only and never reallocate once
    published; old cells are retired via EBR.
13. **Value publication.** A value is written before `keylenx`/permutation
    publication; removal clears value/suffix before publishing the new
    permutation with Release.
14. **Deleted leaves.** A coalesced leaf has `DELETED_BIT`; readers follow its
    `next` or re-reach from the layer root. `deleted_layer` marks a sublayer
    whose parent slot has been cleared.
15. **Count.** `count` increments once per new key (including
    split/layer/empty-leaf paths) and decrements once per successful removal;
    updates do not change it. `len()` is approximate under concurrent updates.
16. **No abandonment.** After `mark_split`, propagation must complete.
    Allocations after that point are infallible (abort on OOM).
17. **Layer pointer.** The parent leaf's slot stores the layer root pointer with
    `keylenx = 128`. If a layer root is promoted to an internode, the slot still
    points to the original left leaf; the new internode is reached by climbing
    that leaf's parent pointer.
18. **Retirement order.** A node/value/suffix bag is retired only after it is
    unreachable. Teardown runs only after `reclaim_all` and with exclusive
    access.
19. **Guard affinity.** Every guard belongs to exactly one tree's collector.
    Cross-tree use panics.
20. **Maintenance.** Empty leaves and dead sublayers are reclaimed only when the
    coalesce queue is processed or the tree is dropped. Long-running workloads
    should call `maintenance` periodically.

---

## 11. Testing and Validation

### 11.1 Test taxonomy

| Layer             | Location                                                                                                               | What it validates                                                                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unit tests        | `src/**/unit_tests.rs`, `src/policy/tests.rs`, `src/tree/coalesce.rs` tests                                            | Individual data structures: nodeversion, permuter, leaf/internode accessors, suffix bags, policy arrays, shard counter, entry API, key arithmetic, alloc/teardown |
| Integration tests | `tests/*.rs`                                                                                                           | Tree-level behavior, concurrency, stress, properties, C++ parity, leaks                                                                                           |
| Property tests    | `key_proptests`, `nodeversion_proptests`, `permuter_proptests`, `tree_proptests`                                       | Invariants over random inputs and operation sequences; differential checks against `BTreeMap`                                                                     |
| Loom              | `src/*/loom_tests.rs`                                                                                                  | Small atomic models of version locking, inline suffix metadata, and internode insert search                                                                       |
| Miri              | `just miri*`; `tests/leak_tests.rs`                                                                                    | UB/leak checks on allocation, suffix, layer, and reclamation paths                                                                                                |
| Stress/regression | `concurrent_*`, `stress_*`, `deleted_leaf_handling`, `phase2_regression`, `split_suffix_regression`, `adaptive_insert` | Races, splits, suffix migration, deleted leaves, coalescing, shared-prefix contention                                                                             |
| C++ parity        | `tests/cpp_tests.rs` + `tests/cpp/`                                                                                    | Workload-level equivalence with the original mttest suite                                                                                                         |
| Benchmarks        | `benches/`                                                                                                             | Performance, tail latency, memory, storage overhead                                                                                                               |

The checkout contains roughly 1,027 test functions: ~625 in `src/` and 402 in
`tests/`. The four proptest suites use 10–500 cases per property block;
`tree_proptests.proptest-regressions` contains four saved regression seeds.

### 11.2 Integration test map

| File                         | Focus                                                                                                                                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adaptive_insert.rs`         | Shared-prefix insert correctness under 4-thread contention; 12-byte keys with an 8-byte shared prefix; batch/long-key variants                                                                                             |
| `batch_insert_tests.rs`      | Empty/single/large batches; updates; duplicates; splits; sequential/reverse/shared-prefix/random/long keys; same ikey different lengths; concurrent batches mixed with single inserts                                      |
| `concurrent_correctness.rs`  | Remove/read stress, permutation integrity, insert/remove cycles, split false negatives, suffix overflow, shared-prefix suffix integrity, scan ordering during writes, 4-thread `BTreeMap` oracle                           |
| `concurrent_regression.rs`   | 2/4/8-thread disjoint/overlapping inserts, read/write races, splits, adjacent keys, long keys, all-value checks, empty-then-read, repeated runs                                                                            |
| `deleted_leaf_handling.rs`   | Readers during coalescing; get-during-coalesce with a 3-thread barrier; stress variant                                                                                                                                     |
| `phase2_regression.rs`       | Suffix migration across splits, 8/16/24-byte boundaries, prefix keys, inline/suffix coexistence, layer growth beyond WIDTH, layer roots becoming internodes                                                                |
| `range_scan_tests.rs`        | Empty/single/bounded ranges, full-scan vs `BTreeMap`, multi-layer ordering, prefix scans, concurrent scans + inserts, deep layers, batch APIs, reverse/forward consistency                                                 |
| `split_suffix_regression.rs` | Multi-layer suffix corruption across splits, variable suffix lengths, inline-to-external overflow during split, mixed inline/suffix keys, layer keys                                                                       |
| `stress_tests.rs`            | 4/8/16 threads, 500–20,000 keys/thread, 16/24/32-byte multi-layer keys, 10k–100k keys, sequential/reverse/random/shared-prefix, mixed reads/writes, repeated runs                                                          |
| `stress_adaptive.rs`         | 8-thread, 5-second extreme same-prefix contention; deadlock/op-count checks                                                                                                                                                |
| `true_inline.rs`             | No leaks with long keys; inline `for_each`, intra-leaf batches, early stop, ranges, multi-layer batch scans                                                                                                                |
| `leak_tests.rs`              | Ten lifecycle patterns × two policies for Miri: single/multi-key sublayers, 2-/3-level twig chains, partial removal, remove-all/reinsert, mixed depths, independent sublayers, repeated cycles, drop with pending coalesce |
| `key_proptests.rs`           | Key extraction, shift/unshift, suffix boundaries, ordering, max length, binary keys                                                                                                                                        |
| `nodeversion_proptests.rs`   | Version-bit invariants, lock/unlock, counters, root/deleted bits, random operation sequences                                                                                                                               |
| `permuter_proptests.rs`      | Bijection, size, insert/remove round trips, width variants 3/7/15                                                                                                                                                          |
| `tree_proptests.rs`          | Empty/get/contains/first/last, insert/update/remove, scans, bounds, forward/reverse equivalence, differential vs `BTreeMap`                                                                                                |

### 11.3 C++ mttest parity

`tests/cpp_tests.rs` is `mod cpp;`. `tests/cpp/mod.rs` declares 28 modules with
88 tests. Every test uses `MassTree15Inline` (WIDTH 15, 8-byte ikeys, 512-byte
inline suffix capacity) and fixed seeds. The workloads mirror the original C++
`mttest` patterns but are not linked against the C++ binary:

- **Reads/writes/scaling:** `r1`, `rscale`, `rw1`, `rw1fixed`, `rw1long`,
  `rw16`, `rw2`, `rw2fixed`, `rw3`, `rw4`, `rw4fixed`, `w1`, `wscale`, `uscale`,
  `same`, `rwsmall24`, `rwsep24`.
- **Scans:** `scan1`, `rscan1`, `conflictscan`.
- **Special patterns:** `splitremove`, `tri1`, `url`, `ycsbk`, `palm`.
- **Writer/deleter chases:** `wd1`, `wd2`, `wd3`.

`palm` reduces the C++ `PALM_N` from 128M to 100k. The parity is
workload-shaped, not a differential oracle.

### 11.4 Loom and Miri

- **Loom** (12 tests): `src/nodeversion/loom_tests.rs` models a simplified
  version word/lock (mutual exclusion, try-lock failure, version visibility,
  `stable()` waiting on dirty bits); `src/inline/loom_tests.rs` models inline
  suffix metadata (concurrent read/assign/clear); `src/internode/loom_tests.rs`
  models internode key insertion/search. These models duplicate the atomic logic
  rather than testing the production types. No loom test covers split
  propagation, coalesce, range traversal, or the full leaf.
- **Miri**: `just miri`, `miri-strict`, `miri-strict-int`, `miri-leaks`,
  `miri-tree-borrows`. `miri-leaks` runs
  `cargo +nightly miri test --test leak_tests` with strict provenance and
  symbolic alignment, so the 20 leak tests are actually executed under Miri.
  Multi-threaded and long-loop tests remain excluded; the exclusions are:

  | File | Exclusions | Reason |
  | --- | --- | --- |
  | `src/nodeversion/concurrent_tests.rs` | module `cfg(not(miri))` | multi-threaded |
  | `src/shard_counter/unit_tests.rs` | 8 `cfg_attr(miri, ignore)` | multi-threaded / 1000-iteration loops |
  | `src/nodeversion/unit_tests.rs` | 2 `cfg_attr(miri, ignore)` | iteration loops too slow |
  | `src/tree/unit_tests.rs` | 1 `cfg_attr(miri, ignore)`, 2 `cfg(not(miri))` | multi-threaded |
  | `src/tree/remove/unit_tests.rs` | 5 `cfg_attr(miri, ignore)`, 12 `cfg(not(miri))` | scheduling / multi-threaded |
  | `src/tree/range/*`, `src/suffix/*` | none | covered by unit tests and the leak target |

  Stress tests lower their thresholds via `cfg!(miri)`. Allocator teardown and
  `true_inline` also provide leak coverage.

### 11.5 Commands

```bash
just test                 # cargo nextest, no-fail-fast, failure extraction
just next                 # nextest with mimalloc
just next-repeat 20       # repeat nextest 20 times to catch flakiness
just test-unit            # cargo test --lib
just test-doc             # cargo test --doc
just test-integration     # cargo test --test '*'
just test-loom            # RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
just miri                 # cargo +nightly miri test --lib
just miri-strict          # strict provenance
just miri-leaks           # strict provenance + symbolic alignment on tests/leak_tests.rs
just miri-tree-borrows    # Tree Borrows
```

Release-mode concurrency tests are recommended by the test headers
(`cargo test --test concurrent_correctness --release`), but note that the
release profile uses `panic = "abort"` and symbol stripping.

### 11.6 Coverage gaps and testing risks

- CI (`.github/workflows/ci.yml`) covers `fmt --check`, strict Clippy, the
  lib/integration/doc suites, the default/no-default/all-features builds, Loom,
  and the Miri leak target.
- Loom models are simplified and do not cover the real tree protocols.
- The C++ parity suite is workload-level, not a linked differential test.
- A few long-running stress tests are `#[ignore]` and not run by default.
- Property-test case counts are modest (10–500 per block).

---

## 12. Performance Architecture

The implementation is explicitly tuned for cache locality, tail latency, and
read scalability.

**Cache-line layout.**

- `LeafNode15` isolates the version/modstate line (CL0) and the permutation line
  (CL1) from the key/value/suffix region. The permutation is the CAS-heavy word;
  separating it avoids false sharing with readers scanning keys.
- `InternodeNode` is exactly 5 cache lines; the header and the first six ikeys
  share CL0, so a routing decision often touches one line.
- All nodes are 64-byte aligned, including pool allocations. The leaf sizes are
  exact multiples of 64 bytes (448, 576, 896, 1152).
- The sharded counter uses 16 × 128-byte aligned shards to prevent false
  sharing.
- The node pool returns cache-line-aligned blocks with no per-node header; the
  freelist pointer is stored in the free block itself.

**Prefetching.**

- `prefetch_read` uses `_mm_prefetch(_MM_HINT_T0)` on x86_64 and
  `prfm pldl1keep` on aarch64.
- Traversal prefetches the selected child and, for internode children, the
  child's header and first child.
- Leaf search prefetches value slots; multi-layer reads prefetch suffix storage
  and layer roots.
- Range scans prefetch next/next-next (forward) and prev/prev-prev (reverse)
  leaves and neighboring slots.
- `no-prefetch` compiles all prefetch helpers to no-ops.

**Avoiding allocation/copies.**

- Inline values (`MassTree`/`MassTree15Inline`) require no allocation or
  retirement; reads return `Copy` values.
- Boxed write-through values (small trivially droppable types) update the
  existing `Box` atomically, avoiding allocation and EBR retirement on updates.
- Batch insert sorts by ikey and inserts runs per leaf under one lock, deferring
  allocation for write-through values.
- Range scans use a fixed 256-byte `CursorKey`, an `ArrayVec` layer stack, and a
  callback/ref API that avoids per-entry key allocation. `ScanEntry` is the only
  allocating iterator form.
- The permutation lets slots move logically without moving key/value bytes.
- Suffix bags use a fixed inline buffer and append-only external bags, so common
  short suffixes never touch the heap.

**Concurrency scalability.**

- Reads never lock; writers are per-node.
- Splits are hand-over-hand and only lock the path being propagated.
- Coalescing is deferred and lock-free queued, spreading cleanup across
  maintenance calls.
- The length counter is sharded and approximate, avoiding a single contended
  cache line.

**Benchmark evidence.** `README.md` reports 2×–3.7× advantages over
`tree_index`, `skipmap`, and `indexset` across read/write, high-impact (long
keys, deep tries, prefix queries), range scan, and tail-latency workloads, plus
a storage-gap comparison between boxed and inline policies. The raw runs are in
`runs/` (e.g. run208 for read/write, run206 for high-impact, run191 for range
scans, run213 for tail latency). These are project-specific measurements; the
benchmark harnesses in `benches/` are the reproducible source.

---

## 13. Feature Matrix and Configuration

| Feature                 | Default | Effect                                                                                                                    |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `debug-print`           | **yes** | Enables `tree/generic/print.rs`; `print_tree`, `print_tree_to`, `Display`                                                 |
| `mimalloc`              | no      | Available to benchmark binaries; the library never installs a global allocator                                           |
| `tracing`               | no      | `init_tracing()` writes NDJSON to a temp dir (console off by default); the library emits insert/get/remove/maintenance events |
| `small-suffix-capacity` | no      | 256-byte inline suffix bag instead of 512; leaf 896 B instead of 1152 B                                                   |
| `sidecar-suffix`        | no      | `InlinePolicy` uses `SidecarSuffix`; leaf 576 B; `embedded_suffix` module excluded                                        |
| `no-prefetch`           | no      | Prefetch helpers become no-ops                                                                                            |
| `debug-routing`         | no      | `eprintln!` tracing of internode routing and split propagation                                                            |

`cfg(loom)` is the only model-checking cfg used.

Profiles:

| Profile              | Settings                                                                  |
| -------------------- | ------------------------------------------------------------------------- |
| `dev`                | opt-level 0, debug assertions and overflow checks on, incremental         |
| `release`            | opt-level 3, fat LTO, 1 codegen unit, `panic = "abort"`, symbols stripped |
| `release-with-debug` | release + debug info, no LTO                                              |
| `profiling`          | release + debug info, no LTO                                              |
| `bench`              | inherits release                                                          |

Tooling: `justfile` provides build, bench, C++/Rust comparison, test, lint,
format, doc, and Miri recipes. `scripts/bump_version.py` updates `Cargo.toml`
and README version mentions; `scripts/compare_benches.py` parses benchmark
output; `scripts/run_cpp_bench.py` orchestrates C++/Rust runs. `rustfmt.toml`
uses edition 2024, width 100, four spaces, Unix newlines. `clippy.toml` sets
MSRV 1.92, cognitive complexity 20, 80-line functions, 6 arguments, and bans
`std::env::set_var`/`remove_var`. README and `clippy.toml` document the MSRV;
check `Cargo.toml` for whether cargo currently enforces it.

---

## 14. Design Notes, Divergences, and Audit Points

### 14.1 Deliberate divergences from the C++ reference

1. **Fixed WIDTH = 15.** The 64-bit permutation format uses 4 bits per slot plus
   a 4-bit size, so 15 slots is the maximum. There is no alternate width or
   parameter sweep. `TreeLeafNode::WIDTH` and `TreeInternode::WIDTH` are 15.
2. **Layers only on suffix conflict.** A short key and a suffix key with the
   same ikey coexist in one leaf; the implementation does not create a layer
   merely because a key has a suffix. Layers are created when two
   _different suffixes_ collide. Long common prefixes are represented by
   single-entry twig chains rather than a layer per chunk.
3. **`keylenx` does not encode suffix length.** It is 64 for any suffix key; the
   suffix length lives in the suffix bag metadata. This differs from C++
   Masstree's layer-length encoding.
4. **No leaf merge or rebalance.** Deletion empties leaves; coalescing unlinks
   empty leaves and cascades empty non-root internodes. Underfull
   leaves/internodes remain. This favors write throughput and simplicity over
   space reclamation.
5. **Lazy coalescing.** Removal only enqueues empty leaves. Unlinking, parent
   cleanup, and sublayer GC happen in maintenance, not in the remove critical
   path. This is a major source of concurrency complexity (deleted leaves,
   B-link following, deferred routes).
6. **Root creation uses a Release store, not a CAS.** Only the old root's lock
   holder can create a new root; the no-abandon protocol guarantees completion.
   A layer-root promotion does not update the parent slot; readers climb parent
   pointers.
7. **`has_changed` ignores only the lock bit.** An in-flight insert now marks
   the version dirty (`OCC-01`) and increments the insert counter on unlock, so
   a reader that snapshotted before the insert sees a version change.
   `has_changed_or_locked` remains available where the dirty window itself
   matters.

### 14.2 API and behavioral notes

- Insert/remove/entry APIs **panic** on internal errors rather than returning
  recoverable errors. `InsertError` now has only the `SplitFailed` variant; the
  split path itself is infallible after the pre-encoding check.
- `RemoveError::RetryLimitExceeded` is the only remove error; not-found is
  `Ok(None)`.
- `len()` is an approximate snapshot (sharded Relaxed loads); it is exact after
  quiescence.
- Range scans are not snapshot-isolated; they provide monotonic keys, not a
  consistent point-in-time view. Bounds are full-key exact (`CursorKey` +
  suffix/layer comparison), not the earlier ikey-only approximation.
- The Entry API is a point-in-time snapshot and is not an atomic
  read-modify-write.
- `BatchInsertResult` has only `inserted`, `updated`, and `old_values`;
  `total()` is `inserted + updated`. A batch either applies an entry or panics
  on an internal invariant violation.
- `debug-print` is enabled by default, so `Display`/`print_tree` code is
  compiled into normal builds. `print_tree` climbs to the true promoted layer
  root before printing, renders values through `P::output_to_value_ref` (an
  atomic copy for write-through policies), and validates the leaf version after
  the whole slot table has been read.
- `tracing` emits insert/get/remove/maintenance events; file logs go to a temp
  directory by default and `MASSTREE_LOG_CONSOLE=1` opts into console output.
- `mimalloc` is recommended for benchmark binaries but the library does not
  install it globally; memory-profiling benches define their own allocator.
- `ValuePtr` is opaque (`Debug` prints the address only); public reads return
  guard-scoped `ValueRef`s. Do not add a safe value accessor back to `ValuePtr`.
- `LeafNode15`/`TreeLeafNode` ownership-bearing mutators (`store_value`,
  `store_layer`, `take_value`, `take_value_for_layer`, `update_value_in_place*`)
  are `unsafe`; safe code can read values but cannot publish a value or layer
  pointer. `load_value`/`take_value` return `None` for layer slots, and
  `store_prepared_value_relaxed` is crate-internal.

### 14.3 Audit points (implementation observations, not guarantees)

These are architectural observations worth keeping in mind when modifying the
code:

1. **Remove revalidates the full key under the lock.**
   `lock_and_verify_for_remove` re-runs the full-key search (ikey, keylenx,
   suffix) after acquiring the lock, so a same-ikey insert/remove cannot shift
   the logical position onto another key.
2. **Sublayer GC reclaims the whole subtree.** `try_remove_sublayer` climbs
   parent pointers to the true layer root, locks and validates every
   leaf/internode top-down, clears the parent slot, marks every node deleted,
   and retires the entire subtree. A promoted sublayer no longer leaks.
3. **Loom coverage is model-level.** The loom tests reimplement simplified
   atomics; they do not exercise `NodeVersion`, `Permuter`, `Linker`, split
   propagation, coalesce, or scans directly.
4. **Layer-root promotion is handled by climbing.** Point reads, range scans,
   the printer, and sublayer GC all resolve a parent slot's original left leaf
   to the parentless internode root before walking siblings. Any change to layer
   promotion must preserve that parent-pointer chain.
5. **Range bounds and scan traversal share full-key ordering.** `CursorKey`,
   `compare_cursor_to_slot`, and the B-link walk in `reach_leaf_for_scan` are
   the single source of truth; do not reintroduce ikey-only comparisons in a
   scan fast path.

### 14.4 Operational guidance

- Call `maintenance(&mut guard)` periodically in long-running single-threaded or
  low-concurrency workloads, or after large delete batches. Otherwise empty
  leaves/sublayers accumulate until the tree is dropped.
- Reuse a guard for bulk operations (loops, batches); creating a guard per
  operation is correct but adds EBR overhead.
- Use `MassTree15<V>` for owned/large values and
  `MassTree`/`MassTree15Inline<V>` for `Copy` values ≤ 8 bytes.
- Use `scan_intra_leaf_batch` (or `scan_batch_ref` for `BoxPolicy`) for
  throughput-oriented scans; use `range` when an owning iterator is convenient.
- Keys are byte slices; for integer keys, use a big-endian encoding
  (`u64::to_be_bytes`) to preserve numeric order.

---

## 15. Glossary

| Term                 | Meaning                                                                           |
| -------------------- | --------------------------------------------------------------------------------- |
| **ikey**             | 8-byte big-endian chunk of a key, stored as `u64`                                 |
| **keylenx**          | Per-slot key discriminator: 0–8 terminal length, 64 suffix, ≥128 layer            |
| **layer / sublayer** | A B+ tree rooted at a leaf slot, indexing the next 8-byte chunk                   |
| **twig**             | A single-entry layer leaf used to compress repeated equal 8-byte chunks           |
| **WIDTH**            | Node fanout; 15 keys for internodes and 15 slots for leaves                       |
| **permutation**      | 64-bit mapping from logical sorted positions to physical slots                    |
| **OCC**              | Optimistic concurrency control: snapshot version, read, validate                  |
| **B-link**           | Leaf chain with `next` pointers; splits install a new sibling and move keys right |
| **split mark**       | LSB of a leaf `next` pointer indicating split/unlink in progress                  |
| **EBR**              | Epoch-based reclamation (`seize`): deferred freeing after readers leave           |
| **guard**            | `seize::LocalGuard` pinning the current epoch; required for tree operations       |
| **coalesce**         | Deferred unlinking of empty leaves and GC of empty sublayers                      |
| **route**            | `Vec<u64>` of per-layer ikeys used to re-find a sublayer parent                   |
| **write-through**    | Updating a small boxed value in place atomically, avoiding allocation             |
| **inline value**     | Terminal value stored in the leaf as 64 bits + sentinel tag                       |
| **sidecar**          | Lazily allocated heap suffix store                                                |
| **stable version**   | A version word with no dirty bits, suitable for OCC                               |

---

## Appendix A — Constants

| Constant                           |                               Value | Location                    |
| ---------------------------------- | ----------------------------------: | --------------------------- |
| `IKEY_SIZE`                        |                                   8 | `key.rs`                    |
| `MAX_KEY_LENGTH`                   |                                 256 | `key.rs`                    |
| `WIDTH` / `WIDTH_15`               |                                  15 | `internode.rs`, `leaf15.rs` |
| `NUM_CHILDREN`                     |                                  16 | `internode.rs`              |
| `MAX_WIDTH`                        |                                  15 | `permuter.rs`               |
| `KSUF_KEYLENX`                     |                                  64 | `leaf15.rs`                 |
| `LAYER_KEYLENX`                    |                                 128 | `leaf15.rs`                 |
| `MODSTATE_QUEUED_BIT`              |                                0x04 | `leaf15.rs`                 |
| `INITIAL_CAPACITY` (suffix)        |                                 128 | `suffix.rs`                 |
| Inline suffix capacity             | 512 / 256 (`small-suffix-capacity`) | `suffix/inline.rs`          |
| `STACK_CAPACITY` (teardown)        |                                 128 | `alloc15.rs`                |
| `MAX_SIZE_CLASSES`                 |                         20 (1280 B) | `node_pool.rs`              |
| `POOL_CAPACITY`                    |                                 512 | `node_pool.rs`              |
| `REFILL_BATCH`                     |                                 128 | `node_pool.rs`              |
| `SHARDS`                           |                                  16 | `shard_counter.rs`          |
| `DEFAULT_RETIRE_BATCH_SIZE`        |                                 256 | `tree/generic.rs`           |
| `MAX_BLINK_HOPS` (insert)          |                                 128 | `tree/generic.rs`           |
| `MAX_RETRIES` (remove layers)      |                                1000 | `tree/remove.rs`            |
| `MAX_REQUEUE_COUNT`                |                                  10 | `tree/coalesce.rs`          |
| `MAX_GC_BLINK_HOPS`                |                                  64 | `tree/coalesce.rs`          |
| `MAX_REPOSITION_RETRIES` (reverse) |                                  16 | `tree/range/reverse_ctx.rs` |
| Reverse `CursorKey` unbounded ikey |                          `u64::MAX` | `cursor_key.rs`             |
| Reverse `CursorKey` sentinel len   |                                   9 | `cursor_key.rs`             |
| Leaf size (Box)                    |                               448 B | `leaf15/layout.rs`          |
| Leaf size (Inline, 512)            |                              1152 B | `leaf15/layout.rs`          |
| Leaf size (Inline, 256)            |                               896 B | `leaf15/layout.rs`          |
| Leaf size (sidecar)                |                               576 B | `leaf15/layout.rs`          |
| Internode size                     |                               320 B | `internode/layout.rs`       |

---

## Appendix B — File Index

| Area                    | Primary files                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Public facade           | `src/lib.rs`, `src/tree.rs`                                                                                                             |
| Keys                    | `src/key.rs`, `src/key/unit_tests.rs`                                                                                                   |
| Routing                 | `src/ksearch.rs`, `src/ksearch/scalar.rs`                                                                                               |
| Version/lock            | `src/nodeversion.rs`, `src/nodeversion/bounded.rs`                                                                                      |
| Permutation             | `src/permuter.rs`                                                                                                                       |
| B-link marking          | `src/link.rs`                                                                                                                           |
| Ordering/hints/prefetch | `src/ordering.rs`, `src/hints.rs`, `src/prefetch.rs`                                                                                    |
| Sharded count           | `src/shard_counter.rs`                                                                                                                  |
| Leaf                    | `src/leaf15.rs`, `src/leaf15/layout.rs`                                                                                                 |
| Internode               | `src/internode.rs`, `src/internode/accessors.rs`, `src/internode/layout.rs`                                                             |
| Leaf/internode traits   | `src/leaf_trait.rs`                                                                                                                     |
| Policies/values         | `src/policy.rs`, `src/policy/{box_values,inline_values,embedded_suffix,sidecar_suffix,suffix_ops}.rs`                                   |
| Suffix                  | `src/suffix.rs`, `src/suffix/{inline,sidecar}.rs`                                                                         |
| Inline encoding         | `src/inline/bits.rs`, `src/inline/sentinel.rs`                                                                                          |
| Allocation/pool         | `src/alloc_trait.rs`, `src/alloc15.rs`, `src/node_pool.rs`                                                                              |
| Reclamation             | `src/retirement.rs`, `src/alloc15.rs`, `src/policy/*`                                                                                   |
| Tree core               | `src/tree/generic.rs`, `src/tree/generic/{search,insert,insert_strategy,layer,optimistic_reads,get_guarded,split,batch,entry,print}.rs` |
| Split propagation       | `src/tree/split.rs`, `src/tree/split/{propagation,propagation_context,parent_locking,root_creation}.rs`                                 |
| Remove                  | `src/tree/remove.rs`, `src/tree/remove/unit_tests.rs`                                                                                   |
| Coalesce                | `src/tree/coalesce.rs`                                                                                                                  |
| Range scans             | `src/tree/range.rs`, `src/tree/range/*`                                                                                                 |
| Batch helpers           | `src/tree/batch_utils.rs`, `src/tree/range/batch_common.rs`, `src/tree/range/iterator/batch_{forward,reverse}.rs`                       |
| Tests                   | `tests/*.rs`, `tests/cpp/*`, `src/**/unit_tests.rs`, `src/**/loom_tests.rs`                                                             |
| Benchmarks              | `benches/*.rs`                                                                                                                          |
| Examples                | `examples/*.rs`                                                                                                                         |

---

## Appendix C — References

- Masstree paper: _Cache Craftiness for Fast Multicore Key-Value Storage_,
  EuroSys 2012 — <https://pdos.csail.mit.edu/papers/masstree:eurosys12.pdf>
- C++ reference implementation: <https://github.com/kohler/masstree-beta>
- `seize` epoch-based reclamation: <https://crates.io/crates/seize>
- Repository README, benchmark tables, and raw runs in `runs/`, `cpp_benches/`,
  `rust_benches/`.

---

_Document generated from the v1.0.0 source tree. When changing the
implementation, update the corresponding sections and, where applicable, the
compile-time layout assertions in `src/leaf15/layout.rs` and
`src/internode/layout.rs`._
