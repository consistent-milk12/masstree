# MassTree Architecture

Complete architectural overview of the MassTree implementation after all phases are complete.

> **Document Status:** This is a **target architecture** document describing the complete design.
> Features are labeled by implementation phase:
> 
> - **(P1)** Phase 1 - Implemented (single-threaded, 0-8 byte keys)
> - **(P2)** Phase 2 - Planned (trie layering, arbitrary key lengths)
> - **(P3)** Phase 3 - Planned (concurrency, epoch-based reclamation)
> - **(P4)** Phase 4 - Planned (scan, remove operations)
> - **(P5)** Phase 5 - Planned (optimizations)
>
> **Current Code:** Phase 1 complete. See `src/lib.rs` for constraints.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Data Structure Hierarchy](#data-structure-hierarchy)
4. [Trie of B+Trees Structure](#trie-of-btrees-structure)
5. [Node Structures](#node-structures)
6. [Operation Flows](#operation-flows)
7. [Concurrency Model](#concurrency-model)
8. [Memory Management](#memory-management)
9. [Value Storage Modes](#value-storage-modes)
10. [Phase Progression](#phase-progression)

---

## High-Level Overview

MassTree is a high-performance concurrent key-value store combining **tries** and **B+trees**:

![01-high-level-overview](docs/images/01-high-level-overview.png)

> **Note:** Empty tree = root leaf with `permutation.size() == 0`. There is no `Empty` variant.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Trie** | Keys split into 8-byte slices; each slice handled by one layer |
| **B+Tree** | Each layer is a B+tree with fanout WIDTH (default 15) |
| **Layer** | When keys share an 8-byte prefix, they share a sublayer B+tree |
| **ikey** | 8-byte key slice stored as big-endian u64 for fast comparison |
| **Suffix** | Remaining bytes after current ikey stored in SuffixBag |

---

## Module Dependency Graph

> **Status:** Phase 1 (Implemented)

![02-module-dependency](docs/images/02-module-dependency.png)

### Module Responsibilities

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `lib.rs` | Public exports | `MassTree`, `MassTreeIndex` |
| `tree.rs` | Tree operations | `MassTree`, `RootNode`, `InsertError` |
| `leaf.rs` | Leaf nodes | `LeafNode`, `LeafValue`, `ModState` |
| `internode.rs` | Internal nodes | `InternodeNode` |
| `key.rs` | Key handling | `Key` (borrowed slice with shift state) |
| `permuter.rs` | Slot ordering | `Permuter` (u64-encoded permutation) |
| `nodeversion.rs` | Versioned locking | `NodeVersion`, `LockGuard` |
| `suffix.rs` | Long key storage | `SuffixBag`, `PermutationProvider` |
| `ksearch.rs` | Binary search | `lower_bound_leaf`, `upper_bound_internode` |

---

## Data Structure Hierarchy

> **Status:** Phase 1 (Implemented) - see inline notes for planned extensions

![03-data-structure-hierarchy](docs/images/03-data-structure-hierarchy.png)

> **Notes:**
>
> - **MassTree methods:** `get`/`insert`/`len` (P1), `remove`/`scan` (P4), `make_new_layer` (P2)
> - `RootNode` has no `Empty` variant; empty tree uses a root leaf with size 0
> - `InternodeNode` uses `child[WIDTH]` + `rightmost_child` to avoid `WIDTH+1` const generics
> - `Key::shift_by(bytes)` will be added in Phase 2 for layer descent
> - `ksuf_matches` currently returns `bool`; Phase 2 will extend to `i32` for layer detection

---

## Trie of B+Trees Structure

> **Status:** Phase 2 (Planned) - not implemented in current code. Phase 1 rejects keys >8 bytes.

![04-trie-structure](docs/images/04-trie-structure.png)

### Layer Creation Rules

![05-layer-creation-rules](docs/images/05-layer-creation-rules.png)

---

## Node Structures

> **Status:** Phase 1 (Implemented)

### LeafNode Memory Layout

![06-leafnode-layout](docs/images/06-leafnode-layout.png)

### InternodeNode Memory Layout

![07-internode-layout](docs/images/07-internode-layout.png)

---

## Operation Flows

> **Status:** Mixed - Basic traversal (P1), concurrent ops (P3), layer descent (P2)

### GET Operation

![08-get-operation](docs/images/08-get-operation.png)

### INSERT Operation

![09-insert-operation](docs/images/09-insert-operation.png)

### SPLIT Propagation

![10-split-propagation](docs/images/10-split-propagation.png)

---

## Concurrency Model

> **Status:** Phase 3 (Planned) - current code is single-threaded with load-then-store locking

### Optimistic Read Protocol

![11-optimistic-read](docs/images/11-optimistic-read.png)

### Write Locking Protocol

> **Note:** Current P1 implementation uses load-then-store; CAS loop shown is the P3 target.

![12-write-locking](docs/images/12-write-locking.png)

### Hand-Over-Hand Locking (Split Propagation)

![13-hand-over-hand](docs/images/13-hand-over-hand.png)

### Epoch-Based Reclamation

![14-epoch-reclamation](docs/images/14-epoch-reclamation.png)

---

## Memory Management

> **Status:** Mixed - Arena allocation (P1), Epoch-Based Reclamation (P3)

### Allocation Strategy

![15-allocation-strategy](docs/images/15-allocation-strategy.png)

### Pointer Discipline

![16-pointer-discipline](docs/images/16-pointer-discipline.png)

---

## Value Storage Modes

> **Status:** Phase 1 (Implemented) - see notes on MassTreeIndex

![17-value-storage-modes](docs/images/17-value-storage-modes.png)

> **Note on MassTreeIndex:** Currently a convenience wrapper over `MassTree<V>` that stores
> `Arc<V>` internally and copies on read. True inline storage is planned for a future release.

---

## Phase Progression

![18-phase-progression](docs/images/18-phase-progression.png)

### Phase Summary

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| **1** | Foundation | Single-threaded core, 0-8 byte keys, get/insert/split |
| **2** | Trie Layering | Suffix storage, layer creation/descent, arbitrary key lengths |
| **3** | Concurrency | Atomic ops, optimistic reads, locked writes, EBR |
| **4** | Full Operations | Scan iterators, remove, layer garbage collection |
| **5** | Optimization | Prefetching, memory ordering, SIMD, benchmarks |

---

## Complete System Diagram

![19-complete-system](docs/images/19-complete-system.png)

---

## Key Invariants

### Structural Invariants

1. **Permuter Invariant** (P1): All 15 slots appear exactly once in the u64 encoding
2. **Slot-0 Rule** (P1): Slot 0's ikey must equal `ikey_bound()` (unless first leaf or same ikey)
3. **Height Semantics** (P1): `height == 0` means children are leaves; `height > 0` means internodes
4. **Layer Root** (P2): Layer roots have `parent == null` and `version.is_root() == true`
5. **Leaf List** (P1): All leaves in a layer form a consistent doubly-linked list

### Concurrency Invariants

1. **Version Ordering** (P3): Writers increment version on unlock; readers retry if version changed
2. **Lock Exclusivity** (P3): Only one writer holds a node's lock at a time
3. **Epoch Safety** (P3): Nodes are not freed while any thread may hold references
4. **Arc Lifetime** (P1): Values outlive nodes via Arc reference counting

### Memory Invariants

1. **Arena Stability** (P1): Pointers remain valid for tree lifetime (arena never shrinks)
2. **Provenance** (P1): All pointer operations preserve strict provenance
3. **Alignment** (P1): Nodes are 64-byte aligned for cache efficiency

---

**Generated from:** CODE_011_FLOW.md, TODO.md, CODE_010.md, CODE_011.md
**Last Updated:** 2025-12-16
**Reviewed and corrected:** Based on ArchReview.md findings - fixed 9 structural/API mismatches
