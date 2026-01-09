//! # [`MassTree`]
//!
//! A high-performance concurrent ordered map for Rust. Stores keys as `&[u8]` and
//! supports variable-length keys by building a trie of B+trees.
//!
//! ## Features
//!
//! - Ordered map for byte keys (lexicographic ordering)
//! - Lock-free reads with version validation
//! - Concurrent inserts and deletes with fine-grained leaf locking
//! - Zero-copy range scans with `scan_ref` and `scan_prefix`
//! - Memory reclamation via epoch-based deferred cleanup
//! - Lazy leaf coalescing for deleted entries
//! - Two node widths: [`MassTree`] (WIDTH=24) and [`MassTree15`] (WIDTH=15)
//!
//! ## Status: v0.3.0 (Core Feature Complete)
//!
//! All core operations implemented and tested. Not yet production-ready—concurrent
//! data structures require extensive stress testing beyond what Miri and proptests provide.
//!
//! | Feature | Status |
//! |---------|--------|
//! | Concurrent get | Lock-free, version-validated |
//! | Concurrent insert | Fine-grained leaf locking |
//! | Concurrent remove | Fine-grained locking + lazy coalescing |
//! | Range scans | `scan`, `scan_ref`, `scan_prefix`, iterator |
//! | Memory reclamation | Seize-based epoch reclamation |
//!
//! **Not yet implemented:** `Entry` API, `DoubleEndedIterator`, `Extend`/`FromIterator`.
//!
//! ## When to Use
//!
//! **May work well for:**
//! - Long keys with shared prefixes (URLs, file paths, UUIDs)
//! - Range scans over ordered data
//! - Mixed read/write workloads
//! - High-contention scenarios (the trie structure helps here)
//!
//! **Consider alternatives for:**
//! - Unordered point lookups → `dashmap`
//! - Pure insert-only workloads → `scc::TreeIndex`
//! - Integer keys only → `congee` (ART-based)
//! - Read-heavy with rare writes → `RwLock<BTreeMap>`
//!
//! ## Variant Selection
//!
//! | Variant | Best For |
//! |---------|----------|
//! | [`MassTree15`] | Range scans, writes, shared-prefix keys, contention |
//! | [`MassTree`] (WIDTH=24) | Random-access reads, single-threaded point ops |
//!
//! [`MassTree15`] tends to perform better due to cheaper u64 atomics and better
//! cache utilization. Consider it for most workloads.
//!
//! ```rust
//! use masstree::{MassTree, MassTree15, MassTree24Inline, MassTree15Inline};
//!
//! // Default: WIDTH=24, Arc-based storage
//! let tree: MassTree<u64> = MassTree::new();
//!
//! // WIDTH=15, Arc-based storage (recommended for most workloads)
//! let tree15: MassTree15<u64> = MassTree15::new();
//!
//! // Inline storage for Copy types (no Arc overhead)
//! let inline: MassTree24Inline<u64> = MassTree24Inline::new();
//! let inline15: MassTree15Inline<u64> = MassTree15Inline::new();
//! ```
//!
//! ## Performance Notes
//!
//! On mixed read/write workloads (90% read, 10% write) at 6 threads, [`MassTree15`]
//! achieves 2-5x higher throughput than `RwLock<BTreeMap>` due to lock-free reads
//! and per-node versioning. The advantage increases under contention.
//!
//! For pure read workloads, `RwLock<BTreeMap>` is competitive or faster—lock-free
//! structures pay complexity costs that only matter when there's actual contention.
//!
//! ## Quick Start
//!
//! ```rust
//! use masstree::MassTree;
//!
//! let tree: MassTree<u64> = MassTree::new();
//! let guard = tree.guard();
//!
//! // Insert
//! tree.insert_with_guard(b"hello", 123, &guard).unwrap();
//!
//! // Point lookup (lock-free)
//! assert_eq!(tree.get_ref(b"hello", &guard), Some(&123));
//!
//! // Remove
//! tree.remove_with_guard(b"hello", &guard).unwrap();
//!
//! // Range scan (zero-copy)
//! use masstree::RangeBound;
//! tree.scan_ref(RangeBound::Included(b"a"), RangeBound::Excluded(b"z"), |key, value| {
//!     println!("{:?} -> {}", key, value);
//!     true // continue scanning
//! }, &guard);
//! ```
//!
//! ## Thread Safety
//!
//! `MassTree<V>` is `Send + Sync` when `V: Send + Sync`. Concurrent access
//! requires using the guard-based API. The guard ties operations to an epoch
//! for safe memory reclamation.
//!
//! ```rust
//! use masstree::MassTree;
//!
//! let tree: MassTree<u64> = MassTree::new();
//! let guard = tree.guard();
//!
//! // Concurrent get (lock-free)
//! let value = tree.get_with_guard(b"key", &guard);
//!
//! // Concurrent insert (fine-grained locking)
//! let old = tree.insert_with_guard(b"key", 42, &guard);
//! ```
//!
//! ## Key Constraints
//!
//! - Keys must be 0-256 bytes. Longer keys will panic.
//! - Keys are byte slices (`&[u8]`), not generic types.
//!
//! ## How It Works
//!
//! Keys are split into 8-byte slices, creating a trie where each node is a B+tree:
//!
//! ```text
//! Key: "users/alice/profile" (19 bytes)
//!      └─ Layer 0: "users/al" (8 bytes)
//!         └─ Layer 1: "ice/prof" (8 bytes)
//!            └─ Layer 2: "ile" (3 bytes)
//! ```
//!
//! Keys with shared prefixes share upper layers, making lookups efficient for
//! hierarchical data.
//!
//! ## Architecture Overview
//!
//! ```text
//!                            ┌─────────────────────────────────────────────────┐
//!                            │              MassTreeGeneric<S,L,A>             │
//!                            │                                                 │
//!                            │  root_ptr ──► Leaf or Internode (Layer 0 root)  │
//!                            │  collector ─► seize::Collector (reclamation)    │
//!                            │  count ─────► ShardedCounter (approx size)      │
//!                            │  coalesce_queue ─► CoalesceQueue (lazy cleanup) │
//!                            └─────────────────────────────────────────────────┘
//!                                                     │
//!                                                     ▼
//!                ┌───────────────────────────────────────────────────────────────┐
//!                │                         LAYER 0                               │
//!                │                                                               │
//!                │    ┌──────────┐     ┌──────────┐     ┌──────────┐             │
//!                │    │Internode │     │Internode │     │Internode │             │
//!                │    └────┬─────┘     └────┬─────┘     └────┬─────┘             │
//!                │         │                │                │                   │
//!                │    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐              │
//!                │    ▼         ▼      ▼         ▼      ▼         ▼              │
//!                │  ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐          │
//!                │  │Leaf│◄─►│Leaf│◄─►│Leaf│◄─►│Leaf│◄─►│Leaf│◄─►│Leaf│ (B-link) │
//!                │  └────┘   └─┬──┘   └────┘   └────┘   └────┘   └────┘          │
//!                │             │ layer_ptr                                       │
//!                └─────────────┼─────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//!                ┌────────────────────────────────────────────────────────────────┐
//!                │                         LAYER 1                                │
//!                │                                                                │
//!                │                   ┌──────────┐                                 │
//!                │                   │ Internode│  (sublayer root)                │
//!                │                   └────┬─────┘                                 │
//!                │                   ┌────┴────┐                                  │
//!                │                   ▼         ▼                                  │
//!                │                 ┌────┐   ┌────┐                                │
//!                │                 │Leaf│◄─►│Leaf│                                │
//!                │                 └────┘   └────┘                                │
//!                └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Node Structure
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────────────────┐
//!  │                         LEAF NODE (WIDTH=15 or 24)                          │
//!  ├─────────────────────────────────────────────────────────────────────────────┤
//!  │  version: NodeVersion  ─────────────────────────────────────────────────────┤
//!  │  ┌────────────────────────────────────────────────────────────────────────┐ │
//!  │  │ bits: u32                                                              │ │
//!  │  │ ├─ bit 0:     LOCK_BIT (exclusive access)                              │ │
//!  │  │ ├─ bit 1:     INSERTING_BIT (insert in progress)                       │ │
//!  │  │ ├─ bit 2:     SPLITTING_BIT (split in progress)                        │ │
//!  │  │ ├─ bits 3-8:  VINSERT (insert version, 6 bits)                         │ │
//!  │  │ ├─ bits 9-27: VSPLIT (split version, 19 bits)                          │ │
//!  │  │ ├─ bit 28:    RESERVED (unused)                                        │ │
//!  │  │ ├─ bit 29:    DELETED_BIT (logically deleted)                          │ │
//!  │  │ ├─ bit 30:    ROOT_BIT (is layer root)                                 │ │
//!  │  │ └─ bit 31:    ISLEAF_BIT (leaf vs internode)                           │ │
//!  │  └────────────────────────────────────────────────────────────────────────┘ │
//!  │                                                                             │
//!  │  permutation: AtomicPermuter15/24  ─────────────────────────────────────────┤
//!  │  ┌────────────────────────────────────────────────────────────────────────┐ │
//!  │  │ Encodes logical→physical slot mapping + size in single atomic          │ │
//!  │  │ WIDTH=15: u64  (4 bits × 15 slots + 4-bit size)                        │ │
//!  │  │ WIDTH=24: u128 (5 bits × 24 slots + 4-bit size)                        │ │
//!  │  └────────────────────────────────────────────────────────────────────────┘ │
//!  │                                                                             │
//!  │  ikeys: [AtomicU64; WIDTH]  ─────────────────────────────────────────────── │
//!  │  keylenx: [AtomicU8; WIDTH] (suffix info: 0-8=inline, 64=suffix, 65=layer)  │
//!  │  values: [AtomicPtr<u8>; WIDTH] (Arc<V> or inline V or layer_ptr)           │
//!  │  suffix: SuffixBag (key bytes beyond 8-byte ikey prefix)                    │
//!  │  prev/next: AtomicPtr<Self> (B-link chain for siblings)                     │
//!  │  parent: AtomicPtr<Internode> (parent pointer for split propagation)        │
//!  └─────────────────────────────────────────────────────────────────────────────┘
//!
//!  ┌─────────────────────────────────────────────────────────────────────────────┐
//!  │                         INTERNODE (WIDTH=15 fixed)                          │
//!  ├─────────────────────────────────────────────────────────────────────────────┤
//!  │  version: NodeVersion                                                       │
//!  │  nkeys: u8 (0..=15 keys)                                                    │
//!  │  ikeys: [AtomicU64; 15] (routing keys)                                      │
//!  │  children: [AtomicPtr<u8>; 16] (child pointers: leaf or internode)          │
//!  │  parent: AtomicPtr<Internode> (parent for propagation)                      │
//!  └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Operation Flows
//!
//! ### GET (Lock-Free Read)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    GET OPERATION - LOCK-FREE WITH OCC                        │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   get_with_guard(key, guard)                                                 │
//!   │          │                                                                   │
//!   │          │  key = Key::new(bytes)  ← Split into 8-byte ikey chunks           │
//!   │          │  layer = 0              ← Start at layer 0                        │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────────────────────────────────────────────────────────────┐   │
//!   │   │                    LAYER LOOP (for multi-layer keys)                 │   │
//!   │   │                                                                      │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │  Load root   │──── AtomicPtr::load(Acquire) from root_ptr         │   │
//!   │   │  │  for layer   │     Layer 0: tree.root, Layer N: sublayer root     │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────────────────────────────────────────────────────┐    │   │
//!   │   │  │              INTERNODE DESCENT LOOP                          │    │   │
//!   │   │  │                                                              │    │   │
//!   │   │  │  ┌──────────────┐                                            │    │   │
//!   │   │  │  │ Check node   │──── version.is_leaf()?                     │    │   │
//!   │   │  │  │ type         │                                            │    │   │
//!   │   │  │  └──────┬───────┘                                            │    │   │
//!   │   │  │         │                                                    │    │   │
//!   │   │  │    ┌────┴────┐                                               │    │   │
//!   │   │  │    │ Leaf    │ Internode                                     │    │   │
//!   │   │  │    ▼         ▼                                               │    │   │
//!   │   │  │  EXIT     ┌──────────────┐                                   │    │   │
//!   │   │  │  LOOP     │ Read version │──── v = version.stable()          │    │   │
//!   │   │  │           │ (Acquire)    │     Spins if DIRTY_MASK set       │    │   │
//!   │   │  │           └──────┬───────┘                                   │    │   │
//!   │   │  │                  │                                           │    │   │
//!   │   │  │                  ▼                                           │    │   │
//!   │   │  │           ┌──────────────┐                                   │    │   │
//!   │   │  │           │ Binary/linear│──── upper_bound_internode()       │    │   │
//!   │   │  │           │ search ikeys │     Find child pointer for ikey   │    │   │
//!   │   │  │           └──────┬───────┘                                   │    │   │
//!   │   │  │                  │                                           │    │   │
//!   │   │  │                  ▼                                           │    │   │
//!   │   │  │           ┌──────────────┐                                   │    │   │
//!   │   │  │           │ Prefetch     │──── prefetch_read(grandchild)     │    │   │
//!   │   │  │           │ grandchild   │     Hides memory latency          │    │   │
//!   │   │  │           └──────┬───────┘                                   │    │   │
//!   │   │  │                  │                                           │    │   │
//!   │   │  │                  ▼                                           │    │   │
//!   │   │  │           ┌──────────────┐                                   │    │   │
//!   │   │  │           │ Version      │──── has_changed(v)?               │    │   │
//!   │   │  │           │ changed?     │                                   │    │   │
//!   │   │  │           └──────┬───────┘                                   │    │   │
//!   │   │  │                  │                                           │    │   │
//!   │   │  │             ┌────┴────┐                                      │    │   │
//!   │   │  │             │ Yes     │ No                                   │    │   │
//!   │   │  │             ▼         ▼                                      │    │   │
//!   │   │  │          RETRY     Descend to child                          │    │   │
//!   │   │  │          from      └───────────────────────────► LOOP        │    │   │
//!   │   │  │          root                                                │    │   │
//!   │   │  │                                                              │    │   │
//!   │   │  └──────────────────────────────────────────────────────────────┘    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼  (reached leaf)                                            │   │
//!   │   │  ┌──────────────────────────────────────────────────────────────┐    │   │
//!   │   │  │                    LEAF SEARCH WITH OCC                      │    │   │
//!   │   │  │                                                              │    │   │
//!   │   │  │  ┌──────────────┐                                            │    │   │
//!   │   │  │  │ v1 = stable()│──── Wait for clean version (Acquire)       │    │   │
//!   │   │  │  └──────┬───────┘                                            │    │   │
//!   │   │  │         │                                                    │    │   │
//!   │   │  │         ▼                                                    │    │   │
//!   │   │  │  ┌──────────────┐                                            │    │   │
//!   │   │  │  │ Check B-link │──── Is ikey >= ikey_bound()?               │    │   │
//!   │   │  │  │ boundary     │     Key might be in next sibling           │    │   │
//!   │   │  │  └──────┬───────┘                                            │    │   │
//!   │   │  │         │                                                    │    │   │
//!   │   │  │    ┌────┴────┐                                               │    │   │
//!   │   │  │    │ Yes     │ No                                            │    │   │
//!   │   │  │    ▼         ▼                                               │    │   │
//!   │   │  │  Follow   ┌──────────────┐                                   │    │   │
//!   │   │  │  B-link   │ Linear search│──── Scan permutation for ikey     │    │   │
//!   │   │  │  to next  │ leaf slots   │     O(WIDTH) with 4x unrolling    │    │   │
//!   │   │  │  sibling  └──────┬───────┘                                   │    │   │
//!   │   │  │                  │                                           │    │   │
//!   │   │  │             ┌────┴────┬──────────────┬─────────────┐         │    │   │
//!   │   │  │             │         │              │             │         │    │   │
//!   │   │  │             ▼         ▼              ▼             ▼         │    │   │
//!   │   │  │        ┌────────┐ ┌────────┐   ┌──────────┐  ┌──────────┐    │    │   │
//!   │   │  │        │ FOUND  │ │  NOT   │   │  LAYER   │  │ VERSION  │    │    │   │
//!   │   │  │        │ value  │ │ FOUND  │   │ POINTER  │  │ CHANGED  │    │    │   │
//!   │   │  │        └───┬────┘ └───┬────┘   └────┬─────┘  └────┬─────┘    │    │   │
//!   │   │  │            │          │             │             │          │    │   │
//!   │   │  │            ▼          ▼             ▼             ▼          │    │   │
//!   │   │  │        ┌────────┐ ┌────────┐   ┌──────────┐  ┌──────────┐    │    │   │
//!   │   │  │        │ Verify │ │ Return │   │key.shift()│ │ Follow   │    │    │   │
//!   │   │  │        │ !has_  │ │ None   │   │layer += 1 │ │ B-link   │    │    │   │
//!   │   │  │        │changed │ │        │   │ CONTINUE  │ │ or RETRY │    │    │   │
//!   │   │  │        │ (v1)   │ │        │   │ LAYER     │ │ from root│    │    │   │
//!   │   │  │        └───┬────┘ └────────┘   │ LOOP      │ └──────────┘    │    │   │
//!   │   │  │            │                   └──────────┘                  │    │   │
//!   │   │  │            ▼                                                 │    │   │
//!   │   │  │        ┌────────┐                                            │    │   │
//!   │   │  │        │ Return │                                            │    │   │
//!   │   │  │        │ &Value │──── Zero-copy reference into tree          │    │   │
//!   │   │  │        └────────┘                                            │    │   │
//!   │   │  │                                                              │    │   │
//!   │   │  └──────────────────────────────────────────────────────────────┘    │   │
//!   │   │                                                                      │   │
//!   │   └──────────────────────────────────────────────────────────────────────┘   │
//!   │                                                                              │
//!   │   COMPLEXITY: O(log n) with layer factor, O(WIDTH) per leaf                  │
//!   │   MEMORY ORDERING: Acquire on version reads, compiler fence on validation    │
//!   │   RETRY POLICY: Bounded retries before falling back to from-root traversal   │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### INSERT (Fine-Grained Locking)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    INSERT OPERATION - FINE-GRAINED LOCKING                   │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   insert_with_guard(key, value, guard)                                       │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────┐                                                           │
//!   │   │ Alloc Arc<V> │──── S::output_to_raw(&value) - done ONCE                  │
//!   │   │ or inline    │     For Arc: heap allocation + refcount                   │
//!   │   │ value_ptr    │     For Inline: encode bits into pointer                  │
//!   │   └──────┬───────┘     (Reused across all retries - no re-allocation)        │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────────────────────────────────────────────────────────────┐   │
//!   │   │                         RETRY LOOP                                   │   │
//!   │   │                                                                      │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │  Optimistic  │──── Same traversal as GET (no locks)               │   │
//!   │   │  │  traversal   │     reach_leaf_concurrent_generic()                │   │
//!   │   │  │  to leaf     │     Prefetch, internode descent, B-link follow     │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │  Lock leaf   │──── version.lock() returns LockGuard               │   │
//!   │   │  │              │     CAS: set LOCK_BIT | INSERTING_BIT              │   │
//!   │   │  │              │     Spins on contention (backoff)                  │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │   Verify     │──── check_membership():                            │   │
//!   │   │  │  membership  │     1. ikey >= ikey_bound? → follow B-link         │   │
//!   │   │  │              │     2. prev != null && ikey < slot[0]? → retry     │   │
//!   │   │  │              │     3. version changed during traversal? → retry   │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │    ┌────┴────┐                                                       │   │
//!   │   │    │ Invalid │ Valid                                                 │   │
//!   │   │    ▼         ▼                                                       │   │
//!   │   │  Unlock   ┌──────────────┐                                           │   │
//!   │   │  RETRY    │ Search leaf  │──── Linear scan for matching ikey         │   │
//!   │   │  LOOP     │ for key      │     Check suffix if keylenx indicates     │   │
//!   │   │           └──────┬───────┘                                           │   │
//!   │   │                  │                                                   │   │
//!   │   │             ┌────┴────────────────────────────┐                      │   │
//!   │   │             │                                 │                      │   │
//!   │   │             ▼                                 ▼                      │   │
//!   │   │  ┌──────────────────────┐          ┌──────────────────────┐          │   │
//!   │   │  │     KEY EXISTS       │          │      NEW KEY         │          │   │
//!   │   │  │     (UPDATE)         │          │      (INSERT)        │          │   │
//!   │   │  ├──────────────────────┤          ├──────────────────────┤          │   │
//!   │   │  │                      │          │                      │          │   │
//!   │   │  │  ┌──────────────┐    │          │  ┌──────────────┐    │          │   │
//!   │   │  │  │ Load old_ptr │    │          │  │ find_usable_ │    │          │   │
//!   │   │  │  │ from slot    │    │          │  │ slot()       │    │          │   │
//!   │   │  │  └──────┬───────┘    │          │  └──────┬───────┘    │          │   │
//!   │   │  │         │            │          │         │            │          │   │
//!   │   │  │         ▼            │          │    ┌────┴────┐       │          │   │
//!   │   │  │  ┌──────────────┐    │          │    │         │       │          │   │
//!   │   │  │  │ Store new_ptr│    │          │    ▼         ▼       │          │   │
//!   │   │  │  │ (Release)    │    │          │ ┌──────┐  ┌──────┐   │          │   │
//!   │   │  │  └──────┬───────┘    │          │ │ HAS  │  │  NO  │   │          │   │
//!   │   │  │         │            │          │ │ SLOT │  │ SLOT │   │          │   │
//!   │   │  │         ▼            │          │ └──┬───┘  └──┬───┘   │          │   │
//!   │   │  │  ┌──────────────┐    │          │    │         │       │          │   │
//!   │   │  │  │ Retire old   │    │          │    │         ▼       │          │   │
//!   │   │  │  │ if NEEDS_    │    │          │    │    ┌────────┐   │          │   │
//!   │   │  │  │ RETIREMENT   │    │          │    │    │ SPLIT  │   │          │   │
//!   │   │  │  │ (Arc only)   │    │          │    │    │ LEAF   │   │          │   │
//!   │   │  │  └──────┬───────┘    │          │    │    │(below) │   │          │   │
//!   │   │  │         │            │          │    │    └────────┘   │          │   │
//!   │   │  │         │            │          │    │                 │          │   │
//!   │   │  │         │            │          │    ▼                 │          │   │
//!   │   │  │         │            │          │ ┌──────────────────┐ │          │   │
//!   │   │  │         │            │          │ │ SLOT-0 RULE:     │ │          │   │
//!   │   │  │         │            │          │ │                  │ │          │   │
//!   │   │  │         │            │          │ │ If slot==0 AND   │ │          │   │
//!   │   │  │         │            │          │ │ prev!=null AND   │ │          │   │
//!   │   │  │         │            │          │ │ ikey!=slot0_ikey │ │          │   │
//!   │   │  │         │            │          │ │ → CANNOT use     │ │          │   │
//!   │   │  │         │            │          │ │   slot 0         │ │          │   │
//!   │   │  │         │            │          │ │ (B-link bound)   │ │          │   │
//!   │   │  │         │            │          │ └────────┬─────────┘ │          │   │
//!   │   │  │         │            │          │          │           │          │   │
//!   │   │  │         │            │          │          ▼           │          │   │
//!   │   │  │         │            │          │ ┌──────────────────┐ │          │   │
//!   │   │  │         │            │          │ │ Install key:     │ │          │   │
//!   │   │  │         │            │          │ │ 1. ikeys[slot]   │ │          │   │
//!   │   │  │         │            │          │ │ 2. keylenx[slot] │ │          │   │
//!   │   │  │         │            │          │ │ 3. suffix if >8B │ │          │   │
//!   │   │  │         │            │          │ │ 4. values[slot]  │ │          │   │
//!   │   │  │         │            │          │ │ 5. CAS permuter  │ │          │   │
//!   │   │  │         │            │          │ │    (atomic pub)  │ │          │   │
//!   │   │  │         │            │          │ └────────┬─────────┘ │          │   │
//!   │   │  │         │            │          │          │           │          │   │
//!   │   │  └─────────┼────────────┘          └──────────┼───────────┘          │   │
//!   │   │            │                                  │                      │   │
//!   │   │            └─────────────┬────────────────────┘                      │   │
//!   │   │                          │                                           │   │
//!   │   │                          ▼                                           │   │
//!   │   │                   ┌──────────────┐                                   │   │
//!   │   │                   │ drop(guard)  │──── Unlock via RAII               │   │
//!   │   │                   │              │     Bumps VINSERT, clears dirty   │   │
//!   │   │                   └──────┬───────┘     (Release ordering)            │   │
//!   │   │                          │                                           │   │
//!   │   │                          ▼                                           │   │
//!   │   │                   ┌──────────────┐                                   │   │
//!   │   │                   │ Return       │                                   │   │
//!   │   │                   │ Ok(old_value)│──── None for new, Some for update │   │
//!   │   │                   └──────────────┘                                   │   │
//!   │   │                                                                      │   │
//!   │   └──────────────────────────────────────────────────────────────────────┘   │
//!   │                                                                              │
//!   │   COMPLEXITY: O(log n) traversal + O(WIDTH) search + O(1) insert             │
//!   │   CONTENTION: Only leaf locked, other leaves/internodes unaffected           │
//!   │   MEMORY: Value allocated once, reused across retries                        │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### SPLIT (Hand-Over-Hand Propagation)
//!
//! ```text
//!   Leaf becomes full during insert
//!          │
//!          ▼
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │ PHASE 1: Create sibling                                                  │
//!   │                                                                          │
//!   │   ┌────────────┐              ┌────────────┐                             │
//!   │   │ Left Leaf  │──── keys ───►│Right Leaf  │  Allocate sibling           │
//!   │   │  (LOCKED)  │  [mid..end]  │  (NEW)     │  Move upper half of keys    │
//!   │   └────────────┘              └────────────┘                             │
//!   │         │                           │                                    │
//!   │         │◄────────── B-link ────────┤                                    │
//!   │         │           (installed)     │                                    │
//!   └─────────┼───────────────────────────┼────────────────────────────────────┘
//!             │                           │
//!             ▼                           ▼
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │ PHASE 2: Hand-over-hand parent locking                                   │
//!   │                                                                          │
//!   │   ┌────────────┐    lock()    ┌────────────┐                             │
//!   │   │ Left Leaf  │─────────────►│  Parent    │                             │
//!   │   │  (LOCKED)  │              │  (LOCKED)  │                             │
//!   │   └────────────┘              └─────┬──────┘                             │
//!   │         │                           │                                    │
//!   │         │       ┌───────────────────┴───────────────────┐                │
//!   │         │       │   Install right sibling pointer       │                │
//!   │         │       │   in parent's children array          │                │
//!   │         │       └───────────────────────────────────────┘                │
//!   │         │                           │                                    │
//!   │    unlock()                    Parent full?                              │
//!   │                                     │                                    │
//!   │                          ┌──────────┴──────────┐                         │
//!   │                          │ No                  │ Yes                     │
//!   │                          ▼                     ▼                         │
//!   │                   ┌────────────┐       ┌────────────────┐                │
//!   │                   │ unlock()   │       │ Propagate up   │                │
//!   │                   │ Done       │       │ (split parent) │                │
//!   │                   └────────────┘       └────────────────┘                │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### REMOVE (Lazy Coalescing)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    REMOVE OPERATION - LAZY COALESCING                        │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   remove_with_guard(key, guard)                                              │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────────────────────────────────────────────────────────────┐   │
//!   │   │                         RETRY LOOP                                   │   │
//!   │   │                                                                      │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │  Optimistic  │──── Same traversal as GET/INSERT                   │   │
//!   │   │  │  traversal   │     reach_leaf_concurrent_generic()                │   │
//!   │   │  │  to leaf     │     May need to descend layers for long keys       │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │  Lock leaf   │──── version.lock() with INSERTING_BIT              │   │
//!   │   │  │              │     (Same lock as insert - mutual exclusion)       │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │   Verify     │──── check_membership() + search for key            │   │
//!   │   │  │  key exists  │     Same B-link checks as insert                   │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │    ┌────┴────────────────────────────┐                               │   │
//!   │   │    │                                 │                               │   │
//!   │   │    ▼                                 ▼                               │   │
//!   │   │ ┌────────────────────┐      ┌────────────────────┐                   │   │
//!   │   │ │    KEY FOUND       │      │   KEY NOT FOUND    │                   │   │
//!   │   │ ├────────────────────┤      ├────────────────────┤                   │   │
//!   │   │ │                    │      │                    │                   │   │
//!   │   │ │  ┌──────────────┐  │      │  ┌──────────────┐  │                   │   │
//!   │   │ │  │ Get value_ptr│  │      │  │ Unlock leaf  │  │                   │   │
//!   │   │ │  │ for return   │  │      │  │ Return       │  │                   │   │
//!   │   │ │  └──────┬───────┘  │      │  │ Ok(None)     │  │                   │   │
//!   │   │ │         │          │      │  └──────────────┘  │                   │   │
//!   │   │ │         ▼          │      │                    │                   │   │
//!   │   │ │  ┌──────────────┐  │      │   OR if layer_ptr: │                   │   │
//!   │   │ │  │ Update perm: │  │      │   Descend layer,   │                   │   │
//!   │   │ │  │              │  │      │   retry in sublayer│                   │   │
//!   │   │ │  │ old_perm =   │  │      │                    │                   │   │
//!   │   │ │  │  perm.load() │  │      └────────────────────┘                   │   │
//!   │   │ │  │              │  │                                               │   │
//!   │   │ │  │ new_perm =   │  │                                               │   │
//!   │   │ │  │  remove_at(  │  │                                               │   │
//!   │   │ │  │   position)  │  │                                               │   │
//!   │   │ │  │              │  │                                               │   │
//!   │   │ │  │ perm.CAS(    │  │                                               │   │
//!   │   │ │  │  old→new,    │  │                                               │   │
//!   │   │ │  │  Release)    │  │                                               │   │
//!   │   │ │  └──────┬───────┘  │                                               │   │
//!   │   │ │         │          │                                               │   │
//!   │   │ │         ▼          │                                               │   │
//!   │   │ │  ┌──────────────┐  │                                               │   │
//!   │   │ │  │ Retire value │  │                                               │   │
//!   │   │ │  │ if NEEDS_    │──┼──── guard.defer_retire(value_ptr)             │   │
//!   │   │ │  │ RETIREMENT   │  │     Value added to thread-local batch         │   │
//!   │   │ │  │ (Arc only)   │  │     Reclaimed when all guards drop            │   │
//!   │   │ │  └──────┬───────┘  │                                               │   │
//!   │   │ │         │          │                                               │   │
//!   │   │ │         ▼          │                                               │   │
//!   │   │ │  ┌──────────────┐  │                                               │   │
//!   │   │ │  │ Leaf empty?  │  │                                               │   │
//!   │   │ │  │ (perm.size   │  │                                               │   │
//!   │   │ │  │  == 0)       │  │                                               │   │
//!   │   │ │  └──────┬───────┘  │                                               │   │
//!   │   │ │         │          │                                               │   │
//!   │   │ │    ┌────┴────┐     │                                               │   │
//!   │   │ │    │ Yes     │ No  │                                               │   │
//!   │   │ │    ▼         ▼     │                                               │   │
//!   │   │ │ ┌────────┐ ┌────┐  │                                               │   │
//!   │   │ │ │Schedule│ │Skip│  │                                               │   │
//!   │   │ │ │coalesce│ │    │  │                                               │   │
//!   │   │ │ └───┬────┘ └──┬─┘  │                                               │   │
//!   │   │ │     │         │    │                                               │   │
//!   │   │ │     ▼         │    │                                               │   │
//!   │   │ │ ┌──────────┐  │    │                                               │   │
//!   │   │ │ │coalesce_ │  │    │                                               │   │
//!   │   │ │ │queue.    │  │    │                                               │   │
//!   │   │ │ │schedule( │  │    │                                               │   │
//!   │   │ │ │ leaf_ptr,│  │    │                                               │   │
//!   │   │ │ │ ikey,    │  │    │                                               │   │
//!   │   │ │ │ layer_ctx│  │    │                                               │   │
//!   │   │ │ │)         │  │    │                                               │   │
//!   │   │ │ └────┬─────┘  │    │                                               │   │
//!   │   │ │      │        │    │                                               │   │
//!   │   │ │      └────┬───┘    │                                               │   │
//!   │   │ │           │        │                                               │   │
//!   │   │ │           ▼        │                                               │   │
//!   │   │ │    ┌──────────────┐│                                               │   │
//!   │   │ │    │ Unlock leaf  ││                                               │   │
//!   │   │ │    │ Return       ││                                               │   │
//!   │   │ │    │ Ok(Some(val))││                                               │   │
//!   │   │ │    └──────────────┘│                                               │   │
//!   │   │ │                    │                                               │   │
//!   │   │ └────────────────────┘                                               │   │
//!   │   │                                                                      │   │
//!   │   └──────────────────────────────────────────────────────────────────────┘   │
//!   │                                                                              │
//!   │   WHY LAZY COALESCING:                                                       │
//!   │   ├─ Inline removal would require locking parent → deadlock risk             │
//!   │   ├─ Empty leaves can be reused by concurrent inserts                        │
//!   │   ├─ Readers already handle deleted nodes via B-link retry                   │
//!   │   └─ Cleanup batched for efficiency (process_coalesce_queue)                 │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    COALESCE (Background Cleanup)                             │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   process_coalesce_queue() / tree.flush(&guard)                              │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────┐                                                           │
//!   │   │ Pop entry    │──── CoalesceEntry { leaf_ptr, ikey_bound, layer_ctx }     │
//!   │   │ from queue   │                                                           │
//!   │   └──────┬───────┘                                                           │
//!   │          │                                                                   │
//!   │     ┌────┴────┐                                                              │
//!   │     │ Empty   │ Entry found                                                  │
//!   │     ▼         ▼                                                              │
//!   │   DONE     ┌──────────────┐                                                  │
//!   │            │ Lock leaf    │                                                  │
//!   │            └──────┬───────┘                                                  │
//!   │                   │                                                          │
//!   │                   ▼                                                          │
//!   │            ┌──────────────┐                                                  │
//!   │            │ Still empty? │──── Concurrent insert may have reused it         │
//!   │            └──────┬───────┘                                                  │
//!   │                   │                                                          │
//!   │              ┌────┴────┐                                                     │
//!   │              │ No      │ Yes                                                 │
//!   │              ▼         ▼                                                     │
//!   │           Unlock    ┌──────────────┐                                         │
//!   │           Skip      │ Check if     │                                         │
//!   │                     │ sublayer root│                                         │
//!   │                     └──────┬───────┘                                         │
//!   │                            │                                                 │
//!   │                       ┌────┴────┐                                            │
//!   │                       │ Yes     │ No (regular leaf)                          │
//!   │                       ▼         ▼                                            │
//!   │                    gc_layer() ┌──────────────┐                               │
//!   │                    (special   │ mark_deleted │──── Set DELETED_BIT           │
//!   │                    cleanup)   └──────┬───────┘                               │
//!   │                                      │                                       │
//!   │                                      ▼                                       │
//!   │                               ┌──────────────┐                               │
//!   │                               │ Unlink from  │                               │
//!   │                               │ B-link chain:│                               │
//!   │                               │              │                               │
//!   │                               │ prev.next =  │                               │
//!   │                               │   self.next  │                               │
//!   │                               │              │                               │
//!   │                               │ next.prev =  │                               │
//!   │                               │   self.prev  │                               │
//!   │                               └──────┬───────┘                               │
//!   │                                      │                                       │
//!   │                                      ▼                                       │
//!   │                               ┌──────────────┐                               │
//!   │                               │ Remove from  │──── NodeCleaner::             │
//!   │                               │ parent       │     remove_leaf_from_         │
//!   │                               │ internode    │     parent_for_coalesce()     │
//!   │                               └──────┬───────┘                               │
//!   │                                      │                                       │
//!   │                                      ▼                                       │
//!   │                               ┌──────────────┐                               │
//!   │                               │ Retire leaf  │──── guard.defer_retire()      │
//!   │                               │ node via     │     Safe: unreachable from    │
//!   │                               │ seize        │     tree after unlinking      │
//!   │                               └──────────────┘                               │
//!   │                                                                              │
//!   │   SAFETY: Leaf retirement is safe because:                                   │
//!   │   1. Marked deleted → readers see DELETED_BIT, follow B-link                 │
//!   │   2. Unlinked from B-link chain → not reachable via sibling traversal        │
//!   │   3. Removed from parent → not reachable via tree traversal                  │
//!   │   4. Epoch guard protects existing references until safe to free             │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### RANGE SCAN (Iterator)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    RANGE SCAN - ZERO-COPY ITERATOR                           │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   scan_ref(start_bound, end_bound, callback, guard)                          │
//!   │          │                                                                   │
//!   │          ▼                                                                   │
//!   │   ┌──────────────────────────────────────────────────────────────────────┐   │
//!   │   │                    INITIALIZATION                                    │   │
//!   │   │                                                                      │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │ Find first   │──── find_leaf_for_scan(start_bound)                │   │
//!   │   │  │ leaf         │     Optimistic traversal to leftmost matching leaf │   │
//!   │   │  │              │     Uses B-link if key > ikey_bound                │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   │         ▼                                                            │   │
//!   │   │  ┌──────────────┐                                                    │   │
//!   │   │  │ Init state:  │                                                    │   │
//!   │   │  │              │                                                    │   │
//!   │   │  │ • leaf_ptr   │──── Current leaf being scanned                     │   │
//!   │   │  │ • ki         │──── Current slot index in permutation              │   │
//!   │   │  │ • perm       │──── Cached permutation snapshot                    │   │
//!   │   │  │ • version    │──── Version at snapshot time                       │   │
//!   │   │  │ • end_bound  │──── Stop condition                                 │   │
//!   │   │  │ • layer_stack│──── For multi-layer key descent                    │   │
//!   │   │  └──────┬───────┘                                                    │   │
//!   │   │         │                                                            │   │
//!   │   └─────────┼────────────────────────────────────────────────────────────┘   │
//!   │             │                                                                │
//!   │             ▼                                                                │
//!   │   ┌──────────────────────────────────────────────────────────────────────┐   │
//!   │   │                    STATE MACHINE LOOP                                │   │
//!   │   │                                                                      │   │
//!   │   │    ┌─────────────────────────────────────────────────────────────┐   │   │
//!   │   │    │                                                             │   │   │
//!   │   │    ▼                                                             │   │   │
//!   │   │  ┌─────────────┐                                                 │   │   │
//!   │   │  │  FIND_NEXT  │◄────────────────────────────────────────────────┤   │   │
//!   │   │  │             │                                                 │   │   │
//!   │   │  │  ki < size? │                                                 │   │   │
//!   │   │  └──────┬──────┘                                                 │   │   │
//!   │   │         │                                                        │   │   │
//!   │   │    YES  │  NO                                                    │   │   │
//!   │   │   ┌─────┴─────┐                                                  │   │   │
//!   │   │   │           │                                                  │   │   │
//!   │   │   ▼           ▼                                                  │   │   │
//!   │   │ ┌───────────────────┐    ┌───────────────────────────────────┐   │   │   │
//!   │   │ │    READ SLOT      │    │           EXHAUSTED               │   │   │   │
//!   │   │ │  perm[ki] → slot  │    │         (ki >= size)              │   │   │   │
//!   │   │ │                   │    │                                   │   │   │   │
//!   │   │ │ ┌───────────────┐ │    │  ┌─────────────────────────────┐  │   │   │   │
//!   │   │ │ │Version check: │ │    │  │ At top layer (stack empty)? │  │   │   │   │
//!   │   │ │ │changed since  │ │    │  └─────────────┬───────────────┘  │   │   │   │
//!   │   │ │ │snapshot?      │ │    │        YES     │      NO          │   │   │   │
//!   │   │ │ └───────┬───────┘ │    │       ┌────────┴────────┐         │   │   │   │
//!   │   │ │    YES  │  NO     │    │       │                 │         │   │   │   │
//!   │   │ │   ┌─────┴─────┐   │    │       ▼                 ▼         │   │   │   │
//!   │   │ │   │           │   │    │  ┌─────────┐     ┌───────────┐    │   │   │   │
//!   │   │ │   ▼           ▼   │    │  │ Follow  │     │ Pop layer │    │   │   │   │
//!   │   │ │ ┌─────┐  ┌──────┐ │    │  │ B-link  │     │ stack,    │    │   │   │   │
//!   │   │ │ │RETRY│  │ Check│ │    │  │ to next │     │ continue  │    │   │   │   │
//!   │   │ │ │     │  │ slot │ │    │  │ leaf    │     │ parent    │────┼───┤   │   │
//!   │   │ │ │Re-  │  │ type │ │    │  │         │     │ scan      │    │   │   │   │
//!   │   │ │ │read │  └──┬───┘ │    │  │ null?   │     └───────────┘    │   │   │   │
//!   │   │ │ │perm,│     │     │    │  │ → DONE  │                      │   │   │   │
//!   │   │ │ │re-  │     │     │    │  └─────────┘                      │   │   │   │
//!   │   │ │ │find │     │     │    │                                   │   │   │   │
//!   │   │ │ │leaf │     │     │    └───────────────────────────────────┘   │   │   │
//!   │   │ │ │if   │     │     │                                            │   │   │
//!   │   │ │ │split│     │     │                                            │   │   │
//!   │   │ │ └──┬──┘     │     │                                            │   │   │
//!   │   │ │    │        │     │                                            │   │   │
//!   │   │ │    │   ┌────┴─────────────────────┐                            │   │   │
//!   │   │ │    │   │                          │                            │   │   │
//!   │   │ │    │   ▼                          ▼                            │   │   │
//!   │   │ │    │ ┌──────────────┐      ┌──────────────┐                    │   │   │
//!   │   │ │    │ │ VALUE SLOT   │      │ LAYER PTR    │                    │   │   │
//!   │   │ │    │ │              │      │              │                    │   │   │
//!   │   │ │    │ │ Has key+val  │      │ Points to    │                    │   │   │
//!   │   │ │    │ │ → go to EMIT │      │ sublayer     │                    │   │   │
//!   │   │ │    │ └──────┬───────┘      └──────┬───────┘                    │   │   │
//!   │   │ │    │        │                     │                            │   │   │
//!   │   │ │    │        │                     ▼                            │   │   │
//!   │   │ │    │        │              ┌──────────────┐                    │   │   │
//!   │   │ │    │        │              │    DOWN:     │                    │   │   │
//!   │   │ │    │        │              │ Push context │                    │   │   │
//!   │   │ │    │        │              │ (leaf, ki+1) │                    │   │   │
//!   │   │ │    │        │              │ to stack,    │                    │   │   │
//!   │   │ │    │        │              │ descend to   │                    │   │   │
//!   │   │ │    │        │              │ sublayer     │────────────────────┤   │   │
//!   │   │ │    │        │              │ root         │                    │   │   │
//!   │   │ │    │        │              └──────────────┘                    │   │   │
//!   │   │ │    │        │                                                  │   │   │
//!   │   │ └────┼────────┼──────────────────────────────────────────────────┘   │   │
//!   │   │      │        │                                                      │   │
//!   │   │      │        ▼                                                      │   │
//!   │   │      │  ┌────────────────────────────────────────────────────────┐   │   │
//!   │   │      │  │                         EMIT                           │   │   │
//!   │   │      │  │                                                        │   │   │
//!   │   │      │  │  ┌──────────────┐                                      │   │   │
//!   │   │      │  │  │ Build key    │──── Reconstruct full key:            │   │   │
//!   │   │      │  │  │ from slot    │     layer prefixes + ikey + suffix   │   │   │
//!   │   │      │  │  └──────┬───────┘                                      │   │   │
//!   │   │      │  │         │                                              │   │   │
//!   │   │      │  │         ▼                                              │   │   │
//!   │   │      │  │  ┌──────────────┐                                      │   │   │
//!   │   │      │  │  │ Check end_   │                                      │   │   │
//!   │   │      │  │  │ bound: key   │                                      │   │   │
//!   │   │      │  │  │ > end?       │                                      │   │   │
//!   │   │      │  │  │ → DONE       │                                      │   │   │
//!   │   │      │  │  └──────┬───────┘                                      │   │   │
//!   │   │      │  │         │                                              │   │   │
//!   │   │      │  │         ▼                                              │   │   │
//!   │   │      │  │  ┌──────────────┐                                      │   │   │
//!   │   │      │  │  │ Load value   │──── Zero-copy: return &V reference   │   │   │
//!   │   │      │  │  │ reference    │     Arc: deref into Arc<V>           │   │   │
//!   │   │      │  │  │              │     Inline: decode from pointer bits │   │   │
//!   │   │      │  │  └──────┬───────┘                                      │   │   │
//!   │   │      │  │         │                                              │   │   │
//!   │   │      │  │         ▼                                              │   │   │
//!   │   │      │  │  ┌──────────────┐                                      │   │   │
//!   │   │      │  │  │ callback(    │                                      │   │   │
//!   │   │      │  │  │  key, &value │                                      │   │   │
//!   │   │      │  │  │ )            │                                      │   │   │
//!   │   │      │  │  └──────┬───────┘                                      │   │   │
//!   │   │      │  │         │                                              │   │   │
//!   │   │      │  │    ┌────┴────┐                                         │   │   │
//!   │   │      │  │    │         │                                         │   │   │
//!   │   │      │  │  true      false                                       │   │   │
//!   │   │      │  │    │         │                                         │   │   │
//!   │   │      │  │    ▼         ▼                                         │   │   │
//!   │   │      │  │  ki += 1   DONE (early termination)                    │   │   │
//!   │   │      │  │    │                                                   │   │   │
//!   │   │      │  │    └───────────────────────────────────────────────────┼───┘   │
//!   │   │      │  │                                                        │       │
//!   │   │      │  └────────────────────────────────────────────────────────┘       │
//!   │   │      │                                                                   │
//!   │   │      └───────────────────────────────────────────────────────────────────┘
//!   │   │                                                                          │
//!   │   └──────────────────────────────────────────────────────────────────────────┘
//!   │                                                                              │
//!   │   GUARANTEES:                                                                │
//!   │   ├─ Keys emitted in lexicographic order                                     │
//!   │   ├─ Each key emitted at most once (snapshot consistency)                    │
//!   │   ├─ Concurrent inserts: may or may not be seen (depends on timing)          │
//!   │   ├─ Zero-copy: values are references, no cloning                            │
//!   │   └─ Handles splits/deletes via version validation + B-link retry            │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Concurrency Model
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    OPTIMISTIC CONCURRENCY CONTROL (OCC)                      │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   ┌─────────────────────────────┐      ┌─────────────────────────────┐       │
//!   │   │     READER (Lock-Free)      │      │    WRITER (Fine-Grained)    │       │
//!   │   ├─────────────────────────────┤      ├─────────────────────────────┤       │
//!   │   │                             │      │                             │       │
//!   │   │  ┌───────────────────────┐  │      │  ┌───────────────────────┐  │       │
//!   │   │  │ v1 = stable()         │  │      │  │ guard = lock()        │  │       │
//!   │   │  │                       │  │      │  │                       │  │       │
//!   │   │  │ • Spin while dirty    │  │      │  │ • CAS: 0 → LOCK|INS   │  │       │
//!   │   │  │ • Return clean version│  │      │  │ • Spin on failure     │  │       │
//!   │   │  │ • Acquire ordering    │  │      │  │ • Acquire ordering    │  │       │
//!   │   │  └───────────┬───────────┘  │      │  └───────────┬───────────┘  │       │
//!   │   │              │              │      │              │              │       │
//!   │   │              ▼              │      │              ▼              │       │
//!   │   │  ┌───────────────────────┐  │      │  ┌───────────────────────┐  │       │
//!   │   │  │ Read node data        │  │      │  │ Modify node data      │  │       │
//!   │   │  │                       │  │      │  │                       │  │       │
//!   │   │  │ • ikeys[slot]         │  │      │  │ • ikeys[slot] = new   │  │       │
//!   │   │  │ • keylenx[slot]       │  │      │  │ • values[slot] = ptr  │  │       │
//!   │   │  │ • values[slot]        │  │      │  │ • permutation CAS     │  │       │
//!   │   │  │ • permutation         │  │      │  │ • suffix bag update   │  │       │
//!   │   │  │                       │  │      │  │                       │  │       │
//!   │   │  │ NO LOCKS HELD         │  │      │  │ EXCLUSIVE ACCESS      │  │       │
//!   │   │  └───────────┬───────────┘  │      │  └───────────┬───────────┘  │       │
//!   │   │              │              │      │              │              │       │
//!   │   │              ▼              │      │              ▼              │       │
//!   │   │  ┌───────────────────────┐  │      │  ┌───────────────────────┐  │       │
//!   │   │  │ has_changed(v1)?      │  │      │  │ drop(guard)           │  │       │
//!   │   │  │                       │  │      │  │                       │  │       │
//!   │   │  │ • Compiler fence      │  │      │  │ • Bump VINSERT/VSPLIT │  │       │
//!   │   │  │ • (v1 ^ cur) >= 512?  │  │      │  │ • Clear LOCK|INS|SPLIT│  │       │
//!   │   │  │                       │  │      │  │ • Release ordering    │  │       │
//!   │   │  │ Yes → RETRY           │  │      │  │                       │  │       │
//!   │   │  │ No  → SUCCESS         │  │      │  │ PANIC-SAFE (RAII)     │  │       │
//!   │   │  └───────────────────────┘  │      │  └───────────────────────┘  │       │
//!   │   │                             │      │                             │       │
//!   │   └─────────────────────────────┘      └─────────────────────────────┘       │
//!   │                                                                              │
//!   │   RACE SCENARIOS AND RESOLUTION:                                             │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │ Scenario 1: Reader overlaps with writer                                │ │
//!   │   │                                                                        │ │
//!   │   │   Reader               Writer                                          │ │
//!   │   │   ──────               ──────                                          │ │
//!   │   │   v1 = stable()                                                        │ │
//!   │   │   read ikey[3]         lock()                                          │ │
//!   │   │                        modify ikey[3]                                  │ │
//!   │   │   read value[3]        unlock()  ← version bumped                      │ │
//!   │   │   has_changed(v1)? → YES → RETRY with fresh data                       │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │ Scenario 2: Reader during split                                        │ │
//!   │   │                                                                        │ │
//!   │   │   Reader               Writer (splitting)                              │ │
//!   │   │   ──────               ─────────────────                               │ │
//!   │   │   v1 = stable()                                                        │ │
//!   │   │                        mark_split()  ← SPLITTING_BIT set               │ │
//!   │   │   search for key       move keys to sibling                            │ │
//!   │   │   key not found        install B-link                                  │ │
//!   │   │   has_changed(v1)? → YES (VSPLIT bumped)                               │ │
//!   │   │   → Follow B-link to find key in sibling                               │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                         B-LINK HELP-ALONG PROTOCOL                           │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   When a split occurs, the B-link chain allows readers to find keys that     │
//!   │   have moved to a sibling, even before the parent is updated.                │
//!   │                                                                              │
//!   │   BEFORE SPLIT:                                                              │
//!   │                                                                              │
//!   │       ┌────────────────────────────────────────────┐                         │
//!   │       │               Leaf A                       │                         │
//!   │       │  ikey_bound = MAX                          │                         │
//!   │       │  keys: [k1, k2, k3, k4, k5, k6, k7, k8]    │                         │
//!   │       │  next → null                               │                         │
//!   │       └────────────────────────────────────────────┘                         │
//!   │                                                                              │
//!   │   DURING/AFTER SPLIT:                                                        │
//!   │                                                                              │
//!   │       ┌───────────────────────┐     ┌───────────────────────┐                │
//!   │       │       Leaf A          │     │       Leaf B          │                │
//!   │       │  ikey_bound = k4      │     │  ikey_bound = MAX     │                │
//!   │       │ keys: [k1, k2, k3, k4]│────►│ keys: [k5, k6, k7, k8]│                │
//!   │       │  next ────────────────────► │  prev → A             │                │
//!   │       └───────────────────────┘     └───────────────────────┘                │
//!   │                                                                              │
//!   │   READER LOOKING FOR k6:                                                     │
//!   │                                                                              │
//!   │     1. Parent routes to Leaf A (old routing)                                 │
//!   │     2. Reader arrives at Leaf A                                              │
//!   │     3. v1 = stable() on Leaf A                                               │
//!   │     4. Search for k6 in Leaf A → NOT FOUND                                   │
//!   │     5. Check: k6 >= ikey_bound(k4)? → YES                                    │
//!   │     6. Follow B-link: A.next → Leaf B                                        │
//!   │     7. v2 = stable() on Leaf B                                               │
//!   │     8. Search for k6 in Leaf B → FOUND                                       │
//!   │     9. Validate !has_changed(v2) → SUCCESS                                   │
//!   │                                                                              │
//!   │   KEY INSIGHT: Readers never block, they "help along" by following B-links.  │
//!   │   The tree is always consistent enough for readers to find their keys.       │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Reclamation
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    SEIZE EPOCH-BASED RECLAMATION                             │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   Seize uses a hyaline-style epoch-based scheme with thread-local batching.  │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    EPOCH TIMELINE                                      │ │
//!   │   │                                                                        │ │
//!   │   │   Time ──────────────────────────────────────────────────────────────► │ │
//!   │   │                                                                        │ │
//!   │   │   Thread 1:                                                            │ │
//!   │   │   ┌────────────────────────────────────────────────────┐               │ │
//!   │   │   │ guard = enter()           drop(guard)              │               │ │
//!   │   │   │      │                         │                   │               │ │
//!   │   │   │      ▼                         ▼                   │               │ │
//!   │   │   │  ┌───────┐ remove(k) ┌──────────────┐              │               │ │
//!   │   │   │  │Epoch 5│──────────►│defer_retire()│              │               │ │
//!   │   │   │  └───────┘    │      └──────────────┘              │               │ │
//!   │   │   │               │             │                      │               │ │
//!   │   │   │               ▼             ▼                      │               │ │
//!   │   │   │        ┌──────────────────────────┐                │               │ │
//!   │   │   │        │ Thread-local batch:      │                │               │ │
//!   │   │   │        │ [ptr1, ptr2, ptr3, ...]  │                │               │ │
//!   │   │   │        └──────────────────────────┘                │               │ │
//!   │   │   │                                                    │               │ │
//!   │   │   └────────────────────────────────────────────────────┘               │ │
//!   │   │                                                                        │ │
//!   │   │   Thread 2:                                                            │ │
//!   │   │   ┌────────────────────────────────────────────────────┐               │ │
//!   │   │   │ guard = enter()              drop(guard)           │               │ │
//!   │   │   │      │                            │                │               │ │
//!   │   │   │      ▼          get(k)            ▼                │               │ │
//!   │   │   │  ┌───────┐  (may see old   ┌───────────────┐       │               │ │
//!   │   │   │  │Epoch 5│   value - SAFE) │Thread 2 exits │       │               │ │
//!   │   │   │  └───────┘                 │epoch 5        │       │               │ │
//!   │   │   │                            └───────────────┘       │               │ │
//!   │   │   └────────────────────────────────────────────────────┘               │ │
//!   │   │                                                                        │ │
//!   │   │   ───────────────────────────────────────────────────────────────────  │ │
//!   │   │                                                                        │ │
//!   │   │   Reclamation (when batch_size reached or all threads exit epoch):     │ │
//!   │   │                                                                        │ │
//!   │   │   ┌────────────────────────────────────────────────────────────────┐   │ │
//!   │   │   │ 1. Check: All threads left epoch 5?                            │   │ │
//!   │   │   │    └─ YES: No thread can hold references from epoch 5          │   │ │
//!   │   │   │                                                                │   │ │
//!   │   │   │ 2. For each ptr in epoch 5's retire list:                      │   │ │
//!   │   │   │    └─ free(ptr)  ← Safe to deallocate                          │   │ │
//!   │   │   │                                                                │   │ │
//!   │   │   │ 3. Advance global epoch                                        │   │ │
//!   │   │   └────────────────────────────────────────────────────────────────┘   │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   SEIZE INTERNALS:                                                           │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                                                                        │ │
//!   │   │   Collector::batch_size = max(CPU_COUNT, 32)                           │ │
//!   │   │   └─ Configurable via MassTree::with_allocator_batch_size()            │ │
//!   │   │                                                                        │ │
//!   │   │   defer_retire(ptr):                                                   │ │
//!   │   │   ├─ Add ptr to thread-local Vec                                       │ │
//!   │   │   ├─ If Vec.len() >= batch_size:                                       │ │
//!   │   │   │  └─ Trigger epoch advancement + reclamation                        │ │
//!   │   │   └─ Cost: ~100-500ns (Vec append)                                     │ │
//!   │   │                                                                        │ │
//!   │   │   Batch trigger (heavy operation):                                     │ │
//!   │   │   ├─ sys_membarrier() syscall (Linux) - expensive!                     │ │
//!   │   │   ├─ Check all thread epochs                                           │ │
//!   │   │   ├─ Reclaim safe batches                                              │ │
//!   │   │   └─ Cost: ~1-5µs                                                      │ │
//!   │   │                                                                        │ │
//!   │   │   Guard reuse (critical for performance):                              │ │
//!   │   │   ├─ guard.enter() → ~µs (first enter)                                 │ │
//!   │   │   ├─ Nested guards → FREE (counter increment)                          │ │
//!   │   │   └─ ALWAYS reuse guards across operations!                            │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   INLINE STORAGE OPTIMIZATION:                                               │
//!   │                                                                              │
//!   │   MassTree15Inline<V: Copy> eliminates most retirement:                      │
//!   │   ├─ Values stored inline (no Arc)                                           │
//!   │   ├─ Updates overwrite in-place (no old value to retire)                     │
//!   │   ├─ Only node retirements remain (splits, coalesces)                        │
//!   │   └─ Result: 77-179% faster on update-heavy workloads                        │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Organization
//!
//! ```text
//!   src/
//!   ├── lib.rs                 ← You are here (entry point, re-exports)
//!   │
//!   ├── tree.rs               ← Core tree: MassTreeGeneric struct
//!   │   └── tree/              ← Tree implementation modules
//!   │       ├── generic.rs     ← Main impl block, guard, flush
//!   │       │   ├── search.rs      ← Key search within nodes
//!   │       │   ├── insert.rs      ← Insert path with slot allocation
//!   │       │   ├── optimistic_reads.rs  ← Lock-free get operations
//!   │       │   ├── batch.rs       ← Batch insert API
//!   │       │   ├── layer.rs       ← Multi-layer (sublayer) handling
//!   │       │   └── split.rs       ← Per-slot split helpers
//!   │       ├── remove.rs      ← Deletion with permutation update
//!   │       ├── split.rs       ← Hand-over-hand split propagation
//!   │       │   ├── propagation.rs         ← Core split loop
//!   │       │   ├── propagation_context.rs ← RAII lock management
//!   │       │   ├── parent_locking.rs      ← Membership validation
//!   │       │   └── root_creation.rs       ← New root/layer-root
//!   │       ├── coalesce.rs    ← Lazy empty leaf cleanup
//!   │       └── range.rs       ← Range iteration module
//!   │           └── range/
//!   │               ├── api.rs         ← scan, scan_ref, scan_prefix
//!   │               ├── iterator.rs    ← RangeIter implementation
//!   │               ├── scan_state.rs  ← State machine for scanning
//!   │               ├── cursor_key.rs  ← Key reconstruction
//!   │               ├── helper.rs      ← Scan utilities
//!   │               └── find.rs        ← Find first/last leaf in range
//!   │
//!   ├── leaf15.rs              ← WIDTH=15 leaf node
//!   ├── leaf24.rs              ← WIDTH=24 leaf node
//!   ├── leaf_trait.rs          ← LayerCapableLeaf trait
//!   ├── internode.rs           ← Internal routing node
//!   ├── nodeversion.rs         ← OCC primitive (lock + version)
//!   │
//!   ├── slot.rs                ← ValueSlot trait (Arc vs Inline)
//!   ├── value.rs               ← LeafValue, LeafValueIndex
//!   ├── suffix.rs              ← SuffixBag for keys > 8 bytes
//!   ├── permuter.rs            ← WIDTH=15 permutation (u64)
//!   ├── permuter24.rs          ← WIDTH=24 permutation (u128)
//!   │
//!   ├── alloc15.rs             ← SeizeAllocator15
//!   ├── alloc24.rs             ← SeizeAllocator24
//!   ├── alloc_trait.rs         ← NodeAllocatorGeneric trait
//!   ├── retirement.rs          ← BatchedRetire, FlushOnDrop
//!   │
//!   ├── key.rs                 ← Key<'a> with layer traversal
//!   ├── ksearch.rs             ← Binary/linear search helpers
//!   ├── link.rs                ← B-link chain utilities
//!   ├── prefetch.rs            ← CPU prefetch hints
//!   └── ordering.rs            ← Memory ordering constants
//! ```
//!
//! ## Value Storage
//!
//! - **`MassTree<V>`**: Stores values as `Arc<V>`. Returns `Arc<V>` on get.
//! - **`MassTreeIndex<V: Copy>`**: Convenience wrapper that copies values.
//!
//! ## Feature Flags
//!
//! - **`mimalloc`**: Uses mimalloc as the global allocator (recommended).
//! - **`tracing`**: Enables structured logging to `logs/masstree.jsonl`.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// We use extensive benchmarking to verify #[inline(always)] placement is correct.
#![allow(clippy::inline_always)]
#![allow(clippy::multiple_crate_versions)]

// Global allocator selection (enabled via features)
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Initialize tracing with file and console output.
///
/// Writes NDJSON logs to `logs/masstree.jsonl` for structured analysis.
/// Also outputs to console unless `MASSTREE_LOG_CONSOLE=0` is set.
///
/// Safe to call multiple times - only the first call takes effect.
///
/// # Environment Variables
///
/// - `RUST_LOG`: Filter directives (default: `masstree=info`). Example: `masstree=debug,masstree::tree::locked=trace`
/// - `MASSTREE_LOG_DIR`: Log directory (default: `logs/`)
/// - `MASSTREE_LOG_CONSOLE`: Set to `0` to disable console output
///
/// # Example
///
/// ```bash
/// # Run with debug logging to file, no console spam
/// RUST_LOG=masstree=debug MASSTREE_LOG_CONSOLE=0 cargo run --features tracing
///
/// # Analyze logs with jq
/// cat logs/masstree.jsonl | jq 'select(.fields.ikey != null)'
/// ```
#[cfg(feature = "tracing")]
pub fn init_tracing() {
    use std::env as StdEnv;
    use std::fs as StdFs;
    use std::sync::OnceLock;
    use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

    // Store the guard in a static to keep the non-blocking writer alive.
    // The guard ensures logs are flushed on process exit.
    static GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

    // OnceLock::get_or_init ensures this runs exactly once
    GUARD.get_or_init(|| {
        // Configuration from environment
        let log_dir = StdEnv::var("MASSTREE_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let console_enabled = false;
        let filter_str = StdEnv::var("RUST_LOG").unwrap_or_else(|_| "masstree=info".to_string());

        // Create log directory
        let _ = StdFs::create_dir_all(&log_dir);

        // File appender - non-rotating, writes to masstree.jsonl (NDJSON)
        let file_appender = tracing_appender::rolling::never(&log_dir, "masstree.jsonl");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        // File layer - JSON format for structured analysis
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .json()
            .with_filter(
                EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("info")),
            );

        // Console layer - compact format for human reading
        let console_layer = if console_enabled {
            Some(
                tracing_subscriber::fmt::layer()
                    .with_thread_ids(true)
                    .compact()
                    .with_filter(
                        EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("trace")),
                    ),
            )
        } else {
            None
        };

        // Use try_init to avoid panic if tests set their own subscriber
        let _ = tracing_subscriber::registry()
            .with(file_layer)
            .with(console_layer)
            .try_init();

        // Return the guard to be stored in OnceLock
        guard
    });
}

/// No-op when tracing feature is disabled.
#[cfg(not(feature = "tracing"))]
pub const fn init_tracing() {}

pub mod alloc15;
pub mod alloc24;
mod alloc_common;
pub mod alloc_trait;
mod error;
pub mod inline;
pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf15;
pub mod leaf24;
pub mod leaf_trait;
pub mod link;
pub mod nodeversion;
pub mod ordering;
pub mod permuter;
pub mod permuter24;
pub mod prefetch;
pub mod ref_value_slot;
mod retirement;
mod shard_counter;
pub mod slot;
pub mod suffix;
pub mod tree;
pub mod value;

pub use error::{AllocError, AllocKind, AllocResult};
pub use retirement::{BatchedRetire, FlushOnDrop};

// Re-export leaf node traits for generic tree operations
pub use leaf_trait::{TreeInternode, TreeLeafNode, TreePermutation};

// Re-export allocator trait for generic tree operations
pub use alloc_trait::NodeAllocatorGeneric;

// Re-export Permuter types
pub use permuter::{AtomicPermuter, AtomicPermuter15, Permuter, Permuter15};
pub use permuter24::{AtomicPermuter24, Permuter24, WIDTH_24};

// Re-export leaf node types
pub use leaf15::{LeafNode15, WIDTH_15};
pub use leaf24::{
    LeafNode24, MODSTATE_DELETED_LAYER, MODSTATE_INSERT, MODSTATE_REMOVE, WIDTH_24 as LEAF24_WIDTH,
};

// Re-export allocator types
pub use alloc15::{SeizeAllocator15, SeizeAllocator15TrueInline};
pub use alloc24::SeizeAllocator24;

// Re-export inline types for true-inline storage
pub use inline::bits::InlineBits;
pub use inline::leaf15_true::LeafNode15TrueInline;
pub use slot::true_inline::TrueInlineSlot;

// Re-export value types
pub use value::{InsertTarget, LeafValue, LeafValueIndex, SplitPoint};

// Re-export link utilities
pub use link::Linker;

// Re-export main types for convenience
pub use ref_value_slot::RefValueSlot;
pub use slot::ValueSlot;
pub use suffix::{InlineSuffixBag, PermutationProvider, SuffixBag};
pub use tree::{InsertError, RemoveError};
pub use tree::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
pub use tree::{
    MassTree, MassTree15, MassTree15Inline, MassTree24, MassTree24Inline, MassTreeGeneric,
    MassTreeIndex,
};

// Re-export batch insert types
pub use tree::{BatchEntry, BatchInsertResult, batch};
