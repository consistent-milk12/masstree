//! Filepath: src/nodeversion.rs
//!
//! Node version for optimistic concurrency control.
//!
//! [`NodeVersion`] combines lock state, version counters, and metadata flags
//! in a single `u32`. Readers use optimistic validation, writers acquire locks.
//!
//! # Concurrency Model
//! 1. Readers: Call `stable()` to get version, perform read, call `has_changed()`
//! 2. Writers: Call `lock()` to get a [`LockGuard`], modify node, let guard drop.
//!
//! # Type-State Pattern
//! The [`LockGuard`] type provides compile-time verification that the lock is held.
//! Operations that require the lock take `&mut LockGuard` as proof. The guard
//! automatically unlocks on drop (panic-safe).
//!
//! ```rust,ignore
//! let mut guard = version.lock();
//! guard.mark_insert();
//! // Lock released when guard drops
//! ```
//!
//! # Bit Layout
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                         NodeVersion u32 Layout                               │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │    31    30    29    28      27 ──────────── 9      8 ──── 3   2   1   0     │
//!   │   ┌────┬────┬────┬────┬─────────────────────┬────────────┬───┬───┬───┬───┐   │
//!   │   │LEAF│ROOT│DEL │RSVD│   VSPLIT (19 bits)  │VINS(6bits) │SPL│INS│LCK│   │   │
//!   │   │    │    │ETED│    │   split version     │insert ver  │IT │ERT│   │   │   │
//!   │   └────┴────┴────┴────┴─────────────────────┴────────────┴───┴───┴───┴───┘   │
//!   │                                                                              │
//!   │   Bit  Name           Description                                            │
//!   │   ───  ────           ───────────                                            │
//!   │    0   LOCK_BIT       Node is locked for modification                        │
//!   │    1   INSERTING_BIT  Insert operation in progress (dirty)                   │
//!   │    2   SPLITTING_BIT  Split operation in progress (dirty)                    │
//!   │   3-8  VINSERT        Insert version counter (6 bits, wraps at 64)           │
//!   │   9-27 VSPLIT         Split version counter (19 bits, ~512K operations)      │
//!   │   28   RESERVED       Reserved/unused bit                                    │
//!   │   29   DELETED_BIT    Node is logically deleted                              │
//!   │   30   ROOT_BIT       Node is a tree/layer root                              │
//!   │   31   ISLEAF_BIT     Node is a leaf (vs internode)                          │
//!   │                                                                              │
//!   │   DIRTY_MASK = INSERTING_BIT | SPLITTING_BIT = bits 1-2                      │
//!   │   - stable() spins while DIRTY_MASK != 0                                     │
//!   │   - has_changed() uses (v1 ^ current) >= VSPLIT_LOWBIT                       │
//!   │     (ignores bits 0-8: LOCK, INSERTING, SPLITTING, VINSERT)                  │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Optimistic Concurrency Control Protocol
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    READER (Lock-Free, Optimistic)                            │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌──────────────┐
//!   │   stable()   │──── Spin while DIRTY_MASK set, return clean version v1
//!   └──────┬───────┘     (Acquire fence on success)
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │  Read data   │──── Access node fields (ikeys, values, permutation)
//!   │  from node   │     No locks held, concurrent writes may occur
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐     Compiler fence + Relaxed load
//!   │ has_changed  │──── Compare (v1 ^ current) >= VSPLIT_LOWBIT (512)
//!   │    (v1)?     │
//!   └──────┬───────┘
//!          │
//!     ┌────┴────┐
//!     │         │
//!     ▼         ▼
//!  ┌──────┐  ┌──────┐
//!  │ true │  │false │
//!  │      │  │      │
//!  │RETRY │  │ USE  │
//!  │      │  │RESULT│
//!  └──────┘  └──────┘
//!
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    WRITER (Fine-Grained Lock)                                │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌──────────────┐
//!   │    lock()    │──── CAS: set LOCK_BIT | INSERTING_BIT atomically
//!   │              │     (Acquire ordering on success)
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ LockGuard    │──── RAII guard proves lock is held
//!   │  returned    │     Type-state pattern: compile-time verification
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ mark_split() │──── Optional: set SPLITTING_BIT if doing split
//!   │ (optional)   │     Required before structural changes
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ Modify node  │──── Write ikeys, values, permutation, pointers
//!   │   data       │     Protected by exclusive lock
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐     Guard Drop:
//!   │ drop(guard)  │──── - If SPLITTING: version += VSPLIT_LOWBIT, clear all
//!   │   (unlock)   │     - Else: version += VINSERT_LOWBIT, clear dirty
//!   └──────────────┘     (Release ordering)
//! ```
//!
//! # Lock State Machine
//!
//! ```text
//!                              ┌─────────────────────────────────────┐
//!                              │                                     │
//!                              ▼                                     │
//!   ┌──────────────────────────────────────────────────────────────┐ │
//!   │                     UNLOCKED (Clean)                         │ │
//!   │                                                              │ │
//!   │   LOCK=0, INSERTING=0, SPLITTING=0                           │ │
//!   │   - Readers: stable() returns immediately                    │ │
//!   │   - Writers: CAS to acquire lock                             │ │
//!   └────────────────────────┬─────────────────────────────────────┘ │
//!                            │                                       │
//!                            │ lock() CAS: set LOCK|INSERTING        │
//!                            │                                       │
//!                            ▼                                       │
//!   ┌──────────────────────────────────────────────────────────────┐ │
//!   │                     LOCKED (Inserting)                       │ │
//!   │                                                              │ │
//!   │   LOCK=1, INSERTING=1, SPLITTING=0                           │ │
//!   │   - Readers: stable() spins (DIRTY_MASK set)                 │ │
//!   │   - Holder: can mark_split() if needed                       │ │
//!   └──────────────┬───────────────────────────────────────────────┘ │
//!                  │                                                 │
//!           ┌──────┴──────┐                                          │
//!           │             │                                          │
//!           ▼             ▼                                          │
//!   ┌────────────┐  ┌────────────┐                                   │
//!   │ mark_split │  │ drop()     │                                   │
//!   │ called     │  │ (unlock)   │                                   │
//!   └─────┬──────┘  └─────┬──────┘                                   │
//!         │               │                                          │
//!         │               │ VINSERT += 1                             │
//!         │               │ clear LOCK|INSERTING                     │
//!         │               │                                          │
//!         ▼               └──────────────────────────────────────────┘
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │                     LOCKED (Splitting)                       │
//!   │                                                              │
//!   │   LOCK=1, INSERTING=1, SPLITTING=1                           │
//!   │   - Readers: stable() spins                                  │
//!   │   - Holder: structural modifications allowed                 │
//!   └──────────────────────────┬───────────────────────────────────┘
//!                              │
//!                              │ drop() (unlock)
//!                              │ VSPLIT += 1
//!                              │ clear LOCK|INSERTING|SPLITTING|ROOT
//!                              │
//!                              └─────────────────────────────────────┐
//!                                                                    │
//!                              ┌─────────────────────────────────────┘
//!                              │
//!                              ▼
//!                         UNLOCKED (Clean, version incremented)
//! ```
//!
//! # [`LockGuard`] Type-State Pattern
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    LockGuard<'a> TYPE-STATE PATTERN                          │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   The LockGuard uses Rust's type system to provide compile-time proof        │
//!   │   that a lock is held. Operations requiring exclusive access take            │
//!   │   `&mut LockGuard` as a capability token.                                    │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                         STRUCT LAYOUT                                  │ │
//!   │   │                                                                        │ │
//!   │   │   struct LockGuard<'a> {                                               │ │
//!   │   │       version: *const NodeVersion,  // Raw ptr to locked version       │ │
//!   │   │       locked_value: u32,            // Snapshot at lock time           │ │
//!   │   │       _lifetime: PhantomData<&'a NodeVersion>,  // Lifetime bound      │ │
//!   │   │       _marker: PhantomData<*mut ()>,            // !Send + !Sync       │ │
//!   │   │   }                                                                    │ │
//!   │   │                                                                        │ │
//!   │   │   Size: 16 bytes (ptr + u32 + padding)                                 │ │
//!   │   │   Alignment: 8 bytes                                                   │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    COMPILE-TIME GUARANTEES                             │ │
//!   │   │                                                                        │ │
//!   │   │   1. LIFETIME BOUND ('a):                                              │ │
//!   │   │      ├─ Guard cannot outlive the NodeVersion it locks                  │ │
//!   │   │      ├─ Prevents use-after-free of version field                       │ │
//!   │   │      └─ Enforced by: PhantomData<&'a NodeVersion>                      │ │
//!   │   │                                                                        │ │
//!   │   │   2. !Send (cannot transfer between threads):                          │ │
//!   │   │      ├─ Lock acquired on thread T must be released on thread T         │ │
//!   │   │      ├─ Prevents cross-thread lock ownership confusion                 │ │
//!   │   │      └─ Enforced by: PhantomData<*mut ()>                              │ │
//!   │   │                                                                        │ │
//!   │   │   3. !Sync (cannot share references between threads):                  │ │
//!   │   │      ├─ &LockGuard cannot be sent to another thread                    │ │
//!   │   │      ├─ Prevents concurrent mutation through shared ref                │ │
//!   │   │      └─ Enforced by: PhantomData<*mut ()>                              │ │
//!   │   │                                                                        │ │
//!   │   │   4. PANIC-SAFE DROP:                                                  │ │
//!   │   │      ├─ Drop impl ALWAYS releases the lock                             │ │
//!   │   │      ├─ Even if panic occurs while holding lock                        │ │
//!   │   │      └─ Prevents deadlock on unwind                                    │ │
//!   │   │                                                                        │ │
//!   │   │   5. #[must_use] ATTRIBUTE:                                            │ │
//!   │   │      ├─ Compiler warns if guard is not used                            │ │
//!   │   │      ├─ Prevents: let _ = version.lock();  // immediate unlock!        │ │
//!   │   │      └─ Forces explicit binding or explicit drop                       │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    AVAILABLE METHODS                                   │ │
//!   │   │                                                                        │ │
//!   │   │   All methods require &mut self (proof of exclusive access):           │ │
//!   │   │                                                                        │ │
//!   │   │   ┌────────────────┬────────────────────────────────────────────────┐  │ │
//!   │   │   │ Method         │ Effect                                         │  │ │
//!   │   │   ├────────────────┼────────────────────────────────────────────────┤  │ │
//!   │   │   │ mark_insert()  │ No-op (INSERTING_BIT set by lock())            │  │ │
//!   │   │   │                │ Exists for semantic clarity                    │  │ │
//!   │   │   ├────────────────┼────────────────────────────────────────────────┤  │ │
//!   │   │   │ mark_split()   │ Sets SPLITTING_BIT                             │  │ │
//!   │   │   │                │ Required before structural changes             │  │ │
//!   │   │   │                │ Causes VSPLIT bump on unlock                   │  │ │
//!   │   │   ├────────────────┼────────────────────────────────────────────────┤  │ │
//!   │   │   │ mark_deleted() │ Sets DELETED_BIT | SPLITTING_BIT               │  │ │
//!   │   │   │                │ Marks node as logically deleted                │  │ │
//!   │   │   │                │ Readers will see deleted and retry             │  │ │
//!   │   │   ├────────────────┼────────────────────────────────────────────────┤  │ │
//!   │   │   │ mark_nonroot() │ Clears ROOT_BIT                                │  │ │
//!   │   │   │                │ Called when node is no longer layer root       │  │ │
//!   │   │   └────────────────┴────────────────────────────────────────────────┘  │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    USAGE PATTERN                                       │ │
//!   │   │                                                                        │ │
//!   │   │   // Correct usage:                                                    │ │
//!   │   │   let mut guard = node.version().lock();  // Acquire                   │ │
//!   │   │   guard.mark_split();                     // Mark operation type       │ │
//!   │   │   // ... modify node fields ...           // Protected region          │ │
//!   │   │   drop(guard);                            // Release (or scope end)    │ │
//!   │   │                                                                        │ │
//!   │   │   // Compile error examples:                                           │ │
//!   │   │   std::thread::spawn(|| guard.mark_split());  // ERROR: !Send          │ │
//!   │   │   let r = &guard; another_fn(r);              // ERROR: !Sync          │ │
//!   │   │   let _ = node.version().lock();              // Warning: #[must_use]  │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    DROP IMPLEMENTATION                                 │ │
//!   │   │                                                                        │ │
//!   │   │   impl Drop for LockGuard<'_> {                                        │ │
//!   │   │       fn drop(&mut self) {                                             │ │
//!   │   │           // Safety: We hold the lock, so we can modify version        │ │
//!   │   │           let version = unsafe { &*self.version };                     │ │
//!   │   │                                                                        │ │
//!   │   │           if self.locked_value & SPLITTING_BIT != 0 {                  │ │
//!   │   │               // Split occurred: bump VSPLIT, clear all dirty bits     │ │
//!   │   │               let new = (self.locked_value + VSPLIT_LOWBIT)            │ │
//!   │   │                         & !(LOCK_BIT | INSERTING_BIT |                 │ │
//!   │   │                             SPLITTING_BIT | ROOT_BIT);                 │ │
//!   │   │               version.0.store(new, Ordering::Release);                 │ │
//!   │   │           } else {                                                     │ │
//!   │   │               // Normal insert: bump VINSERT, clear LOCK|INSERTING     │ │
//!   │   │               let new = (self.locked_value + VINSERT_LOWBIT)           │ │
//!   │   │                         & !(LOCK_BIT | INSERTING_BIT);                 │ │
//!   │   │               version.0.store(new, Ordering::Release);                 │ │
//!   │   │           }                                                            │ │
//!   │   │       }                                                                │ │
//!   │   │   }                                                                    │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Version Comparison Semantics
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    has_changed() vs has_split()                              │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   has_changed(old):                                                          │
//!   │   ─────────────────                                                          │
//!   │   Returns: (old ^ current) > (LOCK_BIT | INSERTING_BIT)                      │
//!   │            = (old ^ current) > 3                                             │
//!   │                                                                              │
//!   │   Detects: Any change to VINSERT or VSPLIT or DELETED or other bits          │
//!   │   Ignores: LOCK_BIT and INSERTING_BIT (bits 0-1)                             │
//!   │                                                                              │
//!   │   Use: Point reads - need to know if ANY modification occurred               │
//!   │                                                                              │
//!   │   ─────────────────────────────────────────────────────────────────────────  │
//!   │                                                                              │
//!   │   has_split(old):                                                            │
//!   │   ────────────────                                                           │
//!   │   Returns: (old ^ current) >= VSPLIT_LOWBIT                                  │
//!   │            = (old ^ current) >= 512                                          │
//!   │                                                                              │
//!   │   Detects: Only structural changes (splits, deletes)                         │
//!   │   Ignores: VINSERT changes, LOCK_BIT, INSERTING_BIT                          │
//!   │                                                                              │
//!   │   Use: Scans - only care if tree structure changed (need re-navigation)      │
//!   │                                                                              │
//!   │   ─────────────────────────────────────────────────────────────────────────  │
//!   │                                                                              │
//!   │   EXAMPLE:                                                                   │
//!   │                                                                              │
//!   │   old = 0x8000_0200  (ISLEAF | VSPLIT=1)                                     │
//!   │   new = 0x8000_0208  (ISLEAF | VSPLIT=1 | VINSERT=1)                         │
//!   │                                                                              │
//!   │   has_changed: (0x8000_0200 ^ 0x8000_0208) = 0x08 > 3 → TRUE (insert)        │
//!   │   has_split:   (0x8000_0200 ^ 0x8000_0208) = 0x08 >= 512 → FALSE (no split)  │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Always-Dirty-On-Lock Strategy
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    Why INSERTING_BIT is Set by lock()                        │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   PROBLEM: Race between CAS insert and locked writers                        │
//!   │                                                                              │
//!   │     Thread A (CAS insert)          Thread B (locked writer)                  │
//!   │     ────────────────────           ─────────────────────────                 │
//!   │     v1 = stable()                                                            │
//!   │                                    lock() ← sets LOCK_BIT only               │
//!   │     ... prepare CAS ...                                                      │
//!   │                                    mark_insert() ← sets INSERTING_BIT        │
//!   │     has_changed(v1)? NO!           ... modifying node ...                    │
//!   │     CAS proceeds ← RACE!                                                     │
//!   │                                                                              │
//!   │   ─────────────────────────────────────────────────────────────────────────  │
//!   │                                                                              │
//!   │   SOLUTION: lock() atomically sets LOCK_BIT | INSERTING_BIT                  │
//!   │                                                                              │
//!   │     Thread A (CAS insert)          Thread B (locked writer)                  │
//!   │     ────────────────────           ─────────────────────────                 │
//!   │     v1 = stable()                                                            │
//!   │                                    lock() ← sets LOCK|INSERTING              │
//!   │     ... prepare CAS ...                                                      │
//!   │     has_changed(v1)? YES!          ... modifying node ...                    │
//!   │     CAS aborts, retry ← SAFE!                                                │
//!   │                                                                              │
//!   │   The INSERTING_BIT in v1 ensures CAS sees the "dirty" state.                │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::hint as StdHint;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic as StdAtomic;
use std::sync::atomic::{AtomicU32, Ordering, fence};
use std::thread as StdThread;
use std::time::{Duration, Instant};

// ============================================================================
//  Bit Constants (matching C++ nodeversion_parameters<uint32_t>)
// ============================================================================

/// Lock bit: node is locked for modification.
const LOCK_BIT: u32 = 1 << 0;

/// Inserting bit: node is being inserted into.
const INSERTING_BIT: u32 = 1 << 1;

/// Splitting bit: node is being split.
const SPLITTING_BIT: u32 = 1 << 2;

/// Dirty mask: either inserting or splitting.
const DIRTY_MASK: u32 = INSERTING_BIT | SPLITTING_BIT;

/// Low bit of insert version counter.
const VINSERT_LOWBIT: u32 = 1 << 3;

/// Low bit of split version counter.
const VSPLIT_LOWBIT: u32 = 1 << 9;

/// Unused bit (reserved).
const UNUSED1_BIT: u32 = 1 << 28;

/// Deleted bit: node is logically deleted.
const DELETED_BIT: u32 = 1 << 29;

/// Root bit: node is a tree root.
const ROOT_BIT: u32 = 1 << 30;

/// Is-leaf bit: node is a leaf (vs internode).
const ISLEAF_BIT: u32 = 1 << 31;

/// Mask for unlock after split: clears root, unused, and version bits below vsplit.
const SPLIT_UNLOCK_MASK: u32 = !(ROOT_BIT | UNUSED1_BIT | (VSPLIT_LOWBIT - 1));

/// Mask for unlock after insert: clears unused and version bits below vinsert.
const UNLOCK_MASK: u32 = !(UNUSED1_BIT | (VINSERT_LOWBIT - 1));

// ============================================================================
//  Backoff (for spin loops)
// ============================================================================

/// Exponential backoff for spin loops.
///
/// Matches C++ `backoff_fence_function` from `reference/compiler.hh:133-143`.
/// Each call to `spin()` executes `count+1` pause instructions, then doubles
/// the count (capped at 15).
///
/// Sequence: 0 → 1 → 3 → 7 → 15 (capped)
struct Backoff {
    count: u32,
}

impl Backoff {
    /// Create a new backoff with count = 0.
    #[inline(always)]
    const fn new() -> Self {
        Self { count: 0 }
    }

    /// Spin for `count+1` iterations using CPU pause hints, then increase count.
    ///
    /// Uses [`StdHint::spin_loop()`] which maps to the x86 `PAUSE` instruction,
    /// improving performance on hyper-threaded CPUs by hinting that we're in
    /// a spin-wait loop.
    fn spin(&mut self) {
        for _ in 0..=self.count {
            StdHint::spin_loop();
        }

        // Double count, cap at 15: 0 -> 1 -> 3 -> 7 -> 15 -> 15
        self.count = ((self.count << 1) | 1) & 15;
    }
}

// ============================================================================
//  NodeVersion
// ============================================================================

/// A versioned lock for tree nodes.
///
/// Combines lock state, dirty flags, version counters, and node metadata.
///
/// # Layout
/// Bit 31: `is_leaf` | Bit 30: `root` | Bit 29: `deleted` | BITS 9-27: `split_version`
/// Bits 3-8: `insert_version` | Bit 2: `splitting` | Bit 1: `inserting` | Bit 0: `locked`
///
/// # Example
///
/// ```rust
/// use masstree::nodeversion::NodeVersion;
///
/// // Create a leaf node version
/// let v = NodeVersion::new(true);
///
/// assert!(v.is_leaf());
/// assert!(!v.is_locked());
/// ```
#[derive(Debug)]
pub struct NodeVersion {
    value: AtomicU32,
}

// ============================================================================
//  LockGuard (Type-State Pattern)
// ============================================================================

/// Zero-sized proof that a lock is held.
///
/// Cannot be constructed except by calling [`NodeVersion::lock()`].
/// Operations that require the lock take `&mut LockGuard` as proof.
/// The lock is automatically released when the guard drops.
///
/// # Panic Safety
/// The guard releases the lock on drop, even during unwinding. This ensures
/// the lock is never held after a panic.
///
/// # Thread Safety
/// Guards are `!Send` and `!Sync` via `PhantomData<*mut ()>` to prevent them
/// from crossing thread boundaries.
///
/// We use `PhantomData`<*mut ()> which makes the type !Send + !Sync because
/// raw pointers are neither Send nor Sync. This is the standard stable Rust
/// pattern for preventing types from being transferred across threads, at least
/// until 1.92.0.
///
/// NOTE: This is sufficient for our use case. The guard holds a reference to
/// `NodeVersion` (via a lifetime marker) which already prevents the guard from
/// outliving the version.
#[derive(Debug)]
#[must_use = "releasing a lock without using the guard is a logic error"]
pub struct LockGuard<'a> {
    version: *const NodeVersion,
    locked_value: u32,

    _lifetime: PhantomData<&'a NodeVersion>,
    // PhantomData<*mut ()> makes this type !Send + !Sync (these are still nightly features)
    _marker: PhantomData<*mut ()>,
}

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        // Version counter increment depends on dirty bits:
        // - If splitting: increment split counter, clear all dirty/lock bits
        // - If inserting: increment insert counter, clear inserting/lock bits
        //
        // With current strategy, INSERTING_BIT is always set (unless SPLITTING_BIT was set),
        // so the version counter is always incremented on unlock.
        let new_value: u32 = if self.locked_value & SPLITTING_BIT != 0 {
            (self.locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            // The expression `(inserting << 2)` equals `vinsert_lowbit` when inserting
            // Currently, INSERTING_BIT is always 1 here, so version increments
            (self.locked_value + ((self.locked_value & INSERTING_BIT) << 2)) & UNLOCK_MASK
        };

        // SAFETY: The guard's lifetime is tied to the `NodeVersion` it was created from.
        // Nodes are only freed via deferred reclamation; holding the lock implies the node
        // remains valid until this guard is dropped.
        unsafe { (*self.version).value.store(new_value, Ordering::Release) };
    }
}

impl LockGuard<'_> {
    #[inline(always)]
    const fn version(&self) -> &NodeVersion {
        // SAFETY: `self.version` is valid for the guard's lifetime because:
        // - LockGuard<'a> holds PhantomData<&'a NodeVersion>, enforcing lifetime
        // - The pointer was created from a valid reference in lock()/try_lock()
        // - Nodes are freed via deferred reclamation, never while locked
        unsafe { &*self.version }
    }

    /// Get the locked version value.
    #[must_use]
    #[inline(always)]
    pub const fn locked_value(&self) -> u32 {
        self.locked_value
    }

    /// Mark the node as being inserted into.
    ///
    /// Sets the inserting dirty bit. Version counter will increment on unlock.
    /// This must be called before modifying node contents during insert.
    ///
    /// # C++ Reference
    /// Matches `nodeversion.hh:143-147` - `mark_insert()` method.
    ///
    /// # Memory Ordering
    /// Uses [`Ordering::Release`] followed by [`Ordering::Acquire`] fence.
    /// This ensures readers calling `stable()` will wait for our modifications.
    ///
    /// # Idempotent
    /// Multiple calls have no additional effect.
    #[inline]
    pub fn mark_insert(&mut self) {
        // Skip if already set (idempotent)
        if (self.locked_value & INSERTING_BIT) != 0 {
            return;
        }

        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version().value.load(Ordering::Relaxed);

        self.version()
            .value
            .store(value | INSERTING_BIT, Ordering::Release);

        // Acquire fence ensures subsequent modifications cannot be reordered
        // before the dirty bit becomes visible to readers.
        fence(Ordering::Acquire);

        // Update tracked value for unlock logic
        self.locked_value |= INSERTING_BIT;
    }

    /// Mark the node as being split.
    ///
    /// Sets the splitting dirty bit. Version counter will increment on unlock.
    ///
    /// NOTE: Must be called explicitly
    ///
    /// Unlike `mark_insert()` (which is now auto set due to new strategy), `mark_split()`
    /// must be called explicitly before split operations. This is because:
    /// 1. Not all inserts require splits
    /// 2. The `SPLITTING_BIT` affects version increment logic differently
    /// 3. Split operations need the split version counter incremented
    ///
    /// # Memory Ordering
    /// Uses [`Ordering::Release`] followed by [`Ordering::Acquire`] fence.
    #[inline]
    pub fn mark_split(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version().value.load(Ordering::Relaxed);

        self.version()
            .value
            .store(value | SPLITTING_BIT, Ordering::Release);

        // Acquire fence ensures subsequent structural modifications
        // cannot be reordered before the dirty bit becomes visible.
        fence(Ordering::Acquire);

        self.locked_value |= SPLITTING_BIT;
    }

    /// Mark the node as deleted.
    ///
    /// Also sets the splitting bit to bump version on unlock.
    ///
    /// # Memory Ordering
    /// Same as [`Self::mark_insert`]: Release store followed by Acquire fence.
    #[inline(always)]
    pub fn mark_deleted(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version().value.load(Ordering::Relaxed);
        let new_value: u32 = value | DELETED_BIT | SPLITTING_BIT;

        self.version().value.store(new_value, Ordering::Release);

        // Acquire fence ensures subsequent structural modifications
        // cannot be reordered before the dirty bit becomes visible.
        fence(Ordering::Acquire);

        self.locked_value = new_value;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub fn mark_nonroot(&mut self) {
        // INVARIANT: lock is held, so no concurrent modifications possible.
        let value: u32 = self.version().value.load(Ordering::Relaxed);

        self.version()
            .value
            .store(value & !ROOT_BIT, Ordering::Release);
        self.locked_value &= !ROOT_BIT;
    }
}

impl NodeVersion {
    /// Create a new node version.
    ///
    /// # Arguments
    /// - `is_leaf` - true for leaf nodes, false for internodes
    #[must_use]
    #[inline(always)]
    pub const fn new(is_leaf: bool) -> Self {
        let initial: u32 = if is_leaf { ISLEAF_BIT } else { 0 };

        Self {
            value: AtomicU32::new(initial),
        }
    }

    /// Create a node version from a raw value.
    ///
    ///  WARN: ONLY FOR TESTING.
    #[must_use]
    #[inline(always)]
    pub const fn from_value(value: u32) -> Self {
        Self {
            value: AtomicU32::new(value),
        }
    }

    // ========================================================================
    //  Flag Accessors
    // ========================================================================

    /// Check if this is a leaf node.
    #[must_use]
    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ISLEAF_BIT) != 0
    }

    /// Check if this is a root node.
    #[must_use]
    #[inline(always)]
    pub fn is_root(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & ROOT_BIT) != 0
    }

    /// Check if this node is logically deleted.
    #[must_use]
    #[inline(always)]
    pub fn is_deleted(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DELETED_BIT) != 0
    }

    /// Check if a version value indicates the node is deleted.
    ///
    /// This is a static check on an already-loaded version value, avoiding
    /// an additional atomic load. Use after calling [`NodeVersion::stable`] when you
    /// need to check both version stability and deleted status.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let version = node.version().stable();
    /// if NodeVersion::is_deleted_version(version) {
    ///     // Node was deleted, need to retry
    /// }
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn is_deleted_version(version: u32) -> bool {
        (version & DELETED_BIT) != 0
    }

    /// Check if this node is locked.
    #[must_use]
    #[inline(always)]
    pub fn is_locked(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & LOCK_BIT) != 0
    }

    /// Check if this node is in initial/unpublished state.
    ///
    /// A node is "unpublished" if it has never been modified (version counters
    /// are zero). This means it was just allocated and hasn't been linked into
    /// the tree yet, so no other thread can see it.
    ///
    /// Used to allow lock-free initialization of newly allocated nodes.
    #[must_use]
    #[inline(always)]
    pub fn is_unpublished(&self) -> bool {
        let v = self.value.load(Ordering::Relaxed);
        // A newly created leaf has value = ISLEAF_BIT (possibly | ROOT_BIT)
        // A newly created internode has value = 0 (possibly | ROOT_BIT)
        // Mask out the allowed initial flags and check if result is zero
        let mask = ISLEAF_BIT | ROOT_BIT;
        (v & !mask) == 0
    }

    /// Check if this node is being inserted into.
    #[must_use]
    #[inline(always)]
    pub fn is_inserting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & INSERTING_BIT) != 0
    }

    /// Check if this node is being split.
    #[must_use]
    #[inline(always)]
    pub fn is_splitting(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & SPLITTING_BIT) != 0
    }

    /// Check if any dirty bit set (inserting or splitting).
    #[must_use]
    #[inline(always)]
    pub fn is_dirty(&self) -> bool {
        (self.value.load(Ordering::Relaxed) & DIRTY_MASK) != 0
    }

    /// Get the raw version value.
    #[must_use]
    #[inline(always)]
    pub fn value(&self) -> u32 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get a raw pointer to this `NodeVersion`.
    ///
    /// Used for APIs that need to lock nodes via raw pointers
    /// (e.g., `PropagationContext::lock_node`).
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const Self {
        StdPtr::from_ref(self)
    }

    // ========================================================================
    // Stable Version (for optimistic reads)
    // ========================================================================

    /// Get a stable version value for optimistic reading.
    ///
    /// Spins while dirty bits (inserting or splitting) are set, then returns
    /// a version with no dirty bits. Use with [`Self::has_changed`] after reading
    /// to detect concurrent modifications.
    ///
    /// # Memory Ordering
    /// Uses `Relaxed` loads during spinning for efficiency (especially on ARM),
    /// then issues an `Acquire` fence only on success. This is equivalent to
    /// `Acquire` on every load on x86, but saves ~1 cycle per spin on ARM.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:36-48` - `stable()` template method
    ///
    /// # Returns
    /// A version value with no dirty bits set.
    #[must_use]
    pub fn stable(&self) -> u32 {
        let mut backoff = Backoff::new();

        loop {
            // Relaxed load like C++ - only need compiler barrier at the end.
            // C++ nodeversion.hh:40-47 uses plain reads with acquire_fence() after loop.
            // On x86, Relaxed compiles to same `mov` as Acquire.
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & DIRTY_MASK) == 0 {
                // Acquire fence after successful read - matches C++ acquire_fence()
                // This establishes synchronizes-with relationship with writer's Release.
                StdAtomic::fence(Ordering::Acquire);

                return value;
            }

            // Exponential backoff reduces cache line contention under heavy load.
            // While C++ uses single PAUSE by default, our testing shows exponential
            // backoff performs better on modern multi-core systems.
            backoff.spin();
        }
    }

    /// Acquire a version value without spinning on dirty bits.
    ///
    /// Unlike [`stable()`](Self::stable), this method returns immediately with
    /// whatever value is currently stored, even if dirty bits are set.
    ///
    /// # Use Cases
    ///
    /// Use this when you want to detect concurrent modification and retry at
    /// a higher level rather than spinning locally. For example:
    ///
    /// ```rust,ignore
    /// let v = version.acquire_raw();
    /// if NodeVersion::is_dirty_value(v) {
    ///     continue 'retry;  // Let outer loop handle retry
    /// }
    /// // ... proceed with read ...
    /// ```
    ///
    /// # Memory Ordering
    ///
    /// Uses `Ordering::Acquire` to synchronize with writer's Release store.
    /// The returned value may have dirty bits set.
    #[must_use]
    #[inline(always)]
    pub fn acquire_raw(&self) -> u32 {
        self.value.load(Ordering::Acquire)
    }

    /// Check if a version value has dirty bits set.
    ///
    /// This is a static helper for checking values returned by [`acquire_raw()`](Self::acquire_raw).
    #[must_use]
    #[inline(always)]
    pub const fn is_dirty_value(v: u32) -> bool {
        (v & DIRTY_MASK) != 0
    }

    /// Try to get a stable version without spinning.
    ///
    /// Returns `Some(version)` if the node is not dirty (not being modified).
    /// Returns `None` if the node is currently being modified (dirty bits set).
    ///
    /// # Use Cases
    ///
    /// Use this for opportunistic reads where you want to skip nodes that are
    /// currently being modified rather than waiting for them:
    ///
    /// ```rust,ignore
    /// loop {
    ///     match version.try_stable() {
    ///         Some(v) => {
    ///             // Node is clean, proceed with read
    ///             // ... read data ...
    ///             if !version.has_changed(v) {
    ///                 return result;
    ///             }
    ///         }
    ///         None => {
    ///             // Node is dirty, retry from higher level
    ///             continue 'retry;
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Memory Ordering
    ///
    /// Uses `Ordering::Acquire` on success to synchronize with writer's Release.
    #[must_use]
    #[inline(always)]
    pub fn try_stable(&self) -> Option<u32> {
        let value: u32 = self.value.load(Ordering::Acquire);

        if (value & DIRTY_MASK) == 0 {
            Some(value)
        } else {
            None
        }
    }

    /// Get a stable version, with a hint to yield on contention.
    ///
    /// This is a variant of [`stable()`](Self::stable) that yields the CPU
    /// after a few spin iterations rather than using exponential backoff.
    /// This can be more efficient when there's high contention on a node.
    ///
    /// # Algorithm
    ///
    /// 1. Try to read a clean version
    /// 2. If dirty, spin briefly with `spin_loop` hints
    /// 3. After a few spins, yield the CPU to other threads
    /// 4. Repeat until clean version obtained
    ///
    /// # Memory Ordering
    ///
    /// Same as [`stable()`](Self::stable): Acquire fence on success.
    #[must_use]
    pub fn stable_yield(&self) -> u32 {
        const SPINS_BEFORE_YIELD: u32 = 4;
        let mut spin_count: u32 = 0;

        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            if (value & DIRTY_MASK) == 0 {
                StdAtomic::fence(Ordering::Acquire);
                return value;
            }

            spin_count += 1;

            if spin_count < SPINS_BEFORE_YIELD {
                for _ in 0..spin_count {
                    StdHint::spin_loop();
                }
            } else {
                StdThread::yield_now();
                spin_count = 0;
            }
        }
    }

    /// Check if the version has changed since `old`.
    ///
    /// Returns true if any version counter bits changed (ignoring lock/inserting bits).
    ///
    /// # C++ Divergence (INTENTIONAL)
    ///
    /// The C++ reference uses `(x.v_ ^ v_) > lock_bit` (ignores only bit 0).
    /// Our Rust implementation uses `> (LOCK_BIT | INSERTING_BIT)` (ignores bits 0-1).
    ///
    /// ## Why This Is Safe
    ///
    /// This divergence is safe due to the "always-dirty-on-lock" strategy:
    ///
    /// 1. **Version counters are the source of truth**: `VINSERT` (bits 3-8) and
    ///    `VSPLIT` (bits 9-27) are the actual change indicators. They are incremented
    ///    atomically when the lock is released, AFTER all modifications are complete.
    ///
    /// 2. **`INSERTING_BIT` is a progress indicator, not a change indicator**:
    ///    - Set atomically with `LOCK_BIT` by `lock()`
    ///    - Cleared when lock releases (version counter increments)
    ///    - If `INSERTING_BIT` is set, modification is in-progress but NOT YET VISIBLE
    ///
    /// 3. **Reader's snapshot is consistent**:
    ///    - Reader got `old` from `stable()` which spins until `DIRTY_MASK == 0`
    ///    - If writer acquires lock AFTER `stable()` but BEFORE `has_changed()`:
    ///      - Writer hasn't modified data yet (just acquired lock)
    ///      - Reader's prior reads are valid (taken before modification)
    ///      - Returning `false` is correct — no actual change to validate
    ///    - If writer releases lock BEFORE `has_changed()`:
    ///      - `VINSERT` or `VSPLIT` incremented → XOR detects it → returns `true`
    ///
    /// 4. **CAS operations use `has_changed_or_locked()` instead**: Operations that
    ///    race with writers (not just read) use the stricter check that detects
    ///    `DIRTY_MASK` being set.
    ///
    /// # Compiler Fence Requirement
    ///
    /// This method includes a **compiler fence** before the version load.
    /// This is critical for correctness: the optimistic read protocol is
    /// "read fields → validate version". Without the fence, the compiler
    /// could reorder field reads to occur AFTER the version check, defeating
    /// the validation.
    ///
    /// ## C++ Reference
    ///
    /// The C++ `nodeversion.hh:72-74` uses `fence()` (compiler barrier):
    /// ```cpp
    /// bool has_changed(nodeversion x) const {
    ///     fence();  // compiler barrier from compiler.hh:77
    ///     return (x.v_ ^ v_) > lock_bit;
    /// }
    /// ```
    ///
    /// ## Why Acquire Alone Is Insufficient
    ///
    /// `Ordering::Acquire` on the load only prevents reordering of operations
    /// AFTER the load. It does NOT prevent the compiler from moving reads
    /// that occurred BEFORE the `has_changed()` call to occur after it.
    ///
    /// The compiler fence ensures all prior reads (slot data) complete before
    /// we load the version for validation.
    #[must_use]
    #[inline(always)]
    pub fn has_changed(&self, old: u32) -> bool {
        // Compiler fence: ensures all prior reads complete before version check.
        // This matches C++ fence() in nodeversion.hh:72.
        StdAtomic::compiler_fence(Ordering::Acquire);

        // Relaxed load like C++ - the compiler fence above prevents reordering.
        //
        // SAFETY (C++ Divergence): We use `> (LOCK_BIT | INSERTING_BIT)` instead of
        // C++'s `> lock_bit`. This ignores both bits 0 and 1. See doc comment above
        // for the full safety argument. TL;DR: version COUNTERS (VINSERT/VSPLIT) are
        // the source of truth; INSERTING_BIT is a progress indicator that's only set
        // while modifications are in-flight (not yet visible to readers).
        (old ^ self.value.load(Ordering::Relaxed)) > (LOCK_BIT | INSERTING_BIT)
    }

    /// Check if a split has occurred since `old`.
    ///
    /// Returns true if the split version counter changed.
    ///
    /// Uses the same compiler fence as `has_changed()` for correctness.
    /// See [`Self::has_changed`] for the full explanation.
    #[must_use]
    #[inline(always)]
    pub fn has_split(&self, old: u32) -> bool {
        // Compiler fence: ensures all prior reads complete before version check.
        // This matches C++ fence() in nodeversion.hh:80.
        StdAtomic::compiler_fence(Ordering::Acquire);

        // Relaxed load like C++ - the compiler fence above prevents reordering.
        (old ^ self.value.load(Ordering::Relaxed)) >= VSPLIT_LOWBIT
    }

    /// Check if a split has occurred since `old`, without a compiler fence.
    ///
    /// This is a faster variant of [`Self::has_split`] that omits the compiler fence.
    /// Use this only when you've already issued a fence (e.g., after an Acquire load)
    /// that ensures all prior reads are complete.
    ///
    /// # Ordering Note
    ///
    /// This method still uses `Ordering::Acquire` on the load, which provides
    /// hardware memory ordering on ARM. The difference from [`Self::has_split`] is
    /// the absence of the **compiler fence** that prevents the compiler from
    /// reordering prior reads to after this check.
    ///
    /// # Safety (Logical)
    ///
    /// The caller must ensure that all reads that need to be validated have
    /// already been completed and are visible before calling this method.
    /// Typically this means you've already done an Acquire load or fence.
    ///
    /// # Reference
    ///
    /// C++ `nodeversion.hh` has `simple_has_split()` for this purpose.
    #[must_use]
    #[inline(always)]
    pub fn has_split_no_compiler_fence(&self, old: u32) -> bool {
        (old ^ self.value.load(Ordering::Acquire)) >= VSPLIT_LOWBIT
    }

    /// Check if the version has changed OR if a modification is in progress.
    ///
    /// This is a stronger check than [`Self::has_changed`] for CAS operations.
    /// It returns true if:
    /// - The version counter has changed (same as `has_changed`), OR
    /// - The node is currently being modified (`INSERTING_BIT` or `SPLITTING_BIT` set)
    ///
    /// CAS inserts should use this instead of `has_changed` to avoid racing
    /// with locked splits. The race scenario:
    /// 1. CAS insert reads version V via `stable()` (no dirty bits)
    /// 2. Locked thread acquires lock, sets `INSERTING_BIT`
    /// 3. CAS insert checks `has_changed(V)` - returns false (ignores `INSERTING_BIT`)
    /// 4. CAS insert proceeds, racing with the split
    ///
    /// By checking `INSERTING_BIT` directly, we catch this race.
    #[must_use]
    #[inline(always)]
    pub fn has_changed_or_locked(&self, old: u32) -> bool {
        // OPTIMIZATION: Acquire fence is sufficient for preventing read reordering.
        StdAtomic::compiler_fence(Ordering::Acquire);

        let current: u32 = self.value.load(Ordering::Acquire);

        // Check if version changed (ignoring lock/dirty bits)
        if (old ^ current) > (LOCK_BIT | INSERTING_BIT) {
            return true;
        }

        // Check if modification in progress (INSERTING_BIT or SPLITTING_BIT set)
        // This catches the race where we got a stable version but then a lock was acquired
        if (current & DIRTY_MASK) != 0 {
            return true;
        }

        false
    }

    // ========================================================================
    // Lock Operations (Type-State Pattern)
    // ========================================================================

    /// Acquire the lock and return a guard.
    ///
    /// Strategy: Always-Dirty-On-Lock
    /// This implementation automatically sets the `INSERTING_BIT` when acquiring the lock.
    /// This eliminates the race window between lock acquisition and explicit dirty marking,
    /// ensuring that CAS insert threads always wait for locked writers to complete.
    ///
    /// 1. `stable()` spins until `DIRTY_MASK == 0` (includes `INSERTING_BIT`)
    /// 2. `lock()` atomically sets `LOCK_BIT | INSERTING_BIT`
    /// 3. Therefore, any thread calling `stable()` will wait for the lock holder
    /// 4. This eliminates the window where a locked writer hasn't called `mark_insert()` yet
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful CAS to synchronize with the
    /// `Release` store in [`Drop::drop`] of the previous lock holder.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:87-109` - `lock()` template method
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock(&self) -> LockGuard<'_> {
        let mut backoff = Backoff::new();

        loop {
            let value: u32 = self.value.load(Ordering::Relaxed);

            // OPTIMISTIC LOCKING: Only wait for LOCK_BIT to clear.
            // We DON'T wait for dirty bits (INSERTING_BIT, SPLITTING_BIT).
            // After acquiring, caller must validate version hasn't changed.
            // This matches C++ nodeversion.hh:96 which only checks lock_bit.
            if (value & LOCK_BIT) == 0 {
                // Only set LOCK_BIT here. INSERTING_BIT is set later via mark_insert().
                // This matches C++ nodeversion.hh:97-98 which only sets lock_bit in lock().
                //
                // Critical for performance: stable() waits for dirty_mask (inserting|splitting).
                // If we set INSERTING_BIT here, other threads calling stable() would spin
                // for the entire lock duration, causing convoy effects under contention.
                // By deferring INSERTING_BIT to mark_insert(), threads can race for lock()
                // without waiting in stable().
                let locked: u32 = value | LOCK_BIT;

                // CAS to acquire lock.
                // Acquire on success ensures we see all prior writes from previous holder.
                // Relaxed on failure is fine, we'll retry.
                if self
                    .value
                    .compare_exchange_weak(value, locked, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return LockGuard {
                        version: StdPtr::from_ref(self),
                        // locked_value tracks current state; mark_insert() will update it
                        locked_value: locked,
                        _lifetime: PhantomData,
                        _marker: PhantomData,
                    };
                }
            }

            // Exponential backoff reduces cache line contention under heavy load.
            backoff.spin();
        }
    }

    /// Try to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was acquired, `None` if the lock
    /// is held or dirty bits are set.
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful CAS.
    ///
    /// # Reference
    /// C++ `nodeversion.hh:111-127` - `try_lock()` template method
    #[must_use]
    pub fn try_lock(&self) -> Option<LockGuard<'_>> {
        let value: u32 = self.value.load(Ordering::Relaxed);

        // OPTIMISTIC: Fail fast only if locked (not if dirty).
        // Matches lock() behavior - caller must validate after acquiring.
        if (value & LOCK_BIT) != 0 {
            return None;
        }

        // Only set LOCK_BIT here. INSERTING_BIT is set later via mark_insert().
        // Same rationale as lock() - avoids convoy effects in stable().
        let locked: u32 = value | LOCK_BIT;

        // Single CAS attempt (use strong CAS for single-shot).
        match self
            .value
            .compare_exchange(value, locked, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(LockGuard {
                version: StdPtr::from_ref(self),
                locked_value: locked,
                _lifetime: PhantomData,
                _marker: PhantomData,
            }),

            Err(_) => None,
        }
    }

    /// Try to acquire the lock with a timeout.
    ///
    /// Returns `Some(guard)` if the lock was acquired within `timeout`,
    /// `None` if the timeout expired.
    ///
    /// # Use Cases
    /// - Deadlock detection in tests
    /// - Bounded wait times in production
    ///
    /// # Example
    /// ```rust,ignore
    /// use std::time::Duration;
    /// use masstree::nodeversion::NodeVersion;
    ///
    /// let version = NodeVersion::new(true);
    /// if let Some(guard) = version.try_lock_for(Duration::from_millis(100)) {
    ///     // Lock acquired within 100ms
    /// } else {
    ///     // Timeout expired, lock not acquired
    /// }
    /// ```
    #[must_use]
    pub fn try_lock_for(&self, timeout: Duration) -> Option<LockGuard<'_>> {
        let deadline = Instant::now() + timeout;
        let mut backoff = Backoff::new();

        loop {
            // Try to acquire.
            if let Some(guard) = self.try_lock() {
                return Some(guard);
            }

            // Check timeout.
            if Instant::now() >= deadline {
                return None;
            }

            backoff.spin();
        }
    }

    /// Acquire the lock using try-lock with yield.
    ///
    /// Unlike [`Self::lock`] which spins with exponential backoff, this method
    /// yields the CPU to other threads when the lock is contended. This is
    /// more efficient for lock convoy situations where multiple threads are
    /// waiting on the same lock.
    ///
    /// # Algorithm
    ///
    /// 1. Try to acquire the lock with `try_lock()`
    /// 2. If failed, do a small number of spin-loop hints
    /// 3. Then yield the CPU with `thread::yield_now()`
    /// 4. Repeat until lock acquired
    ///
    /// # Memory Ordering
    /// Uses `Acquire` ordering on successful lock acquisition.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock_with_yield(&self) -> LockGuard<'_> {
        const SPINS_BEFORE_YIELD: u32 = 4;

        let mut spin_count: u32 = 0;

        loop {
            // Try to acquire the lock
            if let Some(guard) = self.try_lock() {
                return guard;
            }

            spin_count += 1;

            if spin_count < SPINS_BEFORE_YIELD {
                // Brief spin before yielding
                for _ in 0..spin_count {
                    StdHint::spin_loop();
                }
            } else {
                // Yield CPU to other threads - reduces lock convoy
                StdThread::yield_now();
                spin_count = 0; // Reset for next cycle
            }
        }
    }

    // ========================================================================
    // Non-Locking Operations
    // ========================================================================

    /// Mark the node as a root.
    ///
    /// Does not require the lock. Used during tree initialization.
    ///
    /// # Implementation Note
    /// Uses `fetch_or` for atomic read-modify-write. The previous implementation
    /// used separate load/store which could lose concurrent modifications.
    #[inline(always)]
    pub fn mark_root(&self) {
        self.value.fetch_or(ROOT_BIT, Ordering::Release);
    }

    /// Clear the root bit.
    ///
    /// Called when a layer root leaf is demoted (layer root split).
    ///
    /// # Implementation Note
    /// Uses `fetch_and` for atomic read-modify-write. The previous implementation
    /// used separate load/store which could lose concurrent modifications.
    #[inline(always)]
    pub fn mark_nonroot(&self) {
        self.value.fetch_and(!ROOT_BIT, Ordering::Release);
    }

    // ========================================================================
    // Split-Locked Node Creation (Help-Along Protocol)
    // ========================================================================

    /// Create a new node version for a split sibling.
    ///
    /// The new version is:
    /// - Locked (`LOCK_BIT` set)
    /// - Marked as splitting (`SPLITTING_BIT` set)
    /// - Has the same `ISLEAF_BIT` as the source
    /// - Has zeroed version counters (fresh node)
    ///
    /// This is used during splits to create a right sibling that starts locked.
    /// The sibling remains locked until its parent pointer is set, preventing
    /// other threads from trying to split it while parent is NULL.
    ///
    /// # C++ Reference
    ///
    /// Matches `child->assign_version(*n_)` in `masstree_split.hh:198`.
    /// However, we use `SPLITTING_BIT` instead of copying `INSERTING_BIT` because
    /// the right sibling's unlock should increment the split counter.
    ///
    /// # Safety Considerations
    ///
    /// The caller must ensure:
    /// 1. The source node is locked
    /// 2. `unlock_for_split()` will be called on this node exactly once
    /// 3. The new node is not visible to other threads until after `link_sibling()`
    ///
    /// # Memory Ordering
    ///
    /// Uses Relaxed ordering because the new node is not yet visible to other threads.
    /// The fence in `link_sibling()` establishes visibility.
    #[must_use]
    #[inline(always)]
    pub fn new_for_split(source: &Self) -> Self {
        let source_value = source.value.load(Ordering::Relaxed);
        debug_assert!(
            (source_value & LOCK_BIT) != 0,
            "new_for_split: source must be locked"
        );

        // New version has:
        // - ISLEAF_BIT from source (preserved)
        // - LOCK_BIT (locked)
        // - SPLITTING_BIT (will increment split counter on unlock)
        // - Zero version counters (fresh node)
        //
        // We deliberately use SPLITTING_BIT (not INSERTING_BIT) because:
        // 1. This is a split operation
        // 2. unlock_for_split should increment vsplit, not vinsert
        // 3. SPLIT_UNLOCK_MASK clears ROOT_BIT which is correct for split children
        let new_value = (source_value & ISLEAF_BIT) | LOCK_BIT | SPLITTING_BIT;

        Self {
            value: AtomicU32::new(new_value),
        }
    }

    /// Unlock a node that was created with `new_for_split`.
    ///
    /// This performs a split unlock (increments split version counter).
    /// Must be called exactly once after the node's parent pointer is set.
    ///
    /// # C++ Reference
    ///
    /// Matches the unlock in `masstree_split.hh:280`: `child->unlock()`.
    /// The C++ version uses the hand-over-hand pattern where the child
    /// is unlocked after the parent insert completes.
    ///
    /// # Memory Ordering
    ///
    /// Issues a compiler fence before the store to ensure all prior writes
    /// (parent pointer, data) are complete before the unlock is visible.
    /// Uses Release ordering on the store to synchronize with readers'
    /// Acquire loads in `stable()` and `has_changed()`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the node is locked with `SPLITTING_BIT` set.
    #[inline(always)]
    pub fn unlock_for_split(&self) {
        let locked_value = self.value.load(Ordering::Relaxed);

        debug_assert!(
            (locked_value & LOCK_BIT) != 0,
            "unlock_for_split: node must be locked, got value={locked_value:#010x}"
        );

        debug_assert!(
            (locked_value & SPLITTING_BIT) != 0,
            "unlock_for_split: node must have SPLITTING_BIT, got value={locked_value:#010x}"
        );

        // Compute unlocked value: increment split counter, clear dirty/lock/root bits
        // This matches the SPLITTING_BIT branch in LockGuard::drop
        let new_value = (locked_value + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK;

        // Compiler fence: ensures all prior writes are ordered before version store.
        // This is critical - parent pointer and all data must be visible to readers
        // before we unlock. Without this, a reader could see the unlocked version
        // but read stale/missing parent pointer.
        StdAtomic::compiler_fence(Ordering::SeqCst);

        // Release store: synchronizes with Acquire loads in stable()/has_changed()
        self.value.store(new_value, Ordering::Release);
    }

    /// Check if this node was created for a split and hasn't been unlocked yet.
    ///
    /// Returns true if `LOCK_BIT` and `SPLITTING_BIT` are both set.
    /// Used for debugging and assertions.
    #[must_use]
    #[inline(always)]
    pub fn is_split_locked(&self) -> bool {
        let value = self.value.load(Ordering::Relaxed);
        (value & (LOCK_BIT | SPLITTING_BIT)) == (LOCK_BIT | SPLITTING_BIT)
    }
}

impl Clone for NodeVersion {
    fn clone(&self) -> Self {
        Self {
            value: AtomicU32::new(self.value.load(Ordering::Relaxed)),
        }
    }
}

impl Default for NodeVersion {
    /// Creates a new leaf node version.
    fn default() -> Self {
        Self::new(true)
    }
}

// ============================================================================
//  SingleThreadedNodeVersion (for benchmarks)
// ============================================================================

/// A single-threaded node version that skips synchronization.
///
/// This is useful for single-threaded benchmarks where you want to measure
/// the overhead of the data structure without synchronization costs.
///
/// All operations return immediately without any atomic operations or fences.
/// This is NOT thread-safe and must only be used in single-threaded contexts.
///
/// # Reference
/// C++ `nodeversion.hh` has `singlethreaded_nodeversion` for this purpose.
#[derive(Debug, Clone)]
pub struct SingleThreadedNodeVersion {
    value: u32,
}

/// A no-op lock guard for single-threaded usage.
///
/// Does nothing on drop since there's no actual lock to release.
#[derive(Debug)]
#[must_use = "releasing a lock without using the guard is a logic error"]
pub struct SingleThreadedLockGuard<'a> {
    version: &'a mut SingleThreadedNodeVersion,
}

impl Drop for SingleThreadedLockGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        // Same logic as the real LockGuard drop:
        // - If splitting: increment split counter, clear dirty/lock bits
        // - If inserting: increment insert counter, clear inserting/lock bits
        let value: u32 = self.version.value;
        self.version.value = if (value & SPLITTING_BIT) != 0 {
            (value.wrapping_add(VSPLIT_LOWBIT)) & SPLIT_UNLOCK_MASK
        } else {
            value.wrapping_add(VINSERT_LOWBIT) & UNLOCK_MASK
        };
    }
}

impl SingleThreadedLockGuard<'_> {
    /// Mark the node as being inserted into (no-op, for API compatibility).
    #[inline(always)]
    pub const fn mark_insert(&mut self) {
        // No-op - version increments on drop anyway
    }

    /// Mark the node as being split.
    #[inline(always)]
    pub const fn mark_split(&mut self) {
        self.version.value |= SPLITTING_BIT;
    }

    /// Mark the node as deleted.
    #[inline(always)]
    pub const fn mark_deleted(&mut self) {
        self.version.value |= DELETED_BIT | SPLITTING_BIT;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub const fn mark_nonroot(&mut self) {
        self.version.value &= !ROOT_BIT;
    }
}

impl SingleThreadedNodeVersion {
    /// Create a new single-threaded node version.
    #[must_use]
    #[inline(always)]
    pub const fn new(is_leaf: bool) -> Self {
        let initial: u32 = if is_leaf { ISLEAF_BIT } else { 0 };
        Self { value: initial }
    }

    /// Check if this is a leaf node.
    #[must_use]
    #[inline(always)]
    pub const fn is_leaf(&self) -> bool {
        (self.value & ISLEAF_BIT) != 0
    }

    /// Check if this is a root node.
    #[must_use]
    #[inline(always)]
    pub const fn is_root(&self) -> bool {
        (self.value & ROOT_BIT) != 0
    }

    /// Check if this node is logically deleted.
    #[must_use]
    #[inline(always)]
    pub const fn is_deleted(&self) -> bool {
        (self.value & DELETED_BIT) != 0
    }

    /// Check if a version value indicates the node is deleted.
    ///
    /// In single-threaded mode, this is equivalent to `is_deleted()` but
    /// provided for API consistency with concurrent mode.
    #[must_use]
    #[inline(always)]
    pub const fn is_deleted_version(version: u32) -> bool {
        (version & DELETED_BIT) != 0
    }

    /// Get a stable version (returns immediately in single-threaded mode).
    #[must_use]
    #[inline(always)]
    pub const fn stable(&self) -> u32 {
        self.value
    }

    /// Check if the version has changed since `old`.
    #[must_use]
    #[inline(always)]
    pub const fn has_changed(&self, old: u32) -> bool {
        (old ^ self.value) > (LOCK_BIT | INSERTING_BIT)
    }

    /// Check if a split has occurred since `old`.
    #[must_use]
    #[inline(always)]
    pub const fn has_split(&self, old: u32) -> bool {
        (old ^ self.value) >= VSPLIT_LOWBIT
    }

    /// Acquire the "lock" (no-op, returns guard immediately).
    #[inline(always)]
    pub const fn lock(&mut self) -> SingleThreadedLockGuard<'_> {
        SingleThreadedLockGuard { version: self }
    }

    /// Mark the node as a root.
    #[inline(always)]
    pub const fn mark_root(&mut self) {
        self.value |= ROOT_BIT;
    }

    /// Clear the root bit.
    #[inline(always)]
    pub const fn mark_nonroot(&mut self) {
        self.value &= !ROOT_BIT;
    }
}

impl Default for SingleThreadedNodeVersion {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod unit_tests;

// Concurrent tests live in a submodule to keep this file lean.
// Guarded with `#[cfg(not(miri))]` because Miri doesn't support multi-threading well.
#[cfg(test)]
#[cfg(not(miri))]
mod concurrent_tests;

// Loom tests for deterministic concurrency verification.
#[cfg(all(test, loom, not(miri)))]
mod loom_tests;
