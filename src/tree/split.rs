//! Hand-over-hand split propagation.
//!
//! This module implements TRUE hand-over-hand split propagation matching
//! C++ `tcursor::make_split()` from `masstree_split.hh:179-297`.
//!
//! # Key Invariant
//! The left node remains locked while we lock its parent and install the
//! split sibling. This prevents the "split-but-uninstalled" window that
//! causes permanent routing inconsistencies.
//!
//! # Design Choice: [`PropagationContext`]
//! This implementation [`PropagationContext<'op>`] - a unified-lifetime
//! context that allows all [`LockGuard<'op>`] instances to have the same
//! lifetime parameter. This enables lock transer across loop iterations
//! while preserving RAII (panic-safe automatic unlocking).
//!
//! Unlike the previous `PropagationLock` concept:
//! - Guards auto-unlock on drop (RAII preserved)
//! - No `mem::forget` patterns
//! - All locks use existing `NodeVersion::lock()` implementation
//!
//! # Hand-Over-Hand Split Protocol
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    SPLIT PROPAGATION OVERVIEW                                │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   BEFORE SPLIT:                                                              │
//!   │                                                                              │
//!   │                    ┌──────────┐                                              │
//!   │                    │  Parent  │                                              │
//!   │                    │ Internode│                                              │
//!   │                    └────┬─────┘                                              │
//!   │                         │                                                    │
//!   │                         ▼                                                    │
//!   │   ┌───────────────────────────────────────────────────────────────┐          │
//!   │   │                    Left Leaf (FULL)                           │          │
//!   │   │  [k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13] │          │
//!   │   └───────────────────────────────────────────────────────────────┘          │
//!   │                                                                              │
//!   │   ─────────────────────────────────────────────────────────────────────────  │
//!   │                                                                              │
//!   │   AFTER SPLIT:                                                               │
//!   │                                                                              │
//!   │                    ┌──────────────────────────────┐                          │
//!   │                    │          Parent              │                          │
//!   │                    │  [..., split_key, ...]       │                          │
//!   │                    └────────┬───────┬─────────────┘                          │
//!   │                             │       │                                        │
//!   │                   ┌─────────┘       └─────────┐                              │
//!   │                   ▼                           ▼                              │
//!   │   ┌───────────────────────────┐   ┌───────────────────────────┐              │
//!   │   │       Left Leaf           │◄─►│      Right Leaf           │              │
//!   │   │  [k0, k1, ..., k6]        │   │  [k7, k8, ..., k13]       │              │
//!   │   │                           │   │                           │              │
//!   │   │  next ─────────────────────►  │                           │              │
//!   │   │                           │◄─── prev                      │              │
//!   │   └───────────────────────────┘   └───────────────────────────┘              │
//!   │           (B-link chain installed)                                           │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Split Propagation State Machine
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                         PROPAGATION LOOP                                     │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌──────────────┐
//!   │ Leaf full    │──────── Insert triggered split
//!   │ (locked)     │
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ Create right │──────── Allocate sibling, copy upper half of keys
//!   │ sibling      │         Right sibling starts LOCKED (new_for_split)
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ Install      │──────── 1. right.prev = left (init sibling)
//!   │ B-link       │         2. right.next = old_next
//!   │              │         3. Release fence
//!   │              │         4. left.next = right (publish last!)
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │ mark_split() │──────── Set SPLITTING_BIT on left
//!   │ on left      │         Readers will see split in progress
//!   └──────┬───────┘
//!          │
//!          ▼
//!   ┌────────────────────────────────────────────────────────────────────────────┐
//!   │                      HAND-OVER-HAND LOOP                                   │
//!   │                                                                            │
//!   │   ┌──────────────┐                                                         │
//!   │   │ Find parent  │──────── Walk up using parent pointer                    │
//!   │   │ of left      │                                                         │
//!   │   └──────┬───────┘                                                         │
//!   │          │                                                                 │
//!   │     ┌────┴────┐                                                            │
//!   │     │         │                                                            │
//!   │     ▼         ▼                                                            │
//!   │  ┌──────┐  ┌──────────┐                                                    │
//!   │  │Parent│  │No parent │──────► Create new root                             │
//!   │  │exists│  │(is root) │        (RootCreation module)                       │
//!   │  └──┬───┘  └──────────┘                                                    │
//!   │     │                                                                      │
//!   │     ▼                                                                      │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Lock parent  │──────── Acquire parent lock while keeping left locked    │
//!   │  │ (hand-over)  │         This is the "hand-over-hand" pattern             │
//!   │  └──────┬───────┘                                                          │
//!   │         │                                                                  │
//!   │         ▼                                                                  │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Verify left  │──────── Check parent.child[i] still points to left       │
//!   │  │ membership   │         (ParentLocking module)                           │
//!   │  └──────┬───────┘                                                          │
//!   │         │                                                                  │
//!   │    ┌────┴────┐                                                             │
//!   │    │         │                                                             │
//!   │    ▼         ▼                                                             │
//!   │  ┌──────┐  ┌──────────┐                                                    │
//!   │  │Valid │  │ Invalid  │──────► Unlock parent, retry from find_parent       │
//!   │  └──┬───┘  └──────────┘                                                    │
//!   │     │                                                                      │
//!   │     ▼                                                                      │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Insert split │──────── Add split_key and right pointer to parent        │
//!   │  │ into parent  │                                                          │
//!   │  └──────┬───────┘                                                          │
//!   │         │                                                                  │
//!   │         ▼                                                                  │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Unlock left  │──────── Left is done, release lock                       │
//!   │  │ (original)   │                                                          │
//!   │  └──────┬───────┘                                                          │
//!   │         │                                                                  │
//!   │         ▼                                                                  │
//!   │  ┌──────────────┐         ┌────────────────────────────────────────────┐   │
//!   │  │ Parent full? │── Yes ──►│ Recurse: parent becomes new "left"        │   │
//!   │  │              │         │ Continue hand-over-hand upward             │   │
//!   │  └──────┬───────┘         └────────────────────────────────────────────┘   │
//!   │         │ No                                                               │
//!   │         ▼                                                                  │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Unlock right │──────── unlock_for_split() - now visible to readers      │
//!   │  │ sibling      │                                                          │
//!   │  └──────┬───────┘                                                          │
//!   │         │                                                                  │
//!   │         ▼                                                                  │
//!   │  ┌──────────────┐                                                          │
//!   │  │ Unlock parent│                                                          │
//!   │  │ DONE         │                                                          │
//!   │  └──────────────┘                                                          │
//!   │                                                                            │
//!   └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Lock Timeline (Hand-Over-Hand)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    LOCK TIMELINE DURING SPLIT                                │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   Time ──────────────────────────────────────────────────────────────────►   │
//!   │                                                                              │
//!   │   Left Leaf:                                                                 │
//!   │   ┌────────────────────────────────────────────────────────┐                 │
//!   │   │████████████████████████████████████████████████████████│ (locked)        │
//!   │   └────────────────────────────────────────────────────────┘                 │
//!   │   ▲                                              ▲                           │
//!   │   │ lock() at                                    │ unlock()                  │
//!   │   │ insert time                                  │ after parent updated      │
//!   │                                                                              │
//!   │   Right Sibling (new):                                                       │
//!   │                    ┌───────────────────────────────────────────────────┐     │
//!   │                    │███████████████████████████████████████████████████│     │
//!   │                    └───────────────────────────────────────────────────┘     │
//!   │                    ▲                                           ▲             │
//!   │                    │ new_for_split()                           │ unlock_for_ │
//!   │                    │ (starts locked)                           │ split()     │
//!   │                                                                              │
//!   │   Parent Internode:                                                          │
//!   │                           ┌───────────────────────────────────────────────┐  │
//!   │                           │███████████████████████████████████████████████│  │
//!   │                           └───────────────────────────────────────────────┘  │
//!   │                           ▲                                       ▲          │
//!   │                           │ lock() while                          │ unlock() │
//!   │                           │ left still held                       │ after    │
//!   │                           │ (hand-over-hand)                      │ install  │
//!   │                                                                              │
//!   │   KEY INSIGHT:                                                               │
//!   │   ─────────────                                                              │
//!   │   • Left remains locked from insert start until parent is updated            │
//!   │   • This prevents "split-but-uninstalled" window where readers could         │
//!   │     find right sibling via B-link but not via parent traversal               │
//!   │   • Parent lock overlaps with left lock (hand-over-hand)                     │
//!   │   • Right sibling unlocked LAST, only after fully installed                  │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # B-Link Traversal During Split
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────────┐
//!   │                    B-LINK HELP-ALONG DURING SPLIT                            │
//!   ├──────────────────────────────────────────────────────────────────────────────┤
//!   │                                                                              │
//!   │   The B-link protocol allows readers to find keys that have moved to a       │
//!   │   sibling during a split, even when the parent hasn't been updated yet.      │
//!   │   This eliminates the need for readers to acquire locks.                     │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    TREE STATE DURING SPLIT                             │ │
//!   │   │                                                                        │ │
//!   │   │                    ┌─────────────────────┐                             │ │
//!   │   │                    │       Parent        │                             │ │
//!   │   │                    │                     │                             │ │
//!   │   │                    │  children: [Left]   │  ← May not know about       │ │
//!   │   │                    │  (stale routing)    │    Right sibling yet        │ │
//!   │   │                    └──────────┬──────────┘                             │ │
//!   │   │                               │                                        │ │
//!   │   │                               ▼                                        │ │
//!   │   │   ┌─────────────────────────────────────────────────────────────────┐  │ │
//!   │   │   │        Left Leaf                    Right Leaf (NEW)            │  │ │
//!   │   │   │        (LOCKED)                     (LOCKED)                    │  │ │
//!   │   │   │                                                                 │  │ │
//!   │   │   │   ┌─────────────────┐        ┌─────────────────┐                │  │ │
//!   │   │   │   │ version:        │        │ version:        │                │  │ │
//!   │   │   │   │ LOCK|INSERTING| │        │ LOCK|INSERTING  │                │  │ │
//!   │   │   │   │ SPLITTING       │        │                 │                │  │ │
//!   │   │   │   │                 │        │                 │                │  │ │
//!   │   │   │   │ ikey_bound: k6  │        │ ikey_bound: MAX │                │  │ │
//!   │   │   │   │                 │        │                 │                │  │ │
//!   │   │   │   │ keys: [k1..k6]  │        │ keys: [k7..k13] │                │  │ │
//!   │   │   │   │                 │ ◄───── │                 │                │  │ │
//!   │   │   │   │ next ───────────┼───────►│ prev            │                │  │ │
//!   │   │   │   └─────────────────┘        └─────────────────┘                │  │ │
//!   │   │   │                                                                 │  │ │
//!   │   │   │         B-link chain installed BEFORE parent update             │  │ │
//!   │   │   └─────────────────────────────────────────────────────────────────┘  │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   ┌────────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    READER SCENARIOS                                    │ │
//!   │   │                                                                        │ │
//!   │   │   SCENARIO A: Reader looking for k3 (in Left leaf)                     │ │
//!   │   │   ─────────────────────────────────────────────────                    │ │
//!   │   │                                                                        │ │
//!   │   │   1. Parent routes reader to Left                                      │ │
//!   │   │   2. Reader: v1 = Left.version.stable()                                │ │
//!   │   │      └─ Spins if LOCK|INSERTING|SPLITTING set                          │ │
//!   │   │   3. Reader: search for k3                                             │ │
//!   │   │   4. k3 found in Left                                                  │ │
//!   │   │   5. Reader: has_changed(v1)? → Check for concurrent modification      │ │
//!   │   │   6. SUCCESS: Return value                                             │ │
//!   │   │                                                                        │ │
//!   │   │   ─────────────────────────────────────────────────────────────────    │ │
//!   │   │                                                                        │ │
//!   │   │   SCENARIO B: Reader looking for k10 (moved to Right leaf)             │ │
//!   │   │   ─────────────────────────────────────────────────────                │ │
//!   │   │                                                                        │ │
//!   │   │   1. Parent routes reader to Left (stale routing)                      │ │
//!   │   │   2. Reader: v1 = Left.version.stable()                                │ │
//!   │   │      └─ Waits for split to complete                                    │ │
//!   │   │   3. Reader: search for k10 in Left                                    │ │
//!   │   │   4. k10 NOT FOUND in Left                                             │ │
//!   │   │   5. Reader: k10 >= ikey_bound(k6)?                                    │ │
//!   │   │      └─ YES: Key may be in right sibling                               │ │
//!   │   │   6. Reader: follow B-link → Left.next → Right                         │ │
//!   │   │   7. Reader: v2 = Right.version.stable()                               │ │
//!   │   │   8. Reader: search for k10 in Right                                   │ │
//!   │   │   9. k10 FOUND in Right                                                │ │
//!   │   │   10. Reader: has_changed(v2)?                                         │ │
//!   │   │   11. SUCCESS: Return value                                            │ │
//!   │   │                                                                        │ │
//!   │   │   ─────────────────────────────────────────────────────────────────    │ │
//!   │   │                                                                        │ │
//!   │   │   SCENARIO C: Reader during ongoing split                              │ │
//!   │   │   ────────────────────────────────────────                             │ │
//!   │   │                                                                        │ │
//!   │   │   1. Parent routes reader to Left                                      │ │
//!   │   │   2. Reader: v1 = Left.version.stable()                                │ │
//!   │   │      └─ Returns immediately (dirty bits set but stable spins)          │ │
//!   │   │   3. Reader starts searching...                                        │ │
//!   │   │   4. Split completes, version bumped (VSPLIT += 1)                     │ │
//!   │   │   5. Reader: has_changed(v1)? → YES (VSPLIT changed)                   │ │
//!   │   │   6. Reader: RETRY from root OR follow B-link                          │ │
//!   │   │      └─ Depends on split type and key location                         │ │
//!   │   │                                                                        │ │
//!   │   └────────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                              │
//!   │   KEY PROPERTIES:                                                            │
//!   │   ├─ Readers NEVER block writers (no read locks)                             │
//!   │   ├─ Writers NEVER block readers (B-link provides fallback)                  │
//!   │   ├─ Tree always consistent enough for correct traversal                     │
//!   │   ├─ Version validation detects concurrent modifications                     │
//!   │   └─ Retry mechanism handles races safely                                    │
//!   │                                                                              │
//!   └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Module Organization
//!
//! - [`PropagationContext`]: Unified-lifetime context for RAII guards
//! - [`Propagation`]: Core hand-over-hand propagation loop
//! - [`RootCreation`]: Root and layer-root creation helpers
//! - [`ParentLocking`]: Membership validation helpers

mod parent_locking;
mod propagation;
mod propagation_context;
mod root_creation;

pub use propagation::Propagation;
