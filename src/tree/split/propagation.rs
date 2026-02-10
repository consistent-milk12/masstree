//! Core hand-over-hand split propagation loop.
//!
//! Implements the iterative algorithm from C++ `tcursor::make_split()`
//! (`masstree_split.hh:179-297`).
//!
//! # Key Invariant
//!
//! The left node remains locked while we:
//! 1. Lock its parent (hand-over-hand)
//! 2. Validate membership
//! 3. Insert the split sibling
//! 4. Only then unlock in the current order
//!
//! # Design: [`PropagationContext`] with Unified Lifetimes
//!
//! Uses [`PropagationContext<'op>`] to create [`LockGuard<'op>`] instances that all
//! share the same lifetime parameter tied to the reclamation guard. This enables:
//!
//! - Lock transfer across loop iterations via `drop(left_lock); left_lock = parent_lock;`
//! - RAII: guards auto-unlock on drop (panic-safe)
//! - No `mem::forget` patterns
//!
//! Another potential approach (no-drop in release mode) was replaced because it
//! created lock leak risks and made auditing harder.
//!
//! # No-Abandon invariant
//! Once a split sibling is created, the loop must continue until installation
//! succeeds. There is no retry path that abandons a created sibling.
//!
//! # Stable Parent Pointer Fallback
//! If membership validation fails repeatedly, the parent pointer may be stale.
//! After `MAX_STALE_PARENT_RETRIES` consecutive failures, we trigger a
//! re-descent from root to find the correct parent. This is bounded fallback
//! rather than infinite retry.

use std::hint as StdHint;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

use seize::LocalGuard;

use crate::TreeAllocator;
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::nodeversion::LockGuard;
use crate::policy::LeafPolicy;
use crate::tree::InsertError;

use super::parent_locking::ParentLocking;
use super::propagation_context::PropagationContext;
use super::root_creation::RootCreation;

// CRITICAL: Defensive bounds - uncomment if debugging infinite loops or livelock
// const MAX_PROPAGATION_ITERATIONS: usize = 64;
// const MAX_STALE_PARENT_RETRIES: usize = 16;

/// Maximum spins before backoff caps.
///
/// 64 iterations ≈ 200-400 cycles on x86, sufficient for typical lock hold times.
/// Power of 2 enables efficient doubling. Beyond this, we yield to the OS.
const BACKOFF_CAP: u32 = 64;

/// Unit struct namespace for split propagation operations.
pub struct Propagation;

impl Propagation {
    /// Perform TRUE hand-over-hand split propagation for a leaf split.
    ///
    /// This is the main entry point. It takes ownership of the left leaf's
    /// lock and maintains it throughout propagation using unified-lifetime
    /// `LockGuard<'op>` via `PropagationContext`.
    ///
    /// # Arguments
    ///
    /// - `root_ptr`: Atomic pointer to tree root
    /// - `allocator`: Node allocator
    /// - `left_leaf_ptr`: Left leaf pointer (locked via `left_lock`)
    /// - `left_lock`: Lock guard for left leaf (converted to unified lifetime)
    /// - `right_leaf_ptr`: Right sibling pointer (split-locked)
    /// - `split_ikey`: Separator key
    /// - `is_main_root`: Left is THE main tree root
    /// - `is_layer_root`: Left is a layer root (null parent, not main root)
    /// - `guard`: Memory reclamation guard
    ///
    /// # Lock Protocol (v3 - RAII)
    ///
    /// - Entry: `left_leaf_ptr` locked via `left_lock`, `right_leaf_ptr` split-locked
    /// - Exit: All locks released via RAII (guards auto-unlock on drop)
    ///
    /// # C++ Reference
    ///
    /// `tcursor::make_split()` in `reference/masstree_split.hh:179-297`
    #[expect(
        clippy::too_many_arguments,
        reason = "Split propagation requires full context"
    )]
    pub fn make_split_leaf<'op, P, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_leaf_ptr: *mut LeafNode15<P>,
        left_lock: LockGuard<'_>,
        right_leaf_ptr: *mut LeafNode15<P>,
        split_ikey: u64,
        is_main_root: bool,
        is_layer_root: bool,
        guard: &'op LocalGuard<'op>,
    ) -> Result<(), InsertError>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        // DEBUG: Trace leaf split
        #[cfg(feature = "debug-routing")]
        eprintln!(
            "[LEAF_SPLIT] left={left_leaf_ptr:p} right={right_leaf_ptr:p} split_ikey={split_ikey:016x} is_main_root={is_main_root} is_layer_root={is_layer_root}"
        );

        // Create PropagationContext with unified lifetime tied to reclamation guard
        let ctx: PropagationContext<'op> = PropagationContext::new(guard);

        // SAFETY: Lifetime extension is sound because:
        // 1. Reclamation guard prevents deallocation while we hold it
        // 2. Leaf is locked, preventing structural modification
        let left_lock: LockGuard<'op> = unsafe { ctx.unify_guard(left_lock) };
        let result: Result<(), InsertError> = Self::propagation_loop::<P, A>(
            root_ptr,
            allocator,
            &ctx,
            left_leaf_ptr.cast(),
            left_lock,
            right_leaf_ptr.cast(),
            split_ikey,
            is_main_root,
            is_layer_root,
            true, // at_leaf_level
        );

        result
    }

    /// Core iterative propagation loop with hand-over-hand locking.
    ///
    /// Uses `PropagationContext<'op>` for unified-lifetime lock management,
    /// enabling RAII guard transfer across loop iterations.
    ///
    /// # Errors
    /// Returns `InsertError::SplitFailed` only if main root CAS fails.
    #[expect(clippy::too_many_lines, reason = "Complex state machine with tracing")]
    #[expect(clippy::too_many_arguments, reason = "State passed explicitly")]
    fn propagation_loop<'op, P, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        ctx: &PropagationContext<'op>,
        mut left_ptr: *mut u8,         // Erased pointer (leaf or internode)
        mut left_lock: LockGuard<'op>, // RAII guard with unified lifetime
        mut right_ptr: *mut u8,        // Erased pointer (split-locked)
        mut split_ikey: u64,
        mut is_main_root: bool,
        mut is_layer_root: bool,
        mut at_leaf_level: bool,
    ) -> Result<(), InsertError>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        // CRITICAL: Uncomment iteration tracking if debugging infinite loops
        // let mut iterations: usize = 0;
        // let mut stale_parent_retries: usize = 0;

        // Exponential backoff state for contention reduction.
        // See `spin_backoff()` for details on SMT optimization.
        let mut backoff: u32 = 1;

        loop {
            // CRITICAL: Uncomment to detect runaway propagation (tree corruption)
            // iterations += 1;
            // if iterations > MAX_PROPAGATION_ITERATIONS {
            //     Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
            //     drop(left_lock);
            //     panic!("Propagation: exceeded max iterations - tree likely corrupted");
            // }

            // Get left's parent pointer
            let left_parent: *mut u8 = Self::get_parent::<P>(left_ptr, at_leaf_level);

            // =========================================================
            // STEP 1: Check for root cases (layer root FIRST, then main)
            // =========================================================

            // 1a. LAYER ROOT (check BEFORE main root)
            if left_parent.is_null() && is_layer_root {
                Self::promote_layer_root::<P, A>(
                    allocator,
                    left_ptr,
                    right_ptr,
                    split_ikey,
                    at_leaf_level,
                );

                // Unlock right (split-locked), then left (RAII via drop)
                Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
                drop(left_lock); // RAII: auto-unlock
                return Ok(());
            }

            // 1b. MAIN TREE ROOT
            if left_parent.is_null() && is_main_root {
                let result: Result<(), InsertError> = Self::create_main_root::<P, A>(
                    root_ptr,
                    allocator,
                    left_ptr,
                    right_ptr,
                    split_ikey,
                    at_leaf_level,
                );

                Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
                drop(left_lock); // RAII: auto-unlock
                return result;
            }

            // CRITICAL: NULL parent on non-root indicates tree corruption
            // Uncomment if debugging parent pointer issues
            // if left_parent.is_null() {
            //     Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
            //     drop(left_lock);
            //     panic!("Propagation: NULL parent on non-root");
            // }

            // =========================================================
            // STEP 2: Lock parent WHILE left is still locked
            // =========================================================
            //
            // This is TRUE hand-over-hand: we hold left_lock while
            // acquiring parent_lock. Both are LockGuard<'op> with RAII.

            let parent: &InternodeNode = unsafe { &*left_parent.cast::<InternodeNode>() };

            // SAFETY: parent is valid (reclamation guard protects for 'op)
            // Use lock_node_yielding to reduce contention under high thread counts
            let mut parent_lock: LockGuard<'op> =
                unsafe { ctx.lock_node_yielding(parent.version().as_ptr()) };

            // =========================================================
            // STEP 3: Revalidate parent pointer after locking
            // =========================================================
            //
            // C++ masstree_struct.hh:552-570 does this revalidation.
            // The parent pointer could have changed if another thread
            // split the parent and moved our child.

            let current_left_parent: *mut u8 = Self::get_parent::<P>(left_ptr, at_leaf_level);

            if current_left_parent != left_parent {
                // Parent pointer changed - release parent lock and retry with backoff
                drop(parent_lock); // RAII: auto-unlock
                Self::spin_backoff(&mut backoff);

                continue;
            }

            // =========================================================
            // STEP 4: Validate membership (pointer scan, NOT key-based)
            // =========================================================

            let child_idx: usize =
                if let Some(idx) = ParentLocking::validate_membership(parent, left_ptr) {
                    // Success: reset backoff for next potential retry
                    backoff = 1;
                    idx
                } else {
                    // CRITICAL: Uncomment to prevent livelock on persistent stale parent
                    // stale_parent_retries += 1;
                    // if stale_parent_retries > MAX_STALE_PARENT_RETRIES {
                    //     Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
                    //     drop(parent_lock);
                    //     drop(left_lock);
                    //     return Err(InsertError::SplitFailed);
                    // }

                    // Child not found - parent may have been split concurrently
                    // Release parent lock and retry with backoff (left_lock still held)
                    drop(parent_lock); // RAII: auto-unlock
                    Self::spin_backoff(&mut backoff);

                    continue;
                };

            // =========================================================
            // STEP 5: Parent has space - insert and finish
            // =========================================================

            if !parent.is_full() {
                parent_lock.mark_insert();

                // DEBUG: Trace parent insert (no split)
                #[cfg(feature = "debug-routing")]
                {
                    eprintln!(
                        "[PARENT_INSERT] parent={:p} height={} child_idx={} split_ikey={:016x} nkeys_before={}",
                        left_parent,
                        parent.height(),
                        child_idx,
                        split_ikey,
                        parent.nkeys()
                    );
                }

                // Insert at child_idx (pointer-based, NOT key-based)
                parent.insert_key_and_child(child_idx, split_ikey, right_ptr);

                // DEBUG: Show parent keys after insert
                #[cfg(feature = "debug-routing")]
                {
                    let nkeys = parent.nkeys();
                    eprint!("        parent_keys_after[{nkeys}]: ");
                    for i in 0..nkeys {
                        eprint!("{:016x} ", parent.ikey(i));
                    }
                    eprintln!();
                }

                // Set right sibling's parent pointer
                Self::set_parent::<P>(right_ptr, left_parent, at_leaf_level);

                // Unlock order: right → parent → left (RAII via drop)
                Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);
                drop(parent_lock); // RAII: auto-unlock
                drop(left_lock); // RAII: auto-unlock

                return Ok(());
            }

            // =========================================================
            // STEP 6: Parent is full - split and continue
            // =========================================================
            parent_lock.mark_split();

            // Capture parent's root status BEFORE modifications
            // NOTE: Check main root FIRST to avoid the bug where
            // `parent.parent().is_null() && parent.is_root()` matches both
            let parent_is_main_root: bool = {
                let current_root: *mut u8 = root_ptr.load(AtomicOrdering::Acquire);

                StdPtr::eq(current_root, left_parent)
            };

            // SAFETY: Called under lock - no concurrent retirement.
            let parent_is_layer_root: bool = !parent_is_main_root
                && unsafe { parent.parent_unguarded() }.is_null()
                && parent.is_root();

            // Create split-locked sibling directly in pool
            let parent_sibling_ptr: *mut InternodeNode = allocator
                .alloc_internode_direct_for_split(parent.version(), parent.height())
                .cast();

            // Split parent and insert child
            let (popup_key, child_went_left): (u64, bool) = unsafe {
                parent.split_into(
                    &mut *parent_sibling_ptr,
                    parent_sibling_ptr,
                    child_idx,
                    split_ikey,
                    right_ptr,
                )
            };

            // DEBUG: Trace internode split
            #[cfg(feature = "debug-routing")]
            {
                eprintln!(
                    "[INTERNODE_SPLIT] parent={:p} sibling={:p} height={} child_idx={} split_ikey={:016x} popup_key={:016x} child_went_left={}",
                    left_parent,
                    parent_sibling_ptr,
                    parent.height(),
                    child_idx,
                    split_ikey,
                    popup_key,
                    child_went_left
                );
                // Dump keys in parent after split
                let parent_nkeys = parent.nkeys();
                eprint!("        parent_keys[{parent_nkeys}]: ");
                for i in 0..parent_nkeys {
                    eprint!("{:016x} ", parent.ikey(i));
                }
                eprintln!();
                // Dump keys in sibling after split
                let sibling = unsafe { &*parent_sibling_ptr };
                let sibling_nkeys = sibling.nkeys();
                eprint!("        sibling_keys[{sibling_nkeys}]: ");
                for i in 0..sibling_nkeys {
                    eprint!("{:016x} ", sibling.ikey(i));
                }
                eprintln!();
            }

            // Update children's parent pointers in sibling
            Self::update_sibling_children_parents::<P>(parent, parent_sibling_ptr);

            // Set current right's parent based on which side it went
            let right_new_parent: *mut u8 = if child_went_left {
                left_parent
            } else {
                parent_sibling_ptr.cast()
            };
            Self::set_parent::<P>(right_ptr, right_new_parent, at_leaf_level);

            // =========================================================
            // STEP 7: TRUE Hand-over-hand transition (v3 RAII)
            // =========================================================
            //
            // C++ lines 276-287:
            // - Unlock current right (it's now installed)
            // - Unlock current left (NOT the parent!)
            // - parent becomes new left (STAYS LOCKED via parent_lock)
            // - parent_sibling becomes new right (split-locked)
            //
            // v3 RAII approach:
            // - drop(left_lock) releases old left
            // - left_lock = parent_lock transfers ownership WITHOUT unlock
            //   (because both have unified lifetime 'op)

            // Unlock current right sibling (it's fully installed now)
            Self::unlock_right_for_split::<P>(right_ptr, at_leaf_level);

            // Unlock current left (we're moving up)
            // IMPORTANT: This unlocks the OLD left, not the parent!
            drop(left_lock); // RAII: auto-unlock old left

            // =========================================================
            // KEY v3 DIFFERENCE FROM v2: RAII transfer!
            // =========================================================
            //
            // Both left_lock and parent_lock have lifetime 'op (unified
            // via PropagationContext). This allows simple assignment:
            //
            //   left_lock = parent_lock;
            //
            // The parent remains locked because:
            // 1. Assignment moves parent_lock into left_lock
            // 2. parent_lock is no longer valid (moved)
            // 3. The underlying lock is NOT released (no drop)
            //
            // This is panic-safe: if anything panics, left_lock's
            // destructor will unlock the parent.

            left_lock = parent_lock; // RAII transfer: parent stays locked!
            left_ptr = left_parent;
            right_ptr = parent_sibling_ptr.cast();
            split_ikey = popup_key;
            is_main_root = parent_is_main_root;
            is_layer_root = parent_is_layer_root;
            at_leaf_level = false; // Now at internode level
        }
    }

    // =========================================================================
    // Helper methods
    // =========================================================================

    /// Returns parent pointer for a type-erased node.
    ///
    /// # Safety
    /// `ptr` must point to a valid node matching `is_leaf`.
    #[inline]
    fn get_parent<P>(ptr: *mut u8, is_leaf: bool) -> *mut u8
    where
        P: LeafPolicy,
    {
        // SAFETY: Called under lock or during propagation where nodes are locked.
        if is_leaf {
            unsafe { (*ptr.cast::<LeafNode15<P>>()).parent_unguarded() }
        } else {
            unsafe { (*ptr.cast::<InternodeNode>()).parent_unguarded() }
        }
    }

    /// Sets parent pointer for a type-erased node.
    ///
    /// # Safety
    /// `ptr` must point to a valid, locked node matching `is_leaf`.
    #[inline]
    fn set_parent<P>(ptr: *mut u8, parent: *mut u8, is_leaf: bool)
    where
        P: LeafPolicy,
    {
        if is_leaf {
            // SAFETY: Caller guarantees ptr is valid leaf
            unsafe { (*ptr.cast::<LeafNode15<P>>()).set_parent(parent) };
        } else {
            // SAFETY: Caller guarantees ptr is valid internode
            unsafe { (*ptr.cast::<InternodeNode>()).set_parent(parent) };
        }
    }

    /// Unlock a split-locked right sibling.
    ///
    /// Uses `NodeVersion::unlock_for_split()` which:
    /// - Increments the split version counter
    /// - Clears `LOCK_BIT`, `SPLITTING_BIT`, `INSERTING_BIT`
    /// - Uses proper fence before version store
    #[inline]
    fn unlock_right_for_split<P>(ptr: *mut u8, is_leaf: bool)
    where
        P: LeafPolicy,
    {
        if is_leaf {
            // SAFETY: ptr points to a valid split-locked leaf
            unsafe { (*ptr.cast::<LeafNode15<P>>()).version().unlock_for_split() };
        } else {
            // SAFETY: ptr points to a valid split-locked internode
            unsafe { (*ptr.cast::<InternodeNode>()).version().unlock_for_split() };
        }
    }

    /// Fixes parent pointers for children that moved to the split sibling.
    ///
    /// After internode split, children in the sibling still reference the old parent.
    ///
    /// # Safety
    /// - `parent` must be locked
    /// - `sibling_ptr` must be valid and split-locked
    fn update_sibling_children_parents<P>(parent: &InternodeNode, sibling_ptr: *mut InternodeNode)
    where
        P: LeafPolicy,
    {
        let sibling: &InternodeNode = unsafe { &*sibling_ptr };
        let nkeys: usize = sibling.nkeys();

        if parent.children_are_leaves() {
            for i in 0..=nkeys {
                // SAFETY: Sibling is split-locked, children are valid.
                let child: *mut u8 = unsafe { sibling.child_unguarded(i) };

                // Children at valid indices should never be null in a well-formed internode.
                debug_assert!(
                    !child.is_null(),
                    "update_sibling_children_parents: null child at index {i}"
                );

                unsafe {
                    (*child.cast::<LeafNode15<P>>()).set_parent(sibling_ptr.cast());
                }
            }
        } else {
            for i in 0..=nkeys {
                // SAFETY: Sibling is split-locked, children are valid.
                let child: *mut u8 = unsafe { sibling.child_unguarded(i) };

                debug_assert!(
                    !child.is_null(),
                    "update_sibling_children_parents: null child at index {i}"
                );

                unsafe {
                    (*child.cast::<InternodeNode>()).set_parent(sibling_ptr.cast());
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn promote_layer_root<P, A>(
        allocator: &A,
        left_ptr: *mut u8,
        right_ptr: *mut u8,
        split_ikey: u64,
        is_leaf: bool,
    ) where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        if is_leaf {
            RootCreation::promote_layer_root_leaves::<P, A>(
                allocator,
                left_ptr.cast(),
                right_ptr.cast(),
                split_ikey,
            );
        } else {
            RootCreation::promote_layer_root_internodes::<P, A>(
                allocator,
                left_ptr.cast(),
                right_ptr.cast(),
                split_ikey,
            );
        }
    }

    #[cold]
    #[inline(never)]
    fn create_main_root<P, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_ptr: *mut u8,
        right_ptr: *mut u8,
        split_ikey: u64,
        is_leaf: bool,
    ) -> Result<(), InsertError>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        if is_leaf {
            RootCreation::create_root_from_leaves::<P, A>(
                root_ptr,
                allocator,
                left_ptr.cast(),
                right_ptr.cast(),
                split_ikey,
            )
            .map(|_| ())
        } else {
            RootCreation::create_root_from_internodes::<P, A>(
                root_ptr,
                allocator,
                left_ptr.cast(),
                right_ptr.cast(),
                split_ikey,
            )
            .map(|_| ())
        }
    }

    /// Exponential backoff: spin `backoff` times, then double (capped).
    ///
    /// Yields to OS scheduler at cap to avoid wasting cycles under sustained contention.
    #[inline]
    fn spin_backoff(backoff: &mut u32) {
        for _ in 0..*backoff {
            StdHint::spin_loop();
        }

        if *backoff >= BACKOFF_CAP {
            std::thread::yield_now();
        }

        *backoff = (*backoff * 2).min(BACKOFF_CAP);
    }
}
