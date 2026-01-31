use std::ptr as StdPtr;

use crate::leaf_trait::{SplitInsertData, SplitInsertResult};
use crate::{SplitPoint, nodeversion::LockGuard};

use super::{
    AtomicOrdering, InsertError, LayerCapableLeaf, LocalGuard, MassTreeGeneric,
    NodeAllocatorGeneric, Propagation, ValueSlot,
};

// ============================================================================
// Split Preparation Context
// ============================================================================

/// Context gathered before a leaf split operation.
///
/// Extracting this into a struct eliminates code duplication between
/// `handle_leaf_split_generic` and `handle_leaf_split_and_insert_generic`.
struct SplitPreparation {
    /// The calculated split position and separator key.
    split_point: SplitPoint,

    /// True if this leaf is the main tree root (`ROOT_BIT` set AND matches `root_ptr`).
    is_main_root: bool,

    /// True if this leaf is a layer root (`ROOT_BIT` set, null parent, but not main root).
    is_layer_root: bool,

    /// Freshly allocated right sibling (not yet split-locked or visible).
    right_leaf_ptr: *mut u8,
}

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Check if a leaf pointer is the main tree root.
    ///
    /// Returns `true` only if BOTH conditions are met:
    /// 1. The leaf's `ROOT_BIT` is set
    /// 2. The leaf pointer matches `self.root_ptr`
    ///
    /// Uses `Acquire` ordering to synchronize with `Release` stores in root updates.
    #[inline]
    fn check_is_main_root(&self, leaf_ptr: *mut L, root_flag_set: bool) -> bool {
        root_flag_set && {
            let current_root: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            StdPtr::eq(current_root, leaf_ptr)
        }
    }

    /// Prepare for a leaf split by gathering context and allocating the right sibling.
    ///
    /// # Arguments
    ///
    /// - `left_leaf`: Reference to the leaf being split (caller holds lock)
    /// - `left_leaf_ptr`: Raw pointer to the same leaf
    /// - `logical_pos`: Insert position for split point calculation
    /// - `ikey`: Key being inserted (used for split point calculation)
    ///
    /// # Returns
    ///
    /// * `Ok(SplitPreparation)` - Ready to proceed with split
    /// * `Err(InsertError::SplitFailed)` - Could not calculate split point
    ///
    /// # Split Point Calculation Failure
    ///
    /// `calculate_split_point` returns `None` when:
    /// - The leaf is unexpectedly empty (invariant violation)
    /// - The logical position is out of bounds
    /// - Internal permutation state is corrupted
    ///
    /// These represent bugs in the caller or tree corruption, not transient failures.
    ///
    /// Note: Allocations are infallible (abort on OOM like standard Rust).
    fn prepare_split(
        &self,
        left_leaf: &L,
        left_leaf_ptr: *mut L,
        logical_pos: usize,
        ikey: u64,
    ) -> Result<SplitPreparation, InsertError> {
        // Calculate split point - may fail if leaf state is corrupted
        let split_point: SplitPoint = left_leaf
            .calculate_split_point(logical_pos, ikey)
            .ok_or(InsertError::SplitFailed)?;

        // =========================================================================
        // CRITICAL: Capture root status BEFORE mark_split
        // =========================================================================
        //
        // SPLIT_UNLOCK_MASK clears ROOT_BIT on unlock. We must capture both
        // booleans separately BEFORE marking.
        let root_flag_set: bool = left_leaf.version().is_root();
        let parent_is_null: bool = left_leaf.parent().is_null();

        let is_main_root: bool = self.check_is_main_root(left_leaf_ptr, root_flag_set);
        let is_layer_root: bool = root_flag_set && parent_is_null && !is_main_root;

        // Invariant check: ROOT_BIT with non-null parent must be main root
        debug_assert!(
            !root_flag_set || parent_is_null || is_main_root,
            "Invalid state: ROOT_BIT set with non-null parent but not main root"
        );

        // =========================================================================
        // Allocate right sibling BEFORE mark_split
        // =========================================================================
        //
        // Allocation is infallible (aborts on OOM).
        // The leaf is initialized but NOT split-locked yet - that happens in
        // split_into_preallocated() or split_and_insert() after mark_split().
        let right_leaf_ptr: *mut L = self.allocator.alloc_leaf_direct(false, false);

        Ok(SplitPreparation {
            split_point,
            is_main_root,
            is_layer_root,
            right_leaf_ptr: right_leaf_ptr.cast(),
        })
    }

    /// Execute propagation and verify it succeeds.
    ///
    /// Propagation is designed to be infallible after `mark_split()`. Internode
    /// allocations abort on OOM rather than returning errors. This function
    /// wraps propagation and panics on unexpected failures to catch bugs early.
    ///
    /// # Panics
    ///
    /// Panics if propagation fails, as this indicates a bug in the propagation
    /// logic or tree corruption. The no-abandon invariant requires that once
    /// a split sibling is created, propagation MUST complete.
    #[expect(
        clippy::too_many_arguments,
        reason = "Propagation requires full context"
    )]
    fn execute_propagation(
        &self,
        left_leaf_ptr: *mut L,
        lock: LockGuard<'_>,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
        is_main_root: bool,
        is_layer_root: bool,
        guard: &LocalGuard<'_>,
    ) {
        let result = Propagation::make_split_leaf::<S, L, A>(
            &self.root_ptr,
            &self.allocator,
            left_leaf_ptr,
            lock,
            right_leaf_ptr,
            split_ikey,
            is_main_root,
            is_layer_root,
            guard,
        );

        // Propagation is infallible by design - internode allocations abort on OOM.
        // If this fails, it's a bug in the propagation logic or tree corruption.
        if let Err(e) = result {
            // Use cold path to avoid inlining panic code
            #[cold]
            #[inline(never)]
            #[expect(
                clippy::panic,
                reason = "Invariant violation - propagation must succeed"
            )]
            fn propagation_failed(e: &InsertError) -> ! {
                panic!(
                    "Propagation failed unexpectedly: {e:?}. \
                     This indicates a bug - propagation should be infallible after mark_split()"
                );
            }
            propagation_failed(&e);
        }
    }

    // ========================================================================
    // Generic Split Methods
    // ========================================================================

    /// Handle a leaf split with atomic insert.
    ///
    /// This function implements the ATOMIC SPLIT+INSERT pattern matching C++ Masstree:
    ///
    /// # Key Difference from split-then-retry
    ///
    /// Instead of splitting and requiring a retry:
    /// 1. Calculate split point
    /// 2. Allocate new leaf
    /// 3. Mark split in progress
    /// 4. Perform split AND insert atomically
    /// 5. Link leaves
    /// 6. Propagate to parent
    /// 7. **Return success - insert is complete!**
    ///
    /// # Benefits
    ///
    /// - Enables forward-sequential optimization (`split_pos == size`)
    /// - Right leaf is never empty (new key is always inserted)
    /// - No retry needed after split - insert completes in one operation
    ///
    /// # Arguments
    ///
    /// - `left_leaf_ptr`: Pointer to the leaf being split
    /// - `lock`: Lock guard (ownership transferred to propagation)
    /// - `logical_pos`: Insert position for the new key
    /// - `insert_data`: Key and value data to insert
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// * `Ok(SplitInsertResult)` - Split and insert completed successfully.
    ///   Note: Allocation is infallible (aborts on OOM like standard Rust).
    /// * `Err(InsertError::SplitFailed)` - Could not calculate split point (see
    ///   [`prepare_split`] for failure conditions)
    ///
    /// # Lock Protocol
    ///
    /// The left leaf's lock is maintained throughout propagation. The right
    /// sibling is created with a split-locked version in `split_and_insert()`.
    /// This prevents other threads from operating on the right sibling until
    /// its parent pointer is set during propagation.
    ///
    /// # C++ Reference
    ///
    /// Matches `tcursor::make_split()` in `reference/masstree_split.hh` combined with
    /// `leaf::split_into()` atomic insert.
    pub(crate) fn handle_leaf_split_and_insert_generic(
        &self,
        left_leaf_ptr: *mut L,
        mut lock: LockGuard<'_>,
        logical_pos: usize,
        insert_data: SplitInsertData<'_>,
        guard: &LocalGuard<'_>,
    ) -> Result<SplitInsertResult, InsertError> {
        // SAFETY: Caller guarantees left_leaf_ptr is valid and we hold the lock,
        // preventing concurrent modification. The guard protects against deallocation.
        let left_leaf: &L = unsafe { &*left_leaf_ptr };

        // =========================================================================
        // FALLIBLE PHASE: Prepare split context and allocate right sibling
        // =========================================================================
        let prep: SplitPreparation =
            self.prepare_split(left_leaf, left_leaf_ptr, logical_pos, insert_data.ikey)?;

        let right_leaf_ptr: *mut L = prep.right_leaf_ptr.cast();

        // =========================================================================
        // PAST POINT OF NO RETURN: mark_split() and beyond
        // =========================================================================
        //
        // After mark_split(), we MUST complete the split. All subsequent
        // allocations (internode in propagation) are infallible and abort on OOM.
        lock.mark_split();

        // SAFETY: We hold the lock on left_leaf, right_leaf_ptr is freshly allocated
        // and not yet visible to other threads. The guard protects memory reclamation.
        let result: SplitInsertResult = unsafe {
            left_leaf.split_and_insert(
                prep.split_point.pos,
                right_leaf_ptr,
                logical_pos,
                insert_data,
                guard,
            )
        };

        // SAFETY: We hold the lock, right_leaf_ptr is valid and properly initialized.
        unsafe { left_leaf.link_sibling(right_leaf_ptr) };

        // =========================================================================
        // INFALLIBLE: Propagation
        // =========================================================================
        //
        // Use result.split_ikey (from the actual split) for parent propagation.
        // This handles the forward-sequential case where the new key becomes
        // the separator.
        self.execute_propagation(
            left_leaf_ptr,
            lock,
            right_leaf_ptr,
            result.split_ikey,
            prep.is_main_root,
            prep.is_layer_root,
            guard,
        );

        Ok(result)
    }
}
