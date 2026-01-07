use crate::{SplitPoint, nodeversion::LockGuard};

use super::{
    AtomicOrdering, InsertError, LayerCapableLeaf, LocalGuard, MassTreeGeneric,
    NodeAllocatorGeneric, Propagation, ValueSlot,
};

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // ========================================================================
    // Generic Split Methods
    // ========================================================================

    /// Handle a leaf split when the leaf is full.
    ///
    /// This function implements the SPLIT-THEN-RETRY pattern:
    ///
    /// #  FALLIBLE: Allocation (Current Phase)
    ///
    /// The right sibling leaf is allocated using `try_alloc_leaf` BEFORE
    /// marking the split. If allocation fails, we return `Err(AllocationFailed)`
    /// without modifying the tree. The leaf remains full but consistent.
    ///
    /// # Steps
    ///
    /// 1. Calculate split point
    /// 2. **Fallible:** Allocate new leaf (BEFORE `mark_split`)
    /// 3. Mark split in progress
    /// 4. Perform split (sets split-locked version on right sibling)
    /// 5. Link leaves (B-link)
    /// 6. **Infallible:** Propagate to parent (internode allocations abort on OOM)
    /// 7. Return Ok - caller retries insert
    ///
    /// # Arguments
    ///
    /// - `left_leaf_ptr`: Pointer to the leaf being split
    /// - `lock`: Lock guard (ownership transferred to propagation)
    /// - `logical_pos`: Insert position for split point calculation
    /// - `ikey`: Key being inserted
    /// - `guard`: Memory reclamation guard
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Split completed, caller should retry insert
    /// * `Err(InsertError::AllocationFailed)` - Could not allocate sibling leaf
    ///
    /// # Lock Protocol
    ///
    /// The left leaf's lock is maintained throughout propagation. This is the
    /// key difference from the previous (broken) implementation that dropped
    /// the lock before propagation.
    ///
    /// # Split-Locked Right Sibling
    ///
    /// The right sibling is created with a split-locked version in
    /// `split_into_preallocated()`. This is NOT done by the allocator.
    /// The split-locked version prevents other threads from operating on
    /// the right sibling until its parent pointer is set.
    ///
    /// # C++ Reference
    ///
    /// Matches `tcursor::make_split()` in `reference/masstree_split.hh:179-297`.
    pub(crate) fn handle_leaf_split_generic(
        &self,
        left_leaf_ptr: *mut L,
        lock: LockGuard<'_>,
        logical_pos: usize,
        ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        let left_leaf: &L = unsafe { &*left_leaf_ptr };

        // Calculate split point
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

        let is_main_root: bool = root_flag_set && {
            let current_root: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            std::ptr::eq(current_root, left_leaf_ptr)
        };

        let is_layer_root: bool = root_flag_set && parent_is_null && !is_main_root;

        // =========================================================================
        // FALLIBLE POINT: Allocate right sibling BEFORE mark_split
        // =========================================================================
        //
        // This is the only fallible allocation in the split path (Tier 1).
        // If this fails, we return Err without modifying the tree.
        //
        // The leaf is initialized but NOT split-locked yet - that happens in
        // split_into_preallocated() after mark_split().
        let right_leaf_ptr: *mut L = self.allocator.try_alloc_leaf(false, false)?;

        // =========================================================================
        // PAST POINT OF NO RETURN: mark_split() and beyond
        // =========================================================================
        //
        // After mark_split(), we MUST complete the split. All subsequent
        // allocations (internode in propagation) are infallible and abort on OOM.
        let mut lock: LockGuard<'_> = lock;
        lock.mark_split();

        // Perform the split
        // NOTE: split_into_preallocated sets the split-locked version on right_leaf
        let (split_ikey, _) =
            unsafe { left_leaf.split_into_preallocated(split_point.pos, right_leaf_ptr, guard) };

        // Link leaves in B-link order
        unsafe { left_leaf.link_sibling(right_leaf_ptr) };

        // =========================================================================
        //   INFALLIBLE: Propagation (Tier 1)
        // =========================================================================
        //
        // Internode allocations during propagation remain infallible and abort
        // on OOM. This is acceptable because:
        // 1. No-abandon invariant requires completion
        // 2. Internode splits are rare
        // 3. The alternative (corruption) is worse

        let result: Result<(), InsertError> = Propagation::make_split_leaf::<S, L, A>(
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

        result
    }

    /// Propagate a leaf split to the parent.
    ///
    /// # Arguments
    /// * `is_layer_root` - True if the left leaf was a layer root BEFORE the lock was dropped.
    ///   This must be captured before `drop(lock)` because `SPLIT_UNLOCK_MASK` clears `ROOT_BIT`.
    ///
    /// # Help-Along Protocol
    ///
    /// The right sibling (`right_leaf_ptr`) is created with a split-locked version
    /// (`LOCK_BIT` | `SPLITTING_BIT` set). This function unlocks it after setting its
    /// parent pointer. This prevents other threads from trying to split the right
    /// sibling while its parent is NULL.
    ///
    /// All exit paths must call `(*right_leaf_ptr).version().unlock_for_split()`.
    /// Try to find the child index for a given child pointer in an internode.
    ///
    /// Returns `Some(index)` if found, `None` if not found. Use this in retry loops
    /// where not finding the child is a valid transient state during concurrent splits.
    #[allow(dead_code, reason = "traversal helper for future features")]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn try_find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> Option<usize> {
        use crate::leaf_trait::TreeInternode;

        let nkeys = parent.nkeys();
        (0..=nkeys).find(|&i| parent.child(i) == child)
    }

    /// Find the child index for a given child pointer in an internode.
    /// Panics if not found.
    #[allow(dead_code, reason = "traversal helper for future features")]
    #[expect(clippy::expect_used, reason = "FATAL: Fail Fast")]
    fn find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> usize {
        self.try_find_child_index_generic(parent, child)
            .expect("Child not found in parent internode")
    }
}
