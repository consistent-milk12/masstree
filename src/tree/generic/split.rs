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
    /// # Performance
    ///
    /// Marked `#[cold]` because splits are rare (~1 per WIDTH inserts).
    /// Marked `#[inline(never)]` to keep split code out of the hot insert path.
    /// 1. Calculate split point
    /// 2. Allocate new leaf (pre-allocation before marking split)
    /// 3. Mark split in progress
    /// 4. Perform split (creates split-locked right sibling)
    /// 5. Link leaves (B-link)
    /// 6. Propagate to parent using TRUE hand-over-hand
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
    #[cold]
    #[inline(never)]
    pub(crate) fn handle_leaf_split_generic(
        &self,
        left_leaf_ptr: *mut L,
        lock: crate::nodeversion::LockGuard<'_>,
        logical_pos: usize,
        ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        let left_leaf: &L = unsafe { &*left_leaf_ptr };

        // Calculate split point
        let split_point = left_leaf
            .calculate_split_point(logical_pos, ikey)
            .ok_or(InsertError::SplitFailed)?;

        // =========================================================================
        // CRITICAL: Capture root status BEFORE mark_split
        // =========================================================================
        //
        // SPLIT_UNLOCK_MASK clears ROOT_BIT on unlock. We must capture both
        // booleans separately BEFORE marking:
        //
        // - is_main_root: This leaf is THE main tree root (root_ptr points here)
        // - is_layer_root: This leaf is a layer root (null parent, root flag, NOT main)
        //
        // These are MUTUALLY EXCLUSIVE for handling:
        // - Main root: CAS on root_ptr to install new internode
        // - Layer root: NO CAS, just parent pointer updates

        let root_flag_set: bool = left_leaf.version().is_root();
        let parent_is_null: bool = left_leaf.parent().is_null();

        let is_main_root: bool = root_flag_set && {
            let current_root: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            std::ptr::eq(current_root, left_leaf_ptr)
        };

        // Layer root: has root flag, null parent, but is NOT the main tree root
        let is_layer_root: bool = root_flag_set && parent_is_null && !is_main_root;

        // Allocate new leaf directly from pool BEFORE marking split
        // This ensures allocation doesn't happen while we hold the split lock
        // The leaf is initialized but split_into_preallocated will set up the split-locked version
        let right_leaf_ptr: *mut L = self.allocator.alloc_leaf_direct(false, false);

        // Mark split in progress (sets SPLITTING_BIT)
        let mut lock = lock;
        lock.mark_split();

        // Perform the split
        // NOTE: The right sibling receives a split-locked version from
        // split_into_preallocated() - this is NOT done by the allocator!
        // insert_target is ignored - we use SPLIT-THEN-RETRY pattern
        let (split_ikey, _insert_target) =
            unsafe { left_leaf.split_into_preallocated(split_point.pos, right_leaf_ptr, guard) };

        // Link leaves in B-link order (while left is still locked)
        unsafe { left_leaf.link_sibling(right_leaf_ptr) };

        // =========================================================================
        // TRUE HAND-OVER-HAND PROPAGATION
        // =========================================================================
        //
        // Pass ownership of the lock to Propagation::make_split_leaf.
        // The lock is maintained throughout propagation - this is the key
        // difference from the previous (broken) implementation.

        let result: Result<(), InsertError> = Propagation::make_split_leaf::<S, L, A>(
            &self.root_ptr,
            &self.allocator,
            left_leaf_ptr,
            lock, // Ownership transferred - lock maintained during propagation
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
    /// ([`LOCK_BIT`] | [`SPLITTING_BIT`] set). This function unlocks it after setting its
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
