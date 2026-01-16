//! ========================================================================
//!  Generic Insert Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - `#[cold]` on retry/error paths
//! - Unified slot allocation and value update logic

use super::{
    InsertError, InsertSearchResultGeneric, Key, LAYER_KEYLENX, LayerCapableLeaf, Linker,
    LocalGuard, MassTreeGeneric, NodeAllocatorGeneric, TreePermutation, ValueSlot,
};

use crate::nodeversion::LockGuard;

// Conditional instrumentation macros - compile to no-ops when feature disabled
#[cfg(feature = "insert-stats")]
macro_rules! stat {
    (attempt) => {
        crate::insert_stats::record_attempt()
    };

    (validation_failure) => {
        crate::insert_stats::record_validation_failure()
    };

    (membership_failure) => {
        crate::insert_stats::record_membership_failure()
    };

    (null_ptr_retry) => {
        crate::insert_stats::record_null_ptr_retry()
    };

    (split_retry) => {
        crate::insert_stats::record_split_retry()
    };

    (successful_insert) => {
        crate::insert_stats::record_successful_insert()
    };

    (successful_update) => {
        crate::insert_stats::record_successful_update()
    };

    (hop_limit_exceeded) => {
        crate::insert_stats::record_hop_limit_exceeded()
    };

    (deleted_layer_retry) => {
        crate::insert_stats::record_deleted_layer_retry()
    };
}

#[cfg(not(feature = "insert-stats"))]
macro_rules! stat {
    ($name:ident) => {};
}

// ============================================================================
//  FindSlotResult - Slot allocation outcome
// ============================================================================

/// Result of finding a usable slot for insertion.
enum FindSlotResult {
    /// Found a usable slot at the given index with `back_offset` for permutation.
    Found { slot: usize, back_offset: usize },

    /// No usable slot available - need to split the leaf.
    NeedsSplit,
}

// ============================================================================
//  Hot Path Helpers
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Find a usable slot for inserting a new key.
    ///
    /// Handles the slot-0 rule: slot-0 stores `ikey_bound()` and cannot be
    /// reused for a different ikey.
    ///
    /// Returns `FindSlotResult::Found` with the slot and `back_offset`,
    /// or `FindSlotResult::NeedsSplit` if no usable slot is available.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn find_usable_slot(&self, leaf: &L, perm: &L::Perm, ikey: u64) -> FindSlotResult {
        // Check if leaf has space
        if perm.size() >= L::WIDTH {
            return FindSlotResult::NeedsSplit;
        }

        // Get next free slot from back
        let slot: usize = perm.back();

        // Handle slot-0 rule: can't reuse slot-0 for different ikey
        if slot == 0 && !leaf.can_reuse_slot0(ikey) {
            let free_count = L::WIDTH - perm.size();

            for offset in 1..free_count {
                let candidate: usize = perm.back_at_offset(offset);

                if candidate != 0 {
                    return FindSlotResult::Found {
                        slot: candidate,
                        back_offset: offset,
                    };
                }
            }

            // Only slot-0 available - trigger split
            return FindSlotResult::NeedsSplit;
        }

        let back_offset: usize = 0;
        FindSlotResult::Found { slot, back_offset }
    }

    /// Update an existing value in a slot.
    ///
    /// Returns the old value that was replaced.
    ///
    /// # Safety
    ///
    /// Caller must hold the lock on `leaf`.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn update_existing_value(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        slot: usize,
        new_value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> S::Output {
        let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        // Clone old value for return before we store new pointer.
        // SAFETY: old_ptr is non-null and came from output_to_raw
        let old_output: S::Output = unsafe { S::output_from_raw(old_ptr) };
        let new_ptr: *mut u8 = S::output_consume_to_raw(new_value);

        // Mark insert, store value
        // Use update_leaf_value_in_place for updates - this is an optimization
        // that skips the sentinel store for inline values (sentinel already exists).
        lock.mark_insert();
        leaf.update_leaf_value_in_place(slot, new_ptr);

        // Defer retirement of the old value (only for pointer-backed nodes).
        if S::NEEDS_RETIREMENT {
            // Use batched retirement to amortize coordination overhead
            // SAFETY: old_ptr came from output_to_raw and is valid for cleanup
            unsafe {
                crate::BatchedRetire::defer_value::<S>(old_ptr, guard);
            };
        }

        old_output
    }

    /// Insert a new value into a slot and update the permutation.
    ///
    /// # Safety
    ///
    /// Caller must hold the lock on `leaf` and have verified slot availability.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Slot assignment requires full context"
    )]
    fn insert_new_value(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        slot: usize,
        back_offset: usize,
        logical_pos: usize,
        perm: L::Perm,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) {
        // Assign the slot
        self.assign_slot_generic(leaf, lock, slot, key, value, guard);

        // Update permutation
        let mut new_perm = perm;

        if back_offset > 0 {
            let back_pos: usize = L::WIDTH - 1;
            let chosen_pos: usize = back_pos - back_offset;
            new_perm.swap_free_slots(back_pos, chosen_pos);
        }

        let allocated: usize = new_perm.insert_from_back(logical_pos);
        debug_assert_eq!(allocated, slot, "allocated unexpected slot");

        leaf.set_permutation(new_perm);
    }

    // ========================================================================
    //  Empty Leaf Reuse (Lazy Coalescing Optimization)
    // ========================================================================

    /// Check if an empty leaf can be reused for the given key.
    ///
    /// Empty leaves (from which all keys were removed) can be reused instead
    /// of allocating new leaves, which saves memory and improves performance.
    ///
    /// # Rules
    ///
    /// - Leftmost leaves (prev == null): Always reusable, they set `ikey_bound` on insert
    /// - Non-leftmost leaves: `ikey_bound` must match `key.ikey()` for B-link correctness
    /// - Caller must have already verified !`deleted_layer()`
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn can_reuse_empty_leaf(&self, leaf: &L, key: &Key<'_>) -> bool {
        // Leftmost leaves can always be reused - they'll set ikey_bound from the key
        if leaf.prev().is_null() {
            return true;
        }

        // Non-leftmost: ikey_bound must match for B-link chain correctness
        leaf.ikey_bound() == key.ikey()
    }

    /// Insert a value into an empty leaf, reusing it instead of allocating.
    ///
    /// This is the fast path for lazy coalescing: when a leaf becomes empty
    /// after removes, subsequent inserts can reuse it.
    ///
    /// # Returns
    ///
    /// `Ok(None)` - insert succeeded, no old value (leaf was empty)
    #[inline] // Not #[inline(always)] - empty leaf reuse is rare, avoid hot path bloat
    #[expect(clippy::unnecessary_wraps, reason = "Matches insert API return type")]
    fn insert_into_empty_leaf(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, InsertError> {
        // Clear empty state - leaf is being reactivated
        leaf.clear_empty_state();

        // Use slot 0 - it's always available in an empty leaf
        let slot: usize = 0;

        // Assign the slot with key and value
        self.assign_slot_generic(leaf, lock, slot, key, value, guard);

        // Create permutation with single entry at position 0, using slot 0
        let new_perm = L::Perm::make_sorted(1);
        leaf.set_permutation(new_perm);

        // Increment count
        self.count.increment();

        Ok(None)
    }
}

// ============================================================================
//  Cold Path Helpers (Validation / Retry)
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Validate post-lock state: check if version or permutation changed.
    ///
    /// Returns `true` if validation passed, `false` if retry is needed.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn validate_post_lock(
        &self,
        leaf: &L,
        pre_lock_version: u32,
        pre_lock_perm_raw: <L::Perm as TreePermutation>::Raw,
    ) -> bool {
        !leaf.version().has_changed(pre_lock_version) && leaf.permutation_raw() == pre_lock_perm_raw
    }

    /// Validate membership: check if key should be in this leaf or a sibling.
    ///
    /// Returns `Ok(())` if key belongs here, `Err(retry_reason)` if we need
    /// to retry (split in progress or key moved to sibling).
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn validate_membership(&self, leaf: &L, key: &Key<'_>) -> Result<(), MembershipError> {
        let next_raw: *mut L = leaf.next_raw();

        // Check for split in progress
        if Linker::is_marked(next_raw) {
            leaf.wait_for_split();
            return Err(MembershipError::SplitInProgress);
        }

        let next_ptr: *mut L = Linker::unmark_ptr(next_raw);

        if !next_ptr.is_null() {
            // SAFETY: next_ptr is a valid leaf pointer (protected by the guard).
            let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };

            if key.ikey() >= next_bound {
                return Err(MembershipError::KeyMovedToSibling);
            }
        }

        Ok(())
    }

    /// Handle a suffix conflict by creating a new layer.
    ///
    /// Takes ownership of `value` to avoid an unnecessary clone.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Layer created, value inserted
    /// * `Err(InsertError::AllocationFailed)` - Could not allocate layer nodes
    #[cold]
    #[rustfmt::skip]
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Layer creation requires full context"
    )]
    fn handle_suffix_conflict(
        &self,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        slot: usize,
        key: &mut Key<'_>,
        value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        // Mark insert before modifying the node
        lock.mark_insert();

        // Create new layer for the conflicting keys (FALLIBLE)
        //
        // SAFETY: We hold the lock on `leaf`, guard is from this tree's collector
        let layer_ptr: *mut u8 = unsafe {
            self.try_create_layer_concurrent_generic(leaf, slot, key, value, guard)?
        };

        // Retire the existing value in the conflict slot
        let old_ptr: *mut u8 = leaf.take_leaf_value_ptr(slot);

        if !old_ptr.is_null() && S::NEEDS_RETIREMENT {
            // Use batched retirement to amortize coordination overhead
            // SAFETY: old_ptr came from take_leaf_value_ptr
            unsafe {
                crate::BatchedRetire::defer_value::<S>(old_ptr, guard);
            }
        }

        // Clear any existing suffix
        //  SAFETY: We hold the lock
        unsafe {
            leaf.clear_ksuf(slot, guard);
        };

        // Install the layer pointer
        leaf.set_keylenx(slot, LAYER_KEYLENX);
        leaf.set_leaf_value_ptr(slot, layer_ptr);

        // Increment count
        self.count.increment();

        Ok(())
    }
}

/// Errors from membership validation.
enum MembershipError {
    /// A split is in progress - wait and retry.
    SplitInProgress,

    /// Key has moved to a sibling leaf - retry traversal.
    KeyMovedToSibling,
}

// ============================================================================
//  Main Insert Implementation
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Internal concurrent insert with optimistic locking.
    ///
    /// # Single-Layer Fast Path
    ///
    /// When the key is ≤ 8 bytes (no suffix), uses an optimized code path that:
    /// - Uses simplified search (no suffix/layer handling)
    /// - Only handles `Found` and `NotFound` results
    /// - Skips layer descent tracking
    ///
    /// Falls back to multi-layer path if a layer pointer is unexpectedly encountered.
    /// Concurrent insert with B-link local retry optimization.
    ///
    /// Uses two loops:
    /// - `'retry`: Full traversal from layer root (for initial entry, layer changes, hop limit)
    /// - `'forward`: Local retry with B-link advance (for version mismatch, membership failure, splits)
    ///
    /// This matches C++ `goto retry` vs `goto forward` semantics from `masstree_get.hh`.
    #[expect(
        clippy::too_many_lines,
        reason = "Complex concurrency logic with dual-loop structure"
    )]
    pub(super) fn insert_concurrent_generic(
        &self,
        key: &mut Key<'_>,
        value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, InsertError> {
        // Detect single-layer mode: key ≤ 8 bytes means no suffix, no layer operations
        let single_layer_mode: bool = !key.has_suffix();

        // Track current layer root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);

        // Track whether we're in a sublayer (for layer traversal)
        let mut in_sublayer: bool = false;

        // Outer loop: full traversal from layer_root
        'retry: loop {
            // Find the actual layer root (handles layer root promotion for sublayers)
            layer_root = self.maybe_parent_generic(layer_root);

            // Traverse to leaf
            // NOTE: We track leaf_ptr (*mut L) separately to preserve mutable provenance
            // for Miri/Stacked Borrows compliance. Pointers passed to handle_leaf_split_generic
            // must have mutable provenance since they're eventually freed via Box::from_raw.
            let mut leaf_ptr: *mut L =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            // B-link advance if needed (returns *mut L to preserve provenance)
            let (advanced_ptr, exceeded_hop_limit) =
                self.advance_to_key_by_bound_generic(leaf_ptr, key, guard);

            // If we exceeded the hop limit, re-traverse from root
            if exceeded_hop_limit {
                stat!(hop_limit_exceeded);
                layer_root = self.load_root_ptr_generic(guard);
                continue 'retry;
            }

            leaf_ptr = advanced_ptr;
            // SAFETY: leaf_ptr is valid, protected by guard
            let mut leaf: &L = unsafe { &*leaf_ptr };

            // Inner loop: local retry with B-link advance (no full traversal)
            'forward: loop {
                stat!(attempt);

                // ================================================================
                // OPTIMISTIC SEARCH (C++ pattern: search BEFORE locking)
                // ================================================================
                // This reduces critical section time by doing search work
                // while other threads can still access the leaf.
                // C++ masstree_get.hh:85-87 does this before lock().
                let pre_lock_version: u32 = leaf.version().stable();
                let pre_lock_perm_raw = leaf.permutation_raw();
                let pre_lock_perm = leaf.permutation();

                // Do optimistic search BEFORE locking
                let optimistic_search = if single_layer_mode {
                    self.search_for_insert_single_layer(leaf, key, &pre_lock_perm)
                } else {
                    self.search_for_insert_generic(leaf, key, &pre_lock_perm)
                };

                // Lock the leaf (pure spin - yields cause syscall overhead under contention)
                let mut lock = leaf.version().lock();

                // ================================================================
                // POST-LOCK VALIDATION (B-link local retry on failure)
                // ================================================================
                if !self.validate_post_lock(leaf, pre_lock_version, pre_lock_perm_raw) {
                    stat!(validation_failure);
                    drop(lock);

                    // LOCAL RETRY: B-link advance instead of full re-traversal
                    // This is the key optimization matching C++ masstree_get.hh:104
                    let (advanced_ptr, exceeded) =
                        self.advance_to_key_by_bound_generic(leaf_ptr, key, guard);

                    if exceeded {
                        // Too many hops, fall back to full re-traversal
                        stat!(hop_limit_exceeded);
                        layer_root = self.load_root_ptr_generic(guard);
                        continue 'retry;
                    }

                    leaf_ptr = advanced_ptr;
                    leaf = unsafe { &*leaf_ptr };
                    continue 'forward;
                }

                // Check for gc'd sublayer - must restart from main tree root
                // C++ masstree_get.hh:111-115
                if leaf.deleted_layer() {
                    stat!(deleted_layer_retry);
                    drop(lock);
                    key.unshift_all();
                    layer_root = self.load_root_ptr_generic(guard);
                    in_sublayer = false;
                    continue 'retry;
                }

                // Post-lock membership check
                if self.validate_membership(leaf, key).is_err() {
                    stat!(membership_failure);
                    drop(lock);

                    // LOCAL RETRY: B-link advance
                    let (advanced_ptr, exceeded) =
                        self.advance_to_key_by_bound_generic(leaf_ptr, key, guard);

                    if exceeded {
                        stat!(hop_limit_exceeded);
                        layer_root = self.load_root_ptr_generic(guard);
                        continue 'retry;
                    }

                    leaf_ptr = advanced_ptr;
                    leaf = unsafe { &*leaf_ptr };
                    continue 'forward;
                }

                // ================================================================
                // EMPTY LEAF REUSE (Lazy Coalescing Optimization)
                // ================================================================
                // Check if this is an empty leaf that can be reused.
                // This is a fast path for insert-remove-reinsert workloads.
                if leaf.is_empty() && self.can_reuse_empty_leaf(leaf, key) {
                    let result = self.insert_into_empty_leaf(leaf, &mut lock, key, &value, guard);
                    drop(lock);
                    return result;
                }

                // ================================================================
                // USE OPTIMISTIC SEARCH RESULT
                // ================================================================
                // Since validation passed (version and perm unchanged),
                // our optimistic search result is valid. No re-search needed!
                // This matches C++ masstree_get.hh which searches before lock().
                let perm = pre_lock_perm;
                let search_result = optimistic_search;

                // ================================================================
                // Single-layer fast path (keys ≤ 8 bytes)
                // ================================================================
                if single_layer_mode {
                    match search_result {
                        InsertSearchResultGeneric::Found { slot } => {
                            let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                            if old_ptr.is_null() {
                                stat!(null_ptr_retry);
                                drop(lock);
                                continue 'forward;
                            }

                            let old_value =
                                self.update_existing_value(leaf, &mut lock, slot, value, guard);
                            stat!(successful_update);
                            drop(lock);
                            return Ok(Some(old_value));
                        }

                        InsertSearchResultGeneric::NotFound { logical_pos } => {
                            let ikey: u64 = key.ikey();

                            match self.find_usable_slot(leaf, &perm, ikey) {
                                FindSlotResult::Found { slot, back_offset } => {
                                    self.insert_new_value(
                                        leaf,
                                        &mut lock,
                                        slot,
                                        back_offset,
                                        logical_pos,
                                        perm,
                                        key,
                                        &value,
                                        guard,
                                    );

                                    stat!(successful_insert);
                                    drop(lock);
                                    self.count.increment();

                                    return Ok(None);
                                }

                                FindSlotResult::NeedsSplit => {
                                    stat!(split_retry);
                                    // Split the leaf (FALLIBLE allocation for sibling)
                                    // Use leaf_ptr directly to preserve mutable provenance
                                    self.handle_leaf_split_generic(
                                        leaf_ptr,
                                        lock,
                                        logical_pos,
                                        key.ikey(),
                                        guard,
                                    )?;

                                    // After split, use LOCAL retry (B-link advance)
                                    // The key likely moved to the new right sibling
                                    let (advanced_ptr, exceeded) =
                                        self.advance_to_key_by_bound_generic(leaf_ptr, key, guard);

                                    if exceeded {
                                        stat!(hop_limit_exceeded);
                                        layer_root = self.load_root_ptr_generic(guard);
                                        continue 'retry;
                                    }

                                    leaf_ptr = advanced_ptr;
                                    leaf = unsafe { &*leaf_ptr };
                                    continue 'forward;
                                }
                            }
                        }

                        // These can't happen in single-layer mode
                        InsertSearchResultGeneric::Layer { .. }
                        | InsertSearchResultGeneric::Conflict { .. } => {
                            unreachable!("single-layer search never returns Layer or Conflict")
                        }
                    }
                }

                // ================================================================
                // Multi-layer path (handles layer descent and conflicts)
                // ================================================================
                // Note: search_result is already set from optimistic search above

                match search_result {
                    InsertSearchResultGeneric::Found { slot } => {
                        let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                        if old_ptr.is_null() {
                            stat!(null_ptr_retry);
                            drop(lock);
                            continue 'forward;
                        }

                        let old_value =
                            self.update_existing_value(leaf, &mut lock, slot, value, guard);
                        stat!(successful_update);
                        drop(lock);
                        return Ok(Some(old_value));
                    }

                    InsertSearchResultGeneric::NotFound { logical_pos } => {
                        let ikey: u64 = key.ikey();

                        match self.find_usable_slot(leaf, &perm, ikey) {
                            FindSlotResult::Found { slot, back_offset } => {
                                self.insert_new_value(
                                    leaf,
                                    &mut lock,
                                    slot,
                                    back_offset,
                                    logical_pos,
                                    perm,
                                    key,
                                    &value,
                                    guard,
                                );
                                stat!(successful_insert);
                                drop(lock);
                                self.count.increment();
                                return Ok(None);
                            }

                            FindSlotResult::NeedsSplit => {
                                stat!(split_retry);
                                // Use leaf_ptr directly to preserve mutable provenance
                                self.handle_leaf_split_generic(
                                    leaf_ptr,
                                    lock,
                                    logical_pos,
                                    ikey,
                                    guard,
                                )?;

                                // After split, use LOCAL retry (B-link advance)
                                let (advanced_ptr, exceeded) =
                                    self.advance_to_key_by_bound_generic(leaf_ptr, key, guard);

                                if exceeded {
                                    stat!(hop_limit_exceeded);
                                    layer_root = self.load_root_ptr_generic(guard);
                                    continue 'retry;
                                }

                                leaf_ptr = advanced_ptr;
                                leaf = unsafe { &*leaf_ptr };
                            }
                        }
                    }

                    InsertSearchResultGeneric::Layer { slot, .. } => {
                        // Descend into sublayer - requires full traversal
                        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                        drop(lock);

                        key.shift();
                        layer_root = layer_ptr;
                        in_sublayer = true;
                        continue 'retry;
                    }

                    InsertSearchResultGeneric::Conflict { slot } => {
                        // Handle suffix conflict by creating a new layer (FALLIBLE)
                        self.handle_suffix_conflict(leaf, &mut lock, slot, key, value, guard)?;
                        stat!(successful_insert);
                        return Ok(None);
                    }
                }
            } // end 'forward
        } // end 'retry
    }
}
