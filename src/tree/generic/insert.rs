//! ========================================================================
//!  Generic Insert Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - `#[cold]` on retry/error paths
//! - Unified slot allocation and value update logic

use crate::leaf_trait::{SplitInsertData, TreeLeafNode};
use crate::leaf15::KSUF_KEYLENX;
use crate::policy::RetireHandle;

use super::{
    FindSlotResult, InsertError, InsertSearchResultGeneric, Key, LAYER_KEYLENX, LeafPolicy, Linker,
    LocalGuard, MassTreeGeneric, MembershipError, TreeAllocator, TreePermutation,
};

use crate::leaf15::LeafNode15;
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
//  Hot Path Helpers
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
    pub(crate) fn find_usable_slot(
        &self,
        leaf: &LeafNode15<P>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
        ikey: u64,
    ) -> FindSlotResult {
        // Check if leaf has space
        if perm.size() >= LeafNode15::<P>::WIDTH {
            return FindSlotResult::NeedsSplit;
        }

        // Get next free slot from back
        let slot: usize = perm.back();

        // Handle slot-0 rule: can't reuse slot-0 for different ikey
        if slot == 0 && !leaf.can_reuse_slot0(ikey) {
            let free_count = LeafNode15::<P>::WIDTH - perm.size();

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
    pub(crate) fn update_existing_value(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        new_value: &P::Output,
        guard: &LocalGuard<'_>,
    ) -> P::Output {
        // Load old value before overwrite.
        // load_value returns Option<P::Output>; slot is guaranteed occupied (Found path).
        let old_output: P::Output = leaf
            .load_value(slot)
            .expect("slot should have value in Found path");

        lock.mark_insert();

        // Store new value in place; get retirement handle for old data.
        // We hold the lock. Slot contains a terminal value (Found path).
        let retire: RetireHandle = leaf.update_in_place(slot, new_value);

        // Defer retirement of old value data if needed.
        // SAFETY: handle was produced by update_in_place() on this leaf.
        unsafe { P::retire_handle(retire, guard) };

        old_output
    }

    /// Insert a new value into a slot and update the permutation.
    ///
    /// Returns a raw pointer to a suffix bag that must be retired after
    /// releasing the lock, or null if no retirement is needed.
    ///
    /// # Safety
    ///
    /// Caller must hold the lock on `leaf` and have verified slot availability.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Slot assignment requires full context"
    )]
    pub(crate) fn insert_new_value(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        back_offset: usize,
        logical_pos: usize,
        mut perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        key: &Key<'_>,
        value: &P::Output,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> *mut u8 {
        // Assign the slot (returns deferred retire pointer)
        let deferred_retire: *mut u8 =
            self.assign_slot_generic(leaf, lock, slot, key, value, guard, pre_allocated);

        // Update permutation (perm is passed by value, mutate in place)
        if back_offset > 0 {
            let back_pos: usize = LeafNode15::<P>::WIDTH - 1;
            let chosen_pos: usize = back_pos - back_offset;
            perm.swap_free_slots(back_pos, chosen_pos);
        }

        let allocated: usize = perm.insert_from_back(logical_pos);
        debug_assert_eq!(allocated, slot, "allocated unexpected slot");

        // Use Relaxed ordering since we hold the lock - the lock's Release
        // fence on drop provides the necessary synchronization.
        leaf.set_permutation_relaxed(perm);

        deferred_retire
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
    pub(crate) fn can_reuse_empty_leaf(&self, leaf: &LeafNode15<P>, key: &Key<'_>) -> bool {
        // Leftmost leaves can always be reused - they'll set ikey_bound from the key
        // SAFETY: Called under lock - no concurrent retirement.
        if unsafe { leaf.prev_unguarded() }.is_null() {
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
    /// `Ok((None, deferred_ptr))` - insert succeeded, no old value (leaf was empty).
    /// The returned pointer must be retired via `retire_suffix_bag_ptr` after the lock is dropped.
    #[inline] // Not #[inline(always)] - empty leaf reuse is rare, avoid hot path bloat
    #[expect(
        clippy::type_complexity,
        reason = "Returns result + deferred retire pointer"
    )]
    #[expect(clippy::too_many_arguments, reason = "Internals")]
    fn insert_into_empty_leaf(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        key: &Key<'_>,
        value: &P::Output,
        guard: &LocalGuard<'_>,
        pre_allocated: Option<Vec<u8>>,
    ) -> (Result<Option<P::Output>, InsertError>, *mut u8) {
        // Clear empty state - leaf is being reactivated
        leaf.clear_empty_state();

        // Use slot 0 - it's always available in an empty leaf
        let slot: usize = 0;

        // Assign the slot with key and value
        let deferred_retire =
            self.assign_slot_generic(leaf, lock, slot, key, value, guard, pre_allocated);

        // Create permutation with single entry at position 0, using slot 0.
        // Use Relaxed ordering since we hold the lock - the lock's Release
        // fence on drop provides the necessary synchronization.
        let new_perm = <LeafNode15<P> as TreeLeafNode<P>>::Perm::make_sorted(1);
        leaf.set_permutation_relaxed(new_perm);

        (Ok(None), deferred_retire)
    }
}

// ============================================================================
//  Cold Path Helpers (Validation / Retry)
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Validate post-lock state: check if version or permutation changed.
    ///
    /// Returns `true` if validation passed, `false` if retry is needed.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    pub(crate) fn validate_post_lock(
        &self,
        leaf: &LeafNode15<P>,
        pre_lock_version: u32,
        pre_lock_perm_raw: <<LeafNode15<P> as TreeLeafNode<P>>::Perm as TreePermutation>::Raw,
    ) -> bool {
        !leaf.version().has_changed(pre_lock_version) && leaf.permutation_raw() == pre_lock_perm_raw
    }

    /// Validate membership: check if key should be in this leaf or a sibling.
    ///
    /// Returns `Ok(())` if key belongs here, `Err(retry_reason)` if we need
    /// to retry (split in progress, key moved to sibling, or key below lower bound).
    #[inline] // Not #[inline(always)] - 35+ lines with multiple branches
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    pub(crate) fn validate_membership(
        &self,
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
    ) -> Result<(), MembershipError> {
        // SAFETY: Called under lock - no concurrent retirement.
        let next_raw: *mut LeafNode15<P> = unsafe { leaf.next_raw_unguarded() };

        // Check for split in progress
        if Linker::is_marked(next_raw) {
            leaf.wait_for_split();
            return Err(MembershipError::SplitInProgress);
        }

        // Check lower bound for non-leftmost leaves.
        // If key < ikey_bound and prev exists, we're "too far right" due to
        // concurrent splits. Recovery requires restart from root (can't walk left).
        // Skip this check for leftmost leaves (prev == null) since they have no
        // lower bound constraint.
        // SAFETY: Called under lock - no concurrent retirement.
        if !unsafe { leaf.prev_unguarded() }.is_null() {
            let lower_bound: u64 = leaf.ikey_bound();
            if key.ikey() < lower_bound {
                return Err(MembershipError::KeyBelowLowerBound);
            }
        }

        // Check upper bound (key hasn't moved to right sibling)
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

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
    /// Note: Layer allocation is infallible (aborts on OOM like standard Rust).
    #[cold]
    #[rustfmt::skip]
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Layer creation requires full context"
    )]
    fn handle_suffix_conflict(
        &self,
        leaf: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        key: &mut Key<'_>,
        value: P::Output,
        guard: &LocalGuard<'_>,
    ) {
        // Mark insert before modifying the node
        lock.mark_insert();

        // Create new layer for the conflicting keys (infallible - aborts on OOM)
        //
        // SAFETY: We hold the lock on `leaf`, guard is from this tree's collector
        let layer_ptr: *mut u8 = unsafe {
            self.create_layer_concurrent_generic(leaf, slot, key, value, guard)
        };

        // Retire old terminal value, then install layer pointer.
        // SAFETY: We hold the lock. Slot transitions from terminal → layer.
        let retire: RetireHandle = leaf.take_value_for_layer(slot);
        // SAFETY: handle was produced by take_value_for_layer() on this leaf.
        unsafe { P::retire_handle(retire, guard) };

        // Clear any existing suffix before layer installation.
        // SAFETY: We hold the lock
        unsafe {
            leaf.clear_ksuf(slot, guard);
        };

        // Install layer: set keylenx FIRST, then the layer pointer.
        // store_layer only stores the pointer — keylenx must be set separately.
        leaf.set_keylenx(slot, LAYER_KEYLENX);
        leaf.store_layer(slot, layer_ptr);
    }
}

// ============================================================================
//  Main Insert Implementation
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
        value: P::Output,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<P::Output>, InsertError> {
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
            //
            // NOTE: We track leaf_ptr (*mut LeafNode15<P>) separately to preserve mutable provenance
            // for Miri/Stacked Borrows compliance. Pointers passed to handle_leaf_split_generic
            // must have mutable provenance since they're eventually freed via Box::from_raw.
            let mut leaf_ptr: *mut LeafNode15<P> =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            // B-link advance if needed (returns *mut LeafNode15<P> to preserve provenance)
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
            let mut leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            // Pre-allocated suffix storage, hoisted outside the forward loop
            // so it survives B-link retries instead of being re-allocated.
            let mut pre_allocated_vec: Option<Vec<u8>> = None;

            // Inner loop: local retry with B-link advance (no full traversal)
            'forward: loop {
                stat!(attempt);

                // Cache key.has_suffix() once per iteration — key state is
                // stable within a forward loop iteration (shift only happens
                // on Layer descent which does continue 'retry).
                let has_suffix: bool = key.has_suffix();

                // ================================================================
                // OPTIMISTIC SEARCH (C++ pattern: search BEFORE locking)
                // ================================================================
                // This reduces critical section time by doing search work
                // while other threads can still access the leaf.
                // C++ masstree_get.hh:85-87 does this before lock().
                let pre_lock_version: u32 = leaf.version().stable();
                let pre_lock_perm = leaf.permutation();
                let pre_lock_perm_raw = pre_lock_perm.value(); // Avoid second atomic load

                // Do optimistic search BEFORE locking
                let optimistic_search: InsertSearchResultGeneric = if single_layer_mode {
                    self.search_for_insert_single_layer(leaf, key, &pre_lock_perm)
                } else {
                    self.search_for_insert_generic(leaf, key, &pre_lock_perm)
                };

                // Pre-allocate suffix storage outside the lock if we predict
                // inline overflow. This moves the heap allocation out of the
                // critical section.
                //
                // SLACK SIZING: provision for the ENTIRE leaf (WIDTH * suffix_len)
                // so the resulting external bag has room for all future inserts,
                // making the rebuild a one-time event per leaf fill.
                //
                // Skip if we already have a pre-allocation from a previous
                // forward iteration (B-link retry reuse).
                if pre_allocated_vec.is_none() && has_suffix {
                    let suffix_len: usize = key.suffix().len();
                    let inline_capacity: usize = LeafNode15::<P>::INLINE_KSUF_CAPACITY;

                    let threshold: usize = if suffix_len > inline_capacity {
                        0
                    } else {
                        inline_capacity / suffix_len
                    };

                    if pre_lock_perm.size() >= threshold {
                        let estimated_capacity: usize = LeafNode15::<P>::WIDTH * suffix_len;
                        let mut v: Vec<u8> = Vec::new();
                        if v.try_reserve(estimated_capacity).is_ok() {
                            pre_allocated_vec = Some(v);
                        }
                    }
                }

                // Lock the leaf with yield-on-contention to reduce lock convoy effects.
                // Under high contention (e.g., `same` benchmark), pure spin causes:
                // - CPU burning while waiting for lock holder
                // - Thermal throttling on sustained loads
                // - Tail latency spikes when threads queue on same leaf
                let mut lock: LockGuard<'_> = leaf.version().lock_bounded();

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
                match self.validate_membership(leaf, key) {
                    Ok(()) => { /* continue to search */ }
                    Err(MembershipError::SplitInProgress | MembershipError::KeyMovedToSibling) => {
                        // Existing recovery: B-link advance (walk right)
                        stat!(membership_failure);
                        drop(lock);

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
                    Err(MembershipError::KeyBelowLowerBound) => {
                        // Cannot walk left - must restart from layer root
                        stat!(membership_failure);
                        drop(lock);
                        continue 'retry;
                    }
                }

                // ================================================================
                // EMPTY LEAF REUSE (Lazy Coalescing Optimization)
                // ================================================================
                // Check if this is an empty leaf that can be reused.
                // This is a fast path for insert-remove-reinsert workloads.
                // Use pre_lock_perm.size() instead of leaf.is_empty() to avoid
                // another atomic load (we already validated perm hasn't changed).
                if pre_lock_perm.size() == 0 && self.can_reuse_empty_leaf(leaf, key) {
                    let (result, deferred_retire) = self.insert_into_empty_leaf(
                        leaf,
                        &mut lock,
                        key,
                        &value,
                        guard,
                        pre_allocated_vec.take(),
                    );
                    drop(lock);
                    // Retire old suffix bag OUTSIDE the lock
                    if !deferred_retire.is_null() {
                        // SAFETY: deferred_retire came from assign_ksuf.
                        unsafe {
                            LeafNode15::<P>::retire_suffix_bag_ptr(deferred_retire, guard);
                        }
                    }
                    self.count.increment();
                    return result;
                }

                // ================================================================
                // DISPATCH ON SEARCH RESULT
                // ================================================================
                // Since validation passed (version and perm unchanged),
                // our optimistic search result is valid. No re-search needed!
                // This matches C++ masstree_get.hh which searches before lock().
                // Unified path for both single-layer and multi-layer modes.
                // In single-layer mode, Layer/Conflict are unreachable (dead code eliminated).
                match optimistic_search {
                    InsertSearchResultGeneric::Found { slot } => {
                        if leaf.is_value_empty(slot) {
                            stat!(null_ptr_retry);
                            drop(lock);
                            continue 'forward;
                        }

                        let old_value =
                            self.update_existing_value(leaf, &mut lock, slot, &value, guard);
                        stat!(successful_update);
                        drop(lock);
                        return Ok(Some(old_value));
                    }

                    InsertSearchResultGeneric::NotFound { logical_pos } => {
                        let ikey: u64 = key.ikey();

                        match self.find_usable_slot(leaf, &pre_lock_perm, ikey) {
                            FindSlotResult::Found { slot, back_offset } => {
                                let deferred_retire: *mut u8 = self.insert_new_value(
                                    leaf,
                                    &mut lock,
                                    slot,
                                    back_offset,
                                    logical_pos,
                                    pre_lock_perm,
                                    key,
                                    &value,
                                    guard,
                                    pre_allocated_vec.take(),
                                );
                                stat!(successful_insert);
                                drop(lock);

                                // Retire old suffix bag OUTSIDE the lock to avoid
                                // sys_membarrier syscall inside the critical section.
                                if !deferred_retire.is_null() {
                                    // SAFETY: deferred_retire came from assign_ksuf
                                    // and guard is from this tree's collector.
                                    unsafe {
                                        LeafNode15::<P>::retire_suffix_bag_ptr(
                                            deferred_retire,
                                            guard,
                                        );
                                    }
                                }

                                self.count.increment();
                                return Ok(None);
                            }

                            FindSlotResult::NeedsSplit => {
                                // Atomic split+insert - no retry needed!

                                // Compute keylenx from key's current length:
                                // - If has suffix (>8 bytes remaining), keylenx = KSUF_KEYLENX (64)
                                // - Otherwise keylenx = current_len as u8
                                let keylenx: u8 = if has_suffix {
                                    KSUF_KEYLENX
                                } else {
                                    #[expect(
                                        clippy::cast_possible_truncation,
                                        reason = "current_len <= 8 when !has_suffix"
                                    )]
                                    {
                                        key.current_len() as u8
                                    }
                                };

                                // Get suffix if present
                                let suffix: Option<&[u8]> =
                                    if has_suffix { Some(key.suffix()) } else { None };

                                // SplitInsertData<P> carries typed P::Output — no raw pointer conversion.
                                let insert_data = SplitInsertData {
                                    ikey,
                                    keylenx,
                                    suffix,
                                    value,
                                };

                                // Atomic split+insert (FALLIBLE allocation for sibling)
                                // SplitInsertData owns the typed value. On error, its Drop
                                // handles cleanup automatically via P::Output's Drop impl.
                                match self.handle_leaf_split_and_insert_generic(
                                    leaf_ptr,
                                    lock,
                                    logical_pos,
                                    &insert_data,
                                    guard,
                                ) {
                                    Ok(_result) => {}
                                    Err(e) => {
                                        // No cleanup needed — SplitInsertData<P> drops its
                                        // typed value automatically.
                                        return Err(e);
                                    }
                                }

                                // Insert completed during split - done!
                                stat!(successful_insert);
                                self.count.increment();
                                return Ok(None);
                            }
                        }
                    }

                    InsertSearchResultGeneric::Layer { slot, .. } => {
                        // Descend into sublayer - requires full traversal
                        // In single-layer mode, this branch is unreachable (search never returns Layer)
                        debug_assert!(
                            !single_layer_mode,
                            "single-layer search returned Layer variant"
                        );

                        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
                        drop(lock);

                        key.shift();
                        layer_root = layer_ptr;
                        in_sublayer = true;
                        continue 'retry;
                    }

                    InsertSearchResultGeneric::Conflict { slot } => {
                        // Handle suffix conflict by creating a new layer
                        // In single-layer mode, this branch is unreachable (search never returns Conflict)
                        debug_assert!(
                            !single_layer_mode,
                            "single-layer search returned Conflict variant"
                        );

                        self.handle_suffix_conflict(leaf, &mut lock, slot, key, value, guard);
                        stat!(successful_insert);
                        self.count.increment();
                        return Ok(None);
                    }
                }
            } // end 'forward
        } // end 'retry
    }
}
