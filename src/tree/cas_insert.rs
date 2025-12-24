//! CAS-based lock-free insert for [`MassTree`].
//!
//! Provides a fast path for simple inserts that can be completed with atomic
//! slot reservation followed by a compare-and-swap on the permutation.
//!
//! # Protocol
//!
//! ```text
//! 1. Optimistic traversal to find target leaf
//! 2. Get stable version and permutation
//! 3. Search for key position
//! 4. CAS slot value from NULL to claim (NULL-claim semantics)
//! 5. Validate version before writing key data
//! 6. Store key data (ikey, keylenx)
//! 7. Validate version again
//! 8. CAS permutation to atomically publish the insert
//! 9. On failure, restore slot to NULL and retry or fall back
//! ```
//!
//! # Atomic Slot Reservation (NULL-Claim Semantics)
//!
//! Two threads reading the same stale permutation will compute the same
//! `back()` slot. To prevent "slot stealing" where a stale thread overwrites
//! a published entry, we enforce **NULL-claim semantics**:
//!
//! - Slot CAS is always `NULL → our_ptr` (never `existing_ptr → our_ptr`)
//! - If slot is not NULL, treat as contention and retry with fresh state
//! - Only the first thread to CAS from NULL succeeds; others retry
//!
//! This prevents the bug where a stale thread could CAS from another
//! thread's pointer to its own, corrupting the published entry.
//!
//! # When CAS Succeeds
//!
//! - Key is new (not found in leaf)
//! - Leaf has space (`size < WIDTH`)
//! - No slot-0 violation (or slot-0 can be reused)
//! - No suffix needed (inline key ≤ 8 bytes)
//! - Slot CAS succeeds (atomic claim)
//! - Version stable during CAS window
//! - Permutation CAS succeeds
//!
//! # When Falls Back to Locked Path
//!
//! - Key exists (need value update with old value return)
//! - Leaf full (need split)
//! - Layer/suffix conflict (need layer creation)
//! - Slot-0 violation (needs swap logic)
//! - High contention (> `MAX_CAS_RETRIES` failures)

use std::ptr as StdPtr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use seize::LocalGuard;

use crate::alloc::NodeAllocator;
use crate::key::Key;
use crate::leaf::{LeafNode, LeafValue};
use crate::tracing_helpers::{debug_log, trace_log};

use super::MassTree;
use super::locked::InsertSearchResult;

/// Maximum CAS retry attempts before falling back to locked path.
const MAX_CAS_RETRIES: usize = 3;

/// Exponential backoff for CAS retries.
///
/// Reduces contention by spacing out retry attempts. Each retry doubles the
/// number of spin iterations, capped at 64 to avoid excessive latency.
#[inline(always)]
fn backoff(retries: usize) {
    // Exponential: 1, 2, 4, 8, 16, 32, 64, 64, ...
    let spins = 1usize << retries.min(6);

    for _ in 0..spins {
        std::hint::spin_loop();
    }
}

/// Result of a CAS insert attempt.
#[derive(Debug)]
pub(super) enum CasInsertResult<V> {
    /// CAS insert succeeded.
    Success(Option<Arc<V>>),

    /// Key already exists - need locked update to return old value.
    ExistsNeedLock {
        /// Slot where key exists.
        #[allow(dead_code)]
        slot: usize,
    },

    /// Leaf is full - need locked split.
    FullNeedLock,

    /// Layer creation needed - need locked insert.
    LayerNeedLock {
        /// Slot where layer is needed.
        #[allow(dead_code)]
        slot: usize,
    },

    /// Slot-0 violation - need locked path for swap logic.
    ///
    /// This is NOT transient contention. Slot-0 stores `ikey_bound` and can only
    /// be reused if the new key's ikey matches the bound. The locked path has
    /// swap logic to handle this.
    Slot0NeedLock,

    /// High contention or complex case - fall back to locked path.
    ContentionFallback,
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Try CAS-based lock-free insert with atomic slot reservation.
    ///
    /// Attempts to insert a new key-value pair using optimistic concurrency:
    /// 1. CAS slot value to atomically claim the slot
    /// 2. Store key data (ikey, keylenx)
    /// 3. CAS the permutation to atomically publish
    ///
    /// Returns `CasInsertResult` indicating success or reason for fallback.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert
    /// * `value` - The value to insert (wrapped in Arc)
    /// * `guard` - Seize guard for memory safety
    ///
    /// # Returns
    ///
    /// * `Success(None)` - New key inserted successfully
    /// * `ExistsNeedLock` - Key exists, need locked update
    /// * `FullNeedLock` - Leaf full, need split
    /// * `LayerNeedLock` - Layer conflict, need layer creation
    /// * `ContentionFallback` - Too many retries or complex case
    #[expect(clippy::too_many_lines, reason = "Comples concurrency logic")]
    pub(super) fn try_cas_insert(
        &self,
        key: &Key<'_>,
        value: &Arc<V>,
        guard: &LocalGuard<'_>,
    ) -> CasInsertResult<V> {
        let ikey: u64 = key.ikey();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "current_len() <= 8 at each layer"
        )]
        let keylenx: u8 = key.current_len() as u8;

        // Suffix keys require locked path (ksuf allocation is complex)
        if key.has_suffix() {
            return CasInsertResult::ContentionFallback;
        }

        let mut retries: usize = 0;
        let mut leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = StdPtr::null_mut();
        let mut use_reach: bool = true;

        loop {
            // 1. Optimistic traversal to find target leaf
            if use_reach {
                let mut layer_root: *const u8 = self.load_root_ptr(guard);
                layer_root = self.maybe_parent(layer_root);
                leaf_ptr = self.reach_leaf_concurrent(layer_root, key, guard);
            } else {
                use_reach = true;
            }

            let leaf: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*leaf_ptr };

            // If we raced with a split, advance to the correct leaf via B-link.
            let advanced: &LeafNode<LeafValue<V>, WIDTH> =
                self.advance_to_key_by_bound(leaf, key, guard);

            if !StdPtr::eq(advanced, leaf) {
                leaf_ptr = StdPtr::from_ref(advanced).cast_mut();
                use_reach = false;
                continue;
            }

            // 2. Get version (fail-fast if dirty)
            // PERF: Don't call stable() which spins. If a writer has the lock,
            // immediately fall back to locked path instead of spinning.
            // This eliminates the convoy problem where 32 threads spin waiting
            // for one writer to finish.
            let version: u32 = leaf.version().value();
            if leaf.version().is_dirty() {
                trace_log!(ikey, leaf_ptr = ?leaf_ptr, "CAS insert: version dirty, falling back");
                return CasInsertResult::ContentionFallback;
            }

            // Use permutation_try() for freeze safety.
            // If a split is in progress (frozen), fall back to locked path.
            let Ok(perm) = leaf.permutation_try() else {
                trace_log!(ikey, leaf_ptr = ?leaf_ptr, "CAS insert: permutation frozen, falling back");
                return CasInsertResult::ContentionFallback;
            };

            // 3. Search for key position
            let search_result = self.search_for_insert(leaf, key, &perm);

            match search_result {
                InsertSearchResult::Found { slot } => {
                    // Key exists - need locked update to return old value
                    return CasInsertResult::ExistsNeedLock { slot };
                }

                InsertSearchResult::Layer { slot, .. } | InsertSearchResult::Conflict { slot } => {
                    // Layer descent or suffix conflict - need locked path
                    return CasInsertResult::LayerNeedLock { slot };
                }

                InsertSearchResult::NotFound { logical_pos } => {
                    // 4. Check if leaf has space
                    if perm.size() >= WIDTH {
                        return CasInsertResult::FullNeedLock;
                    }

                    // 5. Check slot-0 rule
                    let next_free: usize = perm.back();
                    if next_free == 0 && !leaf.can_reuse_slot0(ikey) {
                        // Slot-0 violation requires swap logic - use locked path
                        // This is NOT transient contention - retrying won't help
                        trace_log!(ikey, leaf_ptr = ?leaf_ptr, next_free, "CAS insert: slot-0 violation, need locked path");
                        return CasInsertResult::Slot0NeedLock;
                    }

                    // 6. Compute new permutation (immutable)
                    let (new_perm, slot) = perm.insert_from_back_immutable(logical_pos);

                    // 7. Prepare our Arc pointer
                    let arc_ptr: *mut u8 = Arc::into_raw(Arc::clone(value)) as *mut u8;

                    trace_log!(
                        ikey,
                        slot,
                        leaf_ptr = ?leaf_ptr,
                        perm_size = perm.size(),
                        "CAS insert: attempting to claim slot"
                    );

                    // FIXED: Enforce NULL-claim semantics.
                    //
                    // CRITICAL: Only claim slots that are empty (NULL). Two threads reading
                    // the same stale permutation will compute the same back() slot. Without
                    // NULL-claim, a stale thread can CAS from ptrA→ptrB, "stealing" a slot
                    // that was already published by another thread. This causes:
                    //   1. The published entry to have wrong data
                    //   2. Keys to go missing after successful insert
                    //
                    // With NULL-claim, only the first thread to CAS NULL→ptr succeeds.
                    // Other threads get contention fallback and retry with fresh state.
                    if let Err(_actual) = leaf.cas_slot_value(slot, StdPtr::null_mut(), arc_ptr) {
                        // Slot already claimed by another thread.
                        // This is expected contention - fall back to get fresh state.
                        // SAFETY: we just created this Arc clone, nobody else has it
                        let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };

                        debug_log!(
                            ikey,
                            slot,
                            leaf_ptr = ?leaf_ptr,
                            "CAS insert: slot not NULL, contention detected"
                        );

                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            trace_log!(ikey, leaf_ptr = ?leaf_ptr, retries, "CAS insert: max retries on slot CAS, falling back");
                            return CasInsertResult::ContentionFallback;
                        }
                        // Backoff before retry to reduce contention
                        backoff(retries);
                        continue;
                    }
                    // FIX: Validate version BEFORE writing key data.
                    //
                    // If we write key data before validation and the version
                    // changed, we may corrupt slot data that another thread
                    // is using. Check version first, then write only if stable.
                    //
                    // CRITICAL: Use has_changed_or_locked() instead of has_changed().
                    // has_changed() ignores INSERTING_BIT, allowing CAS inserts to
                    // race with locked splits. has_changed_or_locked() checks both
                    // version change AND if INSERTING_BIT is set.
                    if leaf.version().has_changed_or_locked(version) {
                        debug_log!(
                            ikey,
                            slot,
                            leaf_ptr = ?leaf_ptr,
                            "CAS insert: version changed after slot claim, aborting (check 1)"
                        );
                        // Version changed - restore slot to NULL and retry
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                // SAFETY: we just created this Arc clone
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }

                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            return CasInsertResult::ContentionFallback;
                        }
                        backoff(retries);
                        continue;
                    }

                    // Version stable - now safe to write key data
                    // SAFETY: we own the slot after successful CAS and version is stable
                    unsafe {
                        leaf.store_key_data_for_cas(slot, ikey, keylenx);
                    }

                    // 9. Secondary version validation (belt-and-suspenders)
                    // This catches any races in the window between the first check
                    // and the permutation CAS.
                    //
                    // CRITICAL: Use has_changed_or_locked() to detect locked splits.
                    if leaf.version().has_changed_or_locked(version) {
                        debug_log!(
                            ikey,
                            slot,
                            leaf_ptr = ?leaf_ptr,
                            "CAS insert: version changed after key write, aborting (check 2)"
                        );
                        // Version changed - try to restore slot to NULL
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) => {
                                // Restored successfully, can reclaim Arc and retry
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                            Err(_) => {
                                // Slot was stolen - just reclaim our Arc
                                // The slot now belongs to someone else
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }

                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            return CasInsertResult::ContentionFallback;
                        }
                        backoff(retries);
                        continue;
                    }

                    // 10. Verify slot ownership before publishing
                    // Another thread could have stolen our slot by CAS'ing from our
                    // value to their value. This can happen when they retry with the
                    // same permutation (which hasn't changed yet).
                    if leaf.load_slot_value(slot) != arc_ptr {
                        // Slot was stolen - just reclaim our Arc
                        // Don't try to restore slot, it belongs to someone else
                        let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };

                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            return CasInsertResult::ContentionFallback;
                        }
                        backoff(retries);
                        continue;
                    }

                    // 11. Final version check before permutation CAS
                    // A locked writer could have acquired the lock between our secondary
                    // check and now. Check again to avoid racing with split_into().
                    if leaf.version().has_changed_or_locked(version) {
                        trace_log!(
                            ikey,
                            slot,
                            "CAS insert: version changed before perm CAS, aborting"
                        );
                        match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                            }
                        }
                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            return CasInsertResult::ContentionFallback;
                        }
                        backoff(retries);
                        continue;
                    }

                    // 12. CAS the permutation to publish
                    //
                    // Use cas_permutation_raw() for freeze safety.
                    // If the CAS fails due to frozen state, fall back to locked path.

                    // TEST HOOK: Allow tests to inject barriers before CAS publish
                    #[cfg(test)]
                    super::test_hooks::call_before_cas_publish_hook();

                    match leaf.cas_permutation_raw(perm, new_perm) {
                        Ok(()) => {
                            // Permutation CAS succeeded - verify slot wasn't stolen
                            // in the tiny window between ownership check and perm CAS
                            if leaf.load_slot_value(slot) != arc_ptr {
                                // Slot was stolen after we published! This is rare but possible.
                                // The entry is in the permutation but has wrong data.
                                //
                                // CRITICAL: We MUST increment count here because:
                                // 1. Our permutation CAS succeeded - slot is now visible in tree
                                // 2. Our key metadata (ikey, keylenx) is in the slot
                                // 3. The locked path retry will find "key exists" and do UPDATE
                                // 4. Updates don't increment count (not a new key)
                                //
                                // If we don't increment here, the key ends up visible but uncounted.
                                // The locked path will fix the value, but we own the count increment.
                                self.count.fetch_add(1, Ordering::Relaxed);
                                debug_log!(
                                    ikey,
                                    slot,
                                    leaf_ptr = ?leaf_ptr,
                                    "CAS insert: slot stolen after publish, count incremented, falling back for value fix"
                                );
                                // Note: we don't reclaim our Arc - it's been overwritten
                                return CasInsertResult::ContentionFallback;
                            }

                            // FIXED: Detect concurrent split after permutation publish.
                            //
                            // A split could have started between our final version check
                            // and the permutation CAS. There are two distinct scenarios:
                            //
                            // A) SPLIT MOVED OUR ENTRY:
                            //    - split_into() read our permutation, copied our entry to right leaf
                            //    - split_into() then truncated left leaf's permutation
                            //    - Our slot is NO LONGER in the current permutation
                            //    - Our entry IS in the tree (in the right leaf)
                            //    - This is SUCCESS - increment count and return
                            //
                            // B) ORPHAN CREATED:
                            //    - Our CAS succeeded but split changed routing
                            //    - Our slot IS still in the current permutation
                            //    - But ikey >= split_ikey means routing goes to right leaf
                            //    - Our entry is unreachable - fall back to locked path
                            //
                            // We distinguish by checking if our slot is in the current perm.
                            {
                                use crate::leaf::link::{is_marked, unmark_ptr};

                                // Helper: check if slot is in permutation
                                let slot_in_perm =
                                    |perm: &crate::permuter::Permuter<WIDTH>, s: usize| -> bool {
                                        for i in 0..perm.size() {
                                            if perm.get(i) == s {
                                                return true;
                                            }
                                        }
                                        false
                                    };

                                // Check 1: Is a SPLIT in progress?
                                if leaf.version().is_splitting() {
                                    // Split in progress - wait for it to complete
                                    // so we can properly determine if we were moved or orphaned
                                    let _ = leaf.version().stable();
                                }

                                let next_raw = leaf.next_raw();

                                // Check 2: Is the B-link being set up?
                                if is_marked(next_raw) {
                                    // Wait for link completion
                                    leaf.wait_for_split();
                                }

                                // Now check the current permutation state.
                                // Use permutation_wait() since we need a valid perm for checking.
                                // At this point the split should have completed (we waited above).
                                let current_perm = leaf.permutation_wait();

                                // Check if our slot is still in the permutation
                                if !slot_in_perm(&current_perm, slot) {
                                    // SCENARIO A: Split moved our entry!
                                    // Our slot was in our new_perm, but split_into() truncated
                                    // the permutation. This means split_into() copied our entry
                                    // to the right leaf before truncating.
                                    //
                                    // Our entry IS in the tree (in the right leaf), so this is
                                    // a successful insert. Increment count and return Success.
                                    debug_log!(
                                        ikey,
                                        slot,
                                        leaf_ptr = ?leaf_ptr,
                                        "CAS insert: slot removed by split (entry moved to right leaf), counting as success"
                                    );
                                    self.count.fetch_add(1, Ordering::Relaxed);
                                    return CasInsertResult::Success(None);
                                }

                                // Our slot IS in the current permutation.
                                // Check if our key belongs in a right sibling.
                                let next_ptr = unmark_ptr(next_raw);
                                if !next_ptr.is_null() {
                                    // SAFETY: next_ptr is valid (not null, unmarked after wait)
                                    let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
                                    if ikey >= next_bound {
                                        // SCENARIO B: Orphan created
                                        // Our slot is in the permutation, but our ikey >= split_ikey.
                                        // This means routing will go to the right leaf, but our
                                        // entry is in the left leaf - unreachable!
                                        //
                                        // Fall back to locked path which will insert correctly.
                                        debug_log!(
                                            ikey,
                                            slot,
                                            leaf_ptr = ?leaf_ptr,
                                            next_bound,
                                            "CAS insert: orphan created (slot in perm but ikey >= next_bound), falling back"
                                        );
                                        // Note: we don't increment count - the fallback will handle it
                                        return CasInsertResult::ContentionFallback;
                                    }
                                }
                            }

                            debug_log!(
                                ikey,
                                slot,
                                leaf_ptr = ?leaf_ptr,
                                old_perm_size = perm.size(),
                                new_perm_size = new_perm.size(),
                                "CAS insert: permutation CAS succeeded"
                            );

                            // Success! Increment count
                            self.count.fetch_add(1, Ordering::Relaxed);
                            return CasInsertResult::Success(None);
                        }

                        Err(failure) => {
                            // Permutation CAS failed - try to restore slot to NULL
                            debug_log!(
                                ikey,
                                slot,
                                leaf_ptr = ?leaf_ptr,
                                old_perm_size = perm.size(),
                                "CAS insert: permutation CAS FAILED, restoring slot"
                            );

                            // Rollback: restore slot to NULL and reclaim Arc
                            match leaf.cas_slot_value(slot, arc_ptr, StdPtr::null_mut()) {
                                Ok(()) | Err(_) => {
                                    // SAFETY: we created this Arc clone, it's ours to reclaim
                                    let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };
                                }
                            }

                            // Check if failure was due to frozen state (split in progress)
                            if failure.is_frozen::<WIDTH>() {
                                trace_log!(
                                    ikey,
                                    slot,
                                    leaf_ptr = ?leaf_ptr,
                                    "CAS insert: permutation frozen during CAS, falling back"
                                );
                                return CasInsertResult::ContentionFallback;
                            }

                            retries += 1;
                            if retries > MAX_CAS_RETRIES {
                                return CasInsertResult::ContentionFallback;
                            }
                            // Backoff before retry
                            backoff(retries);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "fail fast on tests")]
mod tests {
    use super::*;
    use crate::MassTree;

    #[test]
    fn test_cas_insert_basic() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        // CAS insert should work for simple keys
        let key = Key::new(b"hello");
        let value = Arc::new(42u64);

        let result = tree.try_cas_insert(&key, &value, &guard);
        assert!(matches!(result, CasInsertResult::Success(None)));

        // Verify the value was inserted
        assert_eq!(*tree.get(b"hello").unwrap(), 42);
    }

    #[test]
    fn test_cas_insert_existing_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert first (before taking guard)
        let _ = tree.insert(b"key1", 100);

        let guard = tree.guard();

        // CAS insert for existing key should return ExistsNeedLock
        let key = Key::new(b"key1");
        let value = Arc::new(200u64);

        let result = tree.try_cas_insert(&key, &value, &guard);
        assert!(matches!(result, CasInsertResult::ExistsNeedLock { .. }));
    }

    #[test]
    fn test_cas_insert_suffix_key_falls_back() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        // Long key with suffix should fall back
        let key = Key::new(b"this_is_a_very_long_key_with_suffix");
        let value = Arc::new(42u64);

        let result = tree.try_cas_insert(&key, &value, &guard);
        assert!(matches!(result, CasInsertResult::ContentionFallback));
    }

    #[test]
    fn test_cas_insert_multiple_keys() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        // Insert multiple short keys via CAS
        for i in 0..10u64 {
            let key_bytes = format!("k{i}");
            let key = Key::new(key_bytes.as_bytes());
            let value = Arc::new(i * 100);

            let result = tree.try_cas_insert(&key, &value, &guard);
            assert!(
                matches!(result, CasInsertResult::Success(None)),
                "Failed on key {i}"
            );
        }

        // Verify all values
        for i in 0..10u64 {
            let key_bytes = format!("k{i}");
            assert_eq!(*tree.get(key_bytes.as_bytes()).unwrap(), i * 100);
        }
    }
}
