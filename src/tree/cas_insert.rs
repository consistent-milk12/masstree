//! CAS-based lock-free insert for [`MassTree`].
//!
//! Provides a fast path for simple inserts that can be completed with a single
//! compare-and-swap on the permutation, avoiding lock acquisition overhead.
//!
//! # Protocol
//!
//! ```text
//! 1. Optimistic traversal to find target leaf
//! 2. Get stable version and permutation
//! 3. Search for key position
//! 4. Pre-store slot data (ikey, keylenx, value_ptr) with Release ordering
//! 5. CAS permutation to atomically publish the insert
//! 6. On failure, retry or fall back to locked path
//! ```
//!
//! # When CAS Succeeds
//!
//! - Key is new (not found in leaf)
//! - Leaf has space (`size < WIDTH`)
//! - No slot-0 violation (or slot-0 can be reused)
//! - No suffix needed (inline key ≤ 8 bytes)
//! - No concurrent modification (version stable)
//! - Permutation CAS succeeds
//!
//! # When Falls Back to Locked Path
//!
//! - Key exists (need value update with old value return)
//! - Leaf full (need split)
//! - Layer/suffix conflict (need layer creation)
//! - Slot-0 violation (needs swap logic)
//! - High contention (> MAX_CAS_RETRIES failures)

use std::sync::Arc;
use std::sync::atomic::Ordering;

use seize::LocalGuard;

use crate::alloc::NodeAllocator;
use crate::key::Key;
use crate::leaf::{LeafNode, LeafValue};

use super::locked::InsertSearchResult;
use super::MassTree;

/// Maximum CAS retry attempts before falling back to locked path.
const MAX_CAS_RETRIES: usize = 3;

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

    /// High contention or complex case - fall back to locked path.
    ContentionFallback,
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Try CAS-based lock-free insert.
    ///
    /// Attempts to insert a new key-value pair using optimistic concurrency:
    /// 1. Pre-store slot data in a free slot
    /// 2. CAS the permutation to atomically publish
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
    pub(super) fn try_cas_insert(
        &self,
        key: &Key<'_>,
        value: Arc<V>,
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

        loop {
            // 1. Optimistic traversal to find target leaf
            let layer_root: *const u8 = self.load_root_ptr(guard);
            let leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> =
                self.reach_leaf_concurrent(layer_root, key, guard);
            let leaf: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*leaf_ptr };

            // 2. Get stable version and permutation
            let version: u32 = leaf.version().stable();
            let perm = leaf.permutation();

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
                        // Slot-0 violation requires swap logic - fall back
                        return CasInsertResult::ContentionFallback;
                    }

                    // 6. Compute new permutation (immutable)
                    let (new_perm, slot) = perm.insert_from_back_immutable(logical_pos);

                    // 7. Pre-store slot data
                    // Clone the Arc and convert to raw pointer
                    let arc_ptr: *mut u8 = Arc::into_raw(Arc::clone(&value)) as *mut u8;

                    // SAFETY: slot is in the free region (not visible to readers yet)
                    unsafe {
                        leaf.store_slot_for_cas(slot, ikey, keylenx, arc_ptr);
                    }

                    // 8. Validate version unchanged
                    if leaf.version().has_changed(version) {
                        // Version changed - reclaim the Arc we stored
                        // SAFETY: we just stored this pointer, nobody else saw it
                        let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };

                        retries += 1;
                        if retries > MAX_CAS_RETRIES {
                            return CasInsertResult::ContentionFallback;
                        }
                        continue;
                    }

                    // 9. CAS the permutation
                    match leaf.cas_permutation(perm, new_perm) {
                        Ok(()) => {
                            // Success! Increment count
                            self.count.fetch_add(1, Ordering::Relaxed);
                            return CasInsertResult::Success(None);
                        }

                        Err(_current_perm) => {
                            // CAS failed - reclaim the Arc we stored
                            // SAFETY: we stored this pointer, nobody else saw it
                            //         (permutation didn't change to include this slot)
                            let _ = unsafe { Arc::from_raw(arc_ptr as *const V) };

                            retries += 1;
                            if retries > MAX_CAS_RETRIES {
                                return CasInsertResult::ContentionFallback;
                            }
                            // Retry with new state
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
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

        let result = tree.try_cas_insert(&key, value, &guard);
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

        let result = tree.try_cas_insert(&key, value, &guard);
        assert!(matches!(result, CasInsertResult::ExistsNeedLock { .. }));
    }

    #[test]
    fn test_cas_insert_suffix_key_falls_back() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        // Long key with suffix should fall back
        let key = Key::new(b"this_is_a_very_long_key_with_suffix");
        let value = Arc::new(42u64);

        let result = tree.try_cas_insert(&key, value, &guard);
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

            let result = tree.try_cas_insert(&key, value, &guard);
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
