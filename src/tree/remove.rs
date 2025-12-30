//! Deletion operations for MassTree.
//!
//! This module implements the `remove()` operation following the C++
//! reference in `reference/masstree_remove.hh`.
//!
//! # Algorithm Overview
//!
//! 1. Navigate to the target leaf using optimistic traversal
//! 2. Search for the key within the leaf
//! 3. Lock the leaf and verify the key still exists
//! 4. Remove the slot from the permutation
//! 5. Retire the value via seize
//! 6. If leaf is now empty, trigger leaf removal

use std::sync::atomic::Ordering as AtomicOrdering;

use seize::{Guard, LocalGuard};

use crate::{
    alloc_trait::NodeAllocatorGeneric,
    key::Key,
    leaf_trait::{LayerCapableLeaf, TreePermutation},
    leaf24::{KSUF_KEYLENX, LAYER_KEYLENX},
    nodeversion::LockGuard,
    slot::ValueSlot,
    tree::MassTreeGeneric,
};

// ============================================================================
//  Error Types
// ============================================================================

/// Errors that can occur during removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    /// Retry limit exceeded during optimistic concurrency.
    ///
    /// This should be extremely rare. It indicates severe contention
    /// on the target leaf node.
    RetryLimitExceeded,
}

impl std::fmt::Display for RemoveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoveError::RetryLimitExceeded => write!(f, "retry limit exceeded"),
        }
    }
}

impl std::error::Error for RemoveError {}

// ============================================================================
//  Search Result Types
// ============================================================================

/// Result of searching for a key to remove.
#[derive(Debug)]
enum RemoveSearchResult {
    /// Key not found in this leaf.
    NotFound,

    /// Key found at logical position `ki`, physical slot `kp`.
    Found {
        /// Logical position in permutation (0..size).
        ki: usize,
        /// Physical slot index (0..WIDTH).
        kp: usize,
    },

    /// Key might be in sublayer; descend and retry.
    DescendLayer {
        /// Pointer to the layer root.
        layer_ptr: *mut u8,
    },
}

// ============================================================================
//  Constants
// ============================================================================

/// Maximum retries before giving up.
const MAX_RETRIES: usize = 1000;

// ============================================================================
//  Main Entry Point
// ============================================================================

/// Main entry point for concurrent deletion.
///
/// # Algorithm
///
/// 1. Navigate to the target leaf using optimistic traversal
/// 2. Search for the key within the leaf
/// 3. Lock the leaf and verify the key still exists
/// 4. Remove the slot from the permutation
/// 5. Retire the value via seize
/// 6. If leaf is now empty, trigger leaf removal
///
/// # Reference
///
/// C++ `masstree_remove.hh:162-176` - `finish_remove()`
pub fn remove_concurrent_generic<S, L, A>(
    tree: &MassTreeGeneric<S, L, A>,
    key_bytes: &[u8],
    guard: &LocalGuard<'_>,
) -> Result<Option<S::Output>, RemoveError>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    let mut key = Key::new(key_bytes);
    let mut retry_count: usize = 0;

    // Track layer descent for multi-layer keys
    let mut layer_root: *mut u8 = tree.root_ptr.load(AtomicOrdering::Acquire);

    'layer_loop: loop {
        'retry_loop: loop {
            if retry_count >= MAX_RETRIES {
                return Err(RemoveError::RetryLimitExceeded);
            }
            retry_count += 1;

            // Step 1: Navigate to target leaf
            let leaf_ptr: *mut L =
                tree.reach_leaf_concurrent_generic(layer_root, &key, false, guard);
            // SAFETY: reach_leaf_concurrent_generic returns a valid leaf pointer
            let leaf: &L = unsafe { &*leaf_ptr };

            // Step 2: Get stable version and search for slot
            let version: u32 = leaf.version().stable();
            let perm: L::Perm = leaf.permutation();

            let search_result: RemoveSearchResult =
                search_for_remove_generic::<S, L>(leaf, &key, &perm);

            // Step 3: Version validation before locking
            if leaf.version().has_changed(version) {
                continue 'retry_loop;
            }

            match search_result {
                RemoveSearchResult::NotFound => {
                    // Key doesn't exist
                    return Ok(None);
                }

                RemoveSearchResult::Found { ki, kp } => {
                    // Step 4: Lock the leaf
                    let mut lock: LockGuard<'_> = leaf.version().lock();

                    // Step 5: Re-verify after lock (key might have moved)
                    let new_perm: L::Perm = leaf.permutation();
                    if new_perm.size() <= ki {
                        // Slot was removed by concurrent delete
                        drop(lock);
                        continue 'retry_loop;
                    }

                    let new_kp: usize = new_perm.get(ki);
                    let slot_ikey: u64 = leaf.ikey(new_kp);
                    let slot_keylenx: u8 = leaf.keylenx(new_kp);

                    // Verify this is still our key
                    if slot_ikey != key.ikey() {
                        drop(lock);
                        continue 'retry_loop;
                    }

                    // Handle based on key type
                    if slot_keylenx >= LAYER_KEYLENX {
                        // This is a layer pointer, not a value
                        // Need to descend into layer
                        drop(lock);
                        let lp: *mut u8 = leaf.leaf_value_ptr(new_kp);
                        layer_root = lp;
                        key.shift();
                        continue 'layer_loop;
                    }

                    // Step 6: Finish the removal
                    let removed_value: Option<S::Output> =
                        finish_remove_generic::<S, L, A>(tree, leaf, &mut lock, ki, kp, guard);

                    // Step 7: Check if leaf is now empty
                    // NOTE: We intentionally do NOT mark_deleted() here.
                    // Marking a leaf as deleted without updating the tree structure
                    // (parent pointers, B-link chain) causes infinite retry loops
                    // because get/insert see the deleted flag and retry, but the
                    // root still points to the deleted leaf.
                    //
                    // Full leaf removal requires:
                    // 1. Unlinking from B-link chain (btree_leaflink::unlink)
                    // 2. Updating parent internode child pointers
                    // 3. Potentially collapsing empty internodes
                    // 4. gc_layer for empty sublayers
                    //
                    // For now, empty leaves stay in the tree but have size=0,
                    // so searches correctly return not-found.
                    // This is a known limitation documented in KNOWN_BUGS.md.

                    // Lock automatically released on drop
                    return Ok(removed_value);
                }

                RemoveSearchResult::DescendLayer { layer_ptr } => {
                    // Key continues in sublayer - descend and retry
                    layer_root = layer_ptr;
                    key.shift();
                    continue 'layer_loop;
                }
            }
        }
    }
}

// ============================================================================
//  Search for Remove
// ============================================================================

/// Search for a key within a leaf for removal.
///
/// Unlike `search_for_insert`, we need to find an exact match.
///
/// # Algorithm
///
/// 1. Linear scan through permutation slots
/// 2. Compare ikey values
/// 3. If ikey matches, check keylenx and suffix
/// 4. Return position if exact match found
fn search_for_remove_generic<S, L>(leaf: &L, key: &Key<'_>, perm: &L::Perm) -> RemoveSearchResult
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let target_ikey: u64 = key.ikey();
    let size: usize = perm.size();

    for ki in 0..size {
        let kp: usize = perm.get(ki);
        let slot_ikey: u64 = leaf.ikey(kp);

        if slot_ikey < target_ikey {
            continue;
        }

        if slot_ikey > target_ikey {
            // Past the target - key not found
            return RemoveSearchResult::NotFound;
        }

        // ikey matches - check key length/type
        let slot_keylenx: u8 = leaf.keylenx(kp);

        if slot_keylenx >= LAYER_KEYLENX {
            // This is a layer pointer
            if key.has_suffix() {
                // Key continues - need to descend
                let layer_ptr: *mut u8 = leaf.leaf_value_ptr(kp);
                return RemoveSearchResult::DescendLayer { layer_ptr };
            }
            // Short key can't match layer pointer
            return RemoveSearchResult::NotFound;
        }

        // Check inline key length
        #[expect(clippy::cast_possible_truncation, reason = "key.current_len() <= 8")]
        let key_len: u8 = key.current_len() as u8;

        if slot_keylenx == KSUF_KEYLENX {
            // Has suffix - compare suffix
            if !key.has_suffix() {
                continue; // Key too short
            }

            let suffix: &[u8] = key.suffix();
            if leaf.ksuf_equals(kp, suffix) {
                return RemoveSearchResult::Found { ki, kp };
            }
            continue;
        }

        // Inline key (no suffix)
        if key_len <= 8 && slot_keylenx == key_len {
            // Exact match for short key
            return RemoveSearchResult::Found { ki, kp };
        }
    }

    RemoveSearchResult::NotFound
}

// ============================================================================
//  Finish Remove
// ============================================================================

/// Complete the removal of a key from a locked leaf.
///
/// # Preconditions
///
/// - Leaf is locked (caller holds `LockGuard`)
/// - Key exists at logical position `ki`, physical slot `kp`
///
/// # Algorithm
///
/// 1. Extract value for return
/// 2. Schedule value retirement via seize
/// 3. Clear suffix if present
/// 4. Update permutation using `perm.remove(ki)`
/// 5. Store updated permutation
/// 6. Decrement entry count
fn finish_remove_generic<S, L, A>(
    tree: &MassTreeGeneric<S, L, A>,
    leaf: &L,
    lock: &mut LockGuard<'_>,
    ki: usize,
    kp: usize,
    guard: &LocalGuard<'_>,
) -> Option<S::Output>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    // Step 1: Extract the value pointer
    let value_ptr: *mut u8 = leaf.leaf_value_ptr(kp);

    // Step 2: Clone the value for return (before retirement)
    let value: Option<S::Output> = if !value_ptr.is_null() {
        // SAFETY: value_ptr points to valid value created during insert
        // We use try_clone_output which handles Arc cloning properly
        leaf.try_clone_output(kp)
    } else {
        None
    };

    // Step 3: Schedule value retirement
    // The old value pointer needs to be freed after all readers are done
    if !value_ptr.is_null() {
        // SAFETY: value_ptr was created by insert and will be valid until retirement
        unsafe {
            guard.defer_retire(value_ptr, |ptr, _| {
                S::cleanup_value_ptr(ptr);
            });
        }
    }

    // Step 4: Clear suffix if present
    let slot_keylenx: u8 = leaf.keylenx(kp);
    if slot_keylenx == KSUF_KEYLENX {
        // Clear the suffix slot
        // SAFETY: We hold the lock and kp is valid
        unsafe { leaf.clear_ksuf(kp, guard) };
    }

    // Step 5: Update permutation - remove slot at logical position `ki`
    let mut new_perm: L::Perm = leaf.permutation();
    new_perm.remove(ki);
    leaf.set_permutation(new_perm);

    // Step 6: Clear the slot value pointer
    // This prevents accidental access to retired value
    leaf.set_leaf_value_ptr(kp, std::ptr::null_mut());

    // Step 7: Decrement entry count
    tree.dec_count();

    // Mark insert in lock guard for version increment
    // This ensures readers see the removal
    lock.mark_insert();

    value
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Tests")]
#[expect(unused_imports, reason = "Test utilities")]
mod tests {
    use super::*;
    use crate::tree::MassTree24;
    use std::sync::Arc;

    #[test]
    fn test_remove_single_key() {
        let tree: MassTree24<u64> = MassTree24::new();

        tree.insert(b"key1", 42).unwrap();
        assert_eq!(tree.len(), 1);

        let removed = tree.remove(b"key1").unwrap();
        assert_eq!(removed, Some(Arc::new(42)));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_key() {
        let tree: MassTree24<u64> = MassTree24::new();

        tree.insert(b"key1", 42).unwrap();

        let result = tree.remove(b"key2");
        assert!(matches!(result, Ok(None)));

        // Original key still exists
        assert_eq!(tree.get(b"key1"), Some(Arc::new(42)));
    }

    #[test]
    fn test_remove_updates_count() {
        let tree: MassTree24<u64> = MassTree24::new();

        for i in 0..10u64 {
            tree.insert(&i.to_be_bytes(), i).unwrap();
        }
        assert_eq!(tree.len(), 10);

        for i in 0..5u64 {
            tree.remove(&i.to_be_bytes()).unwrap();
        }
        assert_eq!(tree.len(), 5);

        // Verify remaining keys
        for i in 5..10u64 {
            assert!(tree.get(&i.to_be_bytes()).is_some());
        }
        for i in 0..5u64 {
            assert!(tree.get(&i.to_be_bytes()).is_none());
        }
    }

    #[test]
    fn test_remove_returns_old_value() {
        let tree: MassTree24<String> = MassTree24::new();

        tree.insert(b"key", "hello".to_string()).unwrap();
        tree.insert(b"key", "world".to_string()).unwrap();

        let removed = tree.remove(b"key").unwrap();
        assert_eq!(removed, Some(Arc::new("world".to_string())));
    }

    #[test]
    fn test_remove_short_key() {
        let tree: MassTree24<u64> = MassTree24::new();

        // 1-byte key
        tree.insert(&[42], 1).unwrap();
        assert_eq!(tree.remove(&[42]).unwrap(), Some(Arc::new(1)));

        // 8-byte key (max inline)
        let key8 = [1, 2, 3, 4, 5, 6, 7, 8];
        tree.insert(&key8, 8).unwrap();
        assert_eq!(tree.remove(&key8).unwrap(), Some(Arc::new(8)));
    }

    #[test]
    fn test_remove_with_suffix() {
        let tree: MassTree24<u64> = MassTree24::new();

        // 16-byte key (requires suffix)
        let key16 = b"0123456789ABCDEF";
        tree.insert(key16, 16).unwrap();

        let removed = tree.remove(key16).unwrap();
        assert_eq!(removed, Some(Arc::new(16)));
        assert!(tree.get(key16).is_none());
    }

    #[test]
    fn test_remove_all_keys_empties_tree() {
        let tree: MassTree24<u64> = MassTree24::new();

        let keys: Vec<_> = (0..100u64).map(|i| i.to_be_bytes()).collect();

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }
        assert_eq!(tree.len(), 100);

        for key in &keys {
            tree.remove(key).unwrap();
        }
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_remove_in_reverse_order() {
        let tree: MassTree24<u64> = MassTree24::new();

        for i in 0..50u64 {
            tree.insert(&i.to_be_bytes(), i).unwrap();
        }

        // Remove in reverse order
        for i in (0..50u64).rev() {
            let removed = tree.remove(&i.to_be_bytes()).unwrap();
            assert_eq!(removed, Some(Arc::new(i)));
        }

        assert!(tree.is_empty());
    }

    #[test]
    fn test_remove_alternating() {
        let tree: MassTree24<u64> = MassTree24::new();

        for i in 0..100u64 {
            tree.insert(&i.to_be_bytes(), i).unwrap();
        }

        // Remove even keys
        for i in (0..100u64).step_by(2) {
            tree.remove(&i.to_be_bytes()).unwrap();
        }

        assert_eq!(tree.len(), 50);

        // Verify odd keys remain
        for i in (1..100u64).step_by(2) {
            assert!(tree.get(&i.to_be_bytes()).is_some());
        }
    }

    #[test]
    fn test_remove_and_reinsert_same_key() {
        let tree: MassTree24<u64> = MassTree24::new();

        tree.insert(b"key", 1).unwrap();
        tree.remove(b"key").unwrap();

        // Reinsert with different value
        tree.insert(b"key", 2).unwrap();
        assert_eq!(tree.get(b"key"), Some(Arc::new(2)));
    }

    #[test]
    fn test_remove_reinsert_cycle() {
        let tree: MassTree24<u64> = MassTree24::new();
        let key = b"test_key";

        for i in 0..10u64 {
            tree.insert(key, i).unwrap();
            assert_eq!(tree.get(key), Some(Arc::new(i)));

            let removed = tree.remove(key).unwrap();
            assert_eq!(removed, Some(Arc::new(i)));
            assert!(tree.get(key).is_none());
        }
    }

    #[test]
    fn test_remove_from_empty_tree() {
        let tree: MassTree24<u64> = MassTree24::new();
        let result = tree.remove(b"key");
        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn test_remove_empty_key() {
        let tree: MassTree24<u64> = MassTree24::new();

        // Empty key is valid
        tree.insert(&[], 0).unwrap();
        let removed = tree.remove(&[]).unwrap();
        assert_eq!(removed, Some(Arc::new(0)));
    }

    #[test]
    fn test_remove_preserves_other_keys() {
        let tree: MassTree24<u64> = MassTree24::new();

        tree.insert(b"aaa", 1).unwrap();
        tree.insert(b"bbb", 2).unwrap();
        tree.insert(b"ccc", 3).unwrap();

        tree.remove(b"bbb").unwrap();

        assert_eq!(tree.get(b"aaa"), Some(Arc::new(1)));
        assert!(tree.get(b"bbb").is_none());
        assert_eq!(tree.get(b"ccc"), Some(Arc::new(3)));
    }
}
