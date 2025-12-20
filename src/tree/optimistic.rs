//! Optimistic Read Support for [`MassTree`]
//!
//! Implements lock-free optimistic reads using:
//! 1. **seize guards** for protected pointer loads (prevents use-after-free)
//! 2. **Two-store reads** with version protection (stable → reads → `has_changed`)
//! 3. **B-link chain following** for split handling
//!
//! # Protocol
//!
//! ```text
//! 1. guard = tree.collector.enter()
//! 2. Traverse to leaf (version validation at each level)
//! 3. stable() → atomic slot reads → has_changed() → retry if needed
//! 4. Clone Arc AFTER validation (version ensures consistency)
//! 5. drop(guard)
//! ```
//!
//! # Reference
//!
//! C++ `masstree_get.hh:22-57` - `find_unlocked()`

use std::ptr as StdPtr;
use std::sync::Arc;

use seize::LocalGuard;

use crate::alloc::NodeAllocator;
use crate::internode::InternodeNode;
use crate::key::Key;
use crate::ksearch::upper_bound_internode_direct;
use crate::leaf::link::{is_marked, unmark_ptr};
use crate::leaf::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode, LeafValue};
use crate::nodeversion::NodeVersion;

use super::MassTree;

// ============================================================================
//  SearchResult
// ============================================================================

/// Result of searching a leaf node.
enum SearchResult<V> {
    /// Found the value (Arc cloned after version validation).
    Found(Arc<V>),

    /// Key not found in this leaf.
    NotFound,

    /// Slot is a layer pointer - descend into sublayer.
    Layer(*mut u8),

    /// Version changed during search - retry from layer root.
    Retry,
}

// ============================================================================
//  Public API
// ============================================================================

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Get a value by key using an explicit guard.
    ///
    /// Use this when performing multiple operations to amortize guard overhead.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTree::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = tree.guard();
    /// for key in keys {
    ///     if let Some(value) = tree.get_with_guard(&key, &guard) {
    ///         // process value
    ///     }
    /// }
    /// // guard dropped, reclamation can proceed
    /// ```
    #[must_use]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<Arc<V>> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_concurrent(&mut search_key, guard)
    }

    /// Internal concurrent get implementation with layer descent support.
    ///
    /// # Protocol
    ///
    /// 1. Start at tree root
    /// 2. For each layer:
    ///    a. Follow parent pointers to find actual layer root (`maybe_parent`)
    ///    b. Traverse internodes to leaf
    ///    c. Search leaf with version validation
    ///    d. If layer found, shift key and descend
    ///    e. If retry needed, restart from layer root
    fn get_concurrent(&self, key: &mut Key<'_>, guard: &LocalGuard<'_>) -> Option<Arc<V>> {
        // Start at tree root (use atomic pointer for concurrent access)
        let mut layer_root: *const u8 = self.load_root_ptr(guard);

        loop {
            // Find the actual layer root (handles layer root promotion)
            layer_root = self.maybe_parent(layer_root);

            // Traverse to leaf for current layer
            let leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> =
                self.reach_leaf_concurrent(layer_root, key, guard);

            // Search in leaf with version validation
            match self.search_leaf_concurrent(leaf_ptr, key, guard) {
                SearchResult::Found(arc) => return Some(arc),

                SearchResult::NotFound => return None,

                SearchResult::Layer(next_layer) => {
                    // Descend into sublayer
                    key.shift();
                    layer_root = next_layer;
                }

                SearchResult::Retry => {
                    // Retry from current layer root
                }
            }
        }
    }
}

// ============================================================================
//  Internal Implementation
// ============================================================================

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Follow parent pointers to find the actual layer root.
    ///
    /// When a layer root leaf splits and gets a parent internode, the slot
    /// in the parent leaf still points to the old leaf. This function follows
    /// parent pointers to find the actual root of the layer.
    ///
    /// # Reference
    ///
    /// C++ `masstree_struct.hh:83-92` - `maybe_parent()`
    #[expect(clippy::unused_self, reason = "Method signature for API consistency")]
    fn maybe_parent(&self, mut node: *const u8) -> *const u8 {
        loop {
            // SAFETY: node is a valid pointer to a LeafNode or InternodeNode.
            // Both have NodeVersion as first field.
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "node points to LeafNode or InternodeNode, both properly aligned"
            )]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            let parent = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirms this is a LeafNode
                let leaf: &LeafNode<LeafValue<V>, WIDTH> =
                    unsafe { &*(node.cast::<LeafNode<LeafValue<V>, WIDTH>>()) };
                leaf.parent()
            } else {
                // SAFETY: !version.is_leaf() confirms this is an InternodeNode
                let inode: &InternodeNode<LeafValue<V>, WIDTH> =
                    unsafe { &*(node.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };
                inode.parent()
            };

            if parent.is_null() {
                // No parent - this is the layer root
                return node;
            }

            // Has parent - follow it
            node = parent;
        }
    }

    /// Traverse from layer root to target leaf with optimistic concurrency.
    ///
    /// # Algorithm
    ///
    /// 1. Get stable version of current node
    /// 2. If internode: binary search for child, descend
    /// 3. If version changed with split, check if key escaped to sibling
    /// 4. Repeat until leaf reached
    ///
    /// # Reference
    ///
    /// C++ `masstree_struct.hh:633-685` - `reach_leaf` algorithm
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    pub(super) fn reach_leaf_concurrent(
        &self,
        start: *const u8,
        key: &Key<'_>,
        _guard: &LocalGuard<'_>,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        let target_ikey: u64 = key.ikey();
        let mut node: *const u8 = start;

        loop {
            // SAFETY: node is valid, both node types have NodeVersion as first field
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "node points to LeafNode or InternodeNode"
            )]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            // Get stable version (spins if dirty)
            let v: u32 = version.stable();

            if version.is_leaf() {
                // Reached a leaf
                return node as *mut LeafNode<LeafValue<V>, WIDTH>;
            }

            // It's an internode - traverse down
            // SAFETY: !is_leaf() confirmed above
            let inode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(node.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };

            // Binary search for child
            let child_idx: usize = upper_bound_internode_direct(target_ikey, inode);
            let child: *mut u8 = inode.child(child_idx);

            if child.is_null() {
                // Concurrent split in progress - retry from start
                node = start;
                continue;
            }

            // Check if internode changed during our read
            if inode.version().has_changed(v) {
                // Version changed - check for split
                if inode.version().has_split(v) {
                    // Key might have escaped to sibling - retry from start
                    // In full implementation, would check stable_last_key_compare
                    node = start;
                    continue;
                }
                // Just retry this internode
                continue;
            }

            // Descend to child
            node = child;
        }
    }

    /// Search leaf with two-store read protocol.
    ///
    /// # Two-Store Protocol (P0.2)
    ///
    /// For each slot, we atomically read:
    /// 1. `ikey0[slot]` - [`AtomicU64`]
    /// 2. `keylenx[slot]` - [`AtomicU8`] (determines type: value vs layer)
    /// 3. `leaf_values[slot]` - [`AtomicPtr<u8>`] (provenance-safe)
    ///
    /// **CRITICAL**: We only RECORD the snapshot during search. We do NOT
    /// interpret keylenx or clone Arc until AFTER version validation succeeds.
    /// This prevents the race where keylenx and ptr become inconsistent during
    /// a value→layer conversion.
    ///
    /// # Reference
    ///
    /// C++ `masstree_get.hh:22-57` - `find_unlocked()`
    fn search_leaf_concurrent(
        &self,
        leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH>,
        key: &Key<'_>,
        guard: &LocalGuard<'_>,
    ) -> SearchResult<V> {
        // SAFETY: leaf_ptr protected by guard, valid from reach_leaf_concurrent
        let leaf: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*leaf_ptr };

        // Take version snapshot (spins if dirty)
        let mut version: u32 = leaf.version().stable();

        loop {
            // Check for deleted node
            if leaf.version().is_deleted() {
                return SearchResult::Retry;
            }

            // Load permutation (Acquire)
            let perm = leaf.permutation();
            let target_ikey: u64 = key.ikey();

            // Calculate keylenx for search
            #[expect(
                clippy::cast_possible_truncation,
                reason = "current_len() <= 8 at each layer"
            )]
            let search_keylenx: u8 = if key.has_suffix() {
                KSUF_KEYLENX
            } else {
                key.current_len() as u8
            };

            // P0.2: Record slot snapshot, don't interpret yet
            // We store (slot, keylenx, ptr) and interpret AFTER validation
            let mut match_snapshot: Option<(usize, u8, *mut u8)> = None;

            for i in 0..perm.size() {
                let slot: usize = perm.get(i);

                // Two-store read: ikey + keylenx + leaf_values
                let slot_ikey: u64 = leaf.ikey(slot);
                if slot_ikey != target_ikey {
                    continue;
                }

                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                // Empty slot
                if slot_ptr.is_null() {
                    continue;
                }

                // Check for exact match or layer
                if slot_keylenx == search_keylenx {
                    // Potential exact match - check suffix if needed
                    let suffix_match: bool = if slot_keylenx == KSUF_KEYLENX {
                        leaf.ksuf_equals(slot, key.suffix())
                    } else {
                        true
                    };

                    if suffix_match {
                        match_snapshot = Some((slot, slot_keylenx, slot_ptr));
                        break;
                    }
                } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
                    // Layer pointer and key has more bytes - record for descent
                    match_snapshot = Some((slot, slot_keylenx, slot_ptr));
                    break;
                }
                // Same ikey, different key - continue searching
            }

            // Validate version AFTER all reads
            if leaf.version().has_changed(version) {
                // Version changed - follow B-link chain if split occurred
                let (new_leaf, new_version) = self.advance_to_key(leaf, key, version, guard);

                if !StdPtr::eq(new_leaf, leaf) {
                    // Different leaf - search there
                    return self.search_leaf_concurrent(
                        StdPtr::from_ref(new_leaf).cast_mut(),
                        key,
                        guard,
                    );
                }

                // Same leaf, new version - retry search
                version = new_version;
                continue;
            }

            // ================================================================
            // P0.2: VERSION VALIDATED - NOW SAFE TO INTERPRET SNAPSHOT
            // ================================================================
            // At this point, the (keylenx, ptr) pair is consistent.
            // We can safely determine if ptr is an Arc or layer pointer.

            if let Some((_slot, keylenx, ptr)) = match_snapshot {
                if keylenx >= LAYER_KEYLENX {
                    // Layer pointer - return for descent
                    return SearchResult::Layer(ptr);
                }

                // Value Arc - NOW safe to clone
                // SAFETY: version validated, so keylenx correctly identifies ptr as Arc<V>
                let arc: Arc<V> = unsafe {
                    let arc_ptr: *const V = ptr.cast();
                    Arc::increment_strong_count(arc_ptr);
                    Arc::from_raw(arc_ptr)
                };
                return SearchResult::Found(arc);
            }

            // No match found
            return SearchResult::NotFound;
        }
    }

    /// Advance to correct leaf after split detection.
    ///
    /// When `version.has_split(old_version)`, the target key may now be
    /// in a right sibling. Follow B-link chain until correct leaf found.
    ///
    /// # Comparison: ikey-only
    ///
    /// The B-link walk uses **ikey-only** comparison, matching C++.
    /// This works because the split-point algorithm ensures equal ikeys
    /// are never split across siblings.
    ///
    /// # Reference
    ///
    /// C++ `masstree_struct.hh:693-712` - `advance_to_key()`
    #[expect(clippy::unused_self, reason = "Method signature for API consistency")]
    fn advance_to_key<'a>(
        &'a self,
        mut leaf: &'a LeafNode<LeafValue<V>, WIDTH>,
        key: &Key<'_>,
        old_version: u32,
        _guard: &LocalGuard<'_>,
    ) -> (&'a LeafNode<LeafValue<V>, WIDTH>, u32) {
        let mut version: u32 = leaf.version().stable();

        // Only follow chain if split occurred
        if !leaf.version().has_split(old_version) {
            return (leaf, version);
        }

        let key_ikey: u64 = key.ikey();

        while !leaf.version().is_deleted() {
            // Load next pointer
            let next_raw: *mut LeafNode<LeafValue<V>, WIDTH> = leaf.next_raw();

            // Check for marked pointer (split in progress)
            if is_marked(next_raw) {
                // Spin until split completes
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr protected by guard
            let next: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*next_ptr };

            // Compare key.ikey to next leaf's ikey_bound (ikey-only!)
            // This matches C++: compare(ka.ikey(), next->ikey_bound()) >= 0
            let next_bound_ikey: u64 = next.ikey_bound();

            if key_ikey >= next_bound_ikey {
                // Key belongs in next leaf or further
                leaf = next;
                version = leaf.version().stable();
                continue;
            }

            // Key belongs in current leaf
            break;
        }

        (leaf, version)
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn test_get_with_guard_empty_tree() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        assert!(tree.get_with_guard(b"hello", &guard).is_none());
        assert!(tree.get_with_guard(b"", &guard).is_none());
    }

    #[test]
    fn test_get_with_guard_after_insert() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello", 42).unwrap();
        tree.insert(b"world", 123).unwrap();

        let guard = tree.guard();

        assert_eq!(*tree.get_with_guard(b"hello", &guard).unwrap(), 42);
        assert_eq!(*tree.get_with_guard(b"world", &guard).unwrap(), 123);
        assert!(tree.get_with_guard(b"missing", &guard).is_none());
    }

    #[test]
    fn test_get_with_guard_batched() {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in 0..100u64 {
            tree.insert(&i.to_be_bytes(), i * 10).unwrap();
        }

        // Batched reads with single guard
        let guard = tree.guard();
        for i in 0..100u64 {
            let value = tree.get_with_guard(&i.to_be_bytes(), &guard);
            assert_eq!(*value.unwrap(), i * 10);
        }
    }

    #[test]
    fn test_get_with_guard_after_splits() {
        let mut tree: MassTree<u64, 4> = MassTree::new();

        // Insert enough to trigger splits
        for i in 0..20u64 {
            tree.insert(&i.to_be_bytes(), i * 100).unwrap();
        }

        let guard = tree.guard();
        for i in 0..20u64 {
            let value = tree.get_with_guard(&i.to_be_bytes(), &guard);
            assert!(value.is_some(), "Key {i} not found after splits");
            assert_eq!(*value.unwrap(), i * 100);
        }
    }

    #[test]
    fn test_get_with_guard_layers() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Keys that create layers (same 8-byte prefix, different suffixes)
        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();
        tree.insert(b"hello wonder", 3).unwrap();

        let guard = tree.guard();
        assert_eq!(*tree.get_with_guard(b"hello world!", &guard).unwrap(), 1);
        assert_eq!(*tree.get_with_guard(b"hello worm", &guard).unwrap(), 2);
        assert_eq!(*tree.get_with_guard(b"hello wonder", &guard).unwrap(), 3);
        assert!(tree.get_with_guard(b"hello worst", &guard).is_none());
    }

    #[test]
    fn test_get_and_get_with_guard_equivalent() {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in 0..50u64 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i).unwrap();
        }

        // Both methods should return same results
        let guard = tree.guard();
        for i in 0..50u64 {
            let key = format!("{i:08}");
            let v1 = tree.get(key.as_bytes());
            let v2 = tree.get_with_guard(key.as_bytes(), &guard);

            assert_eq!(v1.as_deref(), v2.as_deref());
        }
    }

    #[test]
    fn test_maybe_parent_no_parent() {
        let tree: MassTree<u64> = MassTree::new();

        // Root pointer — load directly, no match needed
        let root_ptr: *const u8 = tree.root_ptr.load(Ordering::Acquire);

        let actual_root = tree.maybe_parent(root_ptr);
        assert_eq!(root_ptr, actual_root);
    }
}
