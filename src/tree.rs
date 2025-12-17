//! Filepath: src/tree.rs
//! `MassTree` - A high-performance concurrent trie of B+trees.
//!
//! This module provides the main `MassTree<V>` and `MassTreeIndex<V>` types.

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::Arc;

use crate::internode::InternodeNode;
use crate::key::Key;
use crate::leaf::{KSUF_KEYLENX, LeafNode, LeafSplitResult, SplitUtils};
use crate::permuter::Permuter;

mod index;
mod layer;
mod split;
mod traverse;

pub use index::MassTreeIndex;

// ============================================================================
//  RootNode Enum
// ============================================================================

/// The root of a `MassTree` layer.
///
/// Can be either a leaf (small tree) or an internode (larger tree).
pub enum RootNode<V, const WIDTH: usize = 15> {
    /// Root is a leaf node (tree has 0 or 1 level).
    Leaf(Box<LeafNode<V, WIDTH>>),

    /// Root is an internode (tree has multiple levels).
    Internode(Box<InternodeNode<V, WIDTH>>),
}

impl<V, const WIDTH: usize> RootNode<V, WIDTH> {
    /// Check if root is a leaf.
    #[inline]
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Check if root is an internode.
    #[inline]
    #[must_use]
    pub const fn is_internode(&self) -> bool {
        matches!(self, Self::Internode(_))
    }

    /// Get leaf reference if root is a leaf.
    #[inline]
    #[must_use]
    pub fn as_leaf(&self) -> Option<&LeafNode<V, WIDTH>> {
        match self {
            Self::Leaf(leaf) => Some(leaf.as_ref()),

            Self::Internode(_) => None,
        }
    }

    /// Get mutable leaf reference if root is a leaf.
    #[inline]
    pub fn as_leaf_mut(&mut self) -> Option<&mut LeafNode<V, WIDTH>> {
        match self {
            Self::Leaf(leaf) => Some(leaf.as_mut()),

            Self::Internode(_) => None,
        }
    }

    /// Get internode reference if root is an internode.
    #[inline]
    #[must_use]
    pub fn as_internode(&self) -> Option<&InternodeNode<V, WIDTH>> {
        match self {
            Self::Leaf(_) => None,

            Self::Internode(node) => Some(node.as_ref()),
        }
    }

    /// Get mutable internode reference if root is an internode.
    #[inline]
    pub fn as_internode_mut(&mut self) -> Option<&mut InternodeNode<V, WIDTH>> {
        match self {
            Self::Leaf(_) => None,

            Self::Internode(node) => Some(node.as_mut()),
        }
    }
}

impl<V, const WIDTH: usize> StdFmt::Debug for RootNode<V, WIDTH> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::Leaf(_) => f.debug_tuple("RootNode::Leaf").field(&"...").finish(),

            Self::Internode(_) => f.debug_tuple("RootNode::Internode").field(&"...").finish(),
        }
    }
}

// ============================================================================
//  MassTree (Default Mode with Arc<V>)
// ============================================================================

// NOTE: MAX_INLINE_KEY_LEN removed - layer support now handles keys of any length
//  up to MAX_KEY_LENGTH (256 bytes, defined in key.rs)

// ============================================================================
//  InsertError
// ============================================================================

/// Errors that can occur during insert operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// Leaf node is full and cannot accept more keys.
    /// Caller should trigger a split.
    LeafFull,

    /// Memory allocation failed.
    AllocationFailed,
}

impl StdFmt::Display for InsertError {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::LeafFull => write!(f, "leaf node is full"),

            Self::AllocationFailed => write!(f, "memory allocation failed"),
        }
    }
}

impl std::error::Error for InsertError {}

/// A high-performance trie of B+trees for key-value storage.
///
/// Keys are byte slices up to 8 bytes in Phase 1. Values are stored
/// as `Arc<V>` for cheap cloning on read operations.
///
/// # Type Parameters
///
/// * `V` - The value type to store
/// * `WIDTH` - Node width (default: 15, max: 15)
///
/// # Phase 1 Limitations
///
/// - Keys must be 0-8 bytes (longer keys rejected with `KeyTooLong`)
/// - Single layer only (no trie traversal)
/// - Single-threaded access only
///
/// # Example
///
/// ```ignore
/// use masstree::tree::MassTree;
///
/// let mut tree: MassTree<u64> = MassTree::new();
/// tree.insert(b"hello", 42).unwrap();
///
/// let value = tree.get(b"hello");
/// assert_eq!(value.map(|v| *v), Some(42));
/// ```
pub struct MassTree<V, const WIDTH: usize = 15> {
    /// Root of the tree.
    root: RootNode<V, WIDTH>,

    /// Arena for leaf nodes (ensures pointer stability).
    /// All leaves except the root leaf are stored here.
    leaf_arena: Vec<Box<LeafNode<V, WIDTH>>>,

    /// Arena for internode nodes (ensures pointer stability).
    internode_arena: Vec<Box<InternodeNode<V, WIDTH>>>,

    /// Number of key-value pairs in the tree.
    /// Updated on insert (new key) and delete operations.
    count: usize,

    /// Marker to make `MassTree` `!Send` and `!Sync`.
    ///
    /// `PhantomData<*const ()>` makes this type `!Send` and `!Sync` because
    /// raw pointers (`*const T`, `*mut T`) are neither `Send` nor `Sync` in Rust,
    /// and `PhantomData<T>` inherits the auto-traits of `T`.
    ///
    /// Phase 1 is single-threaded only. The `NodeVersion` lock semantics
    /// use load-then-store (not CAS), and raw pointers in arenas are not
    /// safe for concurrent access. This marker prevents accidental misuse.
    ///
    /// When Phase 3 implements proper concurrent access with CAS-based locking
    /// and atomic operations, this marker can be removed and proper `Send`/`Sync`
    /// bounds added based on `V`'s bounds.
    _not_send_sync: PhantomData<*const ()>,
}

impl<V, const WIDTH: usize> StdFmt::Debug for MassTree<V, WIDTH> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTree")
            .field("root", &self.root)
            .field("leaf_arena_len", &self.leaf_arena.len())
            .field("internode_arena_len", &self.internode_arena.len())
            .field("count", &self.count)
            .finish()
    }
}

impl<V, const WIDTH: usize> MassTree<V, WIDTH> {
    /// Create a new empty `MassTree`.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: RootNode::Leaf(LeafNode::new_root()),
            leaf_arena: Vec::new(),
            internode_arena: Vec::new(),
            count: 0,
            _not_send_sync: PhantomData,
        }
    }

    /// Store an existing leaf Box in the arena and return a raw pointer.
    ///
    /// The pointer remains valid for the lifetime of the tree.
    fn store_leaf_in_arena(&mut self, leaf: Box<LeafNode<V, WIDTH>>) -> *mut LeafNode<V, WIDTH> {
        self.leaf_arena.push(leaf);
        let idx: usize = self.leaf_arena.len() - 1;

        //  SAFETY: We just pushed, so idx is valid. We derive the pointer AFTER storing
        //  to maintain Stacked Borrows provenance.
        unsafe {
            StdPtr::from_mut::<LeafNode<V, WIDTH>>(self.leaf_arena.get_unchecked_mut(idx).as_mut())
        }
    }

    /// Store an existing internode Box in the arena and return a raw pointer.
    fn store_internode_in_arena(
        &mut self,
        node: Box<InternodeNode<V, WIDTH>>,
    ) -> *mut InternodeNode<V, WIDTH> {
        self.internode_arena.push(node);
        let idx: usize = self.internode_arena.len() - 1;

        //  SAFETY: We just pushed, so idx is valid. We derive the pointer AFTER storing
        //  to maintain Stacked Borrows provenance.
        unsafe {
            StdPtr::from_mut::<InternodeNode<V, WIDTH>>(
                self.internode_arena.get_unchecked_mut(idx).as_mut(),
            )
        }
    }

    // ========================================================================
    //  Split Propagation (moved to split.rs)
    // ========================================================================

    /// Check if the tree is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.root {
            RootNode::Leaf(leaf) => leaf.is_empty(),

            RootNode::Internode(_) => false, // Internode implies at least one key
        }
    }

    /// Get the number of keys in the tree.
    ///
    /// This is O(1) as we track the count incrementally.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    // ========================================================================
    //  Tree Traversal (moved to traverse.rs)
    // ========================================================================

    /// Look up a value by key.
    ///
    /// Returns a clone of the Arc wrapping the value, or None if not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice, up to 256 bytes)
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree: MassTree<u64> = MassTree::new();
    /// // After insert...
    /// if let Some(value) = tree.get(b"key") {
    ///     println!("Found: {}", *value);
    /// }
    /// ```
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<Arc<V>> {
        let mut search_key: Key<'_> = Key::new(key);

        self.get_internal(&mut search_key)
    }

    /// Internal get implementation with layer descent support.
    fn get_internal(&self, key: &mut Key<'_>) -> Option<Arc<V>> {
        // Start at tree root
        let mut current_root: *const u8 = match &self.root {
            RootNode::Leaf(leaf) => StdPtr::from_ref(leaf.as_ref()).cast::<u8>(),

            RootNode::Internode(inode) => StdPtr::from_ref(inode.as_ref()).cast::<u8>(),
        };

        loop {
            // Reach the leaf for current layer
            let leaf: &LeafNode<V, WIDTH> = self.reach_leaf_from_ptr(current_root, key);

            // Search in leaf
            let perm: Permuter<WIDTH> = leaf.permutation();
            let target_ikey: u64 = key.ikey();

            // Calculate keylenx for search
            #[expect(
                clippy::cast_possible_truncation,
                reason = "current_len() <= 8 at each layer"
            )]
            let keylenx: u8 = if key.has_suffix() {
                KSUF_KEYLENX
            } else {
                key.current_len() as u8
            };

            let mut found_layer: Option<*mut u8> = None;
            let mut shift_amount: usize = 0;

            for i in 0..perm.size() {
                let slot: usize = perm.get(i);

                if leaf.ikey(slot) != target_ikey {
                    continue;
                }

                // Found matching ikey, check keylenx/suffix/layer
                let match_result: i32 = leaf.ksuf_match_result(slot, keylenx, key.suffix());

                if match_result == 1 {
                    // Exact match - return value
                    return leaf.leaf_value(slot).try_clone_arc();
                }

                if match_result < 0 {
                    // Layer pointer - prepare to descend
                    // But first check if key has more bytes to compare
                    if !key.has_suffix() {
                        // Key ends at layer boundary - not found
                        // (we need more bytes to distinguish within the layer)
                        return None;
                    }

                    if let Some(layer_ptr) = leaf.get_layer(slot) {
                        #[expect(
                            clippy::cast_sign_loss,
                            reason = "match_result < 0, so -match_result > 0"
                        )]
                        {
                            shift_amount = (-match_result) as usize;
                        }

                        found_layer = Some(layer_ptr);
                    }

                    break;
                }
                // match_result == 0: same ikey but different key, continue searching
            }

            // Handle layer descent outside the loop
            if let Some(layer_ptr) = found_layer {
                key.shift_by(shift_amount);
                current_root = layer_ptr;
                continue; // Continue outer loop with new layer
            }

            // No match found in this layer
            return None;
        }
    }

    /// Reach leaf from a raw pointer (for layer descent).
    fn reach_leaf_from_ptr(&self, root_ptr: *const u8, key: &Key<'_>) -> &LeafNode<V, WIDTH> {
        // Check if it's the main tree root first
        let main_root_ptr: *const u8 = match &self.root {
            RootNode::Leaf(leaf) => StdPtr::from_ref(leaf.as_ref()).cast::<u8>(),

            RootNode::Internode(inode) => StdPtr::from_ref(inode.as_ref()).cast::<u8>(),
        };

        if root_ptr == main_root_ptr {
            return self.reach_leaf(key);
        }

        // It's a layer root (always a leaf in current implementation)
        //  SAFETY: Layer roots are allocated in arena and remain valid
        unsafe { &*(root_ptr.cast::<LeafNode<V, WIDTH>>()) }
    }

    // ========================================================================
    //  Insert Operations
    // ========================================================================

    /// Insert a key-value pair into the tree.
    ///
    /// If the key already exists, the value is updated and the old value returned.
    ///
    /// # Arguments
    ///
    /// * `key` - The key as a byte slice (up to 256 bytes)
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// * `Ok(Some(old_value))` - Key existed, old value returned
    /// * `Ok(None)` - Key inserted (new key)
    ///
    /// # Errors
    ///
    /// * [`InsertError::LeafFull`] - Internal error (shouldn't happen with layer support)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tree: MassTree<u64> = MassTree::new();
    ///
    /// // Insert new key
    /// let result = tree.insert(b"hello", 42);
    /// assert!(result.is_ok());
    /// assert!(result.unwrap().is_none()); // No old value
    ///
    /// // Update existing key
    /// let result = tree.insert(b"hello", 100);
    /// assert!(result.is_ok());
    /// assert_eq!(*result.unwrap().unwrap(), 42); // Old value returned
    /// ```
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<Arc<V>>, InsertError> {
        let mut key: Key<'_> = Key::new(key);
        self.insert_internal(&mut key, value)
    }

    /// Internal insert implementation that wraps value in Arc.
    ///
    /// Delegates to `insert_impl` after wrapping the value.
    fn insert_internal(
        &mut self,
        key: &mut Key<'_>,
        value: V,
    ) -> Result<Option<Arc<V>>, InsertError> {
        self.insert_impl(key, Arc::new(value))
    }

    /// Insert with an existing Arc (avoids double-wrapping).
    ///
    /// Useful when you already have an `Arc<V>` and don't want to clone.
    ///
    /// # Errors
    /// * [`InsertError::LeafFull`] - Internal error (shouldn't happen with layer support)
    pub fn insert_arc(&mut self, key: &[u8], value: Arc<V>) -> Result<Option<Arc<V>>, InsertError> {
        let mut key: Key<'_> = Key::new(key);
        self.insert_impl(&mut key, value)
    }

    /// Unified internal insert implementation with layer support.
    ///
    /// Uses a loop to handle:
    /// 1. Splits: if target leaf is full, split and retry
    /// 2. Layer descent: if existing layer found, descend and retry
    /// 3. Layer creation: if same ikey with different suffix, create new layer
    ///
    /// Both `insert_internal` and `insert_arc` delegate here to avoid code duplication.
    #[expect(clippy::too_many_lines, reason = "")]
    fn insert_impl(
        &mut self,
        key: &mut Key<'_>,
        value: Arc<V>,
    ) -> Result<Option<Arc<V>>, InsertError> {
        // Track current layer root (null means use main tree root)
        let mut layer_root: *mut LeafNode<V, WIDTH> = std::ptr::null_mut();

        loop {
            let ikey: u64 = key.ikey();

            // Calculate keylenx for search and storage
            // For search: use KSUF_KEYLENX if has suffix, else actual length
            // For assign_arc: use min(len, 8) - assign_ksuf will update to KSUF_KEYLENX
            #[expect(
                clippy::cast_possible_truncation,
                reason = "current_len() <= 8 at each layer"
            )]
            let search_keylenx: u8 = if key.has_suffix() {
                KSUF_KEYLENX
            } else {
                key.current_len() as u8
            };

            #[expect(
                clippy::cast_possible_truncation,
                reason = "current_len() <= 8 at each layer"
            )]
            let store_keylenx: u8 = key.current_len().min(8) as u8;

            // Get the target leaf as a raw pointer to avoid borrow issues
            let leaf_ptr: *mut LeafNode<V, WIDTH> = if layer_root.is_null() {
                self.reach_leaf_mut(key) as *mut _
            } else {
                layer_root
            };

            // SAFETY: leaf_ptr is valid (from reach_leaf_mut or layer_root which points to arena)
            let leaf: &mut LeafNode<V, WIDTH> = unsafe { &mut *leaf_ptr };

            // Search for matching ikey in the leaf
            let perm: Permuter<WIDTH> = leaf.permutation();
            let mut found_slot: Option<usize> = None;
            let mut insert_pos: usize = perm.size();
            let mut descend_layer: Option<*mut LeafNode<V, WIDTH>> = None;

            for i in 0..perm.size() {
                let slot: usize = perm.get(i);
                let slot_ikey: u64 = leaf.ikey(slot);

                if slot_ikey == ikey {
                    // Found matching ikey - check for exact match, layer, or suffix conflict
                    let match_result: i32 =
                        leaf.ksuf_match_result(slot, search_keylenx, key.suffix());

                    if match_result == 1 {
                        // Exact match - update value
                        let old_value: Option<Arc<V>> = leaf.swap_value(slot, value);
                        return Ok(old_value);
                    }

                    if match_result < 0 {
                        // Layer pointer - descend into it
                        if let Some(layer_ptr) = leaf.get_layer(slot) {
                            #[expect(
                                clippy::cast_sign_loss,
                                reason = "match_result < 0, so -match_result > 0"
                            )]
                            key.shift_by((-match_result) as usize);
                            descend_layer = Some(layer_ptr.cast::<LeafNode<V, WIDTH>>());
                        } else {
                            return Err(InsertError::LeafFull); // Layer pointer invalid
                        }

                        break;
                    }

                    // Same ikey but different key (keylenx or suffix mismatch)
                    // Check if we need to create a layer:
                    // - If slot has suffix AND new key has suffix -> create layer
                    // - If slot is inline AND new key has suffix -> create layer
                    // - If slot has suffix AND new key is inline -> create layer
                    // - If both are inline with different keylenx -> just different keys, continue
                    let slot_has_suffix: bool = leaf.has_ksuf(slot);
                    let key_has_suffix: bool = key.has_suffix();

                    if slot_has_suffix || key_has_suffix {
                        // At least one has a suffix - need to create layer
                        found_slot = Some(slot);
                        insert_pos = i;
                        break;
                    }
                    // Both are inline keys with same ikey but different keylenx
                    // They can coexist in the same leaf - continue searching
                } else if slot_ikey > ikey {
                    // Found insert position
                    insert_pos = i;
                    break;
                }
            }

            // Handle layer descent
            if let Some(new_layer) = descend_layer {
                layer_root = new_layer;
                continue;
            }

            // Handle layer creation (same ikey, different suffix)
            if let Some(conflict_slot) = found_slot {
                // Create new layer for the conflicting keys
                //  SAFETY: leaf_ptr is valid and we're done reading from leaf
                let leaf_ref: &mut LeafNode<V, WIDTH> = unsafe { &mut *leaf_ptr };
                let (final_leaf_ptr, new_slot) =
                    self.make_new_layer(leaf_ref, conflict_slot, key, Arc::clone(&value));

                // Finish insert in the new layer leaf
                //  SAFETY: make_new_layer returns valid pointer from arena
                unsafe {
                    let final_leaf: &mut LeafNode<V, WIDTH> = &mut *final_leaf_ptr;
                    let mut perm: Permuter<WIDTH> = final_leaf.permutation();
                    let _ = perm.insert_from_back(new_slot);
                    final_leaf.set_permutation(perm);
                }

                self.count += 1;
                return Ok(None);
            }

            // Normal insert path
            let size: usize = leaf.size();

            if leaf.can_insert_directly(ikey) {
                // Has space - insert directly
                let mut perm: Permuter<WIDTH> = leaf.permutation();

                // Check slot-0 rule BEFORE allocation
                let next_free_slot: usize = perm.back();

                if next_free_slot == 0 && !leaf.can_reuse_slot0(ikey) {
                    debug_assert!(
                        size < WIDTH - 1,
                        "should have fallen through to split if only slot 0 is free"
                    );
                    perm.swap_free_slots(WIDTH - 1, size);
                }

                // Allocate slot via insert_from_back
                let slot: usize = perm.insert_from_back(insert_pos);

                // Assign key and value to the slot
                leaf.assign_arc(slot, ikey, store_keylenx, Arc::clone(&value));

                // If key has suffix, store it
                if key.has_suffix() {
                    leaf.assign_ksuf(slot, key.suffix());
                }

                // Commit permutation update
                leaf.set_permutation(perm);

                // Track new entry
                self.count += 1;

                return Ok(None);
            }

            // Leaf is full - calculate split point and split
            let Some(split_point) = SplitUtils::calculate_split_point(leaf, insert_pos, ikey)
            else {
                return Err(InsertError::LeafFull);
            };

            // Convert from post-insert to pre-insert coordinates
            let pre_insert_split_pos: usize = if insert_pos < split_point.pos {
                split_point.pos - 1
            } else {
                split_point.pos
            };

            // Handle edge cases from sequential optimization
            let (new_leaf_box, split_ikey) = if pre_insert_split_pos >= size {
                // Right-sequential: create empty new leaf
                (LeafNode::new(), ikey)
            } else if pre_insert_split_pos == 0 {
                // Left-sequential: move ALL entries to right
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_all_to_right();
                (split_result.new_leaf, split_result.split_ikey)
            } else {
                // Normal split
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_into(pre_insert_split_pos);
                (split_result.new_leaf, split_result.split_ikey)
            };

            // Propagate split up the tree
            self.propagate_split(leaf_ptr, new_leaf_box, split_ikey);

            // Reset layer_root since we're back in the main tree structure
            layer_root = std::ptr::null_mut();

            // Loop continues - reach_leaf_mut will find the correct leaf
        }
    }
}

impl<V, const WIDTH: usize> Default for MassTree<V, WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
#[expect(clippy::cast_possible_truncation, reason = "reasonable in tests")]
#[expect(clippy::cast_sign_loss, reason = "reasonable in tests")]
mod tests {
    use super::*;

    // ========================================================================
    // !Send/!Sync Verification
    // ========================================================================
    //
    // MassTree uses PhantomData<*const ()> to be !Send and !Sync.
    // Raw pointers (*mut T, *const T) are neither Send nor Sync in Rust,
    // and PhantomData<T> inherits the auto-traits of T.
    //
    // To verify this works, uncomment the following and observe the compile error:
    //
    // ```
    // fn require_send<T: Send>() {}
    // fn require_sync<T: Sync>() {}
    //
    // fn test_would_fail() {
    //     require_send::<MassTree<u64>>();  // ERROR: MassTree is !Send
    //     require_sync::<MassTree<u64>>();  // ERROR: MassTree is !Sync
    // }
    // ```

    // ========================================================================
    //  MassTree Basic Tests
    // ========================================================================

    #[test]
    fn test_new_tree_is_empty() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_get_on_empty_tree() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.get(b"hello").is_none());
        assert!(tree.get(b"").is_none());
        assert!(tree.get(b"any key").is_none());
    }

    #[test]
    fn test_root_is_leaf() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.root.is_leaf());
        assert!(!tree.root.is_internode());
    }

    #[test]
    fn test_root_node_accessors() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.root.as_leaf().is_some());
        assert!(tree.root.as_internode().is_none());
    }

    // ========================================================================
    //  MassTreeIndex Basic Tests
    // ========================================================================

    #[test]
    fn test_index_new_is_empty() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::new();

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_index_get_on_empty() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::new();

        assert!(tree.get(b"hello").is_none());
    }

    // ========================================================================
    //  Key Handling Tests
    // ========================================================================

    #[test]
    fn test_reach_leaf_single_node() {
        let tree: MassTree<u64> = MassTree::new();
        let key: Key<'_> = Key::new(b"test");

        let leaf: &LeafNode<u64> = tree.reach_leaf(&key);

        // Should return the root leaf
        assert!(leaf.is_empty());
    }

    #[test]
    fn test_get_with_various_key_lengths() {
        let tree: MassTree<u64> = MassTree::new();

        // All should return None on empty tree
        assert!(tree.get(b"").is_none());
        assert!(tree.get(b"a").is_none());
        assert!(tree.get(b"ab").is_none());
        assert!(tree.get(b"abc").is_none());
        assert!(tree.get(b"abcd").is_none());
        assert!(tree.get(b"abcde").is_none());
        assert!(tree.get(b"abcdef").is_none());
        assert!(tree.get(b"abcdefg").is_none());
        assert!(tree.get(b"abcdefgh").is_none()); // Exactly 8 bytes
    }

    #[test]
    fn test_get_missing_long_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Long keys should return None on empty tree (not inserted)
        assert!(tree.get(b"abcdefghi").is_none()); // 9 bytes
        assert!(tree.get(b"hello world").is_none()); // 11 bytes
        assert!(tree.get(b"this is a long key").is_none());
    }

    // ========================================================================
    //  Default Trait Tests
    // ========================================================================

    #[test]
    fn test_default_trait() {
        let tree: MassTree<u64> = MassTree::default();

        assert!(tree.is_empty());
    }

    #[test]
    fn test_index_default_trait() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::default();

        assert!(tree.is_empty());
    }

    // ========================================================================
    //  Get After Manual Insert Tests
    // ========================================================================

    /// Test get after manually inserting into the leaf.
    /// This bypasses the insert API to test get in isolation.
    #[test]
    fn test_get_after_manual_leaf_insert() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Manually insert into the root leaf (key = "hello", 5 bytes <= 8)
        if let RootNode::Leaf(leaf) = &mut tree.root {
            let key: Key<'_> = Key::new(b"hello");
            let ikey: u64 = key.ikey();
            let keylenx: u8 = key.current_len() as u8;

            // Assign to slot 0 (plain field access, no atomics)
            leaf.assign_value(0, ikey, keylenx, 42u64);

            // Update permutation
            let mut perm = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        // Now test get
        let result = tree.get(b"hello");
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), 42);
    }

    #[test]
    fn test_get_wrong_key_after_insert() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Manually insert "hello" = 42
        if let RootNode::Leaf(leaf) = &mut tree.root {
            let key: Key<'_> = Key::new(b"hello");
            let ikey: u64 = key.ikey();
            let keylenx: u8 = key.current_len() as u8;

            leaf.assign_value(0, ikey, keylenx, 42u64);

            let mut perm: Permuter = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        // Get with wrong key should return None
        assert!(tree.get(b"world").is_none());
        assert!(tree.get(b"hell").is_none());
        assert!(tree.get(b"helloX").is_none()); // 6 bytes, still valid length but different key
    }

    #[test]
    fn test_get_multiple_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert multiple keys manually
        if let RootNode::Leaf(leaf) = &mut tree.root {
            // Insert "aaa" = 100
            let key1: Key<'_> = Key::new(b"aaa");
            leaf.assign_value(0, key1.ikey(), key1.current_len() as u8, 100u64);

            // Insert "bbb" = 200
            let key2: Key<'_> = Key::new(b"bbb");
            leaf.assign_value(1, key2.ikey(), key2.current_len() as u8, 200u64);

            // Insert "ccc" = 300
            let key3: Key<'_> = Key::new(b"ccc");
            leaf.assign_value(2, key3.ikey(), key3.current_len() as u8, 300u64);

            // Update permutation (keys are already sorted)
            let mut perm: Permuter = leaf.permutation();
            let _ = perm.insert_from_back(0); // aaa at position 0
            let _ = perm.insert_from_back(1); // bbb at position 1
            let _ = perm.insert_from_back(2); // ccc at position 2
            leaf.set_permutation(perm);
        }

        // Test retrieval
        assert_eq!(*tree.get(b"aaa").unwrap(), 100);
        assert_eq!(*tree.get(b"bbb").unwrap(), 200);
        assert_eq!(*tree.get(b"ccc").unwrap(), 300);

        // Non-existent keys
        assert!(tree.get(b"ddd").is_none());
        assert!(tree.get(b"aab").is_none());
    }

    // ========================================================================
    //  Index Mode Get Tests
    // ========================================================================

    #[test]
    fn test_index_get_after_manual_insert() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        // Manually insert into the root leaf
        if let RootNode::Leaf(leaf) = &mut tree.inner.root {
            let key = Key::new(b"test");
            let ikey = key.ikey();
            let keylenx = key.current_len() as u8;

            leaf.assign_value(0, ikey, keylenx, 999u64);

            let mut perm = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        // Get returns V directly (not Arc<V>)
        let result = tree.get(b"test");
        assert_eq!(result, Some(999));
    }

    // ========================================================================
    //  Edge Case Tests
    // ========================================================================

    #[test]
    fn test_get_empty_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert empty key
        if let RootNode::Leaf(leaf) = &mut tree.root {
            let key = Key::new(b"");
            leaf.assign_value(0, key.ikey(), 0, 123u64);

            let mut perm = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        assert_eq!(*tree.get(b"").unwrap(), 123);
    }

    #[test]
    fn test_get_max_inline_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert 8-byte key (max inline size)
        let key_bytes = b"12345678";

        if let RootNode::Leaf(leaf) = &mut tree.root {
            let key = Key::new(key_bytes);
            leaf.assign_value(0, key.ikey(), 8, 888u64);

            let mut perm = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        assert_eq!(*tree.get(key_bytes).unwrap(), 888);
    }

    #[test]
    fn test_get_binary_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Binary key with null bytes
        let key_bytes = &[0x00, 0x01, 0x02, 0x00, 0xFF];

        if let RootNode::Leaf(leaf) = &mut tree.root {
            let key = Key::new(key_bytes);
            leaf.assign_value(0, key.ikey(), key.current_len() as u8, 777u64);

            let mut perm = leaf.permutation();
            let _ = perm.insert_from_back(0);
            leaf.set_permutation(perm);
        }

        assert_eq!(*tree.get(key_bytes).unwrap(), 777);
    }

    // ========================================================================
    //  Insert Tests
    // ========================================================================

    #[test]
    fn test_insert_into_empty_tree() {
        let mut tree: MassTree<u64> = MassTree::new();

        let result = tree.insert(b"hello", 42);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No old value
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_insert_and_get() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello", 42).unwrap();

        let value = tree.get(b"hello");
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 42);
    }

    #[test]
    fn test_insert_multiple_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"aaa", 100).unwrap();
        tree.insert(b"bbb", 200).unwrap();
        tree.insert(b"ccc", 300).unwrap();

        assert_eq!(tree.len(), 3);
        assert_eq!(*tree.get(b"aaa").unwrap(), 100);
        assert_eq!(*tree.get(b"bbb").unwrap(), 200);
        assert_eq!(*tree.get(b"ccc").unwrap(), 300);
    }

    #[test]
    fn test_insert_updates_existing() {
        let mut tree: MassTree<u64> = MassTree::new();

        // First insert
        let old = tree.insert(b"key", 100).unwrap();
        assert!(old.is_none());

        // Update
        let old = tree.insert(b"key", 200).unwrap();
        assert_eq!(*old.unwrap(), 100);

        // Verify new value
        assert_eq!(*tree.get(b"key").unwrap(), 200);
        assert_eq!(tree.len(), 1); // Still one key
    }

    #[test]
    fn test_insert_returns_old_arc() {
        let mut tree: MassTree<String> = MassTree::new();

        tree.insert(b"key", "first".to_string()).unwrap();

        let old = tree.insert(b"key", "second".to_string()).unwrap();
        assert_eq!(*old.unwrap(), "first");
    }

    #[test]
    fn test_insert_ascending_order() {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in 0..10 {
            let key = format!("key{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 10);

        // Verify all values
        for i in 0..10 {
            let key = format!("key{i:02}");
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_descending_order() {
        let mut tree: MassTree<u64> = MassTree::new();

        for i in (0..10).rev() {
            let key = format!("key{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 10);

        for i in 0..10 {
            let key = format!("key{i:02}");
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_random_order() {
        let mut tree: MassTree<u64> = MassTree::new();

        let keys = ["dog", "cat", "bird", "fish", "ant", "bee"];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 6);

        for (i, key) in keys.iter().enumerate() {
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_triggers_split() {
        let mut tree: MassTree<u64, 15> = MassTree::new();

        // Fill the leaf (15 slots)
        for i in 0..15 {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Before 16th insert: root is a leaf
        assert!(tree.root.is_leaf());

        // 16th insert should trigger a split
        let result = tree.insert(b"overflow", 999);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // New key, no old value

        // After split: root should be an internode
        assert!(tree.root.is_internode());

        // Verify the new key is accessible
        let value = tree.get(b"overflow");
        assert_eq!(*value.unwrap(), 999);

        // Verify all original keys are still accessible
        for i in 0..15 {
            let key = format!("{i:02}");
            let value = tree.get(key.as_bytes());

            assert!(value.is_some(), "Key {key:?} not found after split");
            assert_eq!(*value.unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_at_capacity() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert exactly WIDTH keys
        for i in 0..15 {
            let key = vec![b'a' + i as u8];
            assert!(tree.insert(&key, i as u64).is_ok());
        }

        assert_eq!(tree.len(), 15);
    }

    #[test]
    fn test_insert_error_display() {
        let err = InsertError::LeafFull;
        assert_eq!(format!("{err}"), "leaf node is full");

        let err = InsertError::AllocationFailed;
        assert_eq!(format!("{err}"), "memory allocation failed");
    }

    #[test]
    fn test_insert_max_key_length() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Exactly 8 bytes should work
        let max_key = b"12345678";
        let result = tree.insert(max_key, 42);
        assert!(result.is_ok());
        assert_eq!(*tree.get(max_key).unwrap(), 42);
    }

    #[test]
    fn test_slot0_reuse_no_predecessor() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Root leaf has no predecessor, so slot 0 should be usable
        tree.insert(b"first", 1).unwrap();

        // Verify the value was stored
        assert_eq!(*tree.get(b"first").unwrap(), 1);
    }

    #[test]
    fn test_slot0_reuse_same_ikey() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert key with ikey X
        tree.insert(b"aaa", 1).unwrap();

        // Insert another key with same ikey prefix (within 8 bytes)
        // This should allow slot 0 reuse if needed
        tree.insert(b"aab", 2).unwrap();

        assert_eq!(*tree.get(b"aaa").unwrap(), 1);
        assert_eq!(*tree.get(b"aab").unwrap(), 2);
    }

    #[test]
    fn test_permutation_updates_correctly() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert keys in reverse order
        tree.insert(b"ccc", 3).unwrap();
        tree.insert(b"bbb", 2).unwrap();
        tree.insert(b"aaa", 1).unwrap();

        // Permutation should maintain sorted order
        if let RootNode::Leaf(leaf) = &tree.root {
            let perm = leaf.permutation();
            assert_eq!(perm.size(), 3);
        }

        // Verify get still works (relies on correct permutation)
        assert_eq!(*tree.get(b"aaa").unwrap(), 1);
        assert_eq!(*tree.get(b"bbb").unwrap(), 2);
        assert_eq!(*tree.get(b"ccc").unwrap(), 3);
    }

    #[test]
    fn test_index_insert_basic() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        let result = tree.insert(b"key", 42);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(tree.get(b"key"), Some(42));
    }

    #[test]
    fn test_index_insert_update() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        tree.insert(b"key", 100).unwrap();

        let old = tree.insert(b"key", 200).unwrap();

        assert_eq!(old, Some(100)); // Old value returned directly (not Arc)
        assert_eq!(tree.get(b"key"), Some(200));
    }

    #[test]
    fn test_insert_empty_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"", 42).unwrap();

        assert_eq!(*tree.get(b"").unwrap(), 42);
    }

    #[test]
    fn test_insert_max_inline_key() {
        let mut tree: MassTree<u64> = MassTree::new();

        let key = b"12345678"; // Exactly 8 bytes

        tree.insert(key, 888).unwrap();

        assert_eq!(*tree.get(key).unwrap(), 888);
    }

    #[test]
    fn test_insert_binary_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        let keys = [
            vec![0x00, 0x01, 0x02],
            vec![0xFF, 0xFE, 0xFD],
            vec![0x00, 0x00, 0x00],
            vec![0xFF, 0xFF, 0xFF],
        ];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }

        for (i, key) in keys.iter().enumerate() {
            assert_eq!(*tree.get(key).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_similar_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"test", 1).unwrap();
        tree.insert(b"test1", 2).unwrap();
        tree.insert(b"test12", 3).unwrap();
        tree.insert(b"tes", 4).unwrap();

        assert_eq!(*tree.get(b"test").unwrap(), 1);
        assert_eq!(*tree.get(b"test1").unwrap(), 2);
        assert_eq!(*tree.get(b"test12").unwrap(), 3);
        assert_eq!(*tree.get(b"tes").unwrap(), 4);
    }

    #[test]
    fn test_insert_arc_basic() {
        let mut tree: MassTree<u64> = MassTree::new();

        let arc = Arc::new(42u64);
        let result = tree.insert_arc(b"key", arc);

        assert!(result.is_ok());
        assert_eq!(*tree.get(b"key").unwrap(), 42);
    }

    #[test]
    fn test_insert_arc_update() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert_arc(b"key", Arc::new(100)).unwrap();
        let old = tree.insert_arc(b"key", Arc::new(200)).unwrap();

        assert_eq!(*old.unwrap(), 100);
        assert_eq!(*tree.get(b"key").unwrap(), 200);
    }

    // ========================================================================
    //  Differential Testing
    // ========================================================================

    #[test]
    fn test_differential_small_sequential() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for i in 0..10 {
            let key = vec![b'a' + i as u8];
            let tree_old = tree.insert(&key, i as u64).unwrap();
            let oracle_old = oracle.insert(key.clone(), i as u64);

            assert_eq!(
                tree_old.map(|arc| *arc),
                oracle_old,
                "Insert mismatch for key {key:?}"
            );
        }
    }

    #[test]
    fn test_differential_updates() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        let key = b"key".to_vec();

        for value in [100, 200, 300] {
            let tree_old = tree.insert(&key, value).unwrap();
            let oracle_old = oracle.insert(key.clone(), value);
            assert_eq!(tree_old.map(|arc| *arc), oracle_old);
        }

        assert_eq!(*tree.get(&key).unwrap(), *oracle.get(&key).unwrap());
    }

    #[test]
    fn test_differential_interleaved() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        tree.insert(b"a", 1).unwrap();
        oracle.insert(b"a".to_vec(), 1);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );

        tree.insert(b"b", 2).unwrap();
        oracle.insert(b"b".to_vec(), 2);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );
        assert_eq!(
            tree.get(b"b").map(|a| *a),
            oracle.get(b"b".as_slice()).copied()
        );

        tree.insert(b"a", 10).unwrap();
        oracle.insert(b"a".to_vec(), 10);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );
    }

    #[test]
    fn test_differential_get_missing() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        assert_eq!(
            tree.get(b"missing").map(|a| *a),
            oracle.get(b"missing".as_slice()).copied()
        );

        tree.insert(b"present", 42).unwrap();

        assert_eq!(tree.get(b"missing").map(|a| *a), None);
        assert_eq!(*tree.get(b"present").unwrap(), 42);
    }

    // ========================================================================
    //  Split Tests (CODE_009.md)
    // ========================================================================

    #[test]
    fn test_insert_100_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert 100 unique keys (will trigger multiple splits)
        for i in 0..100 {
            let key = format!("{i:08}"); // 8-byte keys
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify all keys are retrievable
        for i in 0..100 {
            let key = format!("{i:08}");
            let value = tree.get(key.as_bytes());
            assert!(value.is_some(), "Key {i} not found");
            assert_eq!(*value.unwrap(), i as u64);
        }

        // Root should be internode (multiple splits occurred)
        assert!(tree.root.is_internode());

        // Verify len() works on internode-root trees
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_len_multi_level_tree() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert enough keys to create multiple levels
        for i in 0..200 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify len() returns correct count
        assert_eq!(tree.len(), 200);
        assert!(tree.root.is_internode());

        // Insert more and verify again
        for i in 200..500 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }
        assert_eq!(tree.len(), 500);
    }

    #[test]
    fn test_insert_1000_sequential() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert 1000 keys sequentially
        for i in 0..1000 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
            oracle.insert(key.into_bytes(), i as u64);
        }

        // Differential check: all keys match oracle
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} not found in tree");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_insert_1000_random() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert 1000 keys in "random" order (deterministic pattern)
        for i in 0..1000 {
            // Mix up the order using modular arithmetic
            let idx = (i * 997) % 1000;
            let key = format!("{idx:08}");
            tree.insert(key.as_bytes(), idx as u64).unwrap();
            oracle.insert(key.into_bytes(), idx as u64);
        }

        // Differential check
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} not found");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_split_preserves_values() {
        let mut tree: MassTree<String> = MassTree::new();

        // Insert values with distinct content
        let test_values: Vec<(&[u8], String)> = vec![
            (b"aaa", "first value".to_string()),
            (b"bbb", "second value".to_string()),
            (b"ccc", "third value".to_string()),
            (b"ddd", "fourth value".to_string()),
            (b"eee", "fifth value".to_string()),
        ];

        for (key, value) in &test_values {
            tree.insert(key, value.clone()).unwrap();
        }

        // Fill to trigger split
        for i in 0..20 {
            let key = format!("k{i:02}");
            tree.insert(key.as_bytes(), format!("value{i}")).unwrap();
        }

        // Verify original values survived split
        for (key, expected) in &test_values {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} lost after split");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_split_with_updates() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert and immediately update some keys
        for i in 0..30 {
            let key = format!("{i:04}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Update every other key
        for i in (0..30).step_by(2) {
            let key = format!("{i:04}");
            let old = tree.insert(key.as_bytes(), (i + 1000) as u64).unwrap();
            assert_eq!(*old.unwrap(), i as u64);
        }

        // Verify
        for i in 0..30 {
            let key = format!("{i:04}");
            let expected = if i % 2 == 0 { i + 1000 } else { i };
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), expected as u64);
        }
    }

    #[test]
    fn test_multiple_splits_create_deeper_tree() {
        // With WIDTH=3, many splits needed for 50 keys
        let mut tree: MassTree<u64, 3> = MassTree::new();

        for i in 0..50 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify all keys accessible
        for i in 0..50 {
            let key = format!("{i:08}");
            assert!(
                tree.get(key.as_bytes()).is_some(),
                "Key {i} not found with WIDTH=3"
            );
        }

        // Tree must have internodes
        assert!(tree.root.is_internode());
    }

    #[test]
    fn test_split_empty_right_edge_case() {
        // Test the sequential optimization edge case
        let mut tree: MassTree<u64, 15> = MassTree::new();

        // Fill leaf completely with sequential keys (rightmost optimization)
        for i in 0..15 {
            tree.insert(&[i as u8], i as u64).unwrap();
        }

        // 16th insert at end should create empty right leaf
        tree.insert(&[15u8], 15).unwrap();

        // Verify all keys
        for i in 0..16 {
            assert_eq!(*tree.get(&[i as u8]).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_split_index_mode() {
        // Test splits work with MassTreeIndex too
        let mut tree: MassTreeIndex<u64, 15> = MassTreeIndex::new();

        for i in 0..50 {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        for i in 0..50 {
            let key = format!("{i:02}");
            assert_eq!(tree.get(key.as_bytes()), Some(i as u64));
        }
    }

    #[test]
    fn test_split_maintains_key_order() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert in reverse order
        for i in (0..50).rev() {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Keys should still be retrievable
        for i in 0..50 {
            let key = format!("{i:02}");
            let value = tree.get(key.as_bytes());
            assert!(value.is_some(), "Key {i:02} not found");
            assert_eq!(*value.unwrap(), i as u64);
        }
    }

    // ========================================================================
    //  Layer Tests (CODE_011.md)
    // ========================================================================

    #[test]
    fn test_layer_creation_same_prefix() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert "hello world!" (12 bytes)
        tree.insert(b"hello world!", 1).unwrap();

        // Insert "hello worm" (10 bytes) - same 8-byte prefix "hello wo"
        tree.insert(b"hello worm", 2).unwrap();

        // Both should be retrievable
        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
    }

    #[test]
    fn test_deep_layer_chain() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Two 24-byte keys with 16 bytes common prefix
        let key1 = b"aaaaaaaabbbbbbbbXXXXXXXX"; // 24 bytes
        let key2 = b"aaaaaaaabbbbbbbbYYYYYYYY"; // 24 bytes, differs at byte 16

        tree.insert(key1, 1).unwrap();
        tree.insert(key2, 2).unwrap();

        // Should create 2 intermediate twig nodes:
        // Layer 0: ikey="aaaaaaaa" → twig
        // Layer 1: ikey="bbbbbbbb" → twig
        // Layer 2: ikey="XXXXXXXX" and "YYYYYYYY" (different, final leaf)

        assert_eq!(tree.get(key1).map(|v| *v), Some(1));
        assert_eq!(tree.get(key2).map(|v| *v), Some(2));
    }

    #[test]
    fn test_layer_with_suffixes() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Keys that create layers AND have suffixes
        let key1 = b"prefixAArest1"; // 13 bytes
        let key2 = b"prefixAArest2"; // 13 bytes, same first 8, different suffix

        tree.insert(key1, 1).unwrap();
        tree.insert(key2, 2).unwrap();

        assert_eq!(tree.get(key1).map(|v| *v), Some(1));
        assert_eq!(tree.get(key2).map(|v| *v), Some(2));
        assert_eq!(tree.get(b"prefixAArest3"), None);
    }

    #[test]
    fn test_insert_into_existing_layer() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Create a layer
        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Insert another key with same prefix, different continuation
        tree.insert(b"hello wonder", 3).unwrap();

        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
        assert_eq!(tree.get(b"hello wonder").map(|v| *v), Some(3));
    }

    #[test]
    fn test_update_in_layer() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Update existing key in layer
        let old = tree.insert(b"hello world!", 100).unwrap();
        assert_eq!(old.map(|v| *v), Some(1));

        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(100));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
    }

    #[test]
    fn test_get_nonexistent_in_layer() {
        let mut tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Same prefix but different suffix that wasn't inserted
        assert_eq!(tree.get(b"hello worst"), None);
        assert_eq!(tree.get(b"hello wo"), None); // Exact 8 bytes (partial match)
    }

    #[test]
    fn test_long_key_insert_and_get() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Keys longer than 8 bytes without common prefix
        let key1 = b"abcdefghijklmnop"; // 16 bytes
        let key2 = b"zyxwvutsrqponmlk"; // 16 bytes, different prefix

        tree.insert(key1, 100).unwrap();
        tree.insert(key2, 200).unwrap();

        assert_eq!(tree.get(key1).map(|v| *v), Some(100));
        assert_eq!(tree.get(key2).map(|v| *v), Some(200));
    }

    #[test]
    fn test_mixed_short_and_long_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Mix of short (<=8) and long (>8) keys
        tree.insert(b"short", 1).unwrap();
        tree.insert(b"exactly8", 2).unwrap();
        tree.insert(b"this is longer", 3).unwrap();
        tree.insert(b"another long key", 4).unwrap();

        assert_eq!(tree.get(b"short").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"exactly8").map(|v| *v), Some(2));
        assert_eq!(tree.get(b"this is longer").map(|v| *v), Some(3));
        assert_eq!(tree.get(b"another long key").map(|v| *v), Some(4));
    }

    #[test]
    fn test_layer_differential_vs_btreemap() {
        use std::collections::BTreeMap;

        let mut tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Test with keys that will create layers
        let keys = [
            b"hello world!".to_vec(),
            b"hello worm".to_vec(),
            b"hello wonder".to_vec(),
            b"goodbye world".to_vec(),
            b"goodbye friend".to_vec(),
            b"short".to_vec(),
            b"a".to_vec(),
        ];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
            oracle.insert(key.clone(), i as u64);
        }

        // Verify all oracle entries
        for (key, expected) in &oracle {
            let actual = tree.get(key).map(|v| *v);
            assert_eq!(actual, Some(*expected), "mismatch for key {key:?}");
        }
    }

    #[test]
    fn test_layer_count_tracking() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert keys that create layers
        tree.insert(b"hello world!", 1).unwrap();
        assert_eq!(tree.len(), 1);

        tree.insert(b"hello worm", 2).unwrap();
        assert_eq!(tree.len(), 2);

        tree.insert(b"hello wonder", 3).unwrap();
        assert_eq!(tree.len(), 3);

        // Update shouldn't change count
        tree.insert(b"hello world!", 100).unwrap();
        assert_eq!(tree.len(), 3);
    }
}
