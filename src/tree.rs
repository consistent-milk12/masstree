//! Filepath: src/tree.rs
//! `MassTree` - A high-performance concurrent trie of B+trees.
//!
//! This module provides the main `MassTree<V>` and `MassTreeIndex<V>` types.

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::alloc::{NodeAllocator, SeizeAllocator};
use crate::internode::InternodeNode;
use crate::key::Key;
use crate::ksearch::upper_bound_internode_direct;
use crate::leaf::{KSUF_KEYLENX, LeafNode, LeafSplitResult, LeafValue, SplitUtils};
use crate::nodeversion::NodeVersion;
use crate::permuter::Permuter;
use seize::{Collector, LocalGuard};

mod cas_insert;
mod index;
mod layer;
mod leaf_iterator;
mod locked;
mod optimistic;
mod split;
mod traverse;

#[cfg(all(test, loom, not(miri)))]
mod loom_tests;

#[cfg(all(test, not(miri)))]
mod shuttle_tests;

#[cfg(test)]
pub mod test_hooks;

pub use index::MassTreeIndex;

// Re-export debug counters
pub use optimistic::{
    ADVANCE_BLINK_COUNT, BLINK_SHOULD_FOLLOW_COUNT, CAS_INSERT_FALLBACK_COUNT,
    CAS_INSERT_RETRY_COUNT, CAS_INSERT_SUCCESS_COUNT, DebugCounters, LOCKED_INSERT_COUNT,
    SEARCH_NOT_FOUND_COUNT, SPLIT_COUNT, WRONG_LEAF_INSERT_COUNT, get_all_debug_counters,
    get_debug_counters, reset_debug_counters,
};

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

    /// Root-level split required.
    /// Concurrent root updates require changing `MassTree.root` to `AtomicPtr`.
    /// Fall back to single-threaded insert path.
    RootSplitRequired,

    /// Parent internode is full and needs splitting.
    /// Recursive internode split not yet fully implemented concurrently.
    /// Fall back to single-threaded insert path.
    ParentSplitRequired,
}

impl StdFmt::Display for InsertError {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::LeafFull => write!(f, "leaf node is full"),

            Self::AllocationFailed => write!(f, "memory allocation failed"),

            Self::RootSplitRequired => {
                write!(f, "root-level split required (use single-threaded insert)")
            }

            Self::ParentSplitRequired => {
                write!(f, "parent split required (use single-threaded insert)")
            }
        }
    }
}

impl std::error::Error for InsertError {}

/// A high-performance trie of B+trees for key-value storage.
///
/// Keys are byte slices of any length (0-256 bytes). Values are stored
/// as `Arc<V>` for cheap cloning on read operations.
///
/// # Type Parameters
///
/// * `V` - The value type to store
/// * `WIDTH` - Node width (default: 15, max: 15)
/// * `A` - Node allocator (default: `SeizeAllocator`)
///
/// # Allocator
///
/// The allocator determines how nodes are allocated and (eventually) freed:
///
/// - `SeizeAllocator` (default): Miri-compliant allocator using `Box::into_raw()`
///   for clean pointer provenance. Ready for concurrent access with seize-based
///   deferred reclamation.
///
/// - `ArenaAllocator`: Legacy allocator for testing. Causes Stacked Borrows
///   violations under Miri.
///
/// # Current Limitations
///
/// - Single-threaded access only (Phase 3 will add concurrency)
///
/// # Example
///
/// ```ignore
/// use masstree::MassTree;
///
/// let mut tree: MassTree<u64> = MassTree::new();
/// tree.insert(b"hello", 42).unwrap();
///
/// let value = tree.get(b"hello");
/// assert_eq!(value.map(|v| *v), Some(42));
/// ```
pub struct MassTree<
    V,
    const WIDTH: usize = 15,
    A: NodeAllocator<LeafValue<V>, WIDTH> = SeizeAllocator<LeafValue<V>, WIDTH>,
> {
    /// Memory reclamation collector for safe concurrent access.
    ///
    /// Uses seize's hyaline-based reclamation to safely retire nodes
    /// and values that may still be accessed by concurrent readers.
    collector: Collector,

    /// Node allocator for leaf and internode allocation.
    allocator: A,

    /// Atomic root pointer for concurrent access.
    ///
    /// Points to the same node as `root` but can be atomically updated.
    /// Used by `insert_with_guard`, `get_with_guard`, and concurrent splits.
    /// The node type (leaf vs internode) is determined by the node's version.
    root_ptr: AtomicPtr<u8>,

    /// Number of key-value pairs in the tree (atomic for concurrent access).
    count: AtomicUsize,

    /// Marker to indicate V must be Send + Sync for concurrent access.
    _marker: PhantomData<V>,
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> StdFmt::Debug
    for MassTree<V, WIDTH, A>
{
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTree")
            .field("root_ptr", &self.root_ptr.load(Ordering::Relaxed))
            .field("count", &self.count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<V, const WIDTH: usize> MassTree<V, WIDTH, SeizeAllocator<LeafValue<V>, WIDTH>> {
    /// Create a new empty `MassTree` with the default seize allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn new() -> Self {
        Self::with_allocator(SeizeAllocator::new())
    }
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Create a new empty `MassTree` with a custom allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn with_allocator(mut allocator: A) -> Self {
        // Create root leaf and register with allocator.
        // The allocator tracks it for cleanup — no Box stored in MassTree.
        let root_leaf: Box<LeafNode<LeafValue<V>, WIDTH>> = LeafNode::new_root();
        let root_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = allocator.alloc_leaf(root_leaf);

        Self {
            collector: Collector::new(),
            allocator,
            root_ptr: AtomicPtr::new(root_ptr.cast::<u8>()),
            count: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// Enter a protected region and return a guard.
    ///
    /// The guard protects any pointers loaded during its lifetime from being
    /// reclaimed. Call this before reading tree nodes or values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let guard = tree.guard();
    /// let value = tree.get_with_guard(key, &guard);
    /// // guard dropped, reclamation can proceed
    /// ```
    #[inline]
    #[must_use]
    pub fn guard(&self) -> LocalGuard<'_> {
        self.collector.enter()
    }

    /// Store a leaf in the allocator and return a raw pointer.
    ///
    /// The pointer remains valid for the lifetime of the tree.
    #[inline]
    fn alloc_leaf(
        &mut self,
        leaf: Box<LeafNode<LeafValue<V>, WIDTH>>,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        self.allocator.alloc_leaf(leaf)
    }

    /// Store an internode in the allocator and return a raw pointer.
    #[inline]
    fn alloc_internode(
        &mut self,
        node: Box<InternodeNode<LeafValue<V>, WIDTH>>,
    ) -> *mut InternodeNode<LeafValue<V>, WIDTH> {
        self.allocator.alloc_internode(node)
    }

    // ========================================================================
    //  Atomic Root Access
    // ========================================================================

    /// Load the root pointer atomically.
    ///
    /// Used by concurrent operations to get the current root.
    /// The node type (leaf vs internode) is determined by the node's version.
    #[inline]
    pub(crate) fn load_root_ptr(&self, _guard: &LocalGuard<'_>) -> *const u8 {
        self.root_ptr.load(Ordering::Acquire)
    }

    /// Compare-and-swap the root pointer atomically.
    ///
    /// Returns `Ok(())` if the swap succeeded, `Err(current)` if the current
    /// value was not equal to `expected`.
    ///
    /// # Safety
    ///
    /// The `new` pointer must point to a valid node that will remain valid
    /// for the lifetime of the tree. The old root (if different from new)
    /// must remain valid for concurrent readers (via seize retirement).
    #[inline]
    pub(crate) fn cas_root_ptr(&self, expected: *mut u8, new: *mut u8) -> Result<(), *mut u8> {
        self.root_ptr
            .compare_exchange(expected, new, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
    }

    /// Store a new root pointer atomically.
    ///
    /// Used when we're certain no concurrent root update is happening
    /// (e.g., under a lock or single-threaded context).
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn store_root_ptr(&self, new: *mut u8) {
        self.root_ptr.store(new, Ordering::Release);
    }

    /// Check if the current root is a leaf node.
    ///
    /// # Safety
    /// Reads the version field through a raw pointer. The `root_ptr` must
    /// point to a valid node (guaranteed by construction).
    #[inline]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "root_ptr points to LeafNode or InternodeNode, both have NodeVersion \
                  as first field with proper alignment; we only store *const u8 for type erasure"
    )]
    fn root_is_leaf(&self) -> bool {
        let root: *const u8 = self.root_ptr.load(Ordering::Acquire);

        // SAFETY: `root_ptr` always points to a valid node.
        // `NodeVersion` is the first field of both LeafNode and InternodeNode.
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();

        unsafe { (*version_ptr).is_leaf() }
    }

    /// Get the root pointer as a leaf node pointer.
    ///
    /// # Safety
    /// Caller must ensure `root_is_leaf()` returned true.
    #[inline]
    #[allow(dead_code)]
    unsafe fn root_as_leaf_ptr(&self) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        self.root_ptr.load(Ordering::Acquire).cast()
    }

    /// Get the root pointer as an internode pointer.
    ///
    /// # Safety
    ///
    /// Caller must ensure `root_is_leaf()` returned false.
    #[inline]
    #[allow(dead_code)]
    unsafe fn root_as_internode_ptr(&self) -> *mut InternodeNode<LeafValue<V>, WIDTH> {
        self.root_ptr.load(Ordering::Acquire).cast()
    }

    /// Check if the tree is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        if self.root_is_leaf() {
            let leaf_ptr: *const LeafNode<LeafValue<V>, WIDTH> =
                self.root_ptr.load(Ordering::Acquire).cast();

            unsafe { (*leaf_ptr).is_empty() }
        } else {
            // Internode implies at least one key
            false
        }
    }

    /// Get the number of keys in the tree.
    ///
    /// This is O(1) as we track the count incrementally.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
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
    ///
    /// # Thread Safety
    ///
    /// This method is safe for concurrent use. It internally creates a guard
    /// and uses version-validated reads. For bulk operations, prefer
    /// [`get_with_guard`](Self::get_with_guard) to amortize guard creation cost.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<Arc<V>> {
        let guard = self.guard();
        let mut search_key: Key<'_> = Key::new(key);
        self.get_concurrent(&mut search_key, &guard)
    }

    /// Traverse from an internode down to the target leaf.
    ///
    /// Uses `upper_bound_internode_direct` to find the correct child at each level.
    #[expect(
        clippy::unused_self,
        reason = "Method signature matches reach_leaf pattern"
    )]
    fn reach_leaf_via_internode(
        &self,
        mut inode: &InternodeNode<LeafValue<V>, WIDTH>,
        key: &Key<'_>,
    ) -> &LeafNode<LeafValue<V>, WIDTH> {
        let target_ikey: u64 = key.ikey();

        loop {
            // Find child index
            let child_idx: usize = upper_bound_internode_direct(target_ikey, inode);
            let child_ptr: *mut u8 = inode.child(child_idx);

            // Check child type via NodeVersion
            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to LeafNode or InternodeNode, both properly aligned"
            )]
            let child_version: &NodeVersion = unsafe { &*(child_ptr.cast::<NodeVersion>()) };

            if child_version.is_leaf() {
                // SAFETY: is_leaf() confirms LeafNode
                return unsafe { &*(child_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>()) };
            }

            // Descend to child internode
            // SAFETY: !is_leaf() confirms InternodeNode
            inode = unsafe { &*(child_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };
        }
    }

    /// Mutable version of `reach_leaf_from_ptr`.
    ///
    /// See [`reach_leaf_from_ptr`] for the `maybe_parent` pattern explanation.
    #[inline(always)]
    fn reach_leaf_from_ptr_mut(
        &mut self,
        root_ptr: *mut u8,
        key: &Key<'_>,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        // SAFETY: Both LeafNode and InternodeNode have NodeVersion as first field
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root_ptr points to LeafNode or InternodeNode, both properly aligned"
        )]
        let version: &NodeVersion = unsafe { &*(root_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // It's a leaf - check if the layer was promoted via parent pointer
            // SAFETY: version.is_leaf() confirms this is a LeafNode
            let leaf: &LeafNode<LeafValue<V>, WIDTH> =
                unsafe { &*(root_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>()) };

            let parent_ptr: *mut u8 = leaf.parent();

            if parent_ptr.is_null() {
                // Null parent means this leaf is still the layer root
                root_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>()
            } else {
                // Non-null parent means layer was promoted - follow the parent
                // SAFETY: parent_ptr is the internode created during promotion
                let inode: &InternodeNode<LeafValue<V>, WIDTH> =
                    unsafe { &*(parent_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };

                self.reach_leaf_via_internode_mut(inode, key)
            }
        } else {
            // SAFETY: !version.is_leaf() confirms this is an InternodeNode
            let inode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(root_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };

            self.reach_leaf_via_internode_mut(inode, key)
        }
    }

    /// Mutable traversal from internode to leaf.
    #[expect(
        clippy::unused_self,
        clippy::needless_pass_by_ref_mut,
        reason = "Method signature matches reach_leaf_mut pattern"
    )]
    fn reach_leaf_via_internode_mut(
        &mut self,
        mut inode: &InternodeNode<LeafValue<V>, WIDTH>,
        key: &Key<'_>,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        let target_ikey: u64 = key.ikey();

        loop {
            let child_idx: usize = upper_bound_internode_direct(target_ikey, inode);
            let child_ptr: *mut u8 = inode.child(child_idx);

            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to LeafNode or InternodeNode, both properly aligned"
            )]
            let child_version: &NodeVersion = unsafe { &*(child_ptr.cast::<NodeVersion>()) };

            if child_version.is_leaf() {
                // SAFETY: is_leaf() confirms LeafNode
                return child_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>();
            }

            // SAFETY: !is_leaf() confirms InternodeNode
            inode = unsafe { &*(child_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };
        }
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
        let mut layer_root: *mut LeafNode<LeafValue<V>, WIDTH> = std::ptr::null_mut();

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
            // FIXED: Use reach_leaf_from_ptr_mut for layer roots to handle
            // promoted layers (where the layer root became an internode)
            let leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = if layer_root.is_null() {
                self.reach_leaf_mut(key) as *mut _
            } else {
                self.reach_leaf_from_ptr_mut(layer_root.cast::<u8>(), key)
            };

            // SAFETY: leaf_ptr is valid (from reach_leaf_mut or layer_root which points to arena)
            let leaf: &mut LeafNode<LeafValue<V>, WIDTH> = unsafe { &mut *leaf_ptr };

            // Search for matching ikey in the leaf
            let perm: Permuter<WIDTH> = leaf.permutation();
            let mut found_slot: Option<usize> = None;
            let mut insert_pos: usize = perm.size();
            let mut descend_layer: Option<*mut LeafNode<LeafValue<V>, WIDTH>> = None;

            for i in 0..perm.size() {
                let slot: usize = perm.get(i);
                let slot_ikey: u64 = leaf.ikey(slot);

                if slot_ikey == ikey {
                    // Found matching ikey - check for exact match, layer, or suffix conflict
                    let match_result: i32 =
                        leaf.ksuf_match_result(slot, search_keylenx, key.suffix());

                    if match_result == 1 {
                        // Exact match - update value
                        let guard: LocalGuard<'_> = self.collector.enter();
                        // SAFETY: guard protects concurrent access, slot valid from permuter
                        let old_value: Option<Arc<V>> =
                            unsafe { leaf.swap_value(slot, value, &guard) };
                        return Ok(old_value);
                    }

                    if match_result < 0 {
                        // Layer pointer found
                        //
                        // FIXED: Only descend if key has more bytes to match.
                        // If key terminates at this layer boundary (!key.has_suffix()),
                        // the key is DISTINCT from anything in the layer.
                        // Continue searching - we may need to insert as inline key.
                        //
                        // C++ reference: masstree_insert.hh:36
                        if key.has_suffix() {
                            // Key has more bytes - must descend into layer
                            if let Some(layer_ptr) = leaf.get_layer(slot) {
                                #[expect(
                                    clippy::cast_sign_loss,
                                    reason = "match_result < 0, so -match_result > 0"
                                )]
                                key.shift_by((-match_result) as usize);
                                descend_layer =
                                    Some(layer_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>());
                            } else {
                                return Err(InsertError::LeafFull); // Layer pointer invalid
                            }

                            break;
                        }

                        // !key.has_suffix(): Key terminates at layer boundary.
                        // This is a DIFFERENT key from anything in the layer.
                        // Continue searching - we may need to insert as inline key.
                        continue;
                    }

                    // Same ikey but different key (keylenx or suffix mismatch)
                    // Check if we need to create a layer:
                    //
                    // FIXED: Only create layer if BOTH have suffixes.
                    // Inline (keylenx 0-8) and suffix keys can coexist with same ikey
                    // because they represent different full keys.
                    //
                    // C++ reference: masstree_insert.hh:36-45
                    let slot_has_suffix: bool = leaf.has_ksuf(slot);
                    let key_has_suffix: bool = key.has_suffix();

                    if slot_has_suffix && key_has_suffix {
                        // Both have suffixes with same 8-byte prefix - need layer to distinguish
                        found_slot = Some(slot);
                        insert_pos = i;
                        break;
                    }
                    // One is inline, one has suffix (or both inline with different keylenx)
                    // These are distinct keys that can coexist - continue searching
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
                let leaf_ref: &mut LeafNode<LeafValue<V>, WIDTH> = unsafe { &mut *leaf_ptr };
                // SAFETY: guard protects concurrent access (created inside make_new_layer)
                let (final_leaf_ptr, new_slot) = unsafe {
                    self.make_new_layer(leaf_ref, conflict_slot, key, Arc::clone(&value))
                };

                // Finish insert in the new layer leaf
                //  SAFETY: make_new_layer returns valid pointer from arena
                unsafe {
                    let final_leaf: &mut LeafNode<LeafValue<V>, WIDTH> = &mut *final_leaf_ptr;
                    let mut perm: Permuter<WIDTH> = final_leaf.permutation();
                    let _ = perm.insert_from_back(new_slot);
                    final_leaf.set_permutation(perm);
                }

                self.count.fetch_add(1, Ordering::Relaxed);
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
                    let guard: LocalGuard<'_> = self.collector.enter();
                    // SAFETY: guard protects concurrent access, slot just allocated
                    unsafe { leaf.assign_ksuf(slot, key.suffix(), &guard) };
                }

                // Commit permutation update
                leaf.set_permutation(perm);

                // Track new entry
                self.count.fetch_add(1, Ordering::Relaxed);

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
                let guard: LocalGuard<'_> = self.collector.enter();
                // SAFETY: guard protects concurrent access during splits
                let split_result: LeafSplitResult<LeafValue<V>, WIDTH> =
                    unsafe { leaf.split_all_to_right(&guard) };
                (split_result.new_leaf, split_result.split_ikey)
            } else {
                // Normal split
                let guard: LocalGuard<'_> = self.collector.enter();
                // SAFETY: guard protects concurrent access during splits
                let split_result: LeafSplitResult<LeafValue<V>, WIDTH> =
                    unsafe { leaf.split_into(pre_insert_split_pos, &guard) };
                (split_result.new_leaf, split_result.split_ikey)
            };

            // Propagate split up the tree
            self.propagate_split(leaf_ptr, new_leaf_box, split_ikey);

            // FIXED: Do NOT reset layer_root after split.
            //
            // For main tree splits: layer_root was already null, stays null.
            // On retry, reach_leaf_mut finds the correct leaf.
            //
            // For layer splits: layer_root still points to the old layer leaf.
            // On retry, reach_leaf_from_ptr_mut checks the parent pointer:
            // - If parent is null: leaf is still the layer root (no promotion)
            // - If parent is non-null: layer was promoted, follow parent internode
            //
            // This implements the C++ `maybe_parent()` pattern from
            // reference/masstree_struct.hh:83 and reference/masstree_get.hh:108.

            // Loop continues - reach_leaf_from_ptr_mut handles promotion
        }
    }
}

impl<V, const WIDTH: usize> Default for MassTree<V, WIDTH, SeizeAllocator<LeafValue<V>, WIDTH>> {
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
    // Send/Sync Verification
    // ========================================================================
    //
    // MassTree<V> is Send + Sync when V: Send + Sync.
    //
    // The struct uses:
    // - AtomicPtr<u8> for root_ptr (Send + Sync)
    // - AtomicUsize for count (Send + Sync)
    // - PhantomData<V> which inherits Send/Sync from V
    //
    // This enables concurrent access via Arc<MassTree<V>>.

    fn _assert_send_sync()
    where
        MassTree<u64>: Send + Sync,
    {
    }

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

        assert!(tree.root_is_leaf());
    }

    #[test]
    fn test_root_is_leaf_on_new_tree() {
        let tree: MassTree<u64> = MassTree::new();

        // New tree should have a leaf root
        assert!(tree.root_is_leaf());
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

        let leaf: &LeafNode<LeafValue<u64>> = tree.reach_leaf(&key);

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
        let tree: MassTree<u64> = MassTree::new();

        // Manually insert into the root leaf (key = "hello", 5 bytes <= 8)
        debug_assert!(tree.root_is_leaf());

        // SAFETY: New tree has leaf root - test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.root_ptr.load(Ordering::Acquire).cast() };

        let key: Key<'_> = Key::new(b"hello");
        let ikey: u64 = key.ikey();
        let keylenx: u8 = key.current_len() as u8;

        // Assign to slot 0 (plain field access, no atomics)
        leaf.assign_value(0, ikey, keylenx, 42u64);

        // Update permutation
        let mut perm = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

        // Now test get
        let result = tree.get(b"hello");
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), 42);
    }

    #[test]
    fn test_get_wrong_key_after_insert() {
        let tree: MassTree<u64> = MassTree::new();

        // Manually insert "hello" = 42
        debug_assert!(tree.root_is_leaf());

        // SAFETY: New tree has leaf root test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.root_ptr.load(Ordering::Acquire).cast() };

        let key: Key<'_> = Key::new(b"hello");
        let ikey: u64 = key.ikey();
        let keylenx: u8 = key.current_len() as u8;

        leaf.assign_value(0, ikey, keylenx, 42u64);

        let mut perm: Permuter = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

        assert!(tree.get(b"world").is_none());
        assert!(tree.get(b"hell").is_none());
        assert!(tree.get(b"helloX").is_none());
    }

    #[test]
    fn test_get_multiple_keys() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Use the insert API to add multiple keys
        tree.insert(b"aaa", 100).unwrap();
        tree.insert(b"bbb", 200).unwrap();
        tree.insert(b"ccc", 300).unwrap();

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
        let tree: MassTreeIndex<u64> = MassTreeIndex::new();

        // Manually insert into the root leaf
        debug_assert!(tree.inner.root_is_leaf());

        // SAFETY: New tree has leaf root - test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.inner.root_ptr.load(Ordering::Acquire).cast() };

        let key = Key::new(b"test");
        let ikey = key.ikey();
        let keylenx = key.current_len() as u8;

        leaf.assign_value(0, ikey, keylenx, 999u64);

        let mut perm = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

        // Get returns V directly (not Arc<V>)
        let result = tree.get(b"test");
        assert_eq!(result, Some(999));
    }

    // ========================================================================
    //  Edge Case Tests
    // ========================================================================

    #[test]
    fn test_get_empty_key() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert empty key
        debug_assert!(tree.root_is_leaf());

        // SAFETY: New tree has leaf root - test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.root_ptr.load(Ordering::Acquire).cast() };

        let key = Key::new(b"");
        leaf.assign_value(0, key.ikey(), 0, 123u64);

        let mut perm = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

        assert_eq!(*tree.get(b"").unwrap(), 123);
    }

    #[test]
    fn test_get_max_inline_key() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert 8-byte key (max inline size)
        let key_bytes = b"12345678";

        debug_assert!(tree.root_is_leaf());

        // SAFETY: New tree has leaf root - test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.root_ptr.load(Ordering::Acquire).cast() };

        let key = Key::new(key_bytes);
        leaf.assign_value(0, key.ikey(), 8, 888u64);

        let mut perm = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

        assert_eq!(*tree.get(key_bytes).unwrap(), 888);
    }

    #[test]
    fn test_get_binary_key() {
        let tree: MassTree<u64> = MassTree::new();

        // Binary key with null bytes
        let key_bytes = &[0x00, 0x01, 0x02, 0x00, 0xFF];

        debug_assert!(tree.root_is_leaf());

        // SAFETY: New tree has leaf root - test-only direct manipulation
        let leaf: &mut LeafNode<LeafValue<u64>, 15> =
            unsafe { &mut *tree.root_ptr.load(Ordering::Acquire).cast() };

        let key = Key::new(key_bytes);
        leaf.assign_value(0, key.ikey(), key.current_len() as u8, 777u64);

        let mut perm = leaf.permutation();
        let _ = perm.insert_from_back(0);
        leaf.set_permutation(perm);

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
        assert!(tree.root_is_leaf());

        // 16th insert should trigger a split
        let result = tree.insert(b"overflow", 999);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // New key, no old value

        // After split: root should be an internode
        assert!(!tree.root_is_leaf());

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
        debug_assert!(tree.root_is_leaf());

        // SAFETY: Tree has leaf root after 3 inserts
        let leaf: &LeafNode<LeafValue<u64>, 15> =
            unsafe { &*tree.root_ptr.load(Ordering::Acquire).cast() };
        let perm = leaf.permutation();
        assert_eq!(perm.size(), 3);

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
    //  Split Tests
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
        assert!(!tree.root_is_leaf());

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
        assert!(!tree.root_is_leaf());

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
        assert!(!tree.root_is_leaf());
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
    //  Layer Tests
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

    // ========================================================================
    //  Layer Root Growth Tests
    // ========================================================================

    #[test]
    fn test_layer_growth_beyond_width() {
        // Test that a layer can grow beyond WIDTH entries by promoting to internode
        let mut tree: MassTree<u64> = MassTree::new();

        // All keys share same 8-byte prefix, forcing them into a layer
        let prefix = b"samepfx!";

        // Insert more than WIDTH (15) keys - will trigger layer root split
        for i in 0..20u64 {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            tree.insert(&key, i).unwrap();
        }

        // All keys must be findable
        for i in 0..20u64 {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            let result = tree.get(&key);
            assert!(result.is_some(), "Key with suffix {i} not found");
            assert_eq!(*result.unwrap(), i);
        }
    }

    #[test]
    fn test_layer_root_becomes_internode() {
        // Test with default WIDTH - insert enough keys to trigger layer split
        let mut tree: MassTree<u64> = MassTree::new();

        // Force layer creation with suffix keys
        tree.insert(b"prefix00suffix_a", 1).unwrap();
        tree.insert(b"prefix00suffix_b", 2).unwrap();

        // These should go into the layer and eventually trigger split
        // With WIDTH=15, we need >15 keys in the layer to trigger split
        for i in 0..20u64 {
            let key = format!("prefix00suffix_{i:02}");
            tree.insert(key.as_bytes(), i + 100).unwrap();
        }

        // Verify initial keys still findable
        assert_eq!(*tree.get(b"prefix00suffix_a").unwrap(), 1);
        assert_eq!(*tree.get(b"prefix00suffix_b").unwrap(), 2);

        // Verify all numbered keys findable
        for i in 0..20u64 {
            let key = format!("prefix00suffix_{i:02}");
            let result = tree.get(key.as_bytes());
            assert!(result.is_some(), "Key {key} not found");
        }
    }

    #[test]
    fn test_layer_split_preserves_all_keys() {
        // Comprehensive test: insert many keys with shared prefix, verify all retrievable
        let mut tree: MassTree<u64> = MassTree::new();
        let mut keys_and_values: Vec<(Vec<u8>, u64)> = Vec::new();

        // Create keys that will all go into the same layer (share 8-byte prefix)
        for i in 0..50u64 {
            let mut key = b"testpfx!".to_vec(); // Exactly 8 bytes
            key.extend_from_slice(format!("suffix{i:04}").as_bytes());
            keys_and_values.push((key.clone(), i * 10));
            tree.insert(&key, i * 10).unwrap();
        }

        // Verify all keys
        for (key, expected) in &keys_and_values {
            let result = tree.get(key);
            assert!(result.is_some(), "Key {key:?} not found");
            assert_eq!(*result.unwrap(), *expected);
        }

        assert_eq!(tree.len(), 50);
    }
}
