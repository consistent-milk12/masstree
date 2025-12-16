//! Filepath: src/tree.rs
//! `MassTree` - A high-performance concurrent trie of B+trees.
//!
//! This module provides the main `MassTree<V>` and `MassTreeIndex<V>` types.

use crate::internode::InternodeNode;
use crate::key::Key;
use crate::ksearch::{KeyIndexPosition, lower_bound_leaf, upper_bound_internode_direct};
use crate::leaf::{LAYER_KEYLENX, LeafNode, LeafSplitResult, calculate_split_point};
use crate::permuter::Permuter;
use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::mem as StdMem;
use std::ptr as StdPtr;
use std::sync::Arc;

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

/// Maximum key length for Phase 1 (single layer).
/// Keys longer than this require layer traversal (Phase 2).
pub const MAX_INLINE_KEY_LEN: usize = 8;

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

    /// Key is too long.
    ///
    /// Phase 1: Keys must be <= 8 bytes (single layer only).
    /// Phase 2+: Keys can be up to 256 bytes (multi-layer support).
    KeyTooLong,
}

impl StdFmt::Display for InsertError {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::LeafFull => write!(f, "leaf node is full"),

            Self::AllocationFailed => write!(f, "memory allocation failed"),

            Self::KeyTooLong => write!(f, "key exceeds maximum length"),
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
            _not_send_sync: PhantomData,
        }
    }

    /// Store an existing leaf Box in the arena and return a raw pointer.
    ///
    /// The pointer remains valid for the lifetime of the tree.
    fn store_leaf_in_arena(&mut self, leaf: Box<LeafNode<V, WIDTH>>) -> *mut LeafNode<V, WIDTH> {
        self.leaf_arena.push(leaf);
        let idx: usize = self.leaf_arena.len() - 1;
        // SAFETY: We just pushed, so idx is valid. We derive the pointer AFTER storing
        // to maintain Stacked Borrows provenance.
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

        // SAFETY: We just pushed, so idx is valid. We derive the pointer AFTER storing
        // to maintain Stacked Borrows provenance.
        unsafe {
            StdPtr::from_mut::<InternodeNode<V, WIDTH>>(
                self.internode_arena.get_unchecked_mut(idx).as_mut(),
            )
        }
    }

    /// Find the index of a child pointer in an internode.
    ///
    /// # Panics
    ///
    /// Panics if the child is not found. This indicates a bug in split propagation.
    #[expect(
        clippy::panic,
        reason = "Intentional invariant check - child must exist during split propagation"
    )]
    fn find_child_index(internode: &InternodeNode<V, WIDTH>, child: *mut u8) -> usize {
        let nkeys: usize = internode.size();

        for i in 0..=nkeys {
            if internode.child(i) == child {
                return i;
            }
        }

        // INVARIANT: This is only called during split propagation when we know
        // the child was just inserted into this internode. If we reach here,
        // there's a bug in the split logic. Fail loudly rather than silently
        // returning an incorrect index that could corrupt the tree.
        panic!(
            "Internal invariant violation: child pointer {child:p} not found in internode \
             (nkeys={nkeys}). This indicates a bug in split propagation logic."
        );
    }

    // ========================================================================
    //  Split Propagation
    // ========================================================================

    /// Create a new root internode when the root leaf splits.
    ///
    /// **Root ownership strategy:**
    /// - Root is directly owned by `RootNode` (Box<T>)
    /// - Non-root nodes are stored in arenas (raw pointers)
    /// - When root changes: old root moves to arena, new root is Box in `RootNode`
    ///
    /// **Returns:** Arena-derived pointer to the left leaf (for proper Stacked Borrows provenance).
    /// The caller should use this pointer for leaf linking instead of the original.
    fn create_root_internode(
        &mut self,
        right_leaf: *mut LeafNode<V, WIDTH>,
        split_ikey: u64,
    ) -> *mut LeafNode<V, WIDTH> {
        // Create new root internode with height=0 (children are leaves per CODE_005.md)
        // Don't set children yet - we need arena-derived pointers for proper provenance
        let mut new_root: Box<InternodeNode<V, WIDTH>> = InternodeNode::new_root(0);
        new_root.set_ikey(0, split_ikey);
        new_root.set_nkeys(1);

        // Move old root leaf to arena FIRST to get proper provenance
        // The old RootNode::Leaf is replaced, invalidating any pointers derived from it
        let old_root: RootNode<V, WIDTH> =
            StdMem::replace(&mut self.root, RootNode::Internode(new_root));

        // Get arena-derived pointer for left leaf (proper Stacked Borrows provenance)
        let arena_left_ptr: *mut LeafNode<V, WIDTH> = if let RootNode::Leaf(old_leaf_box) = old_root
        {
            self.leaf_arena.push(old_leaf_box);
            let idx: usize = self.leaf_arena.len() - 1;

            // SAFETY: We just pushed, so idx is valid
            unsafe { StdPtr::from_mut(self.leaf_arena.get_unchecked_mut(idx).as_mut()) }
        } else {
            // This branch shouldn't happen - we only call this when root was a leaf
            debug_assert!(false, "create_root_internode called with non-leaf root");
            StdPtr::null_mut()
        };

        // Now set up children with arena-derived pointer (left) and already-arena pointer (right)
        // Also get the new root pointer for parent updates
        let RootNode::Internode(root_internode) = &mut self.root else {
            unreachable!("We just set root to Internode")
        };

        root_internode.set_child(0, arena_left_ptr.cast::<u8>());
        root_internode.set_child(1, right_leaf.cast::<u8>());

        // Get raw pointer to new root for parent updates
        let new_root_ptr: *mut u8 =
            StdPtr::from_mut::<InternodeNode<V, WIDTH>>(root_internode.as_mut()).cast();

        // Update leaves' parent pointers
        // SAFETY: arena_left_ptr is valid (just derived from arena), right_leaf is arena-backed
        unsafe {
            (*arena_left_ptr).set_parent(new_root_ptr);
            (*right_leaf).set_parent(new_root_ptr);
        }

        arena_left_ptr
    }

    /// Propagate a leaf split up the tree.
    ///
    /// Inserts the split key into the parent internode.
    /// If parent is full, recursively splits.
    /// If there's no parent (root was a leaf), creates new root internode.
    ///
    /// # Arguments
    ///
    /// * `left_leaf` - The original leaf (left sibling after split)
    /// * `right_leaf_box` - The new leaf (right sibling), to be added to arena
    /// * `split_ikey` - The split key to insert into parent
    fn propagate_split(
        &mut self,
        left_leaf: *mut LeafNode<V, WIDTH>,
        right_leaf_box: Box<LeafNode<V, WIDTH>>,
        split_ikey: u64,
    ) {
        // Store the new leaf in the arena to get a stable pointer
        let right_leaf_ptr: *mut LeafNode<V, WIDTH> = self.store_leaf_in_arena(right_leaf_box);

        // Check if left_leaf was the root BEFORE linking (linking uses left_leaf which
        // may have provenance that will be invalidated by create_root_internode)
        let left_was_root: bool = if let RootNode::Leaf(root_leaf) = &self.root {
            StdPtr::eq(root_leaf.as_ref(), &raw const *left_leaf)
        } else {
            false
        };

        if left_was_root {
            // Root was a leaf - create_root_internode will give us arena-derived pointer
            let arena_left_ptr: *mut LeafNode<V, WIDTH> =
                self.create_root_internode(right_leaf_ptr, split_ikey);

            // Link leaves using arena-derived pointer (proper provenance)
            // SAFETY: arena_left_ptr is valid from arena, right_leaf_ptr is from arena
            unsafe {
                let old_next: *mut LeafNode<V, WIDTH> = (*arena_left_ptr).safe_next();

                (*arena_left_ptr).set_next(right_leaf_ptr);
                (*right_leaf_ptr).set_prev(arena_left_ptr);
                (*right_leaf_ptr).set_next(old_next);

                if !old_next.is_null() {
                    (*old_next).set_prev(right_leaf_ptr);
                }
            }
            return;
        }

        // Non-root case: left_leaf has valid provenance (from arena)
        // Link leaves
        // SAFETY: left_leaf is a valid arena pointer, right_leaf_ptr is from our arena
        unsafe {
            let old_next: *mut LeafNode<V, WIDTH> = (*left_leaf).safe_next();

            (*left_leaf).set_next(right_leaf_ptr);
            (*right_leaf_ptr).set_prev(left_leaf);
            (*right_leaf_ptr).set_next(old_next);

            if !old_next.is_null() {
                (*old_next).set_prev(right_leaf_ptr);
            }
        }

        // Get parent from left_leaf
        // SAFETY: left_leaf is a valid pointer we just split
        let parent_ptr: *mut u8 = unsafe { (*left_leaf).parent() };

        // This shouldn't happen if tree invariants are maintained
        assert!(!parent_ptr.is_null(), "Non-root leaf has null parent");

        // Insert split key into parent
        // SAFETY: parent_ptr is valid from leaf's parent pointer
        let parent: &mut InternodeNode<V, WIDTH> =
            unsafe { &mut *parent_ptr.cast::<InternodeNode<V, WIDTH>>() };

        let child_idx: usize = Self::find_child_index(parent, left_leaf.cast::<u8>());

        if parent.size() < WIDTH {
            // Parent has room, just insert
            parent.insert_key_and_child(child_idx, split_ikey, right_leaf_ptr.cast::<u8>());

            // Update right_leaf's parent pointer
            unsafe {
                (*right_leaf_ptr).set_parent(parent_ptr);
            }
        } else {
            // Parent is full, need to split it too (using split+insert API from CODE_005.md)
            // Allocate new internode for the right half
            let new_parent: Box<InternodeNode<V, WIDTH>> = InternodeNode::new(parent.height());
            let new_parent_ptr: *mut InternodeNode<V, WIDTH> =
                self.store_internode_in_arena(new_parent);

            // Split parent AND insert the split_ikey/right_leaf simultaneously
            let (popup_key, insert_went_left) = unsafe {
                parent.split_into(
                    &mut *new_parent_ptr,
                    child_idx,
                    split_ikey,
                    right_leaf_ptr.cast::<u8>(),
                )
            };

            // Set right_leaf's parent based on where the insert went
            unsafe {
                if insert_went_left {
                    (*right_leaf_ptr).set_parent(parent_ptr);
                } else {
                    (*right_leaf_ptr).set_parent(new_parent_ptr.cast::<u8>());
                }
            }

            // Recursively propagate the popup key
            // parent_ptr points to arena-backed internode, safe to use
            self.propagate_internode_split(parent_ptr.cast(), new_parent_ptr, popup_key);
        }
    }

    /// Propagate an internode split up the tree.
    ///
    /// Called after splitting an internode to propagate the split to the parent.
    /// Uses the correct C++ semantics where parent split includes insertion.
    ///
    /// # Arguments
    /// * `left_internode_ptr` - Raw pointer to the left (original) internode after split
    /// * `right_internode_ptr` - Raw pointer to the right sibling (already in arena)
    /// * `popup_key` - The key to insert into the parent
    ///
    /// # Safety notes
    /// Uses raw pointers to avoid Stacked Borrows aliasing issues when the left
    /// internode is the root (we need to access both the internode and self.root).
    #[expect(clippy::too_many_lines, reason = "Due to formatting")]
    fn propagate_internode_split(
        &mut self,
        left_internode_ptr: *mut InternodeNode<V, WIDTH>,
        right_internode_ptr: *mut InternodeNode<V, WIDTH>,
        popup_key: u64,
    ) {
        // Update children's parent pointers in the new internode
        // SAFETY: right_internode_ptr is valid from store_internode_in_arena
        unsafe {
            let right: &InternodeNode<V, WIDTH> = &*right_internode_ptr;

            for i in 0..=right.size() {
                let child: *mut u8 = right.child(i);

                if !child.is_null() {
                    // Update child's parent - height determines child type per CODE_005.md:
                    // - height == 0 means children are leaves
                    // - height > 0 means children are internodes
                    if right.height() == 0 {
                        // Children are leaves
                        (*child.cast::<LeafNode<V, WIDTH>>())
                            .set_parent(right_internode_ptr.cast::<u8>());
                    } else {
                        // Children are internodes
                        (*child.cast::<InternodeNode<V, WIDTH>>())
                            .set_parent(right_internode_ptr.cast::<u8>());
                    }
                }
            }
        }

        // Check if left_internode is root (compare raw pointers without creating references)
        let is_root: bool = if let RootNode::Internode(root) = &self.root {
            StdPtr::eq(root.as_ref(), left_internode_ptr)
        } else {
            false
        };

        if is_root {
            // Root internode split - similar to leaf root split, we need arena-derived pointers
            // SAFETY: left_internode_ptr is valid
            let left_height: u32 = unsafe { (*left_internode_ptr).height() };

            // Create new root above both
            // Height = left_internode.height + 1 (children are internodes, not leaves)
            let mut new_root: Box<InternodeNode<V, WIDTH>> =
                InternodeNode::new_root(left_height + 1);
            new_root.set_ikey(0, popup_key);
            new_root.set_nkeys(1);

            // Move old root to arena FIRST to get proper provenance (same pattern as leaf)
            let RootNode::Internode(old_root_box) =
                StdMem::replace(&mut self.root, RootNode::Internode(new_root))
            else {
                unreachable!("We checked is_root above")
            };

            // Get arena-derived pointer for left internode
            self.internode_arena.push(old_root_box);
            let idx: usize = self.internode_arena.len() - 1;
            // SAFETY: We just pushed, so idx is valid
            let arena_left_ptr: *mut InternodeNode<V, WIDTH> =
                unsafe { self.internode_arena.get_unchecked_mut(idx).as_mut() };

            // CRITICAL: Refresh parent pointers for all children of the demoted root.
            // These children had parent pointers derived from when their parent was
            // in self.root. After demotion, we must update them with the arena-derived
            // pointer to maintain Stacked Borrows provenance validity.
            // SAFETY: arena_left_ptr is valid from arena, children are valid from prior operations
            unsafe {
                let demoted: &InternodeNode<V, WIDTH> = &*arena_left_ptr;
                let arena_left_as_parent: *mut u8 = arena_left_ptr.cast::<u8>();

                for i in 0..=demoted.size() {
                    let child: *mut u8 = demoted.child(i);

                    if !child.is_null() {
                        if demoted.children_are_leaves() {
                            // Children are leaves
                            (*child.cast::<LeafNode<V, WIDTH>>()).set_parent(arena_left_as_parent);
                        } else {
                            // Children are internodes
                            (*child.cast::<InternodeNode<V, WIDTH>>())
                                .set_parent(arena_left_as_parent);
                        }
                    }
                }
            }

            // Now set up children with arena-derived pointers
            let RootNode::Internode(root_internode) = &mut self.root else {
                unreachable!("We just set root to Internode")
            };

            root_internode.set_child(0, arena_left_ptr.cast::<u8>());
            root_internode.set_child(1, right_internode_ptr.cast::<u8>());

            // Get raw pointer to new root for parent updates
            let new_root_ptr: *mut u8 =
                StdPtr::from_mut::<InternodeNode<V, WIDTH>>(root_internode.as_mut()).cast();

            // Update parent pointers for the two direct children
            // SAFETY: arena_left_ptr is valid from arena, right_internode_ptr is arena-backed
            unsafe {
                (*arena_left_ptr).set_parent(new_root_ptr);
                (*right_internode_ptr).set_parent(new_root_ptr);
            }
            return;
        }

        // Not root, insert popup_key into parent (recursive if needed)
        // SAFETY: left_internode_ptr is valid
        let parent_ptr: *mut u8 = unsafe { (*left_internode_ptr).parent() };
        assert!(!parent_ptr.is_null(), "Non-root internode has null parent");

        // SAFETY: parent_ptr is valid from the internode's parent pointer
        let parent: &mut InternodeNode<V, WIDTH> =
            unsafe { &mut *parent_ptr.cast::<InternodeNode<V, WIDTH>>() };
        let child_idx: usize = Self::find_child_index(parent, left_internode_ptr.cast::<u8>());

        if parent.size() < WIDTH {
            // Parent has room - insert directly
            parent.insert_key_and_child(child_idx, popup_key, right_internode_ptr.cast::<u8>());
            unsafe {
                (*right_internode_ptr).set_parent(parent_ptr);
            }
        } else {
            // Parent is full - split with simultaneous insertion (per C++ reference)
            let new_parent: Box<InternodeNode<V, WIDTH>> = InternodeNode::new(parent.height());
            let new_parent_ptr: *mut InternodeNode<V, WIDTH> =
                self.store_internode_in_arena(new_parent);

            // Split parent AND insert the popup_key at child_idx simultaneously
            // SAFETY: new_parent_ptr is valid from store_internode_in_arena
            let (new_popup_key, insert_went_left) = unsafe {
                parent.split_into(
                    &mut *new_parent_ptr,
                    child_idx,
                    popup_key,
                    right_internode_ptr.cast::<u8>(),
                )
            };

            // Set right_internode's parent based on where the insert went
            unsafe {
                if insert_went_left {
                    (*right_internode_ptr).set_parent(parent_ptr);
                } else {
                    (*right_internode_ptr).set_parent(new_parent_ptr.cast::<u8>());
                }
            }

            // Update parent pointers for other children in new_parent
            unsafe {
                let new_parent_ref: &InternodeNode<V, WIDTH> = &*new_parent_ptr;

                for i in 0..=new_parent_ref.size() {
                    let child: *mut u8 = new_parent_ref.child(i);

                    // Skip if this is the child we just set (avoid double-update)
                    if !child.is_null() && child != right_internode_ptr.cast::<u8>() {
                        if new_parent_ref.height() == 0 {
                            // Children are leaves
                            (*child.cast::<LeafNode<V, WIDTH>>())
                                .set_parent(new_parent_ptr.cast::<u8>());
                        } else {
                            // Children are internodes
                            (*child.cast::<InternodeNode<V, WIDTH>>())
                                .set_parent(new_parent_ptr.cast::<u8>());
                        }
                    }
                }
            }

            // Recursively propagate the new popup key
            // parent_ptr points to arena-backed internode, safe to use
            self.propagate_internode_split(parent_ptr.cast(), new_parent_ptr, new_popup_key);
        }
    }

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
    /// Note: This is O(n) as it traverses all leaves.
    /// For a size hint, track count separately.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.root {
            RootNode::Leaf(leaf) => leaf.size(),

            RootNode::Internode(root) => {
                // Traverse to leftmost leaf, then walk the leaf chain
                let leftmost: *const LeafNode<V, WIDTH> = Self::find_leftmost_leaf(root.as_ref());
                let mut count: usize = 0;
                let mut current: *const LeafNode<V, WIDTH> = leftmost;

                // SAFETY: Leaf pointers are valid from arena allocation and
                // linked list structure is maintained by insert/split operations.
                while !current.is_null() {
                    let leaf: &LeafNode<V, WIDTH> = unsafe { &*current };
                    count += leaf.size();
                    current = leaf.safe_next().cast_const().cast::<LeafNode<V, WIDTH>>();
                }

                count
            }
        }
    }

    /// Find the leftmost leaf node in the tree.
    ///
    /// Traverses down the leftmost path from the given internode to find
    /// the first (leftmost) leaf in the tree.
    fn find_leftmost_leaf(root: &InternodeNode<V, WIDTH>) -> *const LeafNode<V, WIDTH> {
        let mut node: *const u8 = root.child(0);
        let mut height: u32 = root.height();

        // Traverse down the leftmost path
        while height > 0 {
            // SAFETY: Child pointers are valid from arena allocation.
            // height > 0 means children are internodes.
            let internode: &InternodeNode<V, WIDTH> =
                unsafe { &*node.cast::<InternodeNode<V, WIDTH>>() };

            node = internode.child(0);
            height -= 1;
        }

        // height == 0 means node points to a leaf
        node.cast::<LeafNode<V, WIDTH>>()
    }

    /// Reach the leaf node that should contain the given key.
    ///
    /// Traverses from root through internodes to find the target leaf.
    /// For single-threaded mode, no version checking is needed.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// Reference to the leaf node that contains or should contain the key.
    #[inline]
    fn reach_leaf(&self, key: &Key<'_>) -> &LeafNode<V, WIDTH> {
        match &self.root {
            RootNode::Leaf(leaf) => leaf.as_ref(),

            RootNode::Internode(internode) => {
                // Traverse internodes to find leaf
                self.reach_leaf_from_internode(internode.as_ref(), key)
            }
        }
    }

    /// Traverse from an internode to the target leaf.
    #[expect(clippy::self_only_used_in_recursion)]
    fn reach_leaf_from_internode(
        &self,
        internode: &InternodeNode<V, WIDTH>,
        key: &Key<'_>,
    ) -> &LeafNode<V, WIDTH> {
        let ikey: u64 = key.ikey();
        let child_idx: usize = upper_bound_internode_direct(ikey, internode);
        let child_ptr: *mut u8 = internode.child(child_idx);

        if internode.children_are_leaves() {
            // Child is a leaf
            // SAFETY: children_are_leaves() guarantees child is LeafNode
            unsafe { &*(child_ptr as *const LeafNode<V, WIDTH>) }
        } else {
            // Child is another internode, recurse
            // SAFETY: !children_are_leaves() guarantees child is InternodeNode
            let child_internode: &InternodeNode<V, WIDTH> =
                unsafe { &*(child_ptr as *const InternodeNode<V, WIDTH>) };

            self.reach_leaf_from_internode(child_internode, key)
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    #[inline]
    fn reach_leaf_mut(&mut self, key: &Key<'_>) -> &mut LeafNode<V, WIDTH> {
        // Check if root is a leaf first (immutable borrow to check)
        let is_leaf: bool = self.root.is_leaf();

        if is_leaf {
            // Now we can borrow mutably
            match &mut self.root {
                RootNode::Leaf(leaf) => return leaf.as_mut(),
                RootNode::Internode(_) => unreachable!(),
            }
        }

        // Root is an internode - extract info with immutable borrow first
        let (start_ptr, children_are_leaves, ikey) = match &self.root {
            RootNode::Internode(internode) => {
                let ikey: u64 = key.ikey();
                let child_idx: usize = upper_bound_internode_direct(ikey, internode.as_ref());
                (
                    internode.child(child_idx),
                    internode.children_are_leaves(),
                    ikey,
                )
            }

            RootNode::Leaf(_) => unreachable!(),
        };

        if children_are_leaves {
            // SAFETY: children_are_leaves() guarantees child is LeafNode
            return unsafe { &mut *start_ptr.cast::<LeafNode<V, WIDTH>>() };
        }

        // Iterative traversal for deeper trees
        // SAFETY: The returned pointer is valid for the tree's lifetime (arena-backed)
        unsafe { &mut *Self::reach_leaf_mut_iterative_static(start_ptr, ikey) }
    }

    /// Iterative leaf reach for deeply nested trees.
    ///
    /// # Safety
    ///
    /// The returned reference is valid for as long as the tree's arenas are not modified.
    /// This is guaranteed by the single-threaded Phase 1 design.
    fn reach_leaf_mut_iterative_static(mut current: *mut u8, ikey: u64) -> *mut LeafNode<V, WIDTH> {
        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &InternodeNode<V, WIDTH> =
                unsafe { &*(current as *const InternodeNode<V, WIDTH>) };
            let child_idx: usize = upper_bound_internode_direct(ikey, internode);
            let child_ptr: *mut u8 = internode.child(child_idx);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is LeafNode
                return child_ptr.cast::<LeafNode<V, WIDTH>>();
            }

            current = child_ptr;
        }
    }

    /// Look up a value by key.
    ///
    /// Returns a clone of the Arc wrapping the value, or None if not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice, max 8 bytes in Phase 1)
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found, or key is too long (>8 bytes)
    ///
    /// # Phase 1 Behavior
    ///
    /// Keys longer than 8 bytes return `None`. Phase 2 will add layer traversal.
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
        // Phase 1: Reject keys longer than 8 bytes
        if key.len() > MAX_INLINE_KEY_LEN {
            return None;
        }

        let key: Key<'_> = Key::new(key);
        self.get_internal(&key)
    }

    /// Internal get implementation with Key struct.
    fn get_internal(&self, key: &Key<'_>) -> Option<Arc<V>> {
        let leaf: &LeafNode<V, WIDTH> = self.reach_leaf(key);

        // Binary search in leaf
        let ikey: u64 = key.ikey();
        // Phase 1: key.current_len() is always <= 8 (checked in get()), so cast is safe
        #[expect(
            clippy::cast_possible_truncation,
            reason = "key length validated <= 8 in public API"
        )]
        let keylenx: u8 = key.current_len() as u8;

        let pos: KeyIndexPosition = lower_bound_leaf(ikey, keylenx, leaf);

        if !pos.is_found() {
            return None;
        }

        let slot: usize = pos.slot();
        let stored_keylenx: u8 = leaf.keylenx(slot);

        // Check for layer pointer (shouldn't happen in Phase 1 with key length restriction)
        if stored_keylenx >= LAYER_KEYLENX {
            // Layer traversal not implemented in Phase 1
            return None;
        }

        // Check keylenx matches exactly
        if stored_keylenx != keylenx {
            return None;
        }

        // Get the value (plain field access, no atomics)
        leaf.leaf_value(slot).try_clone_arc()
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
    /// * `key` - The key as a byte slice (max 8 bytes in Phase 1)
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// * `Ok(Some(old_value))` - Key existed, old value returned
    /// * `Ok(None)` - Key inserted (new key)
    ///
    /// # Errors
    ///
    /// * [`InsertError::KeyTooLong`] - Key exceeds 8 bytes (Phase 1 limit)
    /// * [`InsertError::LeafFull`] - All keys have identical ikey (layer case, not yet supported)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tree: MassTree<u64> = MassTree::new();
    ///
    /// // Insert new key (max 8 bytes in Phase 1)
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
        // Phase 1: Restrict to 8-byte keys
        if key.len() > MAX_INLINE_KEY_LEN {
            return Err(InsertError::KeyTooLong);
        }

        let key: Key<'_> = Key::new(key);
        self.insert_internal(&key, value)
    }

    /// Internal insert implementation with Key struct.
    ///
    /// Uses a loop to handle splits: if the target leaf is full, we split it
    /// and then retry the insert. This matches the Masstree C++ design.
    fn insert_internal(&mut self, key: &Key<'_>, value: V) -> Result<Option<Arc<V>>, InsertError> {
        let ikey: u64 = key.ikey();
        // Phase 1: key.current_len() is always <= 8 (checked in insert()), so cast is safe
        #[expect(
            clippy::cast_possible_truncation,
            reason = "key length validated <= 8 in public API"
        )]
        let keylenx: u8 = key.current_len() as u8;

        loop {
            let leaf: &mut LeafNode<V, WIDTH> = self.reach_leaf_mut(key);

            // Binary search for key
            let pos: KeyIndexPosition = lower_bound_leaf(ikey, keylenx, leaf);

            if pos.is_found() {
                // Key exists, update value
                let slot: usize = pos.slot();
                let new_arc: Arc<V> = Arc::new(value);
                let old_value: Option<Arc<V>> = leaf.swap_value(slot, new_arc);

                return Ok(old_value);
            }

            // Key not found, insert at position pos.i
            let insert_pos: usize = pos.i;
            let size: usize = leaf.size();

            // Check if we can insert directly (has space and slot available)
            let can_insert_directly: bool = if size < WIDTH {
                let perm: Permuter<WIDTH> = leaf.permutation();
                let next_free_slot: usize = perm.back();

                // Can insert if slot 0 isn't at back, or if we can reuse slot 0,
                // or if there are other free slots we can swap to
                next_free_slot != 0 || leaf.can_reuse_slot0(ikey) || size < WIDTH - 1 // Have another free slot to swap with
            } else {
                false
            };

            if can_insert_directly {
                // Has space - insert directly
                let mut perm: Permuter<WIDTH> = leaf.permutation();

                // Check slot-0 rule BEFORE allocation (peek at which slot will be used)
                let next_free_slot: usize = perm.back();

                if next_free_slot == 0 && !leaf.can_reuse_slot0(ikey) {
                    // Can't use slot 0 - swap it out of the back position.
                    // back() returns the slot at position WIDTH - 1.
                    // Swap position WIDTH - 1 with position `size` (first free slot position).
                    // This moves slot 0 away from back() so next allocation skips it.
                    debug_assert!(
                        size < WIDTH - 1,
                        "should have fallen through to split if only slot 0 is free"
                    );
                    perm.swap_free_slots(WIDTH - 1, size);
                }

                // Allocate slot via insert_from_back
                let slot: usize = perm.insert_from_back(insert_pos);

                // Assign key and value to the returned slot
                leaf.assign_value(slot, ikey, keylenx, value);

                // Commit permutation update
                leaf.set_permutation(perm);

                return Ok(None);
            }

            // Leaf is full - calculate split point and split
            let Some(split_point) = calculate_split_point(leaf, insert_pos, ikey) else {
                return Err(InsertError::LeafFull);
            };

            // Convert from post-insert to pre-insert coordinates
            // If insert_pos < split_point.pos, the insert would go in left half
            // and split position needs adjustment since we haven't inserted yet
            let pre_insert_split_pos = if insert_pos < split_point.pos {
                split_point.pos - 1
            } else {
                split_point.pos
            };

            // Handle edge cases from sequential optimization:
            // - split_pos >= size: only new key goes to right (right-sequential)
            // - split_pos == 0: all entries go to right (left-sequential)
            let leaf_ptr: *mut LeafNode<V, WIDTH> = leaf as *mut _;
            let (new_leaf_box, split_ikey) = if pre_insert_split_pos >= size {
                // Right-sequential: create empty new leaf - new key will go there
                (LeafNode::new(), ikey)
            } else if pre_insert_split_pos == 0 {
                // Left-sequential: move ALL entries to right, new key goes to left
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_all_to_right();
                (split_result.new_leaf, split_result.split_ikey)
            } else {
                // Normal split - move entries to new leaf
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_into(pre_insert_split_pos);
                (split_result.new_leaf, split_result.split_ikey)
            };

            // Propagate split up the tree
            self.propagate_split(leaf_ptr, new_leaf_box, split_ikey);

            // Loop continues - reach_leaf_mut will find the correct leaf for our key
        }
    }

    /// Insert with an existing Arc (avoids double-wrapping).
    ///
    /// Useful when you already have an `Arc<V>` and don't want to clone.
    ///
    /// # Errors
    ///
    /// * [`InsertError::KeyTooLong`] - Key exceeds 8 bytes (Phase 1 limit)
    /// * [`InsertError::LeafFull`] - All keys have identical ikey (layer case, not yet supported)
    pub fn insert_arc(&mut self, key: &[u8], value: Arc<V>) -> Result<Option<Arc<V>>, InsertError> {
        // Phase 1: Restrict to 8-byte keys
        if key.len() > MAX_INLINE_KEY_LEN {
            return Err(InsertError::KeyTooLong);
        }

        let key = Key::new(key);
        self.insert_arc_internal(&key, value)
    }

    /// Internal `insert_arc` implementation with split support.
    fn insert_arc_internal(
        &mut self,
        key: &Key<'_>,
        value: Arc<V>,
    ) -> Result<Option<Arc<V>>, InsertError> {
        let ikey: u64 = key.ikey();

        // Phase 1: key.current_len() is always <= 8 (checked in insert_arc()), so cast is safe
        #[expect(
            clippy::cast_possible_truncation,
            reason = "key length validated <= 8 in public API"
        )]
        let keylenx: u8 = key.current_len() as u8;

        loop {
            let leaf: &mut LeafNode<V, WIDTH> = self.reach_leaf_mut(key);

            let pos: KeyIndexPosition = lower_bound_leaf(ikey, keylenx, leaf);

            if pos.is_found() {
                let slot: usize = pos.slot();
                let old_value: Option<Arc<V>> = leaf.swap_value(slot, value);
                return Ok(old_value);
            }

            let insert_pos: usize = pos.i;
            let size: usize = leaf.size();

            // Check if we can insert directly (has space and slot available)
            let can_insert_directly: bool = if size < WIDTH {
                let perm: Permuter<WIDTH> = leaf.permutation();
                let next_free_slot: usize = perm.back();

                // Can insert if slot 0 isn't at back, or if we can reuse slot 0,
                // or if there are other free slots we can swap to
                next_free_slot != 0 || leaf.can_reuse_slot0(ikey) || size < WIDTH - 1 // Have another free slot to swap with
            } else {
                false
            };

            if can_insert_directly {
                // Has space - insert directly
                let mut perm: Permuter<WIDTH> = leaf.permutation();

                // Check slot-0 rule BEFORE allocation (peek at which slot will be used)
                let next_free_slot: usize = perm.back();

                if next_free_slot == 0 && !leaf.can_reuse_slot0(ikey) {
                    // Can't use slot 0 - swap it out of the back position.
                    debug_assert!(
                        size < WIDTH - 1,
                        "should have fallen through to split if only slot 0 is free"
                    );
                    perm.swap_free_slots(WIDTH - 1, size);
                }

                // Use return value of insert_from_back for assignment
                let slot: usize = perm.insert_from_back(insert_pos);

                leaf.assign_arc(slot, ikey, keylenx, value);
                leaf.set_permutation(perm);

                return Ok(None);
            }

            // Leaf is full (or slot 0 is the only free slot and can't be used)
            // Calculate split point and split
            let Some(split_point) = calculate_split_point(leaf, insert_pos, ikey) else {
                return Err(InsertError::LeafFull);
            };

            let pre_insert_split_pos: usize = if insert_pos < split_point.pos {
                split_point.pos - 1
            } else {
                split_point.pos
            };

            // Handle edge cases from sequential optimization
            let leaf_ptr: *mut LeafNode<V, WIDTH> = leaf as *mut _;
            let (new_leaf_box, split_ikey) = if pre_insert_split_pos >= size {
                (LeafNode::new(), ikey)
            } else if pre_insert_split_pos == 0 {
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_all_to_right();
                (split_result.new_leaf, split_result.split_ikey)
            } else {
                let split_result: LeafSplitResult<V, WIDTH> = leaf.split_into(pre_insert_split_pos);
                (split_result.new_leaf, split_result.split_ikey)
            };

            self.propagate_split(leaf_ptr, new_leaf_box, split_ikey);

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
//  MassTreeIndex (Index Mode with V: Copy)
// ============================================================================

/// Convenience wrapper for index-style workloads with copyable values.
///
/// Provides a simpler API that returns `V` directly instead of `Arc<V>`.
/// Best for small, copyable values like `u64`, handles, or pointers.
///
/// # Implementation Note
///
/// **This is currently a wrapper around `MassTree<V>`, NOT true inline storage.**
/// Values are still stored as `Arc<V>` internally; this type simply copies
/// the value out on read. True inline storage is planned for a future release.
///
/// For performance-critical code where Arc overhead matters, use `MassTree<V>`
/// directly and manage the Arc yourself.
///
/// # Type Parameters
///
/// * `V` - The value type to store (must be `Copy`)
/// * `WIDTH` - Node width (default: 15, max: 15)
///
/// # Example
///
/// ```ignore
/// use masstree::tree::MassTreeIndex;
///
/// let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();
/// tree.insert(b"hello", 42).unwrap();
///
/// let value = tree.get(b"hello");
/// assert_eq!(value, Some(42));
/// ```
pub struct MassTreeIndex<V: Copy, const WIDTH: usize = 15> {
    /// Wraps `MassTree` internally. True inline storage is planned for future.
    inner: MassTree<V, WIDTH>,
}

impl<V: Copy, const WIDTH: usize> StdFmt::Debug for MassTreeIndex<V, WIDTH> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTreeIndex")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<V: Copy, const WIDTH: usize> MassTreeIndex<V, WIDTH> {
    /// Create a new empty `MassTreeIndex`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: MassTree::new(),
        }
    }

    /// Check if the tree is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the number of keys in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Look up a value by key.
    ///
    /// Returns a copy of the value, or None if not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice, max 8 bytes in Phase 1)
    ///
    /// # Returns
    ///
    /// * `Some(V)` - If the key was found (value is copied)
    /// * `None` - If the key was not found or key is too long (>8 bytes)
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        // For index mode, we dereference the Arc and copy
        self.inner.get(key).map(|arc: Arc<V>| *arc)
    }

    /// Insert a key-value pair into the tree.
    ///
    /// If the key already exists, the value is updated and the old value returned.
    ///
    /// # Arguments
    ///
    /// * `key` - The key as a byte slice (max 8 bytes in Phase 1)
    /// * `value` - The value to insert (copied)
    ///
    /// # Returns
    ///
    /// * `Ok(Some(old_value))` - Key existed, old value returned (copied)
    /// * `Ok(None)` - Key inserted (new key)
    ///
    /// # Errors
    ///
    /// * [`InsertError::KeyTooLong`] - Key exceeds 8 bytes (Phase 1 limit)
    /// * [`InsertError::LeafFull`] - All keys have identical ikey (layer case, not yet supported)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();
    ///
    /// // Insert new key
    /// assert_eq!(tree.insert(b"hello", 42)?, None);
    ///
    /// // Update existing key
    /// assert_eq!(tree.insert(b"hello", 100)?, Some(42));
    /// ```
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>, InsertError> {
        // Use inner tree's insert, convert Arc<V> to V
        self.inner
            .insert(key, value)
            .map(|opt: Option<Arc<V>>| opt.map(|arc: Arc<V>| *arc))
    }
}

impl<V: Copy, const WIDTH: usize> Default for MassTreeIndex<V, WIDTH> {
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

        // All should return None on empty tree (keys 0-8 bytes)
        assert!(tree.get(b"").is_none());
        assert!(tree.get(b"a").is_none());
        assert!(tree.get(b"ab").is_none());
        assert!(tree.get(b"abc").is_none());
        assert!(tree.get(b"abcd").is_none());
        assert!(tree.get(b"abcde").is_none());
        assert!(tree.get(b"abcdef").is_none());
        assert!(tree.get(b"abcdefg").is_none());
        assert!(tree.get(b"abcdefgh").is_none()); // Exactly 8 bytes - max for Phase 1
    }

    #[test]
    fn test_get_rejects_long_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Keys > 8 bytes should return None in Phase 1
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

        let err = InsertError::KeyTooLong;
        assert_eq!(format!("{err}"), "key exceeds maximum length");
    }

    #[test]
    fn test_insert_key_too_long() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Phase 1: Keys > 8 bytes are rejected
        let long_key = vec![b'x'; 9]; // 9 bytes > 8

        let result = tree.insert(&long_key, 42);
        assert!(matches!(result, Err(InsertError::KeyTooLong)));

        // Even longer keys
        let very_long = vec![b'y'; 100];
        let result = tree.insert(&very_long, 42);
        assert!(matches!(result, Err(InsertError::KeyTooLong)));
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
}
