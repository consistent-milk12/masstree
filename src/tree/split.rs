//! Split propagation logic for `MassTree`.
//!
//! This module handles leaf and internode splits, creating new root nodes when needed.
//! Split propagation is triggered when a node becomes full and must be divided.

use std::ptr as StdPtr;
use std::sync::atomic::Ordering;

use crate::alloc::NodeAllocator;
use crate::internode::InternodeNode;
use crate::leaf::{LeafNode, LeafValue};

use super::MassTree;

/// Update parent pointers for all children of an internode.
///
/// # Safety
///
/// - `internode` must be a valid pointer to an internode
/// - `new_parent` must be the correct parent pointer to set
/// - All child pointers in the internode must be valid
unsafe fn update_children_parent_pointers<V, const WIDTH: usize>(
    internode: *const InternodeNode<LeafValue<V>, WIDTH>,
    new_parent: *mut u8,
    skip_child: Option<*mut u8>,
) {
    // SAFETY: Caller guarantees internode is valid
    let node: &InternodeNode<LeafValue<V>, WIDTH> = unsafe { &*internode };

    for i in 0..=node.size() {
        let child: *mut u8 = node.child(i);

        // Skip null children and optionally a specific child
        if child.is_null() || skip_child == Some(child) {
            continue;
        }

        // - height == 0 means children are leaves
        // - height > 0 means children are internodes
        // SAFETY: Caller guarantees all child pointers are valid
        unsafe {
            if node.children_are_leaves() {
                (*child.cast::<LeafNode<LeafValue<V>, WIDTH>>()).set_parent(new_parent);
            } else {
                (*child.cast::<InternodeNode<LeafValue<V>, WIDTH>>()).set_parent(new_parent);
            }
        }
    }
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Find the index of a child pointer in an internode.
    ///
    /// # Panics
    ///
    /// Panics if the child is not found. This indicates a bug in split propagation.
    #[expect(
        clippy::panic,
        reason = "Intentional invariant check - child must exist during split propagation"
    )]
    pub(super) fn find_child_index(
        internode: &InternodeNode<LeafValue<V>, WIDTH>,
        child: *mut u8,
    ) -> usize {
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

    /// Create a new root internode when the root leaf splits.
    ///
    /// **Root ownership strategy:**
    /// - All nodes (including root) are allocated in arena from the start
    /// - `root_ptr` is the single source of truth for which node is root
    /// - When root changes: just update `root_ptr`, old root stays in arena
    ///
    /// **Returns:** Pointer to the left leaf (the former root, still in arena).
    /// The caller should use this pointer for leaf linking.
    pub(super) fn create_root_internode(
        &mut self,
        right_leaf: *mut LeafNode<LeafValue<V>, WIDTH>,
        split_ikey: u64,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        // Get the current root (a leaf) — it's already in arena
        debug_assert!(
            self.root_is_leaf(),
            "create_root_internode requires leaf root"
        );
        let left_leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> =
            self.root_ptr.load(Ordering::Acquire).cast();

        // Create new internode root (height 0 = children are leaves)
        let new_root_box: Box<InternodeNode<LeafValue<V>, WIDTH>> = InternodeNode::new_root(0);
        new_root_box.set_ikey(0, split_ikey);
        new_root_box.set_nkeys(1);

        // Allocate new root in arena
        let new_root_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> =
            self.alloc_internode(new_root_box);

        // Set up children: left is former root, right is the new split leaf
        // SAFETY: new_root_ptr is valid from alloc_internode
        unsafe {
            (*new_root_ptr).set_child(0, left_leaf_ptr.cast::<u8>());
            (*new_root_ptr).set_child(1, right_leaf.cast::<u8>());
        }

        // Update leaves' parent pointers
        // SAFETY: left_leaf_ptr was root (valid), right_leaf is arena-backed
        unsafe {
            (*left_leaf_ptr).set_parent(new_root_ptr.cast::<u8>());
            (*right_leaf).set_parent(new_root_ptr.cast::<u8>());
        }

        // Update root_ptr to new internode root — single source of truth
        self.root_ptr
            .store(new_root_ptr.cast::<u8>(), Ordering::Release);

        // Return left leaf pointer for caller to use in leaf linking
        left_leaf_ptr
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
    pub(super) fn propagate_split(
        &mut self,
        left_leaf: *mut LeafNode<LeafValue<V>, WIDTH>,
        right_leaf_box: Box<LeafNode<LeafValue<V>, WIDTH>>,
        split_ikey: u64,
    ) {
        // Store the new leaf in the allocator to get a stable pointer
        let right_leaf_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = self.alloc_leaf(right_leaf_box);

        // Check if left_leaf was the root BEFORE linking (linking uses left_leaf which
        // may have provenance that will be invalidated by create_root_internode)
        let left_was_root: bool = self.root_is_leaf() && {
            let root_leaf_ptr: *const LeafNode<LeafValue<V>, WIDTH> =
                self.root_ptr.load(Ordering::Acquire).cast();
            StdPtr::eq(root_leaf_ptr, left_leaf)
        };

        if left_was_root {
            // Root was a leaf - create_root_internode will give us arena-derived pointer
            let arena_left_ptr: *mut LeafNode<LeafValue<V>, WIDTH> =
                self.create_root_internode(right_leaf_ptr, split_ikey);

            // Link leaves using arena-derived pointer (proper provenance)
            // SAFETY: arena_left_ptr is valid from arena, right_leaf_ptr is from arena
            unsafe {
                let old_next: *mut LeafNode<LeafValue<V>, WIDTH> = (*arena_left_ptr).safe_next();

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
            let old_next: *mut LeafNode<LeafValue<V>, WIDTH> = (*left_leaf).safe_next();

            (*left_leaf).set_next(right_leaf_ptr);
            (*right_leaf_ptr).set_prev(left_leaf);
            (*right_leaf_ptr).set_next(old_next);

            if !old_next.is_null() {
                (*old_next).set_prev(right_leaf_ptr);
            }
        }

        // Check if this is a LAYER ROOT split (not main tree root)
        // Layer roots have is_layer_root() == true (is_root flag + null parent)
        // SAFETY: left_leaf is a valid pointer
        let is_layer_root: bool = unsafe { (*left_leaf).is_layer_root() };

        if is_layer_root {
            // LAYER ROOT SPLIT
            // Create internode to become new layer root.
            // The `maybe_parent` pattern means we don't need to update the
            // parent leaf's layer pointer - reach_leaf_from_ptr will handle it.
            self.promote_layer_root(left_leaf, right_leaf_ptr, split_ikey);
            return;
        }

        // Get parent from left_leaf
        // SAFETY: left_leaf is a valid pointer we just split
        let parent_ptr: *mut u8 = unsafe { (*left_leaf).parent() };

        // This shouldn't happen if tree invariants are maintained
        assert!(!parent_ptr.is_null(), "Non-root leaf has null parent");

        // Insert split key into parent
        // SAFETY: parent_ptr is valid from leaf's parent pointer
        let parent: &mut InternodeNode<LeafValue<V>, WIDTH> =
            unsafe { &mut *parent_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>() };

        let child_idx: usize = Self::find_child_index(parent, left_leaf.cast::<u8>());

        if parent.size() < WIDTH {
            // Parent has room, just insert
            parent.insert_key_and_child(child_idx, split_ikey, right_leaf_ptr.cast::<u8>());

            // Update right_leaf's parent pointer
            unsafe {
                (*right_leaf_ptr).set_parent(parent_ptr);
            }
        } else {
            // Allocate new internode for the right half
            let new_parent: Box<InternodeNode<LeafValue<V>, WIDTH>> =
                InternodeNode::new(parent.height());
            let new_parent_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> =
                self.alloc_internode(new_parent);

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
    /// Uses raw pointers throughout. The `root_ptr` is the single source of truth
    /// for the root node.
    pub(super) fn propagate_internode_split(
        &mut self,
        left_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        right_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        popup_key: u64,
    ) {
        // Update children's parent pointers in the new internode
        // SAFETY: right_internode_ptr is valid from store_internode_in_arena
        unsafe {
            update_children_parent_pointers(
                right_internode_ptr,
                right_internode_ptr.cast::<u8>(),
                None,
            );
        }

        // Check if left_internode is the MAIN TREE root (compare raw pointers)
        let is_main_root: bool = !self.root_is_leaf() && {
            let root_internode_ptr: *const InternodeNode<LeafValue<V>, WIDTH> =
                self.root_ptr.load(Ordering::Acquire).cast();

            StdPtr::eq(root_internode_ptr, left_internode_ptr)
        };

        if is_main_root {
            self.propagate_root_internode_split(left_internode_ptr, right_internode_ptr, popup_key);
            return;
        }

        // Check if left_internode is a LAYER ROOT (is_root flag set AND null parent)
        // Must check both conditions - a demoted main tree root may still have is_root set
        // but will have a non-null parent pointer.
        // SAFETY: left_internode_ptr is valid
        let is_layer_root: bool =
            unsafe { (*left_internode_ptr).is_root() && (*left_internode_ptr).parent().is_null() };

        if is_layer_root {
            // Layer root internode split - promote to new layer root
            self.promote_layer_root_internode(left_internode_ptr, right_internode_ptr, popup_key);
            return;
        }

        // Not root, insert popup_key into parent (recursive if needed)
        self.propagate_non_root_internode_split(left_internode_ptr, right_internode_ptr, popup_key);
    }

    /// Handle root internode split - creates new root above both children.
    fn propagate_root_internode_split(
        &mut self,
        left_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        right_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        popup_key: u64,
    ) {
        // SAFETY: left_internode_ptr is valid
        let left_height: u32 = unsafe { (*left_internode_ptr).height() };

        // Create new root above both (height = left + 1)
        let new_root_box: Box<InternodeNode<LeafValue<V>, WIDTH>> =
            InternodeNode::new_root(left_height + 1);
        new_root_box.set_ikey(0, popup_key);
        new_root_box.set_nkeys(1);

        // Allocate new root in arena
        let new_root_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> =
            self.alloc_internode(new_root_box);

        // Old root (left_internode_ptr) is ALREADY in arena — no move needed.
        // Just clear the is_root flag on the demoted internode.
        // SAFETY: left_internode_ptr is valid
        unsafe {
            (*left_internode_ptr).version_mut().mark_nonroot();
        }

        // Update children parent pointers for demoted root
        // SAFETY: left_internode_ptr is valid, children are valid
        unsafe {
            update_children_parent_pointers(
                left_internode_ptr,
                left_internode_ptr.cast::<u8>(),
                None,
            );
        }

        // Set up children of new root
        // SAFETY: new_root_ptr is valid from alloc_internode
        unsafe {
            (*new_root_ptr).set_child(0, left_internode_ptr.cast::<u8>());
            (*new_root_ptr).set_child(1, right_internode_ptr.cast::<u8>());
        }

        // Update parent pointers for the two direct children
        // SAFETY: all pointers are valid from arena
        unsafe {
            (*left_internode_ptr).set_parent(new_root_ptr.cast::<u8>());
            (*right_internode_ptr).set_parent(new_root_ptr.cast::<u8>());
        }

        // Update root_ptr atomically — single source of truth
        self.root_ptr
            .store(new_root_ptr.cast::<u8>(), Ordering::Release);
    }

    /// Handle non-root internode split - insert into parent, recursively if needed.
    fn propagate_non_root_internode_split(
        &mut self,
        left_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        right_internode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH>,
        popup_key: u64,
    ) {
        // SAFETY: left_internode_ptr is valid
        let parent_ptr: *mut u8 = unsafe { (*left_internode_ptr).parent() };
        assert!(!parent_ptr.is_null(), "Non-root internode has null parent");

        // SAFETY: parent_ptr is valid from the internode's parent pointer
        let parent: &mut InternodeNode<LeafValue<V>, WIDTH> =
            unsafe { &mut *parent_ptr.cast::<InternodeNode<LeafValue<V>, WIDTH>>() };
        let child_idx: usize = Self::find_child_index(parent, left_internode_ptr.cast::<u8>());

        if parent.size() < WIDTH {
            // Parent has room - insert directly
            parent.insert_key_and_child(child_idx, popup_key, right_internode_ptr.cast::<u8>());
            unsafe {
                (*right_internode_ptr).set_parent(parent_ptr);
            }
            return;
        }

        // Parent is full - split with simultaneous insertion (per C++ reference)
        let new_parent: Box<InternodeNode<LeafValue<V>, WIDTH>> =
            InternodeNode::new(parent.height());
        let new_parent_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> =
            self.alloc_internode(new_parent);

        // Split parent AND insert the popup_key at child_idx simultaneously
        // SAFETY: new_parent_ptr is valid from alloc_internode
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

        // Update parent pointers for other children in new_parent (skip the one we just set)
        // SAFETY: new_parent_ptr is valid from store_internode_in_arena
        unsafe {
            update_children_parent_pointers(
                new_parent_ptr,
                new_parent_ptr.cast::<u8>(),
                Some(right_internode_ptr.cast::<u8>()),
            );
        }

        // Recursively propagate the new popup key
        // parent_ptr points to arena-backed internode, safe to use
        self.propagate_internode_split(parent_ptr.cast(), new_parent_ptr, new_popup_key);
    }

    /// Promote a layer root leaf to an internode root.
    ///
    /// Called when a layer root leaf splits and needs to become an internode tree.
    ///
    /// # Algorithm
    ///
    /// 1. Create new internode with both leaves as children
    /// 2. Update children's parent pointers to the new internode
    /// 3. Clear root flag on both leaves (they're no longer layer roots)
    ///
    /// # `maybe_parent` Pattern
    ///
    /// The layer pointer in the parent leaf is NOT updated here. Instead:
    ///
    /// 1. The old layer pointer still points to `left_leaf`
    /// 2. `left_leaf` now has `parent != null` (pointing to new internode)
    /// 3. `reach_leaf_from_ptr()` detects this: if leaf has non-null parent,
    ///    it follows the parent internode to find the correct leaf
    ///
    /// This matches the C++ `maybe_parent()` pattern from `masstree_struct.hh:83`.
    ///
    /// # Reference
    ///
    /// C++ `masstree_split.hh:218-226`, `masstree_struct.hh:83`
    fn promote_layer_root(
        &mut self,
        left_leaf: *mut LeafNode<LeafValue<V>, WIDTH>,
        right_leaf: *mut LeafNode<LeafValue<V>, WIDTH>,
        split_ikey: u64,
    ) {
        // Create new internode root for this layer (height=0, children are leaves)
        let mut new_inode: Box<InternodeNode<LeafValue<V>, WIDTH>> = InternodeNode::new(0);

        // Set up children: [left_leaf] -split_ikey- [right_leaf]
        new_inode.set_child(0, left_leaf.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_leaf.cast());
        new_inode.set_nkeys(1);

        // Mark as root (layer roots should have the root flag)
        new_inode.version_mut().mark_root();

        // Allocate internode in arena
        let new_inode_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> =
            self.alloc_internode(new_inode);

        // Update children's parent pointers to the new internode
        // This is the KEY step: now left_leaf.parent() != null
        // SAFETY: left_leaf and right_leaf are valid arena-backed pointers
        unsafe {
            (*left_leaf).set_parent(new_inode_ptr.cast());
            (*right_leaf).set_parent(new_inode_ptr.cast());

            // Clear the root flag on both leaves - they're no longer layer roots
            (*left_leaf).version_mut().mark_nonroot();
            (*right_leaf).version_mut().mark_nonroot();
        }

        // NOTE: We do NOT update the parent leaf's layer pointer here.
        // The `maybe_parent` pattern in `reach_leaf_from_ptr()` handles this:
        // - Layer pointer still points to old left_leaf
        // - But left_leaf.parent() now points to new_inode
        // - reach_leaf_from_ptr detects non-null parent and follows it
    }

    /// Promote a layer root internode to a new layer root internode.
    ///
    /// Called when a layer root internode splits and needs a new parent internode.
    /// This is the internode equivalent of `promote_layer_root` for leaves.
    ///
    /// # Algorithm
    ///
    /// 1. Create new internode with height = left.height + 1
    /// 2. Set both split internodes as children
    /// 3. Mark new internode as layer root
    /// 4. Update children's parent pointers
    /// 5. Clear root flag on both children
    ///
    /// # `maybe_parent` Pattern
    ///
    /// Similar to leaf promotion, we don't update any external layer pointers.
    /// The `reach_leaf_from_ptr()` follows parent pointers to find the current
    /// layer root regardless of how many promotions have occurred.
    fn promote_layer_root_internode(
        &mut self,
        left_inode: *mut InternodeNode<LeafValue<V>, WIDTH>,
        right_inode: *mut InternodeNode<LeafValue<V>, WIDTH>,
        split_ikey: u64,
    ) {
        // SAFETY: left_inode is valid
        let left_height: u32 = unsafe { (*left_inode).height() };

        // Create new internode root for this layer (height + 1, children are internodes)
        let mut new_root: Box<InternodeNode<LeafValue<V>, WIDTH>> =
            InternodeNode::new(left_height + 1);

        // Set up children: [left_inode] -split_ikey- [right_inode]
        new_root.set_child(0, left_inode.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_inode.cast());
        new_root.set_nkeys(1);

        // Mark as root (layer roots should have the root flag)
        new_root.version_mut().mark_root();

        // Allocate internode in arena
        let new_root_ptr: *mut InternodeNode<LeafValue<V>, WIDTH> = self.alloc_internode(new_root);

        // Update children's parent pointers to the new root
        // SAFETY: left_inode and right_inode are valid arena-backed pointers
        unsafe {
            (*left_inode).set_parent(new_root_ptr.cast());
            (*right_inode).set_parent(new_root_ptr.cast());

            // Clear the root flag on both children - they're no longer layer roots
            (*left_inode).version_mut().mark_nonroot();
            (*right_inode).version_mut().mark_nonroot();
        }

        // NOTE: We do NOT update any external layer pointers here.
        // The `maybe_parent` pattern handles this automatically.
    }
}
