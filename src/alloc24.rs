//! Node allocation for [`LeafNode24`] (WIDTH=24).
//!
//! This module provides [`SeizeAllocator24`], a Miri-compliant allocator for
//! 24-slot leaf nodes using `seize` for memory reclamation.
//!
//! # Design
//!
//! This allocator uses **direct retirement** without tracking lists, following
//! papaya's seize integration pattern. Retirement is O(1) - just a seize call.
//! Tree cleanup uses root-based traversal instead of list iteration.
//!
//! # Note on Internode WIDTH
//!
//! While leaves use WIDTH=24 (via Permuter24 with u128), internodes are still
//! limited to WIDTH=15 because they use the original Permuter (u64 with 4-bit slots).
//! This is fine since internodes just hold child pointers; the benefit of WIDTH=24
//! comes from leaves holding more keys and splitting less often.

use std::alloc as StdAlloc;
use std::alloc::Layout;
use std::marker::PhantomData;

use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::error::{AllocError, AllocResult};
use crate::internode::InternodeNode;
use crate::leaf24::{LeafNode24, WIDTH_24};
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;

/// Node allocator using seize for safe memory reclamation (WIDTH=24).
///
/// This allocator relies on tree traversal for teardown rather than
/// maintaining tracking lists, eliminating O(n) overhead per retirement.
///
/// # Design
///
/// - **O(1) retirement**: Direct `guard.defer_retire()` calls
/// - **No tracking overhead**: No mutex locks or linear scans during retirement
/// - **Tree traversal teardown**: Frees all nodes by walking the tree structure
///
/// # Thread Safety
///
/// Stateless after construction - all operations are safe for concurrent use.
#[derive(Debug)]
pub struct SeizeAllocator24<S: ValueSlot> {
    _marker: PhantomData<S>,
}

// SAFETY: Allocator is stateless (just PhantomData), safe to send/share.
unsafe impl<S: ValueSlot + Send + Sync> Send for SeizeAllocator24<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for SeizeAllocator24<S> {}

impl<S: ValueSlot> SeizeAllocator24<S> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<S: ValueSlot> Default for SeizeAllocator24<S> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tree Traversal for Teardown
// =============================================================================

impl<S: ValueSlot> SeizeAllocator24<S> {
    /// Follow parent pointers to find the actual root of a sublayer.
    ///
    /// When a sublayer leaf splits, a new internode becomes the sublayer root,
    /// but the parent layer's slot still points to the OLD leaf. This function
    /// follows parent pointers to find the current root.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must point to a valid leaf or internode
    /// - Caller must have exclusive access (no concurrent readers/writers)
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Callers guarantee proper alignment"
    )]
    unsafe fn find_layer_root(&self, mut node_ptr: *mut u8) -> *mut u8 {
        loop {
            // SAFETY: Both leaves and internodes have NodeVersion at offset 0
            let version_ptr = node_ptr.cast::<NodeVersion>();
            let version: &NodeVersion = unsafe { &*version_ptr };

            // SAFETY: Called during teardown with exclusive access - no concurrent retirement.
            let parent: *mut u8 = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &LeafNode24<S> = unsafe { &*node_ptr.cast::<LeafNode24<S>>() };
                unsafe { leaf.parent_unguarded() }
            } else {
                // SAFETY: !version.is_leaf() confirmed
                let inode: &InternodeNode = unsafe { &*node_ptr.cast::<InternodeNode>() };
                unsafe { inode.parent_unguarded() }
            };

            if parent.is_null() {
                return node_ptr;
            }

            node_ptr = parent;
        }
    }

    /// Traverse tree structure and free all nodes.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must point to a valid leaf or internode
    /// - Caller must have exclusive access (no concurrent readers/writers)
    /// - This is only safe to call during `Drop` when the tree is quiescent
    ///
    /// # Note on Drop Order
    ///
    /// When we encounter a leaf with layer pointers:
    /// 1. We recurse into sublayers first and free them
    /// 2. Then we drop the leaf via `Box::from_raw`
    /// 3. `LeafNode::Drop` runs, but it correctly skips layer slots
    ///    (it only cleans up slots where `keylenx < LAYER_KEYLENX`)
    ///
    /// This is safe because layer pointers are distinguished by keylenx >= 128,
    /// and `LeafNode::Drop` explicitly checks this before cleanup.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Callers guarantee proper alignment"
    )]
    #[expect(clippy::self_only_used_in_recursion)]
    unsafe fn traverse_and_free(&self, node_ptr: *mut u8) {
        if node_ptr.is_null() {
            return;
        }

        // Read version to determine node type.
        // SAFETY: Both leaves and internodes have NodeVersion at offset 0.
        let version_ptr = node_ptr.cast::<NodeVersion>();
        let version = unsafe { &*version_ptr };

        if version.is_leaf() {
            // Handle leaf node
            let leaf_ptr = node_ptr.cast::<LeafNode24<S>>();
            let leaf = unsafe { &*leaf_ptr };

            // Recurse into any layer pointers before freeing this leaf.
            // Layer pointers are sublayer roots that need recursive traversal.
            //
            // IMPORTANT: The layer pointer may point to the ORIGINAL sublayer root
            // leaf, but after sublayer splits, a new internode may be the actual root.
            // We must follow parent pointers to find the current root.
            for slot in 0..WIDTH_24 {
                if leaf.is_layer(slot) {
                    let layer_ptr = leaf.leaf_value_ptr(slot);
                    if !layer_ptr.is_null() {
                        // Find the actual sublayer root (may have changed due to splits)
                        // SAFETY: layer_ptr is valid, we have exclusive access
                        let layer_root = unsafe { self.find_layer_root(layer_ptr) };
                        // Recursively free the sublayer from its current root
                        unsafe { self.traverse_and_free(layer_root) };
                    }
                }
            }

            // Free the leaf itself.
            // LeafNode's Drop impl handles value cleanup (skips layer slots).
            // SAFETY: We have exclusive access and leaf is valid.
            unsafe { drop(Box::from_raw(leaf_ptr)) };
        } else {
            // Handle internode
            let internode_ptr = node_ptr.cast::<InternodeNode>();
            let internode = unsafe { &*internode_ptr };
            let nkeys = internode.nkeys();

            // Recurse into all children (nkeys + 1 children for nkeys keys)
            // SAFETY: During Drop, we have exclusive access - no concurrent retirement.
            for i in 0..=nkeys {
                let child_ptr = unsafe { internode.child_unguarded(i) };
                if !child_ptr.is_null() {
                    unsafe { self.traverse_and_free(child_ptr) };
                }
            }

            // Free the internode.
            // SAFETY: We have exclusive access and internode is valid.
            unsafe { drop(Box::from_raw(internode_ptr)) };
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation
// =============================================================================

impl<S> NodeAllocatorGeneric<S, LeafNode24<S>> for SeizeAllocator24<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        Box::into_raw(node)
    }

    #[inline(always)]
    fn track_leaf(&self, _ptr: *mut LeafNode24<S>) {
        // No tracking - tree traversal handles cleanup
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // Direct retirement - O(1), no tracking overhead.
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(ptr, seize::reclaim::boxed);
        }
    }

    #[inline(always)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // No tracking needed - just return the pointer as-is
        node_ptr
    }

    #[inline(always)]
    fn track_internode_erased(&self, _ptr: *mut u8) {
        // No tracking - tree traversal handles cleanup
    }

    #[inline(always)]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Caller guarantees InternodeNode alignment"
    )]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Direct retirement - O(1), no tracking overhead.
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(ptr.cast::<InternodeNode>(), seize::reclaim::boxed);
        }
    }

    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "Trait contract requires valid pointer from tree root"
    )]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        if root_ptr.is_null() {
            return;
        }
        // SAFETY: root_ptr is valid, we have exclusive access during Drop.
        // Tree traversal frees all nodes including sublayers.
        unsafe { self.traverse_and_free(root_ptr) };
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Retire the subtree root with a reclaimer that traverses and frees.
        // This defers the traversal until seize determines it's safe.
        if root_ptr.is_null() {
            return;
        }

        // SAFETY: Caller ensures subtree is fully unlinked.
        // We defer the traversal to when seize says it's safe to reclaim.
        unsafe {
            guard.defer_retire(root_ptr, |ptr, _collector| {
                // Create a temporary allocator instance for traversal
                let alloc = Self::new();
                // SAFETY: ptr is valid and we have exclusive access (seize guarantees this)
                alloc.traverse_and_free(ptr);
            });
        }
    }

    /// Allocate a leaf directly without Box intermediate.
    ///
    /// Uses raw allocation + `init_at` to avoid stack-to-heap copy.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode24<S> {
        self.try_alloc_leaf(is_root, is_layer_root)
            .unwrap_or_else(|_| {
                let layout: Layout = Layout::new::<LeafNode24<S>>();
                StdAlloc::handle_alloc_error(layout)
            })
    }

    /// Try to allocate a leaf node, returning an error on failure.
    fn try_alloc_leaf(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> AllocResult<*mut LeafNode24<S>> {
        // Step 1: Allocate raw memory
        let layout: Layout = Layout::new::<LeafNode24<S>>();

        // SAFETY: Layout is valid (derived from type)
        let raw_ptr: *mut u8 = unsafe { StdAlloc::alloc(layout) };

        if raw_ptr.is_null() {
            return Err(AllocError::for_leaf::<LeafNode24<S>>());
        }

        let ptr: *mut LeafNode24<S> = raw_ptr.cast();

        // Step 2: Initialize in-place using init_at
        //
        // The is_layer_root flag is passed to init_at as is_root because
        // layer roots also have ROOT_BIT set (they're roots of sublayers).
        //
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            LeafNode24::init_at(ptr, is_root || is_layer_root);
        }

        Ok(ptr)
    }

    /// Allocate an internode directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        use std::alloc::{alloc, Layout};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid (non-zero size)")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr, height);
        }

        ptr.cast()
    }

    /// Allocate an internode as root directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        use std::alloc::{alloc, Layout};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place as root
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr, height);
        }

        ptr.cast()
    }

    /// Allocate an internode for split directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        use std::alloc::{alloc, Layout};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place with split-locked version
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr, parent_version, height);
        }

        ptr.cast()
    }
}

#[cfg(test)]
mod unit_tests;
