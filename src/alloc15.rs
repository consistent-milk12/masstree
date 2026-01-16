//! Node allocation for [`LeafNode15`] with WIDTH=15 leaves and WIDTH=15 internodes.
//!
//! This module provides [`SeizeAllocator15`] and [`SeizeAllocator15TrueInline`],
//! Miri-compliant allocators for masstree nodes using `seize` for memory reclamation.
//!
//! # Design
//!
//! These allocators use **direct retirement** without tracking lists, following
//! papaya's seize integration pattern. Retirement is O(1) - just a seize call.
//! Tree cleanup uses root-based traversal instead of list iteration.

use std::alloc as StdAlloc;
use std::alloc::Layout;
use std::marker::PhantomData;

use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::internode::InternodeNode;
use crate::leaf15::{LeafNode15, WIDTH_15};
use crate::node_pool;
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;
use crate::{AllocError, AllocResult};

// =============================================================================
// SeizeAllocator15 - For LeafNode15<S>
// =============================================================================

/// Node allocator using seize for safe memory reclamation (WIDTH=15).
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
pub struct SeizeAllocator15<S: ValueSlot> {
    _marker: PhantomData<S>,
}

// SAFETY: Allocator is stateless (just PhantomData), safe to send/share.
unsafe impl<S: ValueSlot + Send + Sync> Send for SeizeAllocator15<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for SeizeAllocator15<S> {}

impl<S: ValueSlot> SeizeAllocator15<S> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<S: ValueSlot> Default for SeizeAllocator15<S> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tree Traversal for Teardown - SeizeAllocator15
// =============================================================================

impl<S: ValueSlot> SeizeAllocator15<S> {
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
        clippy::unused_self,
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
                let leaf: &LeafNode15<S> = unsafe { &*node_ptr.cast::<LeafNode15<S>>() };
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
            let leaf_ptr = node_ptr.cast::<LeafNode15<S>>();
            let leaf = unsafe { &*leaf_ptr };

            // Recurse into any layer pointers before freeing this leaf.
            // Layer pointers are sublayer roots that need recursive traversal.
            //
            // IMPORTANT: The layer pointer may point to the ORIGINAL sublayer root
            // leaf, but after sublayer splits, a new internode may be the actual root.
            // We must follow parent pointers to find the current root.
            for slot in 0..WIDTH_15 {
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
            // First drop in place to run destructor (handles value cleanup).
            // SAFETY: We have exclusive access and leaf is valid.
            unsafe { std::ptr::drop_in_place(leaf_ptr) };
            // Then return memory to pool.
            let layout = Layout::new::<LeafNode15<S>>();
            unsafe { node_pool::pool_dealloc(leaf_ptr.cast(), layout) };
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
            unsafe { std::ptr::drop_in_place(internode_ptr) };
            let layout = Layout::new::<InternodeNode>();
            unsafe { node_pool::pool_dealloc(internode_ptr.cast(), layout) };
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation - SeizeAllocator15
// =============================================================================

impl<S> NodeAllocatorGeneric<S, LeafNode15<S>> for SeizeAllocator15<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15<S>>) -> *mut LeafNode15<S> {
        Box::into_raw(node)
    }

    #[inline(always)]
    fn track_leaf(&self, _ptr: *mut LeafNode15<S>) {
        // No tracking - tree traversal handles cleanup
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<S>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        // Use capture-free reclaimer to return to thread-local pool.
        unsafe {
            guard.defer_retire(ptr, node_pool::reclaim_leaf15::<S>);
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
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        // Use capture-free reclaimer to return to thread-local pool.
        unsafe {
            guard.defer_retire(ptr.cast::<InternodeNode>(), node_pool::reclaim_internode);
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
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<S> {
        self.try_alloc_leaf(is_root, is_layer_root)
            .unwrap_or_else(|_| {
                let layout = Layout::new::<LeafNode15<S>>();
                StdAlloc::handle_alloc_error(layout)
            })
    }

    /// Allocate an internode directly using thread-local pool.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr.cast::<InternodeNode>(), height);
        }

        ptr
    }

    /// Allocate an internode as root directly using thread-local pool.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr.cast::<InternodeNode>(), height);
        }

        ptr
    }

    /// Allocate an internode for split directly using thread-local pool.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr.cast::<InternodeNode>(), parent_version, height);
        }

        ptr
    }

    fn try_alloc_leaf(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> AllocResult<*mut LeafNode15<S>> {
        let layout = Layout::new::<LeafNode15<S>>();
        let raw_ptr = node_pool::pool_alloc(layout);

        if raw_ptr.is_null() {
            return Err(AllocError::for_leaf::<LeafNode15<S>>());
        }

        let ptr: *mut LeafNode15<S> = raw_ptr.cast();

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            LeafNode15::init_at(ptr, is_root || is_layer_root);
        }

        Ok(ptr)
    }
}

// =============================================================================
// SeizeAllocator15TrueInline - For LeafNode15TrueInline<V>
// =============================================================================

use crate::inline::bits::InlineBits;
use crate::inline::leaf15_true::LeafNode15TrueInline;
use crate::slot::true_inline::TrueInlineSlot;

/// Allocator for true-inline leaf nodes (WIDTH=15).
///
/// Similar to [`SeizeAllocator15`] but manages [`LeafNode15TrueInline`] nodes
/// instead of [`LeafNode15`]. Since `TrueInlineSlot::NEEDS_RETIREMENT = false`,
/// no value retirement is needed.
///
/// # Layer Support
///
/// True-inline leaves CAN have layer pointers (via `set_layer_ptr()` and
/// `is_layer(slot)`). The traversal must check `is_layer()` for each slot
/// and recurse into sublayers, same as regular leaves.
#[derive(Debug)]
pub struct SeizeAllocator15TrueInline<V: InlineBits> {
    _marker: PhantomData<V>,
}

// SAFETY: Allocator is stateless (just PhantomData), safe to send/share.
unsafe impl<V: InlineBits> Send for SeizeAllocator15TrueInline<V> {}
unsafe impl<V: InlineBits> Sync for SeizeAllocator15TrueInline<V> {}

impl<V: InlineBits> SeizeAllocator15TrueInline<V> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<V: InlineBits> Default for SeizeAllocator15TrueInline<V> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tree Traversal for Teardown - SeizeAllocator15TrueInline
// =============================================================================

impl<V: InlineBits> SeizeAllocator15TrueInline<V> {
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
        clippy::unused_self,
        reason = "Callers guarantee proper alignment"
    )]
    unsafe fn find_layer_root(&self, mut node_ptr: *mut u8) -> *mut u8 {
        use crate::inline::leaf15_true::LeafNode15TrueInline;

        loop {
            // SAFETY: Both leaves and internodes have NodeVersion at offset 0
            let version_ptr = node_ptr.cast::<NodeVersion>();
            let version: &NodeVersion = unsafe { &*version_ptr };

            // SAFETY: Called during teardown with exclusive access - no concurrent retirement.
            let parent: *mut u8 = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &LeafNode15TrueInline<V> =
                    unsafe { &*node_ptr.cast::<LeafNode15TrueInline<V>>() };
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
    /// # Note on True-Inline Leaves
    ///
    /// True-inline leaves CAN have layer pointers (via `set_layer_ptr()` and
    /// `is_layer(slot)`). We must check `is_layer()` for each slot and recurse
    /// into sublayers, same as regular leaves.
    ///
    /// For non-layer slots, `leaf_value_ptr()` returns an encoded pointer
    /// (not a real allocation), so we must NOT try to free it.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Callers guarantee proper alignment"
    )]
    unsafe fn traverse_and_free(&self, node_ptr: *mut u8) {
        use crate::inline::leaf15_true::WIDTH_15;

        if node_ptr.is_null() {
            return;
        }

        // Read version to determine node type.
        // SAFETY: Both leaves and internodes have NodeVersion at offset 0.
        let version_ptr = node_ptr.cast::<NodeVersion>();
        let version = unsafe { &*version_ptr };

        if version.is_leaf() {
            // Handle leaf node
            let leaf_ptr = node_ptr.cast::<LeafNode15TrueInline<V>>();
            let leaf = unsafe { &*leaf_ptr };

            // Recurse into any layer pointers before freeing this leaf.
            // True-inline leaves CAN have layer pointers - check is_layer() first.
            //
            // IMPORTANT: The layer pointer may point to the ORIGINAL sublayer root
            // leaf, but after sublayer splits, a new internode may be the actual root.
            // We must follow parent pointers to find the current root.
            for slot in 0..WIDTH_15 {
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
                // Non-layer slots have encoded inline values, not real allocations
            }

            // Free the leaf itself.
            // First drop in place to run destructor.
            // SAFETY: We have exclusive access and leaf is valid.
            unsafe { std::ptr::drop_in_place(leaf_ptr) };
            // Then return memory to pool.
            let layout = Layout::new::<LeafNode15TrueInline<V>>();
            unsafe { node_pool::pool_dealloc(leaf_ptr.cast(), layout) };
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
            unsafe { std::ptr::drop_in_place(internode_ptr) };
            let layout = Layout::new::<InternodeNode>();
            unsafe { node_pool::pool_dealloc(internode_ptr.cast(), layout) };
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation - SeizeAllocator15TrueInline
// =============================================================================

impl<V: InlineBits + Send + Sync + 'static>
    NodeAllocatorGeneric<TrueInlineSlot<V>, LeafNode15TrueInline<V>>
    for SeizeAllocator15TrueInline<V>
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15TrueInline<V>>) -> *mut LeafNode15TrueInline<V> {
        Box::into_raw(node)
    }

    #[inline(always)]
    fn track_leaf(&self, _ptr: *mut LeafNode15TrueInline<V>) {
        // No tracking - tree traversal handles cleanup
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15TrueInline<V>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        // Use capture-free reclaimer to return to thread-local pool.
        unsafe {
            guard.defer_retire(ptr, node_pool::reclaim_leaf15_true_inline::<V>);
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
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        // Use capture-free reclaimer to return to thread-local pool.
        unsafe {
            guard.defer_retire(ptr.cast::<InternodeNode>(), node_pool::reclaim_internode);
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
        // True-inline leaves CAN have layer pointers - traversal handles them.
        unsafe { self.traverse_and_free(root_ptr) };
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Retire the subtree root with a reclaimer that traverses and frees.
        if root_ptr.is_null() {
            return;
        }

        // SAFETY: Caller ensures subtree is fully unlinked.
        unsafe {
            guard.defer_retire(root_ptr, |ptr, _collector| {
                let alloc = Self::new();
                // SAFETY: ptr is valid and we have exclusive access (seize guarantees this)
                alloc.traverse_and_free(ptr);
            });
        }
    }

    #[inline]
    fn alloc_leaf_direct(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> *mut LeafNode15TrueInline<V> {
        self.try_alloc_leaf(is_root, is_layer_root)
            .unwrap_or_else(|_| {
                let layout = Layout::new::<LeafNode15TrueInline<V>>();
                StdAlloc::handle_alloc_error(layout)
            })
    }

    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr.cast::<InternodeNode>(), height);
        }

        ptr
    }

    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr.cast::<InternodeNode>(), height);
        }

        ptr
    }

    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        let layout = Layout::new::<InternodeNode>();
        let ptr = node_pool::pool_alloc(layout);
        if ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr.cast::<InternodeNode>(), parent_version, height);
        }

        ptr
    }

    fn try_alloc_leaf(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> AllocResult<*mut LeafNode15TrueInline<V>> {
        let layout = Layout::new::<LeafNode15TrueInline<V>>();
        let raw_ptr = node_pool::pool_alloc(layout);

        if raw_ptr.is_null() {
            return Err(AllocError::for_leaf::<LeafNode15TrueInline<V>>());
        }

        let ptr: *mut LeafNode15TrueInline<V> = raw_ptr.cast();

        // SAFETY: ptr is valid, properly aligned, exclusive access
        unsafe {
            LeafNode15TrueInline::init_at(ptr, is_root || is_layer_root);
        }

        Ok(ptr)
    }
}

#[cfg(test)]
mod unit_tests;
