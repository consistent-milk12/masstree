//! Node allocation for [`LeafNode15`] with WIDTH=15 leaves and WIDTH=15 internodes.
//!
//! This module provides [`SeizeAllocator<P>`], a Miri-compliant allocator for
//! masstree nodes using `seize` for memory reclamation.
//!
//! # Design
//!
//! Uses **direct retirement** without tracking lists, following papaya's seize
//! integration pattern. Retirement is O(1) - just a seize call. Tree cleanup
//! uses root-based traversal instead of list iteration.
//!
//! # Performance Optimizations
//!
//! - **Iterative traversal**: Avoids stack overflow on deep trees with sublayers
//! - **No `drop_in_place` for `InternodeNode`**: It has no Drop impl
//! - **Function pointers**: No closure allocations in `retire_subtree_root`

use std::alloc as StdAlloc;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr as StdPtr;

use arrayvec::ArrayVec;
use seize::{Collector, Guard, LocalGuard};

use crate::alloc_trait::TreeAllocator;
use crate::internode::InternodeNode;
use crate::leaf15::{LeafNode15, WIDTH_15};
use crate::node_pool;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;

// =============================================================================
// Iterative Tree Traversal
// =============================================================================

/// Work items for iterative tree traversal during teardown.
///
/// Uses explicit stack to avoid recursion and potential stack overflow
/// on deep trees with many sublayers.
enum TraversalWork {
    /// Visit a node (may be leaf or internode), queuing children/sublayers.
    Visit(*mut u8),

    /// Free a leaf node after its sublayers have been freed.
    FreeLeaf(*mut u8),

    /// Free an internode after its children have been freed.
    FreeInternode(*mut u8),
}

/// `ArrayVec` capacity for traversal stack.
///
/// Handles typical trees entirely on the stack (no heap allocation).
/// 128 entries covers trees with depth 128 and typical branching.
const STACK_CAPACITY: usize = 128;

/// Overflow Vec initial capacity (only allocated when `ArrayVec` fills).
///
/// Sized for deep trees with many sublayers (e.g., rw1long test with long keys).
const OVERFLOW_INITIAL_CAPACITY: usize = 64;

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
#[inline]
#[expect(clippy::cast_ptr_alignment, reason = "Callers guarantee alignment")]
unsafe fn find_layer_root<P: LeafPolicy>(mut node_ptr: *mut u8) -> *mut u8 {
    loop {
        // SAFETY: Both leaves and internodes have NodeVersion at offset 0
        let version: &NodeVersion = unsafe { &*node_ptr.cast::<NodeVersion>() };

        // SAFETY: Called during teardown with exclusive access
        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed
            let leaf: &LeafNode15<P> = unsafe { &*node_ptr.cast::<LeafNode15<P>>() };
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

/// Iterative tree traversal and deallocation.
///
/// Uses `LeafNode15<P>::is_layer()` and `LeafNode15<P>::load_layer_raw()`
/// for policy-agnostic layer detection and pointer extraction.
///
/// # Safety
///
/// - `root_ptr` must point to a valid leaf or internode
/// - Caller must have exclusive access (no concurrent readers/writers)
/// - Only safe during `Drop` when tree is quiescent
#[expect(clippy::cast_ptr_alignment, reason = "Callers guarantee alignment")]
unsafe fn traverse_and_free_iterative<P: LeafPolicy>(root_ptr: *mut u8) {
    // Primary stack: stack-allocated, handles typical trees without heap allocation
    let mut stack: ArrayVec<TraversalWork, STACK_CAPACITY> = ArrayVec::new();
    // Overflow: only allocated for deep trees with many sublayers (e.g., rw1long)
    let mut overflow: Option<Vec<TraversalWork>> = None;

    stack.push(TraversalWork::Visit(root_ptr));

    loop {
        // Pop from ArrayVec first (fast path), fall back to overflow Vec
        let work: TraversalWork = match stack.pop() {
            Some(w) => w,
            None => match overflow.as_mut().and_then(Vec::pop) {
                Some(w) => w,
                None => break, // Both empty, traversal complete
            },
        };

        match work {
            TraversalWork::Visit(node_ptr) => {
                if node_ptr.is_null() {
                    continue;
                }

                // SAFETY: Both leaves and internodes have NodeVersion at offset 0
                let version: &NodeVersion = unsafe { &*node_ptr.cast::<NodeVersion>() };

                if version.is_leaf() {
                    // SAFETY: version.is_leaf() confirmed
                    let leaf: &LeafNode15<P> = unsafe { &*node_ptr.cast::<LeafNode15<P>>() };

                    // Queue leaf for deallocation AFTER sublayers (LIFO order)
                    push_hybrid(&mut stack, &mut overflow, TraversalWork::FreeLeaf(node_ptr));

                    // Queue sublayers (will be processed before FreeLeaf due to LIFO)
                    for slot in 0..WIDTH_15 {
                        if leaf.is_layer(slot) {
                            let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);
                            if !layer_ptr.is_null() {
                                // Find actual sublayer root (may have changed due to splits)
                                // SAFETY: layer_ptr is valid, we have exclusive access
                                let layer_root: *mut u8 =
                                    unsafe { find_layer_root::<P>(layer_ptr) };
                                push_hybrid(
                                    &mut stack,
                                    &mut overflow,
                                    TraversalWork::Visit(layer_root),
                                );
                            }
                        }
                    }
                } else {
                    // SAFETY: !version.is_leaf() confirmed
                    let internode: &InternodeNode = unsafe { &*node_ptr.cast::<InternodeNode>() };
                    let nkeys: usize = internode.nkeys();

                    // Queue internode for deallocation AFTER children
                    push_hybrid(
                        &mut stack,
                        &mut overflow,
                        TraversalWork::FreeInternode(node_ptr),
                    );

                    // Queue children in reverse order for correct traversal
                    // SAFETY: During Drop, we have exclusive access
                    for i in (0..=nkeys).rev() {
                        let child: *mut u8 = unsafe { internode.child_unguarded(i) };
                        if !child.is_null() {
                            push_hybrid(&mut stack, &mut overflow, TraversalWork::Visit(child));
                        }
                    }
                }
            }
            TraversalWork::FreeLeaf(ptr) => {
                // SAFETY: ptr is a valid leaf, we have exclusive access.
                unsafe {
                    StdPtr::drop_in_place(ptr.cast::<LeafNode15<P>>());
                    node_pool::pool_dealloc(ptr, Layout::new::<LeafNode15<P>>());
                }
            }
            TraversalWork::FreeInternode(ptr) => {
                // NO drop_in_place - InternodeNode has no Drop impl
                // Just return memory to pool
                let layout: Layout = Layout::new::<InternodeNode>();
                unsafe { node_pool::pool_dealloc(ptr, layout) };
            }
        }
    }
}

/// Push to hybrid stack: prefer `ArrayVec` (no allocation), fall back to `Vec`.
#[inline]
fn push_hybrid(
    stack: &mut ArrayVec<TraversalWork, STACK_CAPACITY>,
    overflow: &mut Option<Vec<TraversalWork>>,
    work: TraversalWork,
) {
    if let Err(cap_err) = stack.try_push(work) {
        // ArrayVec full, use heap-allocated overflow
        overflow
            .get_or_insert_with(|| Vec::with_capacity(OVERFLOW_INITIAL_CAPACITY))
            .push(cap_err.element());
    }
}

// =============================================================================
// Capture-Free Subtree Reclaimer
// =============================================================================

/// Capture-free subtree reclaimer for `SeizeAllocator<P>`.
///
/// # Safety
///
/// - `ptr` must point to a valid subtree root (leaf or internode)
/// - Must be called only after seize determines it's safe to reclaim
unsafe fn reclaim_subtree<P: LeafPolicy + 'static>(ptr: *mut u8, _collector: &Collector) {
    // SAFETY: ptr is a valid subtree root, safe to traverse and free.
    unsafe {
        traverse_and_free_iterative::<P>(ptr);
    }
}

// =============================================================================
// SeizeAllocator<P> — Unified allocator for all leaf policies
// =============================================================================

/// Unified seize-based allocator for all leaf policies.
///
/// This allocator relies on tree traversal for teardown rather than
/// maintaining tracking lists, eliminating O(n) overhead per retirement.
///
/// # Design
///
/// - **O(1) retirement**: Direct `guard.defer_retire()` calls
/// - **No tracking overhead**: No mutex locks or linear scans during retirement
/// - **Tree traversal teardown**: Frees all nodes by walking the tree structure
/// - **Iterative traversal**: No stack overflow risk on deep trees
///
/// # Thread Safety
///
/// Stateless after construction - all operations are safe for concurrent use.
#[derive(Debug)]
pub struct SeizeAllocator<P: LeafPolicy> {
    _marker: PhantomData<P>,
}

// SAFETY: Allocator is stateless (just PhantomData), safe to send/share.
unsafe impl<P: LeafPolicy> Send for SeizeAllocator<P> {}
unsafe impl<P: LeafPolicy> Sync for SeizeAllocator<P> {}

impl<P: LeafPolicy> SeizeAllocator<P> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Traverse tree structure and free all nodes (iterative).
    ///
    /// # Safety
    ///
    /// - `node_ptr` must point to a valid leaf or internode
    /// - Caller must have exclusive access
    #[expect(clippy::unused_self, reason = "Required by TreeAllocator trait")]
    unsafe fn traverse_and_free(&self, node_ptr: *mut u8) {
        if node_ptr.is_null() {
            return;
        }
        // SAFETY: Caller guarantees valid pointer and exclusive access.
        unsafe {
            traverse_and_free_iterative::<P>(node_ptr);
        }
    }
}

impl<P: LeafPolicy> Default for SeizeAllocator<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: LeafPolicy + 'static> TreeAllocator<P> for SeizeAllocator<P> {
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15<P>>) -> *mut LeafNode15<P> {
        Box::into_raw(node)
    }

    #[inline(always)]
    fn track_leaf(&self, _ptr: *mut LeafNode15<P>) {
        // No tracking - tree traversal handles cleanup
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<P>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(ptr, node_pool::reclaim_leaf15::<P>);
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
        unsafe { self.traverse_and_free(root_ptr) };
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        if root_ptr.is_null() {
            return;
        }

        // SAFETY: Caller ensures subtree is fully unlinked.
        // Use capture-free function pointer to avoid closure allocation.
        unsafe {
            guard.defer_retire(root_ptr, reclaim_subtree::<P>);
        }
    }

    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<P> {
        let layout = Layout::new::<LeafNode15<P>>();
        let raw_ptr = node_pool::pool_alloc(layout);

        if raw_ptr.is_null() {
            StdAlloc::handle_alloc_error(layout);
        }

        let ptr: *mut LeafNode15<P> = raw_ptr.cast();

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            LeafNode15::<P>::init_at(ptr, is_root || is_layer_root);
        }

        ptr
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
}

#[cfg(test)]
mod unit_tests;
