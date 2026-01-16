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
//!
//! # Performance Optimizations
//!
//! - **Iterative traversal**: Avoids stack overflow on deep trees with sublayers
//! - **No `drop_in_place` for `InternodeNode`**: It has no Drop impl
//! - **Function pointers**: No closure allocations in `retire_subtree_root`
//! - **Macro deduplication**: Single implementation for both allocator variants

use std::alloc as StdAlloc;
use std::alloc::Layout;
use std::marker::PhantomData;

use arrayvec::ArrayVec;
use seize::{Collector, Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::inline::bits::InlineBits;
use crate::inline::leaf15_true::LeafNode15TrueInline;
use crate::internode::InternodeNode;
use crate::leaf15::{LeafNode15, WIDTH_15};
use crate::node_pool;
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;
use crate::slot::true_inline::TrueInlineSlot;
use crate::{AllocError, AllocResult};

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
/// # Type Parameters
///
/// - `L`: Leaf type with `parent_unguarded()` method
/// - `parent_fn`: Function to get parent pointer from leaf
///
/// # Safety
///
/// - `node_ptr` must point to a valid leaf or internode
/// - Caller must have exclusive access (no concurrent readers/writers)
#[inline]
#[expect(clippy::cast_ptr_alignment, reason = "Callers guarantee alignment")]
unsafe fn find_layer_root_generic<L, F>(mut node_ptr: *mut u8, parent_fn: F) -> *mut u8
where
    F: Fn(&L) -> *mut u8,
{
    loop {
        // SAFETY: Both leaves and internodes have NodeVersion at offset 0
        let version: &NodeVersion = unsafe { &*node_ptr.cast::<NodeVersion>() };

        // SAFETY: Called during teardown with exclusive access
        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed
            let leaf: &L = unsafe { &*node_ptr.cast::<L>() };
            parent_fn(leaf)
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
/// Uses hybrid stack: `ArrayVec` for typical trees (no heap allocation),
/// with `Vec` fallback for very deep trees with many sublayers.
///
/// # Type Parameters
///
/// - `L`: Leaf node type
/// - `is_layer_fn`: Check if slot contains a layer pointer
/// - `value_ptr_fn`: Get value/layer pointer from slot
/// - `parent_fn`: Get parent pointer from leaf
/// - `free_leaf_fn`: Deallocate a leaf node
///
/// # Safety
///
/// - `root_ptr` must point to a valid leaf or internode
/// - Caller must have exclusive access (no concurrent readers/writers)
/// - Only safe during `Drop` when tree is quiescent
#[expect(clippy::cast_ptr_alignment, reason = "Callers guarantee alignment")]
unsafe fn traverse_and_free_iterative<L, IsLayer, ValuePtr, ParentFn, FreeFn>(
    root_ptr: *mut u8,
    is_layer_fn: IsLayer,
    value_ptr_fn: ValuePtr,
    parent_fn: ParentFn,
    free_leaf_fn: FreeFn,
) where
    IsLayer: Fn(&L, usize) -> bool,
    ValuePtr: Fn(&L, usize) -> *mut u8,
    ParentFn: Fn(&L) -> *mut u8 + Copy,
    FreeFn: Fn(*mut u8),
{
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
                    let leaf: &L = unsafe { &*node_ptr.cast::<L>() };

                    // Queue leaf for deallocation AFTER sublayers (LIFO order)
                    push_hybrid(&mut stack, &mut overflow, TraversalWork::FreeLeaf(node_ptr));

                    // Queue sublayers (will be processed before FreeLeaf due to LIFO)
                    for slot in 0..WIDTH_15 {
                        if is_layer_fn(leaf, slot) {
                            let layer_ptr: *mut u8 = value_ptr_fn(leaf, slot);
                            if !layer_ptr.is_null() {
                                // Find actual sublayer root (may have changed due to splits)
                                // SAFETY: layer_ptr is valid, we have exclusive access
                                let layer_root: *mut u8 =
                                    unsafe { find_layer_root_generic(layer_ptr, parent_fn) };
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
                free_leaf_fn(ptr);
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
// Capture-Free Subtree Reclaimers
// =============================================================================

/// Capture-free subtree reclaimer for `SeizeAllocator15`.
///
/// Used as function pointer in `retire_subtree_root` to avoid closure allocations.
///
/// # Safety
///
/// - `ptr` must point to a valid subtree root (leaf or internode)
/// - Must be called only after seize determines it's safe to reclaim
#[expect(clippy::cast_ptr_alignment, reason = "Seize guarantees alignment")]
unsafe fn reclaim_subtree_15<S: ValueSlot + 'static>(ptr: *mut u8, _collector: &Collector) {
    unsafe {
        traverse_and_free_iterative::<LeafNode15<S>, _, _, _, _>(
            ptr,
            LeafNode15::is_layer,
            LeafNode15::leaf_value_ptr,
            |leaf| leaf.parent_unguarded(),
            |leaf_ptr| {
                std::ptr::drop_in_place(leaf_ptr.cast::<LeafNode15<S>>());
                node_pool::pool_dealloc(leaf_ptr, Layout::new::<LeafNode15<S>>());
            },
        );
    }
}

/// Capture-free subtree reclaimer for `SeizeAllocator15TrueInline`.
///
/// # Safety
///
/// - `ptr` must point to a valid subtree root (leaf or internode)
/// - Must be called only after seize determines it's safe to reclaim
#[expect(clippy::cast_ptr_alignment, reason = "Seize guarantees alignment")]
unsafe fn reclaim_subtree_15_true_inline<V: InlineBits + 'static>(
    ptr: *mut u8,
    _collector: &Collector,
) {
    unsafe {
        traverse_and_free_iterative::<LeafNode15TrueInline<V>, _, _, _, _>(
            ptr,
            LeafNode15TrueInline::is_layer,
            LeafNode15TrueInline::leaf_value_ptr,
            |leaf| leaf.parent_unguarded(),
            |leaf_ptr| {
                std::ptr::drop_in_place(leaf_ptr.cast::<LeafNode15TrueInline<V>>());
                node_pool::pool_dealloc(leaf_ptr, Layout::new::<LeafNode15TrueInline<V>>());
            },
        );
    }
}

// =============================================================================
// Allocator Implementation Macro
// =============================================================================

/// Implements `NodeAllocatorGeneric` for seize-based allocators with WIDTH=15.
///
/// This macro eliminates ~350 lines of duplication between `SeizeAllocator15`
/// and `SeizeAllocator15TrueInline` while maintaining full type safety.
macro_rules! impl_seize_allocator15 {
    (
        name: $name:ident,
        type_param: $T:ident,
        type_bound: $bound:path,
        leaf_type: $leaf:ty,
        slot_type: $slot:ty,
        reclaim_leaf: $reclaim_leaf:expr,
        reclaim_subtree: $reclaim_subtree:expr,
        is_layer_fn: $is_layer:expr,
        value_ptr_fn: $value_ptr:expr,
        parent_fn: $parent_fn:expr,
    ) => {
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
        /// - **Iterative traversal**: No stack overflow risk on deep trees
        ///
        /// # Thread Safety
        ///
        /// Stateless after construction - all operations are safe for concurrent use.
        #[derive(Debug)]
        pub struct $name<$T: $bound> {
            _marker: PhantomData<$T>,
        }

        // SAFETY: Allocator is stateless (just PhantomData), safe to send/share.
        unsafe impl<$T: $bound + Send + Sync> Send for $name<$T> {}
        unsafe impl<$T: $bound + Send + Sync> Sync for $name<$T> {}

        impl<$T: $bound> $name<$T> {
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
            /// - Caller must have exclusive access (no concurrent readers/writers)
            /// - Only safe during `Drop` when tree is quiescent
            unsafe fn traverse_and_free(&self, node_ptr: *mut u8) {
                if node_ptr.is_null() {
                    return;
                }

                // SAFETY: Caller guarantees valid pointer and exclusive access
                unsafe {
                    traverse_and_free_iterative::<$leaf, _, _, _, _>(
                        node_ptr,
                        $is_layer,
                        $value_ptr,
                        $parent_fn,
                        |leaf_ptr| {
                            std::ptr::drop_in_place(leaf_ptr.cast::<$leaf>());
                            node_pool::pool_dealloc(leaf_ptr, Layout::new::<$leaf>());
                        },
                    );
                }
            }
        }

        impl<$T: $bound> Default for $name<$T> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<$T: $bound + Send + Sync + 'static> NodeAllocatorGeneric<$slot, $leaf> for $name<$T> {
            #[inline(always)]
            fn alloc_leaf(&self, node: Box<$leaf>) -> *mut $leaf {
                Box::into_raw(node)
            }

            #[inline(always)]
            fn track_leaf(&self, _ptr: *mut $leaf) {
                // No tracking - tree traversal handles cleanup
            }

            #[inline(always)]
            unsafe fn retire_leaf(&self, ptr: *mut $leaf, guard: &LocalGuard<'_>) {
                // SAFETY: Caller ensures ptr is valid and unreachable from tree.
                unsafe {
                    guard.defer_retire(ptr, $reclaim_leaf);
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
                    guard.defer_retire(root_ptr, $reclaim_subtree);
                }
            }

            #[inline]
            fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut $leaf {
                self.try_alloc_leaf(is_root, is_layer_root)
                    .unwrap_or_else(|_| {
                        let layout = Layout::new::<$leaf>();
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
                    InternodeNode::init_at_for_split(
                        ptr.cast::<InternodeNode>(),
                        parent_version,
                        height,
                    );
                }

                ptr
            }

            #[inline]
            fn try_alloc_leaf(
                &self,
                is_root: bool,
                is_layer_root: bool,
            ) -> AllocResult<*mut $leaf> {
                let layout = Layout::new::<$leaf>();
                let raw_ptr = node_pool::pool_alloc(layout);

                if raw_ptr.is_null() {
                    return Err(AllocError::for_leaf::<$leaf>());
                }

                let ptr: *mut $leaf = raw_ptr.cast();

                // SAFETY: ptr is valid, properly aligned, and we have exclusive access
                unsafe {
                    <$leaf>::init_at(ptr, is_root || is_layer_root);
                }

                Ok(ptr)
            }
        }
    };
}

// =============================================================================
// SeizeAllocator15 - For LeafNode15<S>
// =============================================================================

impl_seize_allocator15! {
    name: SeizeAllocator15,
    type_param: S,
    type_bound: ValueSlot,
    leaf_type: LeafNode15<S>,
    slot_type: S,
    reclaim_leaf: node_pool::reclaim_leaf15::<S>,
    reclaim_subtree: reclaim_subtree_15::<S>,
    is_layer_fn: LeafNode15::is_layer,
    value_ptr_fn: LeafNode15::leaf_value_ptr,
    parent_fn: |leaf: &LeafNode15<S>| leaf.parent_unguarded(),
}

// =============================================================================
// SeizeAllocator15TrueInline - For LeafNode15TrueInline<V>
// =============================================================================

impl_seize_allocator15! {
    name: SeizeAllocator15TrueInline,
    type_param: V,
    type_bound: InlineBits,
    leaf_type: LeafNode15TrueInline<V>,
    slot_type: TrueInlineSlot<V>,
    reclaim_leaf: node_pool::reclaim_leaf15_true_inline::<V>,
    reclaim_subtree: reclaim_subtree_15_true_inline::<V>,
    is_layer_fn: LeafNode15TrueInline::is_layer,
    value_ptr_fn: LeafNode15TrueInline::leaf_value_ptr,
    parent_fn: |leaf: &LeafNode15TrueInline<V>| leaf.parent_unguarded(),
}

#[cfg(test)]
mod unit_tests;
