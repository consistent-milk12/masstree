//! Node allocation for [`LeafNode15`] with WIDTH=24 leaves and WIDTH=15 internodes.
//!
//! This module provides [`SeizeAllocator15`], a Miri-compliant allocator for
//! masstree nodes using `seize` for memory reclamation.
//!
//! # Naming Convention
//!
//! The "15" in [`SeizeAllocator15`] refers to the **internode width** (WIDTH=15),
//! which uses the original Permuter (u64 with 4-bit slots). Despite the name,
//! this allocator manages [`LeafNode15`] nodes which have **leaf width of 24**
//! (via Permuter24 with u128).
//!
//! This asymmetry is intentional: internodes just hold child pointers, so WIDTH=15
//! is sufficient. The benefit of WIDTH=24 comes from leaves holding more keys
//! and splitting less often.

use std::alloc as StdAlloc;
use std::alloc::Layout;
use std::mem as StdMem;

use parking_lot::lock_api::MutexGuard;
use parking_lot::{Mutex, RawMutex};
use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::slot::ValueSlot;
use crate::{AllocError, AllocResult};

/// Miri-compliant allocator for masstree nodes (WIDTH=24 leaves, WIDTH=15 internodes).
///
/// Uses `Box::into_raw()` for clean provenance and `Mutex` for concurrent tracking.
/// Implements [`NodeAllocatorGeneric`] for use with [`crate::MassTreeGeneric`].
///
/// # Naming
///
/// The "15" suffix refers to internode width. See module docs for details.
///
/// # Design
///
/// - Tracks all allocations in `Vec`s protected by `Mutex`
/// - Supports deferred retirement via `seize` guards
/// - Frees all tracked nodes on drop
///
/// # Thread Safety
///
/// All methods use interior mutability via `parking_lot::Mutex`, allowing
/// concurrent allocation from multiple threads with only `&self`.
pub struct SeizeAllocator15<S: ValueSlot> {
    /// Raw pointers to allocated [`LeafNode15`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode15<S>>>,

    /// Raw pointers to allocated internode nodes (WIDTH=15).
    internode_ptrs: Mutex<Vec<*mut InternodeNode>>,
}

// SAFETY: Raw pointers are owned by this allocator and protected by Mutex.
unsafe impl<S: ValueSlot + Send + Sync> Send for SeizeAllocator15<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for SeizeAllocator15<S> {}

impl<S: ValueSlot> std::fmt::Debug for SeizeAllocator15<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let leaf_count = self.leaf_ptrs.lock().len();
        let internode_count = self.internode_ptrs.lock().len();
        f.debug_struct("SeizeAllocator15")
            .field("leaf_count", &leaf_count)
            .field("internode_count", &internode_count)
            .finish()
    }
}

impl<S: ValueSlot> SeizeAllocator15<S> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leaf_ptrs: Mutex::new(Vec::new()),
            internode_ptrs: Mutex::new(Vec::new()),
        }
    }

    /// Return the number of tracked leaf nodes.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaf_ptrs.lock().len()
    }

    /// Return the number of tracked internodes.
    #[must_use]
    pub fn internode_count(&self) -> usize {
        self.internode_ptrs.lock().len()
    }
}

impl<S: ValueSlot> Default for SeizeAllocator15<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ValueSlot> Drop for SeizeAllocator15<S> {
    fn drop(&mut self) {
        // Free all tracked nodes on allocator drop
        for ptr in self.leaf_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in self.internode_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation
// =============================================================================

impl<S> NodeAllocatorGeneric<S, LeafNode15<S>> for SeizeAllocator15<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15<S>>) -> *mut LeafNode15<S> {
        let ptr: *mut LeafNode15<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode15<S>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<S>, guard: &LocalGuard<'_>) {
        // Step 1: Remove from tracking to prevent double-free.
        // The allocator's Drop iterates leaf_ptrs and frees everything,
        // so we must remove the pointer before deferring retirement.
        let found = {
            let mut ptrs = self.leaf_ptrs.lock();
            ptrs.iter().position(|&p| p == ptr).is_some_and(|pos| {
                ptrs.swap_remove(pos);
                true
            })
        };

        debug_assert!(
            found,
            "retire_leaf: pointer {ptr:p} not found in tracking list - possible double-retire"
        );

        // Step 2: Defer retirement via seize.
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    #[inline(always)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // Caller passes a valid pointer to an allocated InternodeNode.
        // Just cast and track - no Box round-trip needed.
        let ptr: *mut InternodeNode = node_ptr.cast();
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        self.internode_ptrs.lock().push(ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        let typed_ptr: *mut InternodeNode = ptr.cast();

        // Step 1: Remove from tracking to prevent double-free.
        let found = {
            let mut ptrs = self.internode_ptrs.lock();
            ptrs.iter()
                .position(|&p| p == typed_ptr)
                .is_some_and(|pos| {
                    ptrs.swap_remove(pos);
                    true
                })
        };

        debug_assert!(
            found,
            "retire_internode_erased: pointer {ptr:p} not found in tracking list - possible double-retire"
        );

        // Step 2: Defer retirement via seize.
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(typed_ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    #[inline(always)]
    fn teardown_tree(&self, _root_ptr: *mut u8) {
        // Free all tracked nodes using interior mutability
        let leaves: Vec<*mut LeafNode15<S>> = StdMem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode> = StdMem::take(&mut *self.internode_ptrs.lock());

        for ptr in leaves {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in internodes {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // Subtree traversal not implemented - nodes remain tracked until allocator drop.
        // This is safe but may delay memory reclamation for replaced layers.
        debug_assert!(
            !root_ptr.is_null(),
            "retire_subtree_root: received null pointer"
        );
        // Nodes will be freed when the allocator is dropped.
        // For eager reclamation, implement subtree traversal here.
    }

    /// Allocate a leaf directly without Box intermediate.
    ///
    /// Uses raw allocation + `init_at` to avoid stack-to-heap copy.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<S> {
        self.try_alloc_leaf(is_root, is_layer_root)
            .unwrap_or_else(|_| {
                let layout = Layout::new::<LeafNode15<S>>();
                StdAlloc::handle_alloc_error(layout)
            })
    }

    /// Allocate an internode directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    /// Allocate an internode as root directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place as root
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    /// Allocate an internode for split directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place with split-locked version
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr, parent_version, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    fn try_alloc_leaf(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> AllocResult<*mut LeafNode15<S>> {
        // Step 1: Allocate raw memory
        let layout: Layout = Layout::new::<LeafNode15<S>>();

        // SAFETY: Layout is valid (derived from type)
        let raw_ptr: *mut u8 = unsafe { StdAlloc::alloc(layout) };

        if raw_ptr.is_null() {
            return Err(AllocError::for_leaf::<LeafNode15<S>>());
        }

        let ptr: *mut LeafNode15<S> = raw_ptr.cast();

        // Step 2: Initialize in-place using init_at
        //
        // This is_layer_root flag is passed to init_at as is_root because
        // layer roots also have ROOT_BIT (they're roots of sublayers).
        //
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            LeafNode15::init_at(ptr, is_root || is_layer_root);
        }

        // Step 3: Reserve space in tracking vector before push
        //
        // CRITICAL: [`Vec::push`] can reallocate and abort if capacity
        // is exhausted. By calling `try_reserve` first, we make tracking fallible.
        {
            let mut ptrs: MutexGuard<'_, RawMutex, Vec<*mut LeafNode15<S>>> = self.leaf_ptrs.lock();

            if ptrs.try_reserve(1).is_err() {
                // Tracking failed, must deallocate the node
                // SAFETY: ptr was just allocated with this layout
                unsafe { StdAlloc::dealloc(raw_ptr, layout) };

                return Err(AllocError::for_tracking(StdMem::size_of::<
                    *mut LeafNode15<S>,
                >()));
            }

            // Now push cannot fail (we just reserved space)
            ptrs.push(ptr);
        }

        Ok(ptr)
    }
}

// =============================================================================
// True-Inline Allocator for LeafNode15TrueInline
// =============================================================================

use crate::inline::bits::InlineBits;
use crate::inline::leaf15_true::LeafNode15TrueInline;
use crate::slot::true_inline::TrueInlineSlot;

/// Allocator for true-inline leaf nodes (WIDTH=15).
///
/// Similar to [`SeizeAllocator15`] but manages [`LeafNode15TrueInline`] nodes
/// instead of [`LeafNode15`]. Since `TrueInlineSlot::NEEDS_RETIREMENT = false`,
/// no value retirement is needed.
pub struct SeizeAllocator15TrueInline<V: InlineBits> {
    /// Raw pointers to allocated [`LeafNode15TrueInline`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode15TrueInline<V>>>,

    /// Raw pointers to allocated internode nodes.
    internode_ptrs: Mutex<Vec<*mut InternodeNode>>,
}

// SAFETY: Raw pointers are owned by this allocator and protected by Mutex.
unsafe impl<V: InlineBits> Send for SeizeAllocator15TrueInline<V> {}
unsafe impl<V: InlineBits> Sync for SeizeAllocator15TrueInline<V> {}

impl<V: InlineBits> std::fmt::Debug for SeizeAllocator15TrueInline<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let leaf_count = self.leaf_ptrs.lock().len();
        let internode_count = self.internode_ptrs.lock().len();
        f.debug_struct("SeizeAllocator15TrueInline")
            .field("leaf_count", &leaf_count)
            .field("internode_count", &internode_count)
            .finish()
    }
}

impl<V: InlineBits> SeizeAllocator15TrueInline<V> {
    /// Create a new allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leaf_ptrs: Mutex::new(Vec::new()),
            internode_ptrs: Mutex::new(Vec::new()),
        }
    }

    /// Return the number of tracked leaf nodes.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaf_ptrs.lock().len()
    }

    /// Return the number of tracked internodes.
    #[must_use]
    pub fn internode_count(&self) -> usize {
        self.internode_ptrs.lock().len()
    }
}

impl<V: InlineBits> Default for SeizeAllocator15TrueInline<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: InlineBits> Drop for SeizeAllocator15TrueInline<V> {
    fn drop(&mut self) {
        for ptr in self.leaf_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in self.internode_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl<V: InlineBits + Send + Sync + 'static>
    NodeAllocatorGeneric<TrueInlineSlot<V>, LeafNode15TrueInline<V>>
    for SeizeAllocator15TrueInline<V>
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15TrueInline<V>>) -> *mut LeafNode15TrueInline<V> {
        let ptr: *mut LeafNode15TrueInline<V> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode15TrueInline<V>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15TrueInline<V>, guard: &LocalGuard<'_>) {
        let found = {
            let mut ptrs = self.leaf_ptrs.lock();
            ptrs.iter().position(|&p| p == ptr).is_some_and(|pos| {
                ptrs.swap_remove(pos);
                true
            })
        };

        debug_assert!(
            found,
            "retire_leaf: pointer {ptr:p} not found in tracking list"
        );

        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    #[inline(always)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        let ptr: *mut InternodeNode = node_ptr.cast();
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        self.internode_ptrs.lock().push(ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        let typed_ptr: *mut InternodeNode = ptr.cast();

        let found = {
            let mut ptrs = self.internode_ptrs.lock();
            ptrs.iter()
                .position(|&p| p == typed_ptr)
                .is_some_and(|pos| {
                    ptrs.swap_remove(pos);
                    true
                })
        };

        debug_assert!(
            found,
            "retire_internode_erased: pointer {ptr:p} not found in tracking list"
        );

        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        unsafe {
            guard.defer_retire(typed_ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    #[inline(always)]
    fn teardown_tree(&self, _root_ptr: *mut u8) {
        let leaves: Vec<*mut LeafNode15TrueInline<V>> = StdMem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode> = StdMem::take(&mut *self.internode_ptrs.lock());

        for ptr in leaves {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in internodes {
            // SAFETY: ptr came from Box::into_raw or alloc()
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        debug_assert!(
            !root_ptr.is_null(),
            "retire_subtree_root: received null pointer"
        );
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
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        #[expect(clippy::cast_ptr_alignment, reason = "Layout guarantees alignment")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr, height);
        }

        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        #[expect(clippy::cast_ptr_alignment, reason = "Layout guarantees alignment")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr, height);
        }

        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode>();
        #[expect(clippy::cast_ptr_alignment, reason = "Layout guarantees alignment")]
        let ptr: *mut InternodeNode = unsafe { alloc(layout).cast::<InternodeNode>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr, parent_version, height);
        }

        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    fn try_alloc_leaf(
        &self,
        is_root: bool,
        is_layer_root: bool,
    ) -> AllocResult<*mut LeafNode15TrueInline<V>> {
        // Step 1: Allocate raw memory
        let layout = Layout::new::<LeafNode15TrueInline<V>>();
        let raw_ptr: *mut u8 = unsafe { StdAlloc::alloc(layout) };

        if raw_ptr.is_null() {
            return Err(AllocError::for_leaf::<LeafNode15TrueInline<V>>());
        }

        let ptr: *mut LeafNode15TrueInline<V> = raw_ptr.cast();

        // Step 2: Initialize in-place
        // SAFETY: ptr is valid, properly aligned, exclusive access
        unsafe {
            LeafNode15TrueInline::init_at(ptr, is_root || is_layer_root);
        }

        // Step 3: Fallible tracking
        {
            let mut ptrs: MutexGuard<'_, RawMutex, Vec<*mut LeafNode15TrueInline<V>>> =
                self.leaf_ptrs.lock();

            if ptrs.try_reserve(1).is_err() {
                unsafe { StdAlloc::dealloc(raw_ptr, layout) };

                return Err(AllocError::for_tracking(StdMem::size_of::<
                    *mut LeafNode15TrueInline<V>,
                >()));
            }

            ptrs.push(ptr);
        }

        Ok(ptr)
    }
}

#[cfg(test)]
mod unit_tests;
