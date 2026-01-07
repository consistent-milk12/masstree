//! Node allocation for [`LeafNode24`] (WIDTH=24).
//!
//! This module provides [`SeizeAllocator24`], a Miri-compliant allocator for
//! 24-slot leaf nodes using `seize` for memory reclamation.
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
use std::mem as StdMem;

use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::error::{AllocError, AllocResult};
use crate::internode::InternodeNode;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;

/// Miri-compliant allocator for 24-slot leaf nodes.
///
/// Uses `Box::into_raw()` for clean provenance and `Mutex` for concurrent tracking.
/// Implements [`NodeAllocatorGeneric`] for use with [`crate::MassTreeGeneric`].
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
pub struct SeizeAllocator24<S: ValueSlot> {
    /// Raw pointers to allocated [`LeafNode24`] nodes, stored as usize for O(1) hashing.
    leaf_ptrs: Mutex<FxHashSet<usize>>,

    /// Raw pointers to allocated internode nodes (WIDTH=15), stored as usize for O(1) hashing.
    internode_ptrs: Mutex<FxHashSet<usize>>,

    /// Marker for the slot type parameter.
    _marker: PhantomData<S>,
}

// SAFETY: Raw pointers are owned by this allocator and protected by Mutex.
unsafe impl<S: ValueSlot + Send + Sync> Send for SeizeAllocator24<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for SeizeAllocator24<S> {}

impl<S: ValueSlot> std::fmt::Debug for SeizeAllocator24<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let leaf_count = self.leaf_ptrs.lock().len();
        let internode_count = self.internode_ptrs.lock().len();
        f.debug_struct("SeizeAllocator24")
            .field("leaf_count", &leaf_count)
            .field("internode_count", &internode_count)
            .finish()
    }
}

impl<S: ValueSlot> SeizeAllocator24<S> {
    /// Create a new allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            leaf_ptrs: Mutex::new(FxHashSet::default()),
            internode_ptrs: Mutex::new(FxHashSet::default()),
            _marker: PhantomData,
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

impl<S: ValueSlot> Default for SeizeAllocator24<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ValueSlot> Drop for SeizeAllocator24<S> {
    fn drop(&mut self) {
        // Free all tracked nodes on allocator drop
        for ptr_addr in self.leaf_ptrs.lock().drain() {
            // SAFETY: ptr_addr came from a valid *mut LeafNode24<S> cast to usize
            unsafe {
                drop(Box::from_raw(ptr_addr as *mut LeafNode24<S>));
            }
        }

        for ptr_addr in self.internode_ptrs.lock().drain() {
            // SAFETY: ptr_addr came from a valid *mut InternodeNode<S> cast to usize
            unsafe {
                drop(Box::from_raw(ptr_addr as *mut InternodeNode<S>));
            }
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
        let ptr: *mut LeafNode24<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().insert(ptr as usize);
        ptr
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode24<S>) {
        self.leaf_ptrs.lock().insert(ptr as usize);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // Step 1: Remove from tracking to prevent double-free (O(1) with FxHashSet).
        let found = self.leaf_ptrs.lock().remove(&(ptr as usize));

        debug_assert!(
            found,
            "retire_leaf: pointer {ptr:p} not found in tracking set - possible double-retire"
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
        // Caller passes a valid pointer to an allocated InternodeNode<S>.
        // Just track - no Box round-trip needed.
        self.internode_ptrs.lock().insert(node_ptr as usize);
        node_ptr
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        self.internode_ptrs.lock().insert(ptr as usize);
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Step 1: Remove from tracking to prevent double-free (O(1) with FxHashSet).
        let found = self.internode_ptrs.lock().remove(&(ptr as usize));

        debug_assert!(
            found,
            "retire_internode_erased: pointer {ptr:p} not found in tracking set - possible double-retire"
        );

        // Step 2: Defer retirement via seize.
        // SAFETY: Caller ensures ptr is valid and unreachable from tree.
        // The pointer was originally allocated as InternodeNode<S>, so alignment is correct.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "ptr originally came from InternodeNode<S> allocation, alignment guaranteed"
        )]
        unsafe {
            guard.defer_retire(ptr.cast::<InternodeNode<S>>(), |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    #[inline(always)]
    fn teardown_tree(&self, _root_ptr: *mut u8) {
        // Free all tracked nodes using interior mutability
        let leaves: FxHashSet<usize> = StdMem::take(&mut *self.leaf_ptrs.lock());
        let internodes: FxHashSet<usize> = StdMem::take(&mut *self.internode_ptrs.lock());

        for ptr_addr in leaves {
            // SAFETY: ptr_addr came from a valid *mut LeafNode24<S> cast to usize
            unsafe {
                drop(Box::from_raw(ptr_addr as *mut LeafNode24<S>));
            }
        }

        for ptr_addr in internodes {
            // SAFETY: ptr_addr came from a valid *mut InternodeNode<S> cast to usize
            unsafe {
                drop(Box::from_raw(ptr_addr as *mut InternodeNode<S>));
            }
        }
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // TODO: Implement subtree traversal for WIDTH=24
        // For now, subtree retirement is not supported for WIDTH=24
        // The allocator's Drop will clean up all nodes
    }

    /// Allocate a leaf directly without Box intermediate.
    ///
    /// Uses raw allocation + `init_at` to avoid stack-to-heap copy.
    /// Delegates to `try_alloc_leaf` and aborts on allocation failure.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode24<S> {
        self.try_alloc_leaf(is_root, is_layer_root)
            .unwrap_or_else(|_| {
                let layout: Layout = Layout::new::<LeafNode24<S>>();
                StdAlloc::handle_alloc_error(layout)
            })
    }

    /// Try to allocate a leaf node, returning an error on failure.
    ///
    /// This is the fallible version of `alloc_leaf_direct`. Use this in
    /// production code paths that need to handle OOM gracefully.
    ///
    /// # Steps
    ///
    /// 1. Allocate raw memory (fallible)
    /// 2. Initialize in-place using `init_at`
    /// 3. Reserve space in tracking vector (fallible)
    /// 4. Push pointer to tracking vector
    ///
    /// # Errors
    ///
    /// Returns `Err(AllocError)` if:
    /// - Raw memory allocation fails
    /// - Tracking vector reservation fails (node is deallocated on this error)
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

        // Step 3: Track the pointer (FxHashSet insertion is fallible via try_reserve)
        {
            let mut ptrs = self.leaf_ptrs.lock();

            if ptrs.try_reserve(1).is_err() {
                // Tracking failed - must deallocate the node
                // SAFETY: ptr was just allocated with this layout
                unsafe { StdAlloc::dealloc(raw_ptr, layout) };

                return Err(AllocError::for_tracking(StdMem::size_of::<usize>()));
            }

            // Now insert cannot fail (we just reserved space)
            ptrs.insert(ptr as usize);
        }

        Ok(ptr)
    }

    /// Allocate an internode directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid (non-zero size)")]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().insert(ptr as usize);
        ptr.cast()
    }

    /// Allocate an internode as root directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid")]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place as root
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().insert(ptr as usize);
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

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout is valid
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid")]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place with split-locked version
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr, parent_version, height);
        }

        // Track for cleanup
        self.internode_ptrs.lock().insert(ptr as usize);
        ptr.cast()
    }
}

#[cfg(test)]
mod unit_tests;
