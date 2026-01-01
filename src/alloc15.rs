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

use std::mem as StdMem;

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::slot::ValueSlot;

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
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S>>>,
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
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
                true
            } else {
                false
            }
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
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Caller guarantees node_ptr is properly aligned for InternodeNode"
    )]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // Caller passes a valid pointer to an allocated InternodeNode<S>.
        // Just cast and track - no Box round-trip needed.
        let ptr: *mut InternodeNode<S> = node_ptr.cast();
        self.internode_ptrs.lock().push(ptr);
        ptr.cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        self.internode_ptrs.lock().push(ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        let typed_ptr: *mut InternodeNode<S> = ptr.cast();

        // Step 1: Remove from tracking to prevent double-free.
        let found = {
            let mut ptrs = self.internode_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == typed_ptr) {
                ptrs.swap_remove(pos);
                true
            } else {
                false
            }
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
        let internodes: Vec<*mut InternodeNode<S>> =
            StdMem::take(&mut *self.internode_ptrs.lock());

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
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<LeafNode15<S>>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<LeafNode15> guarantees alignment"
        )]
        let ptr: *mut LeafNode15<S> = unsafe { alloc(layout).cast::<LeafNode15<S>>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place
        // SAFETY: ptr is valid, properly aligned, and we have exclusive access
        unsafe {
            LeafNode15::init_at(ptr, is_root || is_layer_root);
        }

        // Track for cleanup
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    /// Allocate an internode directly without Box intermediate.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
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

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
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

        let layout = Layout::new::<InternodeNode<S>>();
        // SAFETY: Layout::new::<T>() guarantees proper alignment for T.
        // The global allocator returns memory aligned to layout.align().
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "Layout::new::<InternodeNode> guarantees alignment"
        )]
        let ptr: *mut InternodeNode<S> = unsafe { alloc(layout).cast::<InternodeNode<S>>() };
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    #[test]
    fn test_seize_allocator24_new() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_allocator24_alloc_leaf() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

        let ptr = alloc.alloc_leaf(leaf);
        assert!(!ptr.is_null());
        assert_eq!(alloc.leaf_count(), 1);

        // Verify the pointer is valid
        unsafe {
            assert!((*ptr).is_empty());
        }
    }

    #[test]
    fn test_seize_allocator24_track_leaf() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();
        let ptr: *mut LeafNode15<LeafValue<u64>> = Box::into_raw(leaf);

        alloc.track_leaf(ptr);
        assert_eq!(alloc.leaf_count(), 1);
    }

    #[test]
    fn test_seize_allocator24_drop_frees_nodes() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

        let _ = alloc.alloc_leaf(leaf);
        assert_eq!(alloc.leaf_count(), 1);

        // Drop the allocator - nodes should be freed
        drop(alloc);
        // If this doesn't leak memory, test passes (checked by miri)
    }
}
