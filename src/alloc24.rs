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

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
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
    /// Raw pointers to allocated [`LeafNode24`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode24<S>>>,

    /// Raw pointers to allocated internode nodes (WIDTH=15).
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S>>>,
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

impl<S: ValueSlot> Default for SeizeAllocator24<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ValueSlot> Drop for SeizeAllocator24<S> {
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

impl<S> NodeAllocatorGeneric<S, LeafNode24<S>> for SeizeAllocator24<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        let ptr: *mut LeafNode24<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode24<S>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // Step 1: Remove from tracking to prevent double-free.
        // The allocator's Drop iterates leaf_ptrs and frees everything,
        // so we must remove the pointer before deferring retirement.
        {
            let mut ptrs = self.leaf_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
            }
        }

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
        // SAFETY: Caller passes a valid Box<InternodeNode<S>> as *mut u8.
        // The pointer was originally created from Box::into_raw on an InternodeNode,
        // so alignment is guaranteed.
        let node: Box<InternodeNode<S>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S>>()) };
        let ptr: *mut InternodeNode<S> = Box::into_raw(node);
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
        {
            let mut ptrs = self.internode_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == typed_ptr) {
                ptrs.swap_remove(pos);
            }
        }

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
        let leaves: Vec<*mut LeafNode24<S>> = std::mem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode<S>> =
            std::mem::take(&mut *self.internode_ptrs.lock());

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
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // TODO: Implement subtree traversal for WIDTH=24
        // For now, subtree retirement is not supported for WIDTH=24
        // The allocator's Drop will clean up all nodes
    }

    /// Allocate a leaf directly without Box intermediate.
    ///
    /// Uses raw allocation + `init_at` to avoid stack-to-heap copy.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode24<S> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::new::<LeafNode24<S>>();
        // SAFETY: Layout is valid (non-zero size)
        #[expect(clippy::cast_ptr_alignment, reason = "Layout is valid (non-zero size)")]
        let ptr: *mut LeafNode24<S> = unsafe { alloc(layout).cast::<LeafNode24<S>>() };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Initialize in-place
        // SAFETY: ptr is valid, aligned, and we have exclusive access
        unsafe {
            LeafNode24::init_at(ptr, is_root || is_layer_root);
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
        self.internode_ptrs.lock().push(ptr);
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
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_allocator24_alloc_leaf() {
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

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
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf);

        alloc.track_leaf(ptr);
        assert_eq!(alloc.leaf_count(), 1);
    }

    #[test]
    fn test_seize_allocator24_drop_frees_nodes() {
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let _ = alloc.alloc_leaf(leaf);
        assert_eq!(alloc.leaf_count(), 1);

        // Drop the allocator - nodes should be freed
        drop(alloc);
        // If this doesn't leak memory, test passes (checked by miri)
    }
}
