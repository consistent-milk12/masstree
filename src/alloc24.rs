//! Node allocation for [`LeafNode24`] (WIDTH=24).
//!
//! This module provides allocators that work with the 24-slot leaf nodes,
//! mirroring the design of `alloc.rs` but for the larger WIDTH.

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::internode::InternodeNode;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;

/// Width constant for clarity.
const WIDTH_24: usize = 24;

/// Trait for allocating and deallocating 24-slot tree nodes.
///
/// Similar to [`NodeAllocator`](crate::alloc::NodeAllocator) but for `LeafNode24`.
///
/// # Type Parameters
///
/// * `S` - The slot type implementing [`ValueSlot`]
pub trait NodeAllocator24<S: ValueSlot> {
    /// Allocate a [`LeafNode24`] and return a stable raw pointer.
    fn alloc_leaf24(&mut self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S>;

    /// Allocate an internode and return a stable raw pointer.
    ///
    /// Note: Internodes are WIDTH=24 to match the tree structure.
    fn alloc_internode24(
        &mut self,
        node: Box<InternodeNode<S, WIDTH_24>>,
    ) -> *mut InternodeNode<S, WIDTH_24>;

    /// Track a leaf24 pointer for cleanup (concurrent-safe via `&self`).
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_leaf24(&self, ptr: *mut LeafNode24<S>) {
        // Default: no-op
    }

    /// Track an internode pointer for cleanup (concurrent-safe via `&self`).
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_internode24(&self, ptr: *mut InternodeNode<S, WIDTH_24>) {
        // Default: no-op
    }

    /// Retire a leaf24 node that has been unlinked from the tree.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree by any new traversal.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_leaf24(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // Default: no-op
    }

    /// Retire an internode that has been unlinked from the tree.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree by any new traversal.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_internode24(
        &self,
        ptr: *mut InternodeNode<S, WIDTH_24>,
        guard: &LocalGuard<'_>,
    ) {
        // Default: no-op
    }

    /// Teardown reachable nodes at tree drop.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn teardown_tree24(&mut self, root_ptr: *mut u8) {
        // Default: no-op
    }
}

/// Miri-compliant allocator for 24-slot leaf nodes.
///
/// Uses `Box::into_raw()` for clean provenance and `Mutex` for concurrent tracking.
pub struct SeizeAllocator24<S: ValueSlot> {
    /// Raw pointers to allocated [`LeafNode24`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode24<S>>>,

    /// Raw pointers to allocated internode nodes.
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S, WIDTH_24>>>,
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

    /// Return the number of tracked leaf24 nodes.
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

impl<S: ValueSlot> NodeAllocator24<S> for SeizeAllocator24<S> {
    fn alloc_leaf24(&mut self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        let ptr: *mut LeafNode24<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    fn alloc_internode24(
        &mut self,
        node: Box<InternodeNode<S, WIDTH_24>>,
    ) -> *mut InternodeNode<S, WIDTH_24> {
        let ptr: *mut InternodeNode<S, WIDTH_24> = Box::into_raw(node);
        self.internode_ptrs.lock().push(ptr);
        ptr
    }

    fn track_leaf24(&self, ptr: *mut LeafNode24<S>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    fn track_internode24(&self, ptr: *mut InternodeNode<S, WIDTH_24>) {
        self.internode_ptrs.lock().push(ptr);
    }

    unsafe fn retire_leaf24(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller ensures ptr is valid and unreachable
        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    unsafe fn retire_internode24(
        &self,
        ptr: *mut InternodeNode<S, WIDTH_24>,
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: Caller ensures ptr is valid and unreachable
        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    fn teardown_tree24(&mut self, _root_ptr: *mut u8) {
        // Free all tracked nodes
        let leaves: Vec<*mut LeafNode24<S>> = std::mem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode<S, WIDTH_24>> =
            std::mem::take(&mut *self.internode_ptrs.lock());

        for ptr in leaves {
            // SAFETY: ptr came from Box::into_raw in alloc_leaf24
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in internodes {
            // SAFETY: ptr came from Box::into_raw in alloc_internode24
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl<S: ValueSlot> Drop for SeizeAllocator24<S> {
    fn drop(&mut self) {
        // Free all tracked nodes on allocator drop
        for ptr in self.leaf_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw in alloc_leaf24
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in self.internode_ptrs.lock().drain(..) {
            // SAFETY: ptr came from Box::into_raw in alloc_internode24
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation
// =============================================================================

impl<S> crate::alloc_trait::NodeAllocatorGeneric<S, LeafNode24<S>> for SeizeAllocator24<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline]
    fn alloc_leaf(&mut self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        NodeAllocator24::alloc_leaf24(self, node)
    }

    #[inline]
    fn track_leaf(&self, ptr: *mut LeafNode24<S>) {
        NodeAllocator24::track_leaf24(self, ptr);
    }

    #[inline]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid and unreachable
        unsafe { NodeAllocator24::retire_leaf24(self, ptr, guard) }
    }

    #[inline]
    fn alloc_internode_erased(&mut self, node_ptr: *mut u8) -> *mut u8 {
        // SAFETY: Caller passes a valid Box<InternodeNode<S, 24>> as *mut u8
        let node: Box<InternodeNode<S, WIDTH_24>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S, WIDTH_24>>()) };
        NodeAllocator24::alloc_internode24(self, node).cast()
    }

    #[inline]
    fn track_internode_erased(&self, ptr: *mut u8) {
        NodeAllocator24::track_internode24(self, ptr.cast());
    }

    #[inline]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid InternodeNode<S, 24>
        unsafe { NodeAllocator24::retire_internode24(self, ptr.cast(), guard) }
    }

    #[inline]
    fn teardown_tree(&mut self, root_ptr: *mut u8) {
        NodeAllocator24::teardown_tree24(self, root_ptr);
    }

    #[inline]
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // TODO: Implement subtree traversal for WIDTH=24
        // For now, subtree retirement is not supported for WIDTH=24
        // The allocator's Drop will clean up all nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    #[test]
    fn test_seize_allocator24_new() {
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_allocator24_alloc_leaf() {
        let mut alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let ptr = alloc.alloc_leaf24(leaf);
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

        alloc.track_leaf24(ptr);
        assert_eq!(alloc.leaf_count(), 1);
    }

    #[test]
    fn test_seize_allocator24_drop_frees_nodes() {
        let mut alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let _ = alloc.alloc_leaf24(leaf);
        assert_eq!(alloc.leaf_count(), 1);

        // Drop the allocator - nodes should be freed
        drop(alloc);
        // If this doesn't leak memory, test passes (checked by miri)
    }
}
