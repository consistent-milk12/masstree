//! Node allocation for [`LeafNode24`] (WIDTH=24).
//!
//! This module provides allocators that work with the 24-slot leaf nodes,
//! mirroring the design of `alloc.rs` but for the larger WIDTH.
//!
//! # Note on Internode WIDTH
//!
//! While leaves use WIDTH=24 (via Permuter24 with u128), internodes are still
//! limited to WIDTH=15 because they use the original Permuter (u64 with 4-bit slots).
//! This is fine since internodes just hold child pointers; the benefit of WIDTH=24
//! comes from leaves holding more keys and splitting less often.

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::internode::InternodeNode;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;

/// Width constant for internodes (limited by 4-bit permutation slots).
const INTERNODE_WIDTH: usize = 15;

/// Trait for allocating and deallocating 24-slot tree nodes.
///
/// Similar to [`NodeAllocator`](crate::alloc::NodeAllocator) but for `LeafNode24`.
///
/// # Type Parameters
///
/// * `S` - The slot type implementing [`ValueSlot`]
///
/// # Note
///
/// Uses interior mutability (`parking_lot::Mutex`) so all methods take `&self`.
pub trait NodeAllocator24<S: ValueSlot> {
    /// Allocate a [`LeafNode24`] and return a stable raw pointer.
    fn alloc_leaf24(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S>;

    /// Allocate an internode and return a stable raw pointer.
    ///
    /// Note: Internodes are WIDTH=15 (max for 4-bit permutation slots).
    fn alloc_internode24(
        &self,
        node: Box<InternodeNode<S, INTERNODE_WIDTH>>,
    ) -> *mut InternodeNode<S, INTERNODE_WIDTH>;

    /// Track a leaf24 pointer for cleanup (concurrent-safe via `&self`).
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_leaf24(&self, ptr: *mut LeafNode24<S>) {
        // Default: no-op
    }

    /// Track an internode pointer for cleanup (concurrent-safe via `&self`).
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_internode24(&self, ptr: *mut InternodeNode<S, INTERNODE_WIDTH>) {
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
        ptr: *mut InternodeNode<S, INTERNODE_WIDTH>,
        guard: &LocalGuard<'_>,
    ) {
        // Default: no-op
    }

    /// Teardown reachable nodes at tree drop.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn teardown_tree24(&self, root_ptr: *mut u8) {
        // Default: no-op
    }
}

/// Miri-compliant allocator for 24-slot leaf nodes.
///
/// Uses `Box::into_raw()` for clean provenance and `Mutex` for concurrent tracking.
pub struct SeizeAllocator24<S: ValueSlot> {
    /// Raw pointers to allocated [`LeafNode24`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode24<S>>>,

    /// Raw pointers to allocated internode nodes (WIDTH=15).
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S, INTERNODE_WIDTH>>>,
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
    fn alloc_leaf24(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        let ptr: *mut LeafNode24<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    fn alloc_internode24(
        &self,
        node: Box<InternodeNode<S, INTERNODE_WIDTH>>,
    ) -> *mut InternodeNode<S, INTERNODE_WIDTH> {
        let ptr: *mut InternodeNode<S, INTERNODE_WIDTH> = Box::into_raw(node);
        self.internode_ptrs.lock().push(ptr);
        ptr
    }

    fn track_leaf24(&self, ptr: *mut LeafNode24<S>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    fn track_internode24(&self, ptr: *mut InternodeNode<S, INTERNODE_WIDTH>) {
        self.internode_ptrs.lock().push(ptr);
    }

    unsafe fn retire_leaf24(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
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

    unsafe fn retire_internode24(
        &self,
        ptr: *mut InternodeNode<S, INTERNODE_WIDTH>,
        guard: &LocalGuard<'_>,
    ) {
        // Step 1: Remove from tracking to prevent double-free.
        {
            let mut ptrs = self.internode_ptrs.lock();
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

    fn teardown_tree24(&self, _root_ptr: *mut u8) {
        // Free all tracked nodes using interior mutability
        let leaves: Vec<*mut LeafNode24<S>> = std::mem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode<S, INTERNODE_WIDTH>> =
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
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        NodeAllocator24::alloc_leaf24(self, node)
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode24<S>) {
        NodeAllocator24::track_leaf24(self, ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid and unreachable
        unsafe { NodeAllocator24::retire_leaf24(self, ptr, guard) }
    }

    #[inline(always)]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Caller guarantees node_ptr is properly aligned for InternodeNode"
    )]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // SAFETY: Caller passes a valid Box<InternodeNode<S, INTERNODE_WIDTH>> as *mut u8.
        // The pointer was originally created from Box::into_raw on an InternodeNode,
        // so alignment is guaranteed.
        let node: Box<InternodeNode<S, INTERNODE_WIDTH>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S, INTERNODE_WIDTH>>()) };
        NodeAllocator24::alloc_internode24(self, node).cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        NodeAllocator24::track_internode24(self, ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid InternodeNode<S, 15>
        unsafe { NodeAllocator24::retire_internode24(self, ptr.cast(), guard) }
    }

    #[inline(always)]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        NodeAllocator24::teardown_tree24(self, root_ptr);
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // TODO: Implement subtree traversal for WIDTH=24
        // For now, subtree retirement is not supported for WIDTH=24
        // The allocator's Drop will clean up all nodes
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
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let _ = alloc.alloc_leaf24(leaf);
        assert_eq!(alloc.leaf_count(), 1);

        // Drop the allocator - nodes should be freed
        drop(alloc);
        // If this doesn't leak memory, test passes (checked by miri)
    }
}
