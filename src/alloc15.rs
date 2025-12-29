//! Node allocation for [`LeafNode15`] (WIDTH=15).
//!
//! This module provides allocators that work with 15-slot leaf nodes,
//! matching the C++ reference implementation's default configuration.

#![allow(missing_docs)] // Internal module matching alloc24.rs API

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::internode::InternodeNode;
use crate::leaf15::LeafNode15;
use crate::slot::ValueSlot;

/// Width constant for internodes (limited by 4-bit permutation slots).
const INTERNODE_WIDTH: usize = 15;

/// Trait for allocating and deallocating 15-slot tree nodes.
pub trait NodeAllocator15<S: ValueSlot> {
    /// Allocate a [`LeafNode15`] and return a stable raw pointer.
    fn alloc_leaf15(&self, node: Box<LeafNode15<S>>) -> *mut LeafNode15<S>;

    /// Allocate an internode and return a stable raw pointer.
    fn alloc_internode15(
        &self,
        node: Box<InternodeNode<S, INTERNODE_WIDTH>>,
    ) -> *mut InternodeNode<S, INTERNODE_WIDTH>;

    /// Track a leaf15 pointer for cleanup.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_leaf15(&self, ptr: *mut LeafNode15<S>) {}

    /// Track an internode pointer for cleanup.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_internode15(&self, ptr: *mut InternodeNode<S, INTERNODE_WIDTH>) {}

    /// Retire a leaf15 node.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_leaf15(&self, ptr: *mut LeafNode15<S>, guard: &LocalGuard<'_>) {}

    /// Retire an internode.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_internode15(
        &self,
        ptr: *mut InternodeNode<S, INTERNODE_WIDTH>,
        guard: &LocalGuard<'_>,
    ) {
    }

    /// Teardown reachable nodes at tree drop.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn teardown_tree15(&self, root_ptr: *mut u8) {}
}

/// Allocator for 15-slot leaf nodes.
pub struct SeizeAllocator15<S: ValueSlot> {
    /// Raw pointers to allocated [`LeafNode15`] nodes.
    leaf_ptrs: Mutex<Vec<*mut LeafNode15<S>>>,

    /// Raw pointers to allocated internode nodes.
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S, INTERNODE_WIDTH>>>,
}

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

    /// Return the number of tracked leaf15 nodes.
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

impl<S: ValueSlot> NodeAllocator15<S> for SeizeAllocator15<S> {
    fn alloc_leaf15(&self, node: Box<LeafNode15<S>>) -> *mut LeafNode15<S> {
        let ptr: *mut LeafNode15<S> = Box::into_raw(node);
        self.leaf_ptrs.lock().push(ptr);
        ptr
    }

    fn alloc_internode15(
        &self,
        node: Box<InternodeNode<S, INTERNODE_WIDTH>>,
    ) -> *mut InternodeNode<S, INTERNODE_WIDTH> {
        let ptr: *mut InternodeNode<S, INTERNODE_WIDTH> = Box::into_raw(node);
        self.internode_ptrs.lock().push(ptr);
        ptr
    }

    fn track_leaf15(&self, ptr: *mut LeafNode15<S>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    fn track_internode15(&self, ptr: *mut InternodeNode<S, INTERNODE_WIDTH>) {
        self.internode_ptrs.lock().push(ptr);
    }

    unsafe fn retire_leaf15(&self, ptr: *mut LeafNode15<S>, guard: &LocalGuard<'_>) {
        {
            let mut ptrs = self.leaf_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
            }
        }

        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    unsafe fn retire_internode15(
        &self,
        ptr: *mut InternodeNode<S, INTERNODE_WIDTH>,
        guard: &LocalGuard<'_>,
    ) {
        {
            let mut ptrs = self.internode_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
            }
        }

        unsafe {
            guard.defer_retire(ptr, |p, _| {
                drop(Box::from_raw(p));
            });
        }
    }

    fn teardown_tree15(&self, _root_ptr: *mut u8) {
        let leaves: Vec<*mut LeafNode15<S>> = std::mem::take(&mut *self.leaf_ptrs.lock());
        let internodes: Vec<*mut InternodeNode<S, INTERNODE_WIDTH>> =
            std::mem::take(&mut *self.internode_ptrs.lock());

        for ptr in leaves {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in internodes {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl<S: ValueSlot> Drop for SeizeAllocator15<S> {
    fn drop(&mut self) {
        for ptr in self.leaf_ptrs.lock().drain(..) {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in self.internode_ptrs.lock().drain(..) {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation
// =============================================================================

impl<S> crate::alloc_trait::NodeAllocatorGeneric<S, LeafNode15<S>> for SeizeAllocator15<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode15<S>>) -> *mut LeafNode15<S> {
        NodeAllocator15::alloc_leaf15(self, node)
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode15<S>) {
        NodeAllocator15::track_leaf15(self, ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<S>, guard: &LocalGuard<'_>) {
        unsafe { NodeAllocator15::retire_leaf15(self, ptr, guard) }
    }

    #[inline(always)]
    #[expect(clippy::cast_ptr_alignment)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        let node: Box<InternodeNode<S, INTERNODE_WIDTH>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S, INTERNODE_WIDTH>>()) };
        NodeAllocator15::alloc_internode15(self, node).cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        NodeAllocator15::track_internode15(self, ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        unsafe { NodeAllocator15::retire_internode15(self, ptr.cast(), guard) }
    }

    #[inline(always)]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        NodeAllocator15::teardown_tree15(self, root_ptr);
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // Not implemented for WIDTH=15 yet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    #[test]
    fn test_seize_allocator15_new() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_allocator15_alloc_leaf() {
        let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

        let ptr = alloc.alloc_leaf15(leaf);
        assert!(!ptr.is_null());
        assert_eq!(alloc.leaf_count(), 1);

        unsafe {
            assert!((*ptr).is_empty());
        }
    }
}
