//! Generic allocator trait for tree nodes.
//!
//! This module defines [`TreeAllocator`] that abstracts over allocators
//! for different leaf policies.

use seize::LocalGuard;

use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;

/// Trait for allocating and deallocating tree nodes generically.
///
/// Enables tree operations to work with any leaf policy implementing [`LeafPolicy`].
///
/// # Safety
///
/// Implementations must uphold the tree's ownership and layout invariants:
///
/// - Every allocation method must return a non-null, correctly aligned node of
///   the requested kind, fully initialized for the requested root/split state.
/// - Retired pointers must be valid nodes previously allocated by this
///   allocator and unreachable from the tree.
/// - `teardown_tree` must only be called with exclusive access after all
///   readers have left and the collector has been drained.
/// - Allocation failures must abort; returning null or a dummy pointer is
///   undefined behavior in the tree.
/// - Allocation methods must never panic/unwind. A panic after `mark_split`
///   (or while a split sibling is linked and split-locked) leaves a node that
///   readers following `next` can spin on forever and that has no parent.
///   On OOM or any other failure, abort the process instead of unwinding.
///
/// The trait is `unsafe` because a safe implementation returning an invalid
/// pointer would let safe tree code dereference it; the no-panic requirement
/// is part of the same contract.
///
/// # Type Parameters
///
/// - `P`: The leaf policy implementing [`LeafPolicy`]
///
/// # Internode Handling
///
/// Internode pointers use `*mut u8` for type erasure. Implementations must ensure
/// internodes have the same WIDTH as leaves (invariant enforced by construction).
pub unsafe trait TreeAllocator<P: LeafPolicy>: Send + Sync {
    // ========================================================================
    // Leaf Allocation
    // ========================================================================

    /// Allocate a leaf node directly from the pool and initialize it in-place.
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<P>;

    /// Retire a leaf node for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid leaf allocated by this allocator
    /// - `ptr` must be unreachable from the tree (unlinked)
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<P>, guard: &LocalGuard<'_>);

    // ========================================================================
    // Internode Allocation
    // ========================================================================

    /// Allocate an internode as a root node directly from the pool.
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8;

    /// Allocate an internode for a split operation directly from the pool.
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &NodeVersion,
        height: u32,
    ) -> *mut u8;

    /// Retire an internode for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid internode
    /// - `ptr` must be unreachable from the tree
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>);

    // ========================================================================
    // Tree Lifecycle
    // ========================================================================

    /// Teardown all reachable nodes at tree drop.
    ///
    /// # Safety
    ///
    /// The caller must have exclusive access to the tree (no live guards or
    /// readers), all deferred retirements must have been flushed, and
    /// `root_ptr` must be null or a valid root allocated by this allocator.
    unsafe fn teardown_tree(&self, root_ptr: *mut u8);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
