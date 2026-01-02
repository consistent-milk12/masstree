//! Root and layer-root creation helpers.
//!
//! Provides atomic root installation for both main tree roots and layer roots.
//! Main tree roots use atomic store (we hold the lock on current root).
//! Layer roots use parent pointer updates only (no modification to `root_ptr`).

use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering, fence as atomic_fence};

use crate::NodeAllocatorGeneric;
use crate::leaf_trait::{LayerCapableLeaf, TreeInternode};
use crate::slot::ValueSlot;
use crate::tree::InsertError;

/// Unit struct namespace for root creation operations.
///
/// All methods are stateless. Root creation is separated from propagation
/// to keep the propagation loop focused on the core algorithm.
pub struct RootCreation;

impl RootCreation {
    // =========================================================================
    // Main Tree Root Creation (atomic store on root_ptr - we hold the lock)
    // =========================================================================

    /// Create a new main tree root internode from two leaves.
    ///
    /// Atomically installs a new root via store on `root_ptr`. Parent pointers
    /// are updated after the store to avoid dangling references.
    ///
    /// # Arguments
    ///
    /// - `root_ptr`: Atomic pointer to tree root
    /// - `allocator`: Node allocator
    /// - `left_leaf_ptr`: Left leaf (expected current root)
    /// - `right_leaf_ptr`: Right leaf (split sibling, split-locked)
    /// - `split_ikey`: Separator key
    ///
    /// # Returns
    ///
    /// `Ok(new_root_ptr)` on success. `Err(InsertError::SplitFailed)` if CAS
    /// fails (another thread installed a root first).
    ///
    /// # Note
    ///
    /// Caller is responsible for unlocking `right_leaf_ptr` after this returns.
    #[expect(clippy::unnecessary_wraps, reason = "API Consistency")]
    pub fn create_root_from_leaves<S, L, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> Result<*mut L::Internode, InsertError>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Create new root internode directly in pool (height=0, children are leaves)
        let new_root_ptr: *mut u8 = allocator.alloc_internode_direct_root(0);
        let new_root: &L::Internode = unsafe { &*new_root_ptr.cast::<L::Internode>() };

        // Set up children: [left] -split_ikey- [right]
        new_root.set_child(0, left_leaf_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_leaf_ptr.cast());
        new_root.set_nkeys(1);

        // Atomically install new root
        // CRITICAL: We hold lock on left (current root), so this store is safe.
        // Using Release ordering to ensure new_root is fully visible before the swap.
        root_ptr.store(new_root_ptr, AtomicOrdering::Release);

        unsafe {
            (*left_leaf_ptr).set_parent(new_root_ptr);
            (*right_leaf_ptr).set_parent(new_root_ptr);
            (*left_leaf_ptr).version().mark_nonroot();
        }

        Ok(new_root_ptr.cast())
    }

    /// Create a new main tree root internode from two internodes.
    ///
    /// Used when the existing root internode splits.
    #[expect(clippy::unnecessary_wraps, reason = "API Consistency")]
    pub fn create_root_from_internodes<S, L, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_inode_ptr: *mut L::Internode,
        right_inode_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> Result<*mut L::Internode, InsertError>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let left: &L::Internode = unsafe { &*left_inode_ptr };

        // Create new root directly in pool (height = left.height + 1)
        let new_root_ptr: *mut u8 = allocator.alloc_internode_direct_root(left.height() + 1);
        let new_root: &L::Internode = unsafe { &*new_root_ptr.cast::<L::Internode>() };

        new_root.set_child(0, left_inode_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_inode_ptr.cast());
        new_root.set_nkeys(1);

        // Atomically install new root
        // CRITICAL: We hold lock on left (current root), so this store is safe.
        root_ptr.store(new_root_ptr, AtomicOrdering::Release);

        unsafe {
            (*left_inode_ptr).set_parent(new_root_ptr);
            (*right_inode_ptr).set_parent(new_root_ptr);
            (*left_inode_ptr).version().mark_nonroot();
        }

        Ok(new_root_ptr.cast())
    }

    // =========================================================================
    // Layer Root Creation (NO CAS on root_ptr)
    // =========================================================================

    /// Promote a layer root leaf to a new layer internode.
    ///
    /// Layer roots are NOT the main tree root. They are created when a leaf
    /// that was a layer root (null parent, root flag set) splits.
    ///
    /// Layer root promotion does NOT use CAS on `root_ptr` - it only updates
    /// parent pointers. This is the key difference from main root creation.
    pub fn promote_layer_root_leaves<S, L, A>(
        allocator: &A,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> *mut L::Internode
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Create new internode directly in pool (height=0, children are leaves)
        // Mark as layer root (has root flag, but not main tree root)
        let new_inode_ptr: *mut u8 = allocator.alloc_internode_direct_root(0);
        let new_inode: &L::Internode = unsafe { &*new_inode_ptr.cast::<L::Internode>() };

        new_inode.set_child(0, left_leaf_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_leaf_ptr.cast());
        new_inode.set_nkeys(1);

        // Update parent pointers - NO CAS needed
        unsafe {
            atomic_fence(AtomicOrdering::Release);

            (*left_leaf_ptr).set_parent(new_inode_ptr);
            (*right_leaf_ptr).set_parent(new_inode_ptr);

            // Clear root flags on both leaves
            (*left_leaf_ptr).version().mark_nonroot();
            (*right_leaf_ptr).version().mark_nonroot();
        }

        new_inode_ptr.cast()
    }

    /// Promote a layer root internode to a new layer internode.
    pub fn promote_layer_root_internodes<S, L, A>(
        allocator: &A,
        left_inode_ptr: *mut L::Internode,
        right_inode_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> *mut L::Internode
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let left: &L::Internode = unsafe { &*left_inode_ptr };

        // Create new internode directly in pool with root flag
        let new_inode_ptr: *mut u8 = allocator.alloc_internode_direct_root(left.height() + 1);
        let new_inode: &L::Internode = unsafe { &*new_inode_ptr.cast::<L::Internode>() };

        new_inode.set_child(0, left_inode_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_inode_ptr.cast());
        new_inode.set_nkeys(1);

        unsafe {
            atomic_fence(AtomicOrdering::Release);

            (*left_inode_ptr).set_parent(new_inode_ptr);
            (*right_inode_ptr).set_parent(new_inode_ptr);
            (*left_inode_ptr).version().mark_nonroot();
        }

        new_inode_ptr.cast()
    }
}
