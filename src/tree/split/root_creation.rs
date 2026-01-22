//! Root and layer-root creation helpers.
//!
//! Provides atomic root installation for both main tree roots and layer roots.
//! Main tree roots use atomic store (we hold the lock on current root).
//! Layer roots use parent pointer updates only (no modification to `root_ptr`).

use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

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
    /// Caller must unlock `right_leaf_ptr` after this returns.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Returns Result for API consistency"
    )]
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
        // Allocate root internode (height=0, leaf children)
        let new_root_ptr: *mut u8 = allocator.alloc_internode_direct_root(0);
        let new_root: &L::Internode = unsafe { &*new_root_ptr.cast::<L::Internode>() };

        // Layout: [left] -split_ikey- [right]
        new_root.set_child(0, left_leaf_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_leaf_ptr.cast());
        new_root.set_nkeys(1);

        // Install new root. We hold left's lock, so store is safe.
        root_ptr.store(new_root_ptr, AtomicOrdering::Release);

        // SAFETY: Both leaves are valid and locked (left by caller, right split-locked).
        unsafe {
            (*left_leaf_ptr).set_parent(new_root_ptr);
            (*right_leaf_ptr).set_parent(new_root_ptr);
            (*left_leaf_ptr).version().mark_nonroot();
        }

        Ok(new_root_ptr.cast())
    }

    /// Create a new main tree root internode from two internodes.
    ///
    /// Used when the existing root internode splits. Atomically installs via
    /// store on `root_ptr` (we hold the lock on current root).
    ///
    /// # Note
    /// Caller must unlock `right_inode_ptr` after this returns.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Matches create_root_from_leaves signature"
    )]
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

        // Allocate root internode (height = left.height + 1)
        let new_root_ptr: *mut u8 = allocator.alloc_internode_direct_root(left.height() + 1);
        let new_root: &L::Internode = unsafe { &*new_root_ptr.cast::<L::Internode>() };

        // Layout: [left] -split_ikey- [right]
        new_root.set_child(0, left_inode_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_inode_ptr.cast());
        new_root.set_nkeys(1);

        // Install new root. We hold left's lock, so store is safe.
        root_ptr.store(new_root_ptr, AtomicOrdering::Release);

        // SAFETY: Both internodes are valid and locked (left by caller, right split-locked).
        unsafe {
            (*left_inode_ptr).set_parent(new_root_ptr);
            (*right_inode_ptr).set_parent(new_root_ptr);
            (*left_inode_ptr).version().mark_nonroot();
        }

        Ok(new_root_ptr.cast())
    }

    // =========================================================================
    // Layer Root Creation (no root_ptr modification)
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
        // Allocate layer root internode (height=0, leaf children)
        let new_inode_ptr: *mut u8 = allocator.alloc_internode_direct_root(0);
        let new_inode: &L::Internode = unsafe { &*new_inode_ptr.cast::<L::Internode>() };

        // Layout: [left] -split_ikey- [right]
        new_inode.set_child(0, left_leaf_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_leaf_ptr.cast());
        new_inode.set_nkeys(1);

        // SAFETY: Both leaves are valid and locked (left by caller, right split-locked).
        // set_parent uses Release ordering internally.
        unsafe {
            (*left_leaf_ptr).set_parent(new_inode_ptr);
            (*right_leaf_ptr).set_parent(new_inode_ptr);
            (*left_leaf_ptr).version().mark_nonroot();
            (*right_leaf_ptr).version().mark_nonroot();
        }

        new_inode_ptr.cast()
    }

    /// Promote a layer root internode to a new layer internode.
    ///
    /// Used when an existing layer root internode splits. Updates parent
    /// pointers only (no `root_ptr` modification).
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

        // Allocate layer root internode (height = left.height + 1)
        let new_inode_ptr: *mut u8 = allocator.alloc_internode_direct_root(left.height() + 1);
        let new_inode: &L::Internode = unsafe { &*new_inode_ptr.cast::<L::Internode>() };

        // Layout: [left] -split_ikey- [right]
        new_inode.set_child(0, left_inode_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_inode_ptr.cast());
        new_inode.set_nkeys(1);

        // SAFETY: Both internodes are valid and locked (left by caller, right split-locked).
        // set_parent uses Release ordering internally.
        unsafe {
            (*left_inode_ptr).set_parent(new_inode_ptr);
            (*right_inode_ptr).set_parent(new_inode_ptr);
            (*left_inode_ptr).version().mark_nonroot();
        }

        new_inode_ptr.cast()
    }
}
