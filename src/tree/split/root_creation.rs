//! Root and layer-root creation helpers.
//!
//! Provides atomic root installation for both main tree roots and layer roots.
//! Layer roots use parent pointer updates only (no CAS on root_ptr).
//!
//! # CAS Failure Policy
//!
//! When CAS fails during main root creation, the allocated internode is
//! NOT retired. It remains tracked by the allocator and will be freed
//! when the allocator drops. This prevents double-free.

use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering, fence as atomic_fence};

use crate::NodeAllocatorGeneric;
use crate::leaf_trait::{LayerCapableLeaf, TreeInternode};
use crate::tree::InsertError;
use crate::value::LeafValue;

/// Unit struct namespace for root creation operations.
///
/// All methods are stateless. Root creation is separated from propagation
/// to keep the propagation loop focused on the core algorithm.
pub struct RootCreation;

impl RootCreation {
    // =========================================================================
    // Main Tree Root Creation (uses CAS on root_ptr)
    // =========================================================================

    /// Create a new main tree root internode from two leaves.
    ///
    /// Atomically installs a new root via CAS on `root_ptr`. Parent pointers
    /// are only updated after CAS succeeds to avoid dangling references.
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
    /// # CAS Failure Policy
    ///
    /// On CAS failure, the allocated internode is NOT retired. It remains
    /// tracked by the allocator and will be freed when the allocator drops.
    /// This prevents double-free (see SpecAnalysis.md §2.3).
    ///
    /// # Note
    ///
    /// Caller is responsible for unlocking `right_leaf_ptr` after this returns.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "debug",
            skip(root_ptr, allocator),
            fields(
                left = ?left_leaf_ptr,
                right = ?right_leaf_ptr,
                split_ikey = %format_args!("{:016x}", split_ikey)
            )
        )
    )]
    pub fn create_root_from_leaves<V, L, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> Result<*mut L::Internode, InsertError>
    where
        V: Send + Sync + 'static,
        L: LayerCapableLeaf<V>,
        A: NodeAllocatorGeneric<LeafValue<V>, L>,
    {
        #[cfg(feature = "tracing")]
        tracing::debug!("RootCreation: creating root from leaves");

        // Create new root internode (height=0, children are leaves)
        let new_root: Box<L::Internode> = L::Internode::new_root_boxed(0);

        // Set up children: [left] -split_ikey- [right]
        new_root.set_child(0, left_leaf_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_leaf_ptr.cast());
        new_root.set_nkeys(1);

        // Allocate and track
        let new_root_ptr: *mut u8 =
            allocator.alloc_internode_erased(Box::into_raw(new_root).cast());

        #[cfg(feature = "tracing")]
        tracing::debug!(new_root_ptr = ?new_root_ptr, "RootCreation: allocated");

        // Atomically install via CAS
        let expected: *mut u8 = left_leaf_ptr.cast();
        match root_ptr.compare_exchange(
            expected,
            new_root_ptr,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => {
                // CAS succeeded - update parent pointers
                unsafe {
                    // Release fence: ensure internode is fully constructed
                    // before it becomes visible via parent pointers
                    atomic_fence(AtomicOrdering::Release);

                    (*left_leaf_ptr).set_parent(new_root_ptr);
                    (*right_leaf_ptr).set_parent(new_root_ptr);
                    (*left_leaf_ptr).version().mark_nonroot();
                }

                #[cfg(feature = "tracing")]
                tracing::info!(new_root_ptr = ?new_root_ptr, "RootCreation: root installed");

                Ok(new_root_ptr.cast())
            }
            Err(current) => {
                // Under TRUE hand-over-hand, CAS failure should be unreachable.
                // This indicates an invariant violation - panic with diagnostics.
                // (See §2.1 - unified panic policy per SpecAnalysis2 §7.2)
                //
                // The allocated node remains tracked by the allocator (no leak).
                panic!(
                    "RootCreation::create_root_from_leaves: CAS failed unexpectedly. \
                     expected={expected:?}, current={current:?}. \
                     This indicates an invariant violation - the root was modified \
                     while we held the lock."
                );
            }
        }
    }

    /// Create a new main tree root internode from two internodes.
    ///
    /// Used when the existing root internode splits.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "debug",
            skip(root_ptr, allocator),
            fields(
                left = ?left_inode_ptr,
                right = ?right_inode_ptr,
                split_ikey = %format_args!("{:016x}", split_ikey)
            )
        )
    )]
    pub fn create_root_from_internodes<V, L, A>(
        root_ptr: &AtomicPtr<u8>,
        allocator: &A,
        left_inode_ptr: *mut L::Internode,
        right_inode_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> Result<*mut L::Internode, InsertError>
    where
        V: Send + Sync + 'static,
        L: LayerCapableLeaf<V>,
        A: NodeAllocatorGeneric<LeafValue<V>, L>,
    {
        let left: &L::Internode = unsafe { &*left_inode_ptr };

        #[cfg(feature = "tracing")]
        tracing::debug!(
            left_height = left.height(),
            "RootCreation: creating root from internodes"
        );

        // Create new root (height = left.height + 1)
        let new_root: Box<L::Internode> = L::Internode::new_root_boxed(left.height() + 1);

        new_root.set_child(0, left_inode_ptr.cast());
        new_root.set_ikey(0, split_ikey);
        new_root.set_child(1, right_inode_ptr.cast());
        new_root.set_nkeys(1);

        let new_root_ptr: *mut u8 =
            allocator.alloc_internode_erased(Box::into_raw(new_root).cast());

        let expected: *mut u8 = left_inode_ptr.cast();
        match root_ptr.compare_exchange(
            expected,
            new_root_ptr,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => {
                unsafe {
                    atomic_fence(AtomicOrdering::Release);
                    (*left_inode_ptr).set_parent(new_root_ptr);
                    (*right_inode_ptr).set_parent(new_root_ptr);
                    (*left_inode_ptr).version().mark_nonroot();
                }

                #[cfg(feature = "tracing")]
                tracing::info!(
                    new_root_ptr = ?new_root_ptr,
                    new_height = left.height() + 1,
                    "RootCreation: internode root installed"
                );

                Ok(new_root_ptr.cast())
            }
            Err(current) => {
                // Panic on CAS failure - see §2.1 and create_root_from_leaves
                panic!(
                    "RootCreation::create_root_from_internodes: CAS failed unexpectedly. \
                     expected={expected:?}, current={current:?}. Invariant violation."
                );
            }
        }
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
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "debug",
            skip(allocator),
            fields(
                left = ?left_leaf_ptr,
                right = ?right_leaf_ptr,
                split_ikey = %format_args!("{:016x}", split_ikey)
            )
        )
    )]
    pub fn promote_layer_root_leaves<V, L, A>(
        allocator: &A,
        left_leaf_ptr: *mut L,
        right_leaf_ptr: *mut L,
        split_ikey: u64,
    ) -> *mut L::Internode
    where
        V: Send + Sync + 'static,
        L: LayerCapableLeaf<V>,
        A: NodeAllocatorGeneric<LeafValue<V>, L>,
    {
        #[cfg(feature = "tracing")]
        tracing::debug!("RootCreation: promoting layer root leaves");

        // Create new internode (height=0, children are leaves)
        let new_inode: Box<L::Internode> = L::Internode::new_boxed(0);

        new_inode.set_child(0, left_leaf_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_leaf_ptr.cast());
        new_inode.set_nkeys(1);

        // Mark as layer root (has root flag, but not main tree root)
        new_inode.version().mark_root();

        let new_inode_ptr: *mut u8 =
            allocator.alloc_internode_erased(Box::into_raw(new_inode).cast());

        // Update parent pointers - NO CAS needed
        unsafe {
            atomic_fence(AtomicOrdering::Release);

            (*left_leaf_ptr).set_parent(new_inode_ptr);
            (*right_leaf_ptr).set_parent(new_inode_ptr);

            // Clear root flags on both leaves
            (*left_leaf_ptr).version().mark_nonroot();
            (*right_leaf_ptr).version().mark_nonroot();
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            new_inode_ptr = ?new_inode_ptr,
            "RootCreation: layer root internode created"
        );

        new_inode_ptr.cast()
    }

    /// Promote a layer root internode to a new layer internode.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "debug",
            skip(allocator),
            fields(
                left = ?left_inode_ptr,
                right = ?right_inode_ptr,
                split_ikey = %format_args!("{:016x}", split_ikey)
            )
        )
    )]
    pub fn promote_layer_root_internodes<V, L, A>(
        allocator: &A,
        left_inode_ptr: *mut L::Internode,
        right_inode_ptr: *mut L::Internode,
        split_ikey: u64,
    ) -> *mut L::Internode
    where
        V: Send + Sync + 'static,
        L: LayerCapableLeaf<V>,
        A: NodeAllocatorGeneric<LeafValue<V>, L>,
    {
        let left: &L::Internode = unsafe { &*left_inode_ptr };

        #[cfg(feature = "tracing")]
        tracing::debug!(
            left_height = left.height(),
            "RootCreation: promoting layer root internodes"
        );

        let new_inode: Box<L::Internode> = L::Internode::new_boxed(left.height() + 1);

        new_inode.set_child(0, left_inode_ptr.cast());
        new_inode.set_ikey(0, split_ikey);
        new_inode.set_child(1, right_inode_ptr.cast());
        new_inode.set_nkeys(1);

        new_inode.version().mark_root();

        let new_inode_ptr: *mut u8 =
            allocator.alloc_internode_erased(Box::into_raw(new_inode).cast());

        unsafe {
            atomic_fence(AtomicOrdering::Release);

            (*left_inode_ptr).set_parent(new_inode_ptr);
            (*right_inode_ptr).set_parent(new_inode_ptr);
            (*left_inode_ptr).version().mark_nonroot();
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            new_inode_ptr = ?new_inode_ptr,
            new_height = left.height() + 1,
            "RootCreation: layer root internode created"
        );

        new_inode_ptr.cast()
    }
}
