//! Parent membership validation.
//!
//! Provides helpers for validating that a child is still in its parent
//! during split propagation.

use crate::leaf_trait::{LayerCapableLeaf, TreeInternode};
use crate::slot::ValueSlot;

/// Unit struct namespace for parent validation operations.
pub struct ParentLocking;

impl ParentLocking {
    /// Find child index in parent by pointer scan.
    ///
    /// This is the ONLY correct way to find insertion position during split
    /// propagation. Key-based search is wrong because seperator keys may be
    /// inconsistent during concurrent splits.
    ///
    /// # Returns
    /// `Some(index)` if child found, [`None`] otherwise.
    #[inline(always)]
    pub fn find_child_index<S, L>(parent: &L::Internode, child_ptr: *mut u8) -> Option<usize>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        let nkeys: usize = parent.nkeys();

        (0..=nkeys).find(|i: &usize| parent.child(*i) == child_ptr)
    }

    /// Validate that child is still in parent (memership check).
    ///
    /// Must be called after locking parent, before inserting.
    #[inline(always)]
    pub fn validate_membership<S, L>(parent: &L::Internode, child_ptr: *mut u8) -> Option<usize>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        Self::find_child_index::<S, L>(parent, child_ptr)
    }
}
