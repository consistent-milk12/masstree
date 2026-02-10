//! Parent membership validation.
//!
//! Provides helpers for validating that a child is still in its parent
//! during split propagation.

use crate::internode::InternodeNode;

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
    pub fn find_child_index(parent: &InternodeNode, child_ptr: *mut u8) -> Option<usize> {
        let nkeys: usize = parent.nkeys();

        // SAFETY: Parent is locked - no concurrent retirement of children.
        (0..=nkeys).find(|i: &usize| unsafe { parent.child_unguarded(*i) } == child_ptr)
    }

    /// Validate that child is still in parent (membership check).
    ///
    /// Must be called after locking parent, before inserting.
    #[inline(always)]
    pub fn validate_membership(parent: &InternodeNode, child_ptr: *mut u8) -> Option<usize> {
        Self::find_child_index(parent, child_ptr)
    }
}
