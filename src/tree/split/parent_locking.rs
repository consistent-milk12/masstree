//! Parent membership validation.

use crate::internode::InternodeNode;

/// Unit struct namespace for parent validation operations.
pub struct ParentLocking;

impl ParentLocking {
    /// Find child index in parent by pointer scan.
    #[inline(always)]
    pub fn find_child_index(parent: &InternodeNode, child_ptr: *mut u8) -> Option<usize> {
        let nkeys: usize = parent.nkeys();

        // SAFETY: Parent is locked - no concurrent retirement of children.
        (0..=nkeys).find(|i: &usize| unsafe { parent.child_unguarded(*i) } == child_ptr)
    }

    /// Validate that child is still in parent (membership check).
    #[inline(always)]
    pub fn validate_membership(parent: &InternodeNode, child_ptr: *mut u8) -> Option<usize> {
        Self::find_child_index(parent, child_ptr)
    }
}
