//! Orphaned slot detection for debugging and diagnostics.
//!
//! An "orphaned slot" is a physical slot with a non-NULL value pointer that is
//! NOT referenced by the current permutation. This can occur due to:
//!
//! - Phase 3 Race: CAS insert publishes, then split clobbers permutation
//! - Bugs in slot assignment logic
//! - Incomplete rollback after failed CAS
//!
//! These methods are diagnostic tools for detecting such anomalies.
//!
//! # Quiescence Requirements
//!
//! Orphan detection is only reliable when the leaf is quiescent:
//! - No concurrent CAS inserts in progress
//! - No split in progress (not frozen, not splitting)
//!
//! Using these methods on an actively modified leaf may yield false positives.

#![allow(dead_code)] // Diagnostic methods may not be used in all builds

use crate::{LeafFreezeUtils, ValueSlot, permuter::Permuter};

use super::LeafNode;

/// Result of orphan detection.
#[derive(Debug, Clone)]
pub struct OrphanInfo {
    /// Physical slot index where orphan was found.
    pub slot: usize,
    /// The orphaned pointer value (for debugging).
    /// Note: This is exposed as `usize` (address only) to avoid provenance issues.
    pub ptr_addr: usize,
}

/// Mark slots referenced by permutation in the `in_perm` array.
///
/// Returns early if a slot index from the permutation is out of bounds.
fn mark_perm_slots<const WIDTH: usize>(perm: Permuter<WIDTH>, in_perm: &mut [bool; WIDTH]) {
    for i in 0..perm.size() {
        let slot: usize = perm.get(i);
        if let Some(entry) = in_perm.get_mut(slot) {
            *entry = true;
        }
    }
}

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    /// Check if this leaf has any orphaned slots.
    ///
    /// An orphaned slot is one with a non-NULL value pointer that is not
    /// referenced by the current permutation.
    ///
    /// # Quiescence Requirements
    ///
    /// This method is only reliable when:
    /// - The leaf is not being modified (no concurrent inserts/splits)
    /// - The permutation is not frozen
    ///
    /// # Returns
    ///
    /// `true` if any orphaned slots were detected, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the permutation is frozen (split in progress).
    #[must_use]
    pub fn has_orphaned_slots(&self) -> bool {
        // Load permutation - must not be frozen
        let raw: u64 = self.permutation.load(crate::ordering::READ_ORD);
        assert!(
            !LeafFreezeUtils::is_frozen::<WIDTH>(raw),
            "has_orphaned_slots: cannot check while permutation is frozen"
        );
        let perm: Permuter<WIDTH> = Permuter::from_value(raw);

        // Build set of slots referenced by permutation
        let mut in_perm: [bool; WIDTH] = [false; WIDTH];
        mark_perm_slots(perm, &mut in_perm);

        // Check all physical slots using iterator
        in_perm
            .iter()
            .enumerate()
            .filter(|(_, is_in_perm)| !**is_in_perm)
            .any(|(slot, _)| !self.leaf_value_ptr(slot).is_null())
    }

    /// Find all orphaned slots in this leaf.
    ///
    /// Returns a vector of `OrphanInfo` for each orphaned slot found.
    /// An empty vector means no orphans were detected.
    ///
    /// # Quiescence Requirements
    ///
    /// Same as [`Self::has_orphaned_slots()`].
    ///
    /// # Panics
    ///
    /// Panics if the permutation is frozen (split in progress).
    #[must_use]
    pub fn find_orphaned_slots(&self) -> Vec<OrphanInfo> {
        // Load permutation - must not be frozen
        let raw: u64 = self.permutation.load(crate::ordering::READ_ORD);
        assert!(
            !LeafFreezeUtils::is_frozen::<WIDTH>(raw),
            "find_orphaned_slots: cannot check while permutation is frozen"
        );
        let perm: Permuter<WIDTH> = Permuter::from_value(raw);

        // Build set of slots referenced by permutation
        let mut in_perm: [bool; WIDTH] = [false; WIDTH];
        mark_perm_slots(perm, &mut in_perm);

        // Collect orphans using iterator
        in_perm
            .iter()
            .enumerate()
            .filter(|(_, is_in_perm)| !**is_in_perm)
            .filter_map(|(slot, _)| {
                let ptr: *mut u8 = self.leaf_value_ptr(slot);
                if ptr.is_null() {
                    None
                } else {
                    Some(OrphanInfo {
                        slot,
                        ptr_addr: ptr.addr(),
                    })
                }
            })
            .collect()
    }

    /// Check if this leaf has any orphaned slots, skipping if frozen.
    ///
    /// Unlike [`Self::has_orphaned_slots()`], this method returns `None`
    /// if the permutation is frozen, rather than panicking.
    ///
    /// # Returns
    ///
    /// - `Some(true)` if orphans were found
    /// - `Some(false)` if no orphans were found
    /// - `None` if the permutation is frozen (cannot check)
    #[must_use]
    pub fn has_orphaned_slots_if_safe(&self) -> Option<bool> {
        let raw: u64 = self.permutation.load(crate::ordering::READ_ORD);
        if LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
            return None;
        }

        let perm: Permuter<WIDTH> = Permuter::from_value(raw);

        let mut in_perm: [bool; WIDTH] = [false; WIDTH];
        mark_perm_slots(perm, &mut in_perm);

        let has_orphan = in_perm
            .iter()
            .enumerate()
            .filter(|(_, is_in_perm)| !**is_in_perm)
            .any(|(slot, _)| !self.leaf_value_ptr(slot).is_null());

        Some(has_orphan)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "Fail fast with message")]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    type TestLeaf = LeafNode<LeafValue<u64>, 4>;

    #[test]
    fn test_no_orphans_in_fresh_leaf() {
        let leaf = TestLeaf::new();
        assert!(!leaf.has_orphaned_slots());
        assert!(leaf.find_orphaned_slots().is_empty());
    }

    #[test]
    fn test_no_orphans_after_proper_insert() {
        let leaf = TestLeaf::new();

        // Insert a value properly (slot in permutation)
        leaf.assign_value(0, 0x1234, 4, 42u64);
        let perm = Permuter::<4>::make_sorted(1);
        leaf.set_permutation(perm);

        assert!(!leaf.has_orphaned_slots());
        assert!(leaf.find_orphaned_slots().is_empty());
    }

    #[test]
    fn test_detects_orphan_slot() {
        let leaf = TestLeaf::new();

        // Set up a slot with data but NOT in permutation (simulating the bug: It is noted in
        // internal docs and tracing logs: KNOWN_BUGS.md and logs/masstree.json)
        leaf.assign_value(2, 0x5678, 4, 99u64);
        // Permutation is empty, so slot 2 is orphaned
        leaf.set_permutation(Permuter::empty());

        assert!(leaf.has_orphaned_slots());
        let orphans = leaf.find_orphaned_slots();
        assert_eq!(orphans.len(), 1);

        let first_orphan = orphans.first().expect("should have one orphan");
        assert_eq!(first_orphan.slot, 2);
    }
}
