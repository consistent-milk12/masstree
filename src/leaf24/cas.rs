//! CAS permutation operations for [`LeafNode24`].

use std::fmt as StdFmt;

use crate::freeze24::Freeze24Utils;
use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
use crate::permuter24::Permuter24;
use crate::slot::ValueSlot;

use super::LeafNode24;

/// Error returned when a permutation CAS fails for [`LeafNode24`].
///
/// Contains the current raw permutation value (u128). This is a newtype rather
/// than a bare `u128` to prevent accidental misuse:
/// - The raw value may be frozen (invalid as a `Permuter24`)
/// - Using a newtype forces explicit handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CasPermutationFailure24(u128);

impl CasPermutationFailure24 {
    /// Get the current raw permutation value that caused the CAS to fail.
    #[must_use]
    #[inline(always)]
    pub const fn current_raw(self) -> u128 {
        self.0
    }

    /// Check if the current permutation is frozen.
    ///
    /// If frozen, the caller should fall back to a locked path rather than retry.
    #[must_use]
    #[inline(always)]
    pub const fn is_frozen(self) -> bool {
        Freeze24Utils::is_frozen(self.0)
    }

    /// Try to interpret the failure as a valid permutation.
    ///
    /// Returns `Some(permuter)` if the raw value is not frozen,
    /// or `None` if frozen.
    #[must_use]
    #[inline(always)]
    pub const fn as_permuter(self) -> Option<Permuter24> {
        if self.is_frozen() {
            None
        } else {
            Some(Permuter24::from_value(self.0))
        }
    }
}

impl StdFmt::Display for CasPermutationFailure24 {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(
            f,
            "CAS failed: permutation changed (raw={:#034x}; may be frozen)",
            self.0
        )
    }
}

impl<S: ValueSlot> LeafNode24<S> {
    /// Compare-and-swap the permutation atomically.
    ///
    /// Returns `Err(CasPermutationFailure24)` on failure. The failure contains
    /// the current raw permutation value, which may be frozen.
    ///
    /// # Handling Failures
    /// Use `failure.is_frozen()` to check if a split is in progress.
    /// If frozen, fall back to locked path. Otherwise, retry with updated expected.
    #[inline(always)]
    pub(crate) fn cas_permutation_raw(
        &self,
        expected: Permuter24,
        new: Permuter24,
    ) -> Result<(), CasPermutationFailure24> {
        self.permutation
            .compare_exchange(expected, new, CAS_SUCCESS, CAS_FAILURE)
            .map(|_| ())
            .map_err(|old| CasPermutationFailure24(old.value()))
    }

    /// Compare-and-swap the permutation (non-freeze contexts only).
    ///
    /// # Errors
    /// If Failure returned frozen raw.
    ///
    /// # Panics
    /// Panics if the failure is due to a frozen permutation. Use `cas_permutation_raw`
    /// in code paths where freezing may occur.
    #[inline(always)]
    pub fn cas_permutation(&self, expected: Permuter24, new: Permuter24) -> Result<(), Permuter24> {
        match self.cas_permutation_raw(expected, new) {
            Ok(()) => Ok(()),

            Err(failure) => {
                assert!(
                    !failure.is_frozen(),
                    "cas_permutation(): failure returned frozen raw, use cas_permutation_raw"
                );

                Err(Permuter24::from_value(failure.current_raw()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    #[test]
    fn test_cas_permutation_success() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let expected = leaf.permutation();

        let mut new_perm = expected;
        let _slot = new_perm.insert_from_back(0);

        let result = leaf.cas_permutation(expected, new_perm);
        assert!(result.is_ok());
        assert_eq!(leaf.size(), 1);
    }

    #[test]
    fn test_cas_permutation_failure() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let expected = leaf.permutation();

        // Modify the permutation to make CAS fail
        let mut wrong_expected = expected;
        let _slot = wrong_expected.insert_from_back(0);
        leaf.set_permutation(wrong_expected);

        let mut new_perm = expected;
        let _slot = new_perm.insert_from_back(0);

        let result = leaf.cas_permutation(expected, new_perm);
        assert!(result.is_err());
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn test_cas_failure_is_not_frozen() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let expected = leaf.permutation();

        // Modify to make CAS fail
        let mut new_current = expected;
        let _ = new_current.insert_from_back(0);
        leaf.set_permutation(new_current);

        let mut new_perm = expected;
        let _ = new_perm.insert_from_back(0);

        let result = leaf.cas_permutation_raw(expected, new_perm);
        assert!(result.is_err());

        let failure = result.unwrap_err();
        assert!(!failure.is_frozen());
        assert!(failure.as_permuter().is_some());
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn test_cas_failure_frozen() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let expected = leaf.permutation();

        // Freeze the permutation
        let frozen_raw = Freeze24Utils::freeze_raw(expected.value());
        leaf.permutation_store_raw_release(frozen_raw);

        let mut new_perm = expected;
        let _ = new_perm.insert_from_back(0);

        let result = leaf.cas_permutation_raw(expected, new_perm);
        assert!(result.is_err());

        let failure = result.unwrap_err();
        assert!(failure.is_frozen());
        assert!(failure.as_permuter().is_none());
    }
}
