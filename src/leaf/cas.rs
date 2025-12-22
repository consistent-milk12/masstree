//! CAS permutation failure and internals

use std::fmt as StdFmt;

use crate::{
    LeafFreezeUtils, ValueSlot,
    leaf::LeafNode,
    ordering::{CAS_FAILURE, CAS_SUCCESS},
    permuter::Permuter,
};

/// Error returned when a permutation CAS fails.
///
/// Contains the current raw permutation value. This is a newtype rather than
/// a bare `u64` to prevent accidental misuse:
/// - The raw value may be frozen (invalid as a `Permuter`)
/// - Using a newtype forces explicit handling
///
/// # Checking for Frozen State
///
/// ```rust,ignore
/// match leaf.cas_permutation_raw(expected, new) {
///     Ok(()) => { /* success */ }
///     Err(failure) => {
///         if failure.is_frozen::<WIDTH>() {
///             // Split in progress, fall back
///         } else {
///             // Normal contention, retry with failure.current_raw()
///         }
///     }
/// }
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CasPermutationFailure(u64);

impl CasPermutationFailure {
    /// Get the current raw permutation value that caused the CAS to fail.
    #[must_use]
    #[inline(always)]
    pub const fn current_raw(self) -> u64 {
        self.0
    }

    /// Check if the current permutation is frozen.
    ///
    /// If frozen, the caller should fall back to a locked path rather than retry.
    #[must_use]
    #[inline(always)]
    pub const fn is_frozen<const WIDTH: usize>(self) -> bool {
        LeafFreezeUtils::is_frozen::<WIDTH>(self.0)
    }

    /// Try to interpret the failure as a valid permutation.
    ///
    /// Returns `Some(permuter)` if the raw value is not frozen,
    /// or `None` if frozen.
    ///
    /// # Use Case
    /// For retry loops that need to compute a new `expected` value.
    #[must_use]
    #[inline(always)]
    pub const fn as_permuter<const WIDTH: usize>(self) -> Option<Permuter<WIDTH>> {
        if self.is_frozen::<WIDTH>() {
            None
        } else {
            Some(Permuter::from_value(self.0))
        }
    }
}

impl StdFmt::Display for CasPermutationFailure {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        // NOTE: Whether `raw` is frozen WIDTH-dependent. Avoid embedding a WIDTH-specific
        // classification in Display, callers should check `failure.is_frozen::<WIDTH>()`.
        write!(
            f,
            "CAS failed: permutation changed (raw={:#018x}; may be frozen)",
            self.0
        )
    }
}

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    /// Compare-and-swap the permutation atomically.
    ///
    /// Returns `Err(CasPermutationFailure)` on failure. The failure contains the
    /// current raw permutation value, which may be frozen.
    ///
    /// # Handling Failures
    /// Use `failure.is_frozen::<WIDTH>()` to check if a split is in progress.
    /// If frozen, fall back to locked path. Otherwise, retry with updated expected.
    pub(crate) fn cas_permutation_raw(
        &self,
        expected: Permuter<WIDTH>,
        new: Permuter<WIDTH>,
    ) -> Result<(), CasPermutationFailure> {
        self.permutation
            .compare_exchange(expected.value(), new.value(), CAS_SUCCESS, CAS_FAILURE)
            .map(|_| ())
            .map_err(CasPermutationFailure)
    }
}
