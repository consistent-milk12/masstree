//! Permutation Freeze

use std::fmt as StdFmt;

use crate::{ValueSlot, leaf::LeafNode, permuter::Permuter};

/// Permutation freeze related utils for leaf structures
#[derive(Debug)]
pub struct LeafFreezeUtils;

impl LeafFreezeUtils {
    /// Compute bit shift for the freeze marker nibble.
    ///
    /// The freeze marker is placed at position `WIDTH - 1` (the highest slot position).
    /// For WIDTH=15, this is bits 60-63.
    #[must_use]
    #[inline(always)]
    const fn freeze_shift<const WIDTH: usize>() -> usize {
        ((WIDTH - 1) * 4) + 4
    }

    /// Compute the ask for the freeze nibble.
    #[must_use]
    #[inline(always)]
    const fn freeze_nibble_mask<const WIDTH: usize>() -> u64 {
        0xF_u64 << Self::freeze_shift::<WIDTH>()
    }

    /// Check if a raw permutation value is frozen.
    ///
    /// A permutation is frozen if its highest slot nibble equals `OxF`,
    /// which is an invalid slot index for `WIDTH <= 15`.
    #[must_use]
    #[inline(always)]
    pub const fn is_frozen<const WIDTH: usize>(raw: u64) -> bool {
        ((raw >> Self::freeze_shift::<WIDTH>()) & 0xF) == 0xF
    }

    /// Freeze a raw permutation value by setting a sentinel nibble.
    ///
    /// Returns a value that will fail any `cas_permutation()` with a valid expected value.
    #[must_use]
    #[inline(always)]
    pub const fn freeze_raw<const WIDTH: usize>(raw: u64) -> u64 {
        (raw & !Self::freeze_nibble_mask::<WIDTH>()) | (0xF_u64 << Self::freeze_shift::<WIDTH>())
    }
}

/// Error returned when attempting to read a frozen permutation.
///
/// This indicates a split is in progress. The caller should either:
/// - Wait for unfreeze using `permutation_wait()`, or
/// - Fall back to a locked code path
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Frozen;

impl StdFmt::Display for Frozen {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(f, "permutation is frozen (split in progress)")
    }
}

impl std::error::Error for Frozen {}

/// Error returned when attempting to freeze an already-frozen permutation.
///
/// This is a diagnostic error for [`LeafNode::try_freeze_permutation()`].
/// Under normal operation, this should not occur because:
/// - Freeze requires holding the leaf lock
/// - The lock holder always unfreezes before releasing
///
/// Observing this error indicates an internal invariant violation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlreadyFrozen {
    /// The raw permutation value that was already frozen.
    pub raw: u64,
}

impl StdFmt::Display for AlreadyFrozen {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(
            f,
            "permutation already frozen (raw={:#018x}); possible invariant violation",
            self.raw
        )
    }
}

impl std::error::Error for AlreadyFrozen {}

/// RAII guard for a frozen permutation.
///
/// Created by [`LeafNode::freeze_permutation()`]. The guard:
/// - Holds a snapshot of the pre-freeze permutation value
/// - Automatically restores a valid permutation on drop (panic safety)
/// - Must be consumed via [`unfreeze_get_permutation()`] on the success path
///
/// # Panic Safety
/// If the guard is dropped without being consumed (e.g., during unwinding),
/// it restores the original snapshot permutation. This ensures readers don't
/// spin forever, but may leave the tree in an inconsistent state if the split
/// had already performed destructive moves.
///
/// CRITICAL: Split code must be structured so that panics cannot occur after
/// destructive mutations begin.
#[must_use = "FreezeGuard must be consumed vai unfreeze_get_permutation()"]
pub struct FreezeGuard<'a, S: ValueSlot, const WIDTH: usize> {
    leaf: &'a LeafNode<S, WIDTH>,
    snapshot_raw: u64,
    active: bool,
}

impl<'a, S: ValueSlot, const WIDTH: usize> FreezeGuard<'a, S, WIDTH> {
    /// Construct a new [`FreezeGuard`]
    #[inline(always)]
    pub const fn new(leaf: &'a LeafNode<S, WIDTH>, snapshot_raw: u64, active: bool) -> Self {
        Self {
            leaf,
            snapshot_raw,
            active,
        }
    }

    /// Get the permutation snapshot captured at freeze time.
    ///
    /// This is the authoritative membership for split computation.
    /// It includes all CAS inserts that published before freeze succeeded.
    #[must_use]
    #[inline(always)]
    pub const fn snapshot(&self) -> Permuter<WIDTH> {
        Permuter::from_value(self.snapshot_raw)
    }

    /// Get the raw snapshot value (for debugging/logging).
    #[must_use]
    #[inline(always)]
    pub const fn snapshot_raw(&self) -> u64 {
        self.snapshot_raw
    }

    /// Set whether the guard is active.
    #[inline(always)]
    pub const fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl<S: ValueSlot, const WIDTH: usize> Drop for FreezeGuard<'_, S, WIDTH> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        // Fail-stop recovery: restore a valid permutation so readers don't spin forever.
        //
        // This is only correct if the split didn't perform any destructive moves.
        // The split code must be structured to be panic-free after freeze.
        self.leaf.permutation_store_raw_release(self.snapshot_raw);
    }
}

impl<S: ValueSlot, const WIDTH: usize> StdFmt::Debug for FreezeGuard<'_, S, WIDTH> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("FreezeGuard")
            .field("snapshot_raw", &format_args!("{:#018x}", self.snapshot_raw))
            .field("active", &self.active)
            .finish()
    }
}

// ============================================================================
// FreezeGuardOps Implementation
// ============================================================================

impl<S: ValueSlot, const WIDTH: usize> crate::leaf_trait::FreezeGuardOps<Permuter<WIDTH>>
    for FreezeGuard<'_, S, WIDTH>
{
    #[inline(always)]
    fn snapshot(&self) -> Permuter<WIDTH> {
        FreezeGuard::snapshot(self)
    }

    #[inline(always)]
    fn snapshot_raw(&self) -> u64 {
        FreezeGuard::snapshot_raw(self)
    }

    #[inline(always)]
    fn set_active(&mut self, active: bool) {
        FreezeGuard::set_active(self, active);
    }
}
