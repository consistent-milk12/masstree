//! Permutation Freeze related utils

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

    #[must_use]
    #[inline(always)]
    pub const fn freeze_raw<const WIDTH: usize>(raw: u64) -> u64 {
        (raw & !Self::freeze_nibble_mask::<WIDTH>()) | (0xF_u64 << Self::freeze_shift::<WIDTH>())
    }
}
