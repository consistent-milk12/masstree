//! Freeze utilities for Permuter24 (u128-based).
//!
//! This module provides freeze/unfreeze operations for the 24-slot permuter.
//! The freeze mechanism uses slot 23 as a sentinel position, setting it to
//! an invalid value (31) to indicate the permutation is frozen.
//!
//! # Usage
//!
//! Freezing is used during split operations to prevent concurrent CAS inserts
//! from modifying the permutation while the split is in progress.

/// Freeze utilities for Permuter24 (u128-based).
///
/// Uses slot 23 as the freeze sentinel position. When frozen, slot 23
/// contains 0x1F (31), which is invalid since valid slots are 0-23.
#[derive(Debug)]
pub struct Freeze24Utils;

impl Freeze24Utils {
    /// Bit shift for slot 23 (freeze position).
    /// Slot 23 is at bits 120-124 (23 * 5 + 5 = 120).
    const FREEZE_SHIFT: usize = 23 * 5 + 5; // = 120

    /// Mask for slot 23.
    const FREEZE_MASK: u128 = 0x1F_u128 << Self::FREEZE_SHIFT;

    /// Sentinel value (31 in slot 23 position).
    const SENTINEL: u128 = 0x1F_u128 << Self::FREEZE_SHIFT;

    /// Check if raw permutation is frozen.
    ///
    /// Frozen when slot 23 equals 31 (0x1F), which is invalid for WIDTH=24.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let raw = perm.value();
    /// if Freeze24Utils::is_frozen(raw) {
    ///     // Split in progress, retry or wait
    /// }
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn is_frozen(raw: u128) -> bool {
        (raw & Self::FREEZE_MASK) == Self::SENTINEL
    }

    /// Freeze a raw permutation value.
    ///
    /// Sets slot 23 to sentinel value (31). This makes the permutation
    /// appear "frozen" to concurrent readers, who should wait or retry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let old_raw = perm.value();
    /// let frozen = Freeze24Utils::freeze_raw(old_raw);
    /// atomic_perm.store_raw(frozen, Ordering::Release);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn freeze_raw(raw: u128) -> u128 {
        (raw & !Self::FREEZE_MASK) | Self::SENTINEL
    }

    /// Extract the original slot 23 value from a frozen permutation.
    ///
    /// This is needed to restore the original value after unfreezing.
    ///
    /// # Note
    ///
    /// This should only be called on unfrozen permutations. For frozen
    /// permutations, slot 23 contains the sentinel value (31).
    #[must_use]
    #[inline(always)]
    pub const fn get_slot23(raw: u128) -> u128 {
        (raw >> Self::FREEZE_SHIFT) & 0x1F
    }

    /// Unfreeze a permutation by restoring the original slot 23 value.
    ///
    /// # Arguments
    ///
    /// * `frozen` - The frozen raw permutation value
    /// * `original_slot23` - The original value of slot 23 before freezing
    ///
    /// # Example
    ///
    /// ```ignore
    /// let original_slot23 = Freeze24Utils::get_slot23(old_raw);
    /// let frozen = Freeze24Utils::freeze_raw(old_raw);
    /// // ... do split work ...
    /// let restored = Freeze24Utils::unfreeze_raw(frozen, original_slot23);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn unfreeze_raw(frozen: u128, original_slot23: u128) -> u128 {
        (frozen & !Self::FREEZE_MASK) | (original_slot23 << Self::FREEZE_SHIFT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freeze_unfrozen_value() {
        // A normal permutation should not be frozen
        let normal: u128 = 0x1234_5678_9ABC_DEF0;
        assert!(!Freeze24Utils::is_frozen(normal));
    }

    #[test]
    fn test_freeze_and_check() {
        let normal: u128 = 0x0000_0000_0000_0000;
        let frozen = Freeze24Utils::freeze_raw(normal);

        assert!(Freeze24Utils::is_frozen(frozen));
        assert!(!Freeze24Utils::is_frozen(normal));
    }

    #[test]
    fn test_freeze_preserves_other_bits() {
        let original: u128 = 0x0123_4567_89AB_CDEF_0123_4567_89AB_CDEF;
        let frozen = Freeze24Utils::freeze_raw(original);

        // Check that only slot 23 changed
        let slot23_mask = Freeze24Utils::FREEZE_MASK;
        assert_eq!(original & !slot23_mask, frozen & !slot23_mask);
    }

    #[test]
    fn test_unfreeze_restores_original() {
        let original: u128 = 0x0ABC_DEF0_1234_5678_9ABC_DEF0_1234_5678;
        let slot23 = Freeze24Utils::get_slot23(original);

        let frozen = Freeze24Utils::freeze_raw(original);
        assert!(Freeze24Utils::is_frozen(frozen));

        let restored = Freeze24Utils::unfreeze_raw(frozen, slot23);
        assert!(!Freeze24Utils::is_frozen(restored));
        assert_eq!(restored, original);
    }

    #[test]
    fn test_get_slot23() {
        // Slot 23 at bits 120-124
        let with_slot23_15: u128 = 0x0F << 120;
        assert_eq!(Freeze24Utils::get_slot23(with_slot23_15), 15);

        let with_slot23_0: u128 = 0x00 << 120;
        assert_eq!(Freeze24Utils::get_slot23(with_slot23_0), 0);

        let with_slot23_23: u128 = 0x17 << 120;
        assert_eq!(Freeze24Utils::get_slot23(with_slot23_23), 23);
    }
}
