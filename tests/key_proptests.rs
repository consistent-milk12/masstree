//! Property-based tests for the `key` module.
//!
//! These tests verify invariants and properties that should hold for all inputs.

use madtree::key::{Key, IKEY_SIZE, MAX_KEY_LENGTH};
use proptest::prelude::*;
use std::cmp::Ordering;

// ============================================================================
//  Strategies
// ============================================================================

/// Strategy for generating valid key data (1 to MAX_KEY_LENGTH bytes).
fn key_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=MAX_KEY_LENGTH)
}

/// Strategy for generating key data with guaranteed suffix (> 8 bytes).
fn key_data_with_suffix() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), (IKEY_SIZE + 1)..=MAX_KEY_LENGTH)
}

/// Strategy for generating short key data (1 to 8 bytes, no suffix).
fn key_data_short() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=IKEY_SIZE)
}

/// Strategy for generating multi-layer key data (> 16 bytes, allows 2+ shifts).
fn key_data_multi_layer() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), (IKEY_SIZE * 2 + 1)..=MAX_KEY_LENGTH)
}

// ============================================================================
//  Ikey Extraction Properties
// ============================================================================

proptest! {
    /// The ikey should match manual big-endian u64 conversion of first 8 bytes.
    #[test]
    fn ikey_matches_big_endian_conversion(data in key_data()) {
        let key = Key::new(&data);
        let expected = Key::read_ikey(&data, 0);
        prop_assert_eq!(key.ikey(), expected);
    }

    /// read_ikey with offset 0 should equal the key's ikey().
    #[test]
    fn read_ikey_at_zero_equals_ikey(data in key_data()) {
        let key = Key::new(&data);
        prop_assert_eq!(Key::read_ikey(&data, 0), key.ikey());
    }

    /// Keys with identical first 8 bytes should have identical ikeys.
    #[test]
    fn identical_prefix_means_identical_ikey(
        prefix in prop::collection::vec(any::<u8>(), IKEY_SIZE..=IKEY_SIZE),
        suffix1 in prop::collection::vec(any::<u8>(), 0..16),
        suffix2 in prop::collection::vec(any::<u8>(), 0..16)
    ) {
        let mut data1 = prefix.clone();
        data1.extend(&suffix1);
        let mut data2 = prefix;
        data2.extend(&suffix2);

        // Truncate if needed
        data1.truncate(MAX_KEY_LENGTH);
        data2.truncate(MAX_KEY_LENGTH);

        let key1 = Key::new(&data1);
        let key2 = Key::new(&data2);

        prop_assert_eq!(key1.ikey(), key2.ikey());
    }
}

// ============================================================================
//  Shift/Unshift Roundtrip Properties
// ============================================================================

proptest! {
    /// Shifting then unshifting should restore the original state.
    #[test]
    fn shift_unshift_roundtrip(data in key_data_with_suffix()) {
        let mut key = Key::new(&data);
        let original_ikey = key.ikey();
        let original_shift_count = key.shift_count();
        let original_suffix_start = key.suffix_start();

        key.shift();
        prop_assert!(key.is_shifted());
        prop_assert_eq!(key.shift_count(), 1);

        key.unshift();
        prop_assert_eq!(key.ikey(), original_ikey);
        prop_assert_eq!(key.shift_count(), original_shift_count);
        prop_assert_eq!(key.suffix_start(), original_suffix_start);
    }

    /// Multiple shifts followed by unshift_all should restore original state.
    #[test]
    fn multiple_shifts_unshift_all_roundtrip(data in key_data_multi_layer()) {
        let mut key = Key::new(&data);
        let original_ikey = key.ikey();
        let original_suffix_start = key.suffix_start();

        // Shift as many times as possible
        let mut shift_count = 0;
        while key.has_suffix() {
            key.shift();
            shift_count += 1;
        }
        prop_assert!(shift_count >= 2, "Expected at least 2 shifts for multi-layer data");

        // Reset all
        key.unshift_all();
        prop_assert_eq!(key.ikey(), original_ikey);
        prop_assert_eq!(key.shift_count(), 0);
        prop_assert_eq!(key.suffix_start(), original_suffix_start);
    }

    /// After shifting, current_len decreases by IKEY_SIZE.
    #[test]
    fn shift_decreases_current_len(data in key_data_with_suffix()) {
        let mut key = Key::new(&data);
        let len_before = key.current_len();

        key.shift();
        let len_after = key.current_len();

        prop_assert_eq!(len_after, len_before - IKEY_SIZE);
    }

    /// shift_count increments by 1 on each shift.
    #[test]
    fn shift_increments_count(data in key_data_with_suffix()) {
        let mut key = Key::new(&data);
        let count_before = key.shift_count();

        key.shift();

        prop_assert_eq!(key.shift_count(), count_before + 1);
    }
}

// ============================================================================
//  Suffix Properties
// ============================================================================

proptest! {
    /// Keys with len <= IKEY_SIZE should not have a suffix.
    #[test]
    fn short_keys_have_no_suffix(data in key_data_short()) {
        let key = Key::new(&data);
        prop_assert!(!key.has_suffix());
        prop_assert_eq!(key.suffix_len(), 0);
        prop_assert!(key.suffix().is_empty());
    }

    /// Keys with len > IKEY_SIZE should have a suffix.
    #[test]
    fn long_keys_have_suffix(data in key_data_with_suffix()) {
        let key = Key::new(&data);
        prop_assert!(key.has_suffix());
        prop_assert!(key.suffix_len() > 0);
        prop_assert_eq!(key.suffix_len(), data.len() - IKEY_SIZE);
    }

    /// suffix_len + suffix_start should equal total length.
    #[test]
    fn suffix_len_plus_start_equals_len(data in key_data()) {
        let key = Key::new(&data);
        prop_assert_eq!(key.suffix_start() + key.suffix_len(), key.len());
    }

    /// current_len should equal len - shift_count * IKEY_SIZE.
    #[test]
    fn current_len_formula(data in key_data_multi_layer()) {
        let mut key = Key::new(&data);
        let total_len = key.len();

        // Test at various shift levels
        let mut shifts = 0;
        loop {
            prop_assert_eq!(key.current_len(), total_len.saturating_sub(shifts * IKEY_SIZE));

            if !key.has_suffix() {
                break;
            }
            key.shift();
            shifts += 1;
        }
    }
}

// ============================================================================
//  Lexicographic Ordering Properties
// ============================================================================

proptest! {
    /// Big-endian u64 comparison should match lexicographic byte comparison.
    #[test]
    fn ikey_ordering_matches_lexicographic(
        a in prop::collection::vec(any::<u8>(), IKEY_SIZE..=IKEY_SIZE),
        b in prop::collection::vec(any::<u8>(), IKEY_SIZE..=IKEY_SIZE)
    ) {
        let key_a = Key::new(&a);
        let key_b = Key::new(&b);

        let ikey_ordering = key_a.ikey().cmp(&key_b.ikey());
        let byte_ordering = a.cmp(&b);

        prop_assert_eq!(ikey_ordering, byte_ordering);
    }

    /// compare_ikey should be consistent with direct u64 comparison.
    #[test]
    fn compare_ikey_consistent(a: u64, b: u64) {
        let result = Key::compare_ikey(a, b);
        let expected = a.cmp(&b);
        prop_assert_eq!(result, expected);
    }

    /// compare_ikey should be reflexive: compare(x, x) == Equal.
    #[test]
    fn compare_ikey_reflexive(x: u64) {
        prop_assert_eq!(Key::compare_ikey(x, x), Ordering::Equal);
    }

    /// compare_ikey should be antisymmetric.
    #[test]
    fn compare_ikey_antisymmetric(a: u64, b: u64) {
        let ab = Key::compare_ikey(a, b);
        let ba = Key::compare_ikey(b, a);

        match ab {
            Ordering::Less => prop_assert_eq!(ba, Ordering::Greater),
            Ordering::Greater => prop_assert_eq!(ba, Ordering::Less),
            Ordering::Equal => prop_assert_eq!(ba, Ordering::Equal),
        }
    }
}

// ============================================================================
//  Key::compare Properties
// ============================================================================

proptest! {
    /// compare should be reflexive when keylenx matches current_len.
    #[test]
    fn compare_reflexive(data in key_data()) {
        let key = Key::new(&data);
        let result = key.compare(key.ikey(), key.current_len());
        prop_assert_eq!(result, Ordering::Equal);
    }

    /// Longer keys should be Greater than shorter keys with same ikey (when both <= 8).
    #[test]
    fn compare_by_length_when_ikey_equal(
        len1 in 1usize..=IKEY_SIZE,
        len2 in 1usize..=IKEY_SIZE,
        fill_byte: u8
    ) {
        // Create keys of different lengths but same ikey prefix
        let data1: Vec<u8> = vec![fill_byte; len1];

        let key1 = Key::new(&data1);

        // Use key1's ikey as the "stored" ikey, compare against len2
        let result = key1.compare(key1.ikey(), len2);

        match len1.cmp(&len2) {
            Ordering::Less => prop_assert_eq!(result, Ordering::Less),
            Ordering::Greater => prop_assert_eq!(result, Ordering::Greater),
            Ordering::Equal => prop_assert_eq!(result, Ordering::Equal),
        }
    }

    /// Keys with different ikeys should compare by ikey regardless of length.
    #[test]
    fn compare_by_ikey_first(
        a in prop::collection::vec(any::<u8>(), 1..=IKEY_SIZE),
        b in prop::collection::vec(any::<u8>(), 1..=IKEY_SIZE)
    ) {
        let key_a = Key::new(&a);
        let ikey_b = Key::read_ikey(&b, 0);

        // When ikeys differ, comparison should follow ikey order
        if key_a.ikey() != ikey_b {
            let result = key_a.compare(ikey_b, b.len());
            let expected = key_a.ikey().cmp(&ikey_b);
            prop_assert_eq!(result, expected);
        }
    }
}

// ============================================================================
//  Full Data Invariants
// ============================================================================

proptest! {
    /// full_data should always return the original data regardless of shifts.
    #[test]
    fn full_data_unchanged_by_shifts(data in key_data_multi_layer()) {
        let mut key = Key::new(&data);

        // Verify before shifts
        prop_assert_eq!(key.full_data(), &data[..]);

        // Shift multiple times
        while key.has_suffix() {
            key.shift();
            prop_assert_eq!(key.full_data(), &data[..]);
        }

        // Verify after unshift_all
        key.unshift_all();
        prop_assert_eq!(key.full_data(), &data[..]);
    }

    /// len() should always return the original length.
    #[test]
    fn len_unchanged_by_shifts(data in key_data_with_suffix()) {
        let mut key = Key::new(&data);
        let original_len = key.len();

        key.shift();
        prop_assert_eq!(key.len(), original_len);

        key.unshift();
        prop_assert_eq!(key.len(), original_len);
    }
}

// ============================================================================
//  Edge Cases
// ============================================================================

proptest! {
    /// Empty suffix after exact 8-byte boundary should be handled correctly.
    #[test]
    fn exact_8_byte_no_suffix(
        data in prop::collection::vec(any::<u8>(), IKEY_SIZE..=IKEY_SIZE)
    ) {
        let key = Key::new(&data);
        prop_assert!(!key.has_suffix());
        prop_assert_eq!(key.suffix_len(), 0);
        prop_assert_eq!(key.current_len(), IKEY_SIZE);
    }

    /// 9-byte key should have exactly 1-byte suffix.
    #[test]
    fn nine_byte_one_suffix(
        data in prop::collection::vec(any::<u8>(), (IKEY_SIZE + 1)..=(IKEY_SIZE + 1))
    ) {
        let key = Key::new(&data);
        prop_assert!(key.has_suffix());
        prop_assert_eq!(key.suffix_len(), 1);
        prop_assert_eq!(key.suffix(), &data[IKEY_SIZE..]);
    }

    /// Keys at MAX_KEY_LENGTH should work correctly.
    #[test]
    fn max_length_keys_work(
        data in prop::collection::vec(any::<u8>(), MAX_KEY_LENGTH..=MAX_KEY_LENGTH)
    ) {
        let key = Key::new(&data);
        prop_assert_eq!(key.len(), MAX_KEY_LENGTH);
        prop_assert!(key.has_suffix());

        // Should be able to shift multiple times
        let max_shifts = (MAX_KEY_LENGTH - 1) / IKEY_SIZE;
        let mut key_mut = key;
        for _ in 0..max_shifts {
            prop_assert!(key_mut.has_suffix() || key_mut.current_len() == IKEY_SIZE);
            if key_mut.has_suffix() {
                key_mut.shift();
            }
        }
    }
}

// ============================================================================
//  Binary Data Handling
// ============================================================================

proptest! {
    /// Keys containing null bytes should be handled correctly.
    #[test]
    fn handles_null_bytes(
        prefix in prop::collection::vec(0u8..=0u8, 1..=4),
        middle in prop::collection::vec(any::<u8>(), 1..=8),
        suffix in prop::collection::vec(0u8..=0u8, 0..=4)
    ) {
        let mut data = prefix;
        data.extend(&middle);
        data.extend(&suffix);
        data.truncate(MAX_KEY_LENGTH);

        let key = Key::new(&data);

        // Basic properties should still hold
        prop_assert_eq!(key.len(), data.len());
        prop_assert_eq!(key.ikey(), Key::read_ikey(&data, 0));
    }

    /// All-zero keys should work.
    #[test]
    fn handles_all_zeros(len in 1usize..=32) {
        let data: Vec<u8> = vec![0; len];
        let key = Key::new(&data);

        prop_assert_eq!(key.ikey(), 0);
        prop_assert_eq!(key.len(), len);
    }

    /// All-0xFF keys should work.
    #[test]
    fn handles_all_ones(len in 1usize..=32) {
        let data: Vec<u8> = vec![0xFF; len];
        let key = Key::new(&data);

        let expected_ikey = if len >= IKEY_SIZE {
            u64::MAX
        } else {
            // Partial: 0xFF repeated `len` times, then zeros
            let mut bytes = [0u8; 8];
            for b in bytes.iter_mut().take(len) {
                *b = 0xFF;
            }
            u64::from_be_bytes(bytes)
        };

        prop_assert_eq!(key.ikey(), expected_ikey);
    }
}
