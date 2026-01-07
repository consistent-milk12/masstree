use super::{INITIAL_CAPACITY, InlineSuffixBag, PermutationProvider, SuffixBag};
use crate::permuter24::Permuter24;

// ========================================================================
//  Basic Tests
// ========================================================================

#[test]
fn test_new_suffix_bag() {
    let bag: SuffixBag<15> = SuffixBag::new();

    assert_eq!(bag.count(), 0);
    assert!(bag.capacity() >= INITIAL_CAPACITY);
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_with_capacity() {
    let bag: SuffixBag<15> = SuffixBag::with_capacity(256);

    assert!(bag.capacity() >= 256);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_default() {
    let bag: SuffixBag<15> = SuffixBag::default();

    assert_eq!(bag.count(), 0);
}

// ========================================================================
//  Assign and Get Tests
// ========================================================================

#[test]
fn test_assign_and_get() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");
    bag.assign(5, b"world");
    bag.assign(10, b"!");

    assert_eq!(bag.get(0), Some(b"hello".as_slice()));
    assert_eq!(bag.get(5), Some(b"world".as_slice()));
    assert_eq!(bag.get(10), Some(b"!".as_slice()));
    assert_eq!(bag.get(1), None);
    assert_eq!(bag.count(), 3);
}

#[test]
fn test_empty_suffix() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"");

    assert_eq!(bag.get(0), Some(b"".as_slice()));
    assert!(bag.has_suffix(0));
}

#[test]
fn test_get_or_empty() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");

    assert_eq!(bag.get_or_empty(0), b"hello".as_slice());
    assert_eq!(bag.get_or_empty(1), b"".as_slice());
}

#[test]
fn test_overwrite_suffix() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");
    assert_eq!(bag.get(0), Some(b"hello".as_slice()));

    bag.assign(0, b"goodbye");
    assert_eq!(bag.get(0), Some(b"goodbye".as_slice()));

    // Old data still in buffer (not compacted)
    assert!(bag.used() >= "hello".len() + "goodbye".len());
}

// ========================================================================
//  Clear Tests
// ========================================================================

#[test]
fn test_clear_suffix() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");
    assert!(bag.has_suffix(0));

    bag.clear(0);

    assert!(!bag.has_suffix(0));
    assert_eq!(bag.get(0), None);
}

#[test]
fn test_clear_already_empty() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Clearing an empty slot should not panic
    bag.clear(0);

    assert!(!bag.has_suffix(0));
}

// ========================================================================
//  Compact Tests
// ========================================================================

#[test]
fn test_compact() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Add several suffixes
    bag.assign(0, b"aaaa");
    bag.assign(1, b"bbbb");
    bag.assign(2, b"cccc");
    bag.assign(3, b"dddd");

    let before: usize = bag.used();
    assert_eq!(before, 16);

    // Compact keeping only slots 0 and 2
    let reclaimed: usize = bag.compact([0, 2].into_iter());

    assert!(reclaimed > 0);
    assert_eq!(bag.get(0), Some(b"aaaa".as_slice()));
    assert_eq!(bag.get(2), Some(b"cccc".as_slice()));
    assert_eq!(bag.get(1), None);
    assert_eq!(bag.get(3), None);
    assert_eq!(bag.used(), 8);
}

#[test]
fn test_compact_empty() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Compact with no active slots should work
    let reclaimed: usize = bag.compact(std::iter::empty());

    assert_eq!(reclaimed, 0);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_compact_all() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"test");
    bag.assign(1, b"data");

    // Compact keeping all
    let reclaimed: usize = bag.compact([0, 1].into_iter());

    // No garbage to collect
    assert_eq!(reclaimed, 0);
    assert_eq!(bag.get(0), Some(b"test".as_slice()));
    assert_eq!(bag.get(1), Some(b"data".as_slice()));
}

#[test]
fn test_compact_with_permuter() {
    // Create a mock permuter
    struct MockPerm {
        slots: Vec<usize>,
    }

    impl PermutationProvider for MockPerm {
        fn size(&self) -> usize {
            self.slots.len()
        }

        fn get(&self, i: usize) -> usize {
            self.slots[i]
        }
    }

    let mut bag: SuffixBag<15> = SuffixBag::new();
    bag.assign(0, b"keep0");
    bag.assign(1, b"drop1");
    bag.assign(2, b"keep2");

    let perm = MockPerm {
        slots: vec![0, 2], // Only slots 0 and 2 are active
    };

    bag.compact_with_permuter(&perm, None);

    assert_eq!(bag.get(0), Some(b"keep0".as_slice()));
    assert_eq!(bag.get(1), None);
    assert_eq!(bag.get(2), Some(b"keep2".as_slice()));
}

#[test]
fn test_compact_with_exclude() {
    struct MockPerm {
        slots: Vec<usize>,
    }

    impl PermutationProvider for MockPerm {
        fn size(&self) -> usize {
            self.slots.len()
        }

        fn get(&self, i: usize) -> usize {
            self.slots[i]
        }
    }

    let mut bag: SuffixBag<15> = SuffixBag::new();
    bag.assign(0, b"keep");
    bag.assign(1, b"exclude");
    bag.assign(2, b"keep2");

    let perm = MockPerm {
        slots: vec![0, 1, 2],
    };

    // Exclude slot 1 from compaction
    bag.compact_with_permuter(&perm, Some(1));

    assert_eq!(bag.get(0), Some(b"keep".as_slice()));
    assert_eq!(bag.get(1), None); // Excluded
    assert_eq!(bag.get(2), Some(b"keep2".as_slice()));
}

// ========================================================================
//  Growth Tests
// ========================================================================

#[test]
fn test_growth() {
    let mut bag: SuffixBag<15> = SuffixBag::with_capacity(16);

    // Fill past capacity
    for i in 0..15 {
        bag.assign(i, b"12345678"); // 8 bytes each = 120 bytes total
    }

    assert!(bag.capacity() > 16);
    assert_eq!(bag.used(), 120);

    // All suffixes should still be accessible
    for i in 0..15 {
        assert_eq!(bag.get(i), Some(b"12345678".as_slice()));
    }
}

#[test]
fn test_long_suffix() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    let long_suffix: Vec<u8> = vec![b'x'; 1000];
    bag.assign(0, &long_suffix);

    assert_eq!(bag.get(0), Some(long_suffix.as_slice()));
}

// ========================================================================
//  Comparison Tests
// ========================================================================

#[test]
fn test_suffix_equals() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");

    assert!(bag.suffix_equals(0, b"hello"));
    assert!(!bag.suffix_equals(0, b"world"));
    assert!(!bag.suffix_equals(0, b"hell"));
    assert!(!bag.suffix_equals(1, b"hello")); // No suffix at slot 1
}

#[test]
fn test_suffix_compare() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");

    assert_eq!(
        bag.suffix_compare(0, b"hello"),
        Some(std::cmp::Ordering::Equal)
    );
    assert_eq!(
        bag.suffix_compare(0, b"hella"),
        Some(std::cmp::Ordering::Greater)
    );
    assert_eq!(
        bag.suffix_compare(0, b"hellz"),
        Some(std::cmp::Ordering::Less)
    );
    assert_eq!(bag.suffix_compare(1, b"hello"), None);
}

// ========================================================================
//  Clone Tests
// ========================================================================

#[test]
fn test_clone() {
    let mut bag: SuffixBag<15> = SuffixBag::new();
    bag.assign(0, b"hello");
    bag.assign(5, b"world");

    let cloned: SuffixBag<15> = bag.clone();

    assert_eq!(cloned.get(0), Some(b"hello".as_slice()));
    assert_eq!(cloned.get(5), Some(b"world".as_slice()));
    assert_eq!(cloned.count(), 2);
}

// ========================================================================
//  Width Variants Tests
// ========================================================================

#[test]
fn test_width_7() {
    let mut bag: SuffixBag<7> = SuffixBag::new();

    bag.assign(0, b"test0");
    bag.assign(6, b"test6");

    assert_eq!(bag.get(0), Some(b"test0".as_slice()));
    assert_eq!(bag.get(6), Some(b"test6".as_slice()));
    assert_eq!(bag.count(), 2);
}

#[test]
fn test_width_3() {
    let mut bag: SuffixBag<3> = SuffixBag::new();

    bag.assign(0, b"a");
    bag.assign(1, b"b");
    bag.assign(2, b"c");

    assert_eq!(bag.count(), 3);
}

// ========================================================================
//  In-Place Assignment Tests
// ========================================================================

#[test]
fn test_try_assign_in_place_fresh_bag() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Fresh bag has capacity, should succeed
    assert!(bag.try_assign_in_place(0, b"hello"));
    assert_eq!(bag.get(0), Some(b"hello".as_slice()));
}

#[test]
fn test_try_assign_in_place_reuse_slot() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Assign a longer suffix first
    bag.assign(0, b"hello world");
    let used_before: usize = bag.used();

    // Assign a shorter suffix - should reuse the slot
    assert!(bag.try_assign_in_place(0, b"hi"));
    assert_eq!(bag.get(0), Some(b"hi".as_slice()));

    // Used bytes should not increase (reused existing space)
    assert_eq!(bag.used(), used_before);
}

#[test]
fn test_try_assign_in_place_append() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Assign to slot 0
    assert!(bag.try_assign_in_place(0, b"first"));

    // Assign to slot 1 - should append
    assert!(bag.try_assign_in_place(1, b"second"));

    assert_eq!(bag.get(0), Some(b"first".as_slice()));
    assert_eq!(bag.get(1), Some(b"second".as_slice()));
}

#[test]
fn test_try_assign_in_place_fails_when_full() {
    // Create a bag with very small capacity
    let mut bag: SuffixBag<15> = SuffixBag::with_capacity(10);

    // First assignment should succeed
    assert!(bag.try_assign_in_place(0, b"12345"));

    // Second assignment that exceeds capacity should fail
    assert!(!bag.try_assign_in_place(1, b"678901234567890"));

    // First slot should still be valid
    assert_eq!(bag.get(0), Some(b"12345".as_slice()));
    // Second slot should not exist
    assert_eq!(bag.get(1), None);
}

#[test]
fn test_try_assign_in_place_same_length() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hello");
    let used_before: usize = bag.used();

    // Same length should reuse slot
    assert!(bag.try_assign_in_place(0, b"world"));
    assert_eq!(bag.get(0), Some(b"world".as_slice()));
    assert_eq!(bag.used(), used_before);
}

#[test]
fn test_try_assign_in_place_longer_suffix_needs_append() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    bag.assign(0, b"hi");
    let used_before: usize = bag.used();

    // Longer suffix can't reuse slot, needs append
    assert!(bag.try_assign_in_place(0, b"hello world"));
    assert_eq!(bag.get(0), Some(b"hello world".as_slice()));

    // Used bytes should increase
    assert!(bag.used() > used_before);
}

#[test]
fn test_try_assign_in_place_mixed_usage() {
    let mut bag: SuffixBag<15> = SuffixBag::new();

    // Fill several slots
    for i in 0..5 {
        assert!(bag.try_assign_in_place(i, b"test"));
    }

    // Reuse slot 2 with shorter suffix
    assert!(bag.try_assign_in_place(2, b"ab"));
    assert_eq!(bag.get(2), Some(b"ab".as_slice()));

    // Other slots unchanged
    assert_eq!(bag.get(0), Some(b"test".as_slice()));
    assert_eq!(bag.get(1), Some(b"test".as_slice()));
    assert_eq!(bag.get(3), Some(b"test".as_slice()));
    assert_eq!(bag.get(4), Some(b"test".as_slice()));
}

// ========================================================================
//  InlineSuffixBag Tests
// ========================================================================

#[test]
fn test_inline_new() {
    let bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    assert_eq!(bag.capacity(), 256);
    assert_eq!(bag.used(), 0);
    assert_eq!(bag.remaining(), 256);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_inline_default() {
    let bag: InlineSuffixBag<24, 256> = InlineSuffixBag::default();

    assert_eq!(bag.count(), 0);
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_inline_try_assign_basic() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    assert!(bag.try_assign(0, b"hello"));
    assert!(bag.try_assign(5, b"world"));

    assert_eq!(bag.get(0), Some(b"hello".as_slice()));
    assert_eq!(bag.get(5), Some(b"world".as_slice()));
    assert_eq!(bag.get(1), None);
    assert_eq!(bag.count(), 2);
    assert_eq!(bag.used(), 10);
}

#[test]
fn test_inline_try_assign_reuse_slot() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    // Assign longer suffix first
    assert!(bag.try_assign(0, b"hello world"));
    let used_before = bag.used();

    // Shorter suffix should reuse slot's space
    assert!(bag.try_assign(0, b"hi"));
    assert_eq!(bag.get(0), Some(b"hi".as_slice()));

    // Used bytes should not increase
    assert_eq!(bag.used(), used_before);
}

#[test]
fn test_inline_try_assign_append() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    assert!(bag.try_assign(0, b"first"));
    assert!(bag.try_assign(1, b"second"));

    assert_eq!(bag.used(), 11); // 5 + 6
    assert_eq!(bag.get(0), Some(b"first".as_slice()));
    assert_eq!(bag.get(1), Some(b"second".as_slice()));
}

#[test]
fn test_inline_try_assign_fails_when_full() {
    let mut bag: InlineSuffixBag<24, 32> = InlineSuffixBag::new();

    // Fill most of the capacity
    assert!(bag.try_assign(0, b"12345678901234567890")); // 20 bytes
    assert!(bag.try_assign(1, b"1234567890")); // 10 bytes, total 30

    // This should fail - only 2 bytes remaining
    assert!(!bag.try_assign(2, b"abc"));

    // First two slots should still be valid
    assert_eq!(bag.get(0), Some(b"12345678901234567890".as_slice()));
    assert_eq!(bag.get(1), Some(b"1234567890".as_slice()));
    assert_eq!(bag.get(2), None);
}

#[test]
fn test_inline_clear() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");
    assert!(bag.has_suffix(0));

    bag.clear(0);

    assert!(!bag.has_suffix(0));
    assert_eq!(bag.get(0), None);
    // Used bytes NOT reclaimed by clear
    assert_eq!(bag.used(), 5);
}

#[test]
fn test_inline_clear_all() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");
    bag.try_assign(1, b"world");

    bag.clear_all();

    assert_eq!(bag.count(), 0);
    assert_eq!(bag.used(), 0);
    assert!(!bag.has_suffix(0));
    assert!(!bag.has_suffix(1));
}

#[test]
fn test_inline_get_or_empty() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert_eq!(bag.get_or_empty(0), b"hello".as_slice());
    assert_eq!(bag.get_or_empty(1), b"".as_slice());
}

#[test]
fn test_inline_suffix_equals() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert!(bag.suffix_equals(0, b"hello"));
    assert!(!bag.suffix_equals(0, b"world"));
    assert!(!bag.suffix_equals(1, b"hello"));
}

#[test]
fn test_inline_suffix_compare() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert_eq!(
        bag.suffix_compare(0, b"hello"),
        Some(std::cmp::Ordering::Equal)
    );
    assert_eq!(
        bag.suffix_compare(0, b"hella"),
        Some(std::cmp::Ordering::Greater)
    );
    assert_eq!(bag.suffix_compare(1, b"hello"), None);
}

#[test]
fn test_inline_drain_to_external() {
    let mut bag: InlineSuffixBag<24, 64> = InlineSuffixBag::new();

    // Fill inline bag
    bag.try_assign(0, b"suffix0");
    bag.try_assign(1, b"suffix1");
    bag.try_assign(2, b"suffix2");

    // Create a permutation with 3 sorted entries (slots 0, 1, 2)
    let perm = Permuter24::make_sorted(3);

    // Drain to external with a new suffix for slot 3
    #[expect(
        clippy::expect_used,
        reason = "test code - panic on failure is intended"
    )]
    let external = bag
        .drain_to_external(&perm, 3, b"new_suffix")
        .expect("drain_to_external should succeed");

    // Inline bag slots should be cleared (count = 0)
    // Note: used() may still be non-zero since clear() doesn't compact
    assert_eq!(bag.count(), 0);

    // External bag should have all suffixes
    assert_eq!(external.get(0), Some(b"suffix0".as_slice()));
    assert_eq!(external.get(1), Some(b"suffix1".as_slice()));
    assert_eq!(external.get(2), Some(b"suffix2".as_slice()));
    assert_eq!(external.get(3), Some(b"new_suffix".as_slice()));
}

#[test]
fn test_inline_drain_replaces_slot() {
    let mut bag: InlineSuffixBag<24, 64> = InlineSuffixBag::new();

    bag.try_assign(0, b"old_suffix");
    bag.try_assign(1, b"keep_this");

    // Create a permutation with 2 sorted entries (slots 0, 1)
    let perm = Permuter24::make_sorted(2);

    // Replace slot 0's suffix during drain
    #[expect(
        clippy::expect_used,
        reason = "test code - panic on failure is intended"
    )]
    let external = bag
        .drain_to_external(&perm, 0, b"new_suffix")
        .expect("drain_to_external should succeed");

    // External should have new suffix for slot 0
    assert_eq!(external.get(0), Some(b"new_suffix".as_slice()));
    assert_eq!(external.get(1), Some(b"keep_this".as_slice()));
}

#[test]
fn test_inline_clone() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();
    bag.try_assign(0, b"hello");
    bag.try_assign(5, b"world");

    let cloned = bag.clone();

    assert_eq!(cloned.get(0), Some(b"hello".as_slice()));
    assert_eq!(cloned.get(5), Some(b"world".as_slice()));
    assert_eq!(cloned.count(), 2);
    assert_eq!(cloned.used(), bag.used());
}

#[test]
fn test_inline_empty_suffix() {
    let mut bag: InlineSuffixBag<24, 256> = InlineSuffixBag::new();

    // Empty suffix should work
    assert!(bag.try_assign(0, b""));
    assert_eq!(bag.get(0), Some(b"".as_slice()));
    assert!(bag.has_suffix(0));
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_inline_various_widths() {
    // Test with different WIDTH parameters
    let mut bag7: InlineSuffixBag<7, 128> = InlineSuffixBag::new();
    bag7.try_assign(0, b"test");
    bag7.try_assign(6, b"last");
    assert_eq!(bag7.get(0), Some(b"test".as_slice()));
    assert_eq!(bag7.get(6), Some(b"last".as_slice()));

    let mut bag15: InlineSuffixBag<15, 128> = InlineSuffixBag::new();
    bag15.try_assign(14, b"slot14");
    assert_eq!(bag15.get(14), Some(b"slot14".as_slice()));
}

#[test]
fn test_inline_size_calculation() {
    // Verify the size calculation from the doc comment
    // InlineSuffixBag<24, 256>: 24*4 + 2 + 256 = 354 bytes
    assert_eq!(
        std::mem::size_of::<InlineSuffixBag<24, 256>>(),
        24 * 4 + 2 + 256
    );

    // InlineSuffixBag<15, 128>: 15*4 + 2 + 128 = 190 bytes
    assert_eq!(
        std::mem::size_of::<InlineSuffixBag<15, 128>>(),
        15 * 4 + 2 + 128
    );
}
