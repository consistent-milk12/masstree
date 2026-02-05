use std::cmp::Ordering;
use std::iter as StdIter;
use std::mem as StdMem;

use super::{InlineSuffixBag, PermutationProvider, SuffixBag, SuffixSidecar, INITIAL_CAPACITY};
use crate::permuter::Permuter15;
// Note: AllocError removed - allocations are now infallible

// ========================================================================
//  Basic Tests
// ========================================================================

#[test]
fn test_new_suffix_bag() {
    let bag: SuffixBag = SuffixBag::new();

    assert_eq!(bag.count(), 0);
    assert!(bag.capacity() >= INITIAL_CAPACITY);
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_with_capacity() {
    let bag: SuffixBag = SuffixBag::with_capacity(256);

    assert!(bag.capacity() >= 256);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_default() {
    let bag: SuffixBag = SuffixBag::default();

    assert_eq!(bag.count(), 0);
}

// ========================================================================
//  Assign and Get Tests
// ========================================================================

#[test]
fn test_assign_and_get() {
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"");

    assert_eq!(bag.get(0), Some(b"".as_slice()));
    assert!(bag.has_suffix(0));
}

#[test]
fn test_get_or_empty() {
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"hello");

    assert_eq!(bag.get_or_empty(0), b"hello".as_slice());
    assert_eq!(bag.get_or_empty(1), b"".as_slice());
}

#[test]
fn test_overwrite_suffix() {
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"hello");
    assert!(bag.has_suffix(0));

    bag.clear(0);

    assert!(!bag.has_suffix(0));
    assert_eq!(bag.get(0), None);
}

#[test]
fn test_clear_already_empty() {
    let mut bag: SuffixBag = SuffixBag::new();

    // Clearing an empty slot should not panic
    bag.clear(0);

    assert!(!bag.has_suffix(0));
}

// ========================================================================
//  Compact Tests
// ========================================================================

#[test]
fn test_compact() {
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::new();

    // Compact with no active slots should work
    let reclaimed: usize = bag.compact(StdIter::empty());

    assert_eq!(reclaimed, 0);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_compact_all() {
    let mut bag: SuffixBag = SuffixBag::new();

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

    let mut bag: SuffixBag = SuffixBag::new();
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

    let mut bag: SuffixBag = SuffixBag::new();
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
    let mut bag: SuffixBag = SuffixBag::with_capacity(16);

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
    let mut bag: SuffixBag = SuffixBag::new();

    let long_suffix: Vec<u8> = vec![b'x'; 1000];
    bag.assign(0, &long_suffix);

    assert_eq!(bag.get(0), Some(long_suffix.as_slice()));
}

// ========================================================================
//  Comparison Tests
// ========================================================================

#[test]
fn test_suffix_equals() {
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"hello");

    assert!(bag.suffix_equals(0, b"hello"));
    assert!(!bag.suffix_equals(0, b"world"));
    assert!(!bag.suffix_equals(0, b"hell"));
    assert!(!bag.suffix_equals(1, b"hello")); // No suffix at slot 1
}

#[test]
fn test_suffix_compare() {
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"hello");

    assert_eq!(bag.suffix_compare(0, b"hello"), Some(Ordering::Equal));
    assert_eq!(bag.suffix_compare(0, b"hella"), Some(Ordering::Greater));
    assert_eq!(bag.suffix_compare(0, b"hellz"), Some(Ordering::Less));
    assert_eq!(bag.suffix_compare(1, b"hello"), None);
}

// ========================================================================
//  Clone Tests
// ========================================================================

#[test]
fn test_clone() {
    let mut bag: SuffixBag = SuffixBag::new();
    bag.assign(0, b"hello");
    bag.assign(5, b"world");

    let cloned: SuffixBag = bag.clone();

    assert_eq!(cloned.get(0), Some(b"hello".as_slice()));
    assert_eq!(cloned.get(5), Some(b"world".as_slice()));
    assert_eq!(cloned.count(), 2);
}

// ========================================================================
//  In-Place Assignment Tests
// ========================================================================

#[test]
fn test_try_assign_in_place_fresh_bag() {
    let mut bag: SuffixBag = SuffixBag::new();

    // Fresh bag has capacity, should succeed
    assert!(bag.try_assign_in_place(0, b"hello"));
    assert_eq!(bag.get(0), Some(b"hello".as_slice()));
}

#[test]
fn test_try_assign_in_place_reuse_slot() {
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::with_capacity(10);

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
    let mut bag: SuffixBag = SuffixBag::new();

    bag.assign(0, b"hello");
    let used_before: usize = bag.used();

    // Same length should reuse slot
    assert!(bag.try_assign_in_place(0, b"world"));
    assert_eq!(bag.get(0), Some(b"world".as_slice()));
    assert_eq!(bag.used(), used_before);
}

#[test]
fn test_try_assign_in_place_longer_suffix_needs_append() {
    let mut bag: SuffixBag = SuffixBag::new();

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
    let mut bag: SuffixBag = SuffixBag::new();

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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    #[cfg(not(feature = "large-suffix-capacity"))]
    {
        assert_eq!(bag.capacity(), 256);
        assert_eq!(bag.remaining(), 256);
    }

    #[cfg(feature = "large-suffix-capacity")]
    {
        assert_eq!(bag.capacity(), 512);
        assert_eq!(bag.remaining(), 512);
    }

    assert_eq!(bag.used(), 0);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_inline_default() {
    let bag: InlineSuffixBag = InlineSuffixBag::default();

    assert_eq!(bag.count(), 0);
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_inline_try_assign_basic() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    assert!(bag.try_assign(0, b"first"));
    assert!(bag.try_assign(1, b"second"));

    assert_eq!(bag.used(), 11); // 5 + 6
    assert_eq!(bag.get(0), Some(b"first".as_slice()));
    assert_eq!(bag.get(1), Some(b"second".as_slice()));
}

#[test]
fn test_inline_clear() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert_eq!(bag.get_or_empty(0), b"hello".as_slice());
    assert_eq!(bag.get_or_empty(1), b"".as_slice());
}

#[test]
fn test_inline_suffix_equals() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert!(bag.suffix_equals(0, b"hello"));
    assert!(!bag.suffix_equals(0, b"world"));
    assert!(!bag.suffix_equals(1, b"hello"));
}

#[test]
fn test_inline_suffix_compare() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");

    assert_eq!(bag.suffix_compare(0, b"hello"), Some(Ordering::Equal));
    assert_eq!(bag.suffix_compare(0, b"hella"), Some(Ordering::Greater));
    assert_eq!(bag.suffix_compare(1, b"hello"), None);
}

#[test]
fn test_inline_drain_to_external() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    // Fill inline bag
    bag.try_assign(0, b"suffix0");
    bag.try_assign(1, b"suffix1");
    bag.try_assign(2, b"suffix2");

    // Create a permutation with 3 sorted entries (slots 0, 1, 2)
    let perm = Permuter15::make_sorted(3);

    // Drain to external with a new suffix for slot 3
    let external = bag.drain_to_external(&perm, 3, b"new_suffix");

    // Inline bag state is PRESERVED (orphaned inline optimization).
    // This is intentional - clearing inline before external publication
    // creates a race where readers see neither inline nor external data.
    // The inline data becomes "orphaned" but is still valid for readers.
    assert_eq!(bag.count(), 3);

    // External bag should have all suffixes
    assert_eq!(external.get(0), Some(b"suffix0".as_slice()));
    assert_eq!(external.get(1), Some(b"suffix1".as_slice()));
    assert_eq!(external.get(2), Some(b"suffix2".as_slice()));
    assert_eq!(external.get(3), Some(b"new_suffix".as_slice()));
}

#[test]
fn test_inline_drain_replaces_slot() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"old_suffix");
    bag.try_assign(1, b"keep_this");

    // Create a permutation with 2 sorted entries (slots 0, 1)
    let perm = Permuter15::make_sorted(2);

    // Replace slot 0's suffix during drain
    let external = bag.drain_to_external(&perm, 0, b"new_suffix");

    // External should have new suffix for slot 0
    assert_eq!(external.get(0), Some(b"new_suffix".as_slice()));
    assert_eq!(external.get(1), Some(b"keep_this".as_slice()));
}

#[test]
fn test_inline_drain_to_external_init() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    // Simulate split initialization: slots 0, 1, 2 filled sequentially
    bag.try_assign(0, b"suffix_0");
    bag.try_assign(1, b"suffix_1");
    bag.try_assign(2, b"suffix_2");

    // Drain to external for slot 3 (slots 0..3 are filled)
    let external = bag.drain_to_external_init(3, b"new_suffix_3");

    // Inline bag state is PRESERVED (orphaned inline optimization).
    // See drain_to_external test for rationale.
    assert_eq!(bag.count(), 3);

    // External should have all suffixes
    assert_eq!(external.get(0), Some(b"suffix_0".as_slice()));
    assert_eq!(external.get(1), Some(b"suffix_1".as_slice()));
    assert_eq!(external.get(2), Some(b"suffix_2".as_slice()));
    assert_eq!(external.get(3), Some(b"new_suffix_3".as_slice()));
}

#[test]
fn test_inline_drain_to_external_init_replaces_slot() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    // Simulate: slots 0, 1 filled, now replacing slot 1
    bag.try_assign(0, b"keep_this");
    bag.try_assign(1, b"old_suffix");

    // Drain with new_slot=1 means slots 0..1 are "filled" (just slot 0)
    // and we're assigning to slot 1
    let external = bag.drain_to_external_init(1, b"new_suffix");

    // External should have slot 0 preserved and slot 1 with new suffix
    assert_eq!(external.get(0), Some(b"keep_this".as_slice()));
    assert_eq!(external.get(1), Some(b"new_suffix".as_slice()));
}

#[test]
fn test_inline_clone() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();
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
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    // Empty suffix should work
    assert!(bag.try_assign(0, b""));
    assert_eq!(bag.get(0), Some(b"".as_slice()));
    assert!(bag.has_suffix(0));
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_inline_size_calculation() {
    // InlineSuffixBag layout: 15 * 4 (slots) + 2 (size) + 1 (count) + 1 (pad) + CAPACITY
    // Default (256):  64 + 256 = 320 bytes
    // Large (512):    64 + 512 = 576 bytes
    #[cfg(not(feature = "large-suffix-capacity"))]
    assert_eq!(StdMem::size_of::<InlineSuffixBag>(), 320);

    #[cfg(feature = "large-suffix-capacity")]
    assert_eq!(StdMem::size_of::<InlineSuffixBag>(), 576);
}

// ============================================================================
//  Tests for suffix_count maintenance
// ============================================================================

#[test]
fn test_suffix_count_assign_clear() {
    let mut bag: SuffixBag = SuffixBag::new();
    assert_eq!(bag.count(), 0);

    bag.assign(0, b"hello");
    assert_eq!(bag.count(), 1);

    bag.assign(1, b"world");
    assert_eq!(bag.count(), 2);

    // Reassign to same slot, count should stay unchanged
    bag.assign(0, b"hi");
    assert_eq!(bag.count(), 2);

    bag.clear(0);
    assert_eq!(bag.count(), 1);

    bag.clear(1);
    assert_eq!(bag.count(), 0);

    // Clear already empty slot - count stays 0
    bag.clear(0);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_suffix_count_try_assign_in_place() {
    let mut bag: SuffixBag = SuffixBag::new();
    assert_eq!(bag.count(), 0);

    assert!(bag.try_assign_in_place(0, b"hello"));
    assert_eq!(bag.count(), 1);

    assert!(bag.try_assign_in_place(1, b"world"));
    assert_eq!(bag.count(), 2);

    // Reassign to same slot (shorter suffix reuses space) - count unchanged
    assert!(bag.try_assign_in_place(0, b"hi"));
    assert_eq!(bag.count(), 2);

    // Reassign to same slot (longer suffix appends) - count unchanged
    assert!(bag.try_assign_in_place(1, b"hello world"));
    assert_eq!(bag.count(), 2);
}

#[test]
fn test_suffix_count_after_compact() {
    let mut bag: SuffixBag = SuffixBag::new();
    bag.assign(0, b"hello");
    bag.assign(1, b"world");
    bag.assign(2, b"test");
    assert_eq!(bag.count(), 3);

    bag.clear(1);
    assert_eq!(bag.count(), 2);

    // Compact keeping only slots 0 and 2
    bag.compact([0, 2].into_iter());
    assert_eq!(bag.count(), 2);

    // Compact keeping only slot 0
    bag.compact(StdIter::once(0));
    assert_eq!(bag.count(), 1);

    // Compact with empty iterator
    bag.compact(StdIter::empty());
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_suffix_count_clone() {
    let mut bag: SuffixBag = SuffixBag::new();
    bag.assign(0, b"hello");
    bag.assign(1, b"world");
    bag.clear(1); // Clear but don't compact - garbage in data

    let cloned = bag.clone();
    assert_eq!(cloned.count(), 1);
    assert_eq!(cloned.get(0), Some(b"hello".as_slice()));
    assert_eq!(cloned.get(1), None);

    // Clone should have compacted - used bytes should be minimal
    assert_eq!(cloned.used(), 5); // Only "hello"
}

#[test]
fn test_inline_suffix_count_try_assign() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();
    assert_eq!(bag.count(), 0);

    assert!(bag.try_assign(0, b"hello"));
    assert_eq!(bag.count(), 1);

    assert!(bag.try_assign(1, b"world"));
    assert_eq!(bag.count(), 2);

    // Reassign shorter suffix (reuses space) - count unchanged
    assert!(bag.try_assign(0, b"hi"));
    assert_eq!(bag.count(), 2);
}

#[test]
fn test_inline_suffix_count_clear() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");
    bag.try_assign(1, b"world");
    assert_eq!(bag.count(), 2);

    bag.clear(0);
    assert_eq!(bag.count(), 1);

    bag.clear(1);
    assert_eq!(bag.count(), 0);

    // Clear already empty slot
    bag.clear(0);
    assert_eq!(bag.count(), 0);
}

#[test]
fn test_inline_suffix_count_clear_all() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    bag.try_assign(0, b"hello");
    bag.try_assign(1, b"world");
    assert_eq!(bag.count(), 2);

    bag.clear_all();
    assert_eq!(bag.count(), 0);
}

// ============================================================================
//  Tests for try_assign with compaction
// ============================================================================

#[test]
fn test_try_assign_compacts_before_grow() {
    // Create a bag with INITIAL_CAPACITY to avoid capacity changes from compaction
    let mut bag: SuffixBag = SuffixBag::with_capacity(INITIAL_CAPACITY);

    // Fill most of it
    for i in 0..12 {
        bag.assign(i, b"1234567890"); // 10 bytes each = 120 bytes
    }

    // Clear half the slots, creating 60 bytes of garbage
    for i in (0..12).step_by(2) {
        bag.clear(i);
    }
    // Now: 60 bytes active (slots 1,3,5,7,9,11), 60 bytes garbage

    let old_used: usize = bag.used();
    assert_eq!(old_used, 120); // All data still in buffer

    // Try to assign - should compact first (reclaiming 60 bytes)
    // Returns false if buffer grew (which it shouldn't since compaction freed space)
    let _did_not_grow: bool = bag.try_assign(12, b"newdata!!!");

    // After compaction + new assignment: 60 + 10 = 70 bytes
    assert!(bag.used() < old_used); // Compaction happened
    assert_eq!(bag.count(), 7); // slots 1,3,5,7,9,11,12

    // Verify data integrity
    assert_eq!(bag.get(12), Some(b"newdata!!!".as_slice()));
    assert_eq!(bag.get(1), Some(b"1234567890".as_slice()));
}

#[test]
fn test_try_assign_grows_when_compact_insufficient() {
    let mut bag: SuffixBag = SuffixBag::with_capacity(16);

    // Fill completely with no garbage
    bag.assign(0, b"12345678"); // 8 bytes
    bag.assign(1, b"12345678"); // 8 bytes = 16 total, no garbage

    let old_capacity = bag.capacity();

    // Try to assign - compaction won't help, must grow
    // Returns false because buffer grew
    let did_grow: bool = !bag.try_assign(2, b"newdata!");
    assert!(did_grow);

    // Capacity should have grown
    assert!(bag.capacity() > old_capacity);
    assert_eq!(bag.count(), 3);
}

// ============================================================================
//  Tests for compacting clone
// ============================================================================

#[test]
fn test_clone_compacts_garbage() {
    let mut bag: SuffixBag = SuffixBag::new();

    // Add and remove several suffixes to create garbage
    bag.assign(0, b"aaaaaaaaaa"); // 10 bytes
    bag.assign(1, b"bbbbbbbbbb"); // 10 bytes
    bag.assign(2, b"cccccccccc"); // 10 bytes
    bag.assign(0, b"ddddd"); // 5 bytes (old 10 bytes now garbage)
    bag.clear(1); // 10 more bytes of garbage

    // Original has garbage
    let original_used: usize = bag.used();
    assert!(original_used >= 35); // At least all the data written

    let cloned: SuffixBag = bag.clone();

    // Clone should only have active data
    assert_eq!(cloned.used(), 15); // 5 (slot 0) + 10 (slot 2)
    assert_eq!(cloned.count(), 2);
    assert_eq!(cloned.get(0), Some(b"ddddd".as_slice()));
    assert_eq!(cloned.get(1), None);
    assert_eq!(cloned.get(2), Some(b"cccccccccc".as_slice()));
}

#[test]
fn test_clone_empty_bag() {
    let bag: SuffixBag = SuffixBag::new();
    let cloned: SuffixBag = bag;

    assert_eq!(cloned.count(), 0);
    assert_eq!(cloned.used(), 0);
}

// ============================================================================
//  SuffixSidecar Tests (Miri-friendly - no iterations, tests Drop correctness)
// ============================================================================

/// Test sidecar creation and drop with only inline suffixes.
/// Miri will detect any memory leaks or invalid drops.
#[test]
fn test_sidecar_drop_inline_only() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // Add a few inline suffixes
    sidecar.inline.try_assign(0, b"hello");
    sidecar.inline.try_assign(1, b"world");
    sidecar.inline.try_assign(2, b"test");

    assert!(sidecar.has_suffix(0));
    assert!(sidecar.has_suffix(1));
    assert!(sidecar.has_suffix(2));
    assert!(!sidecar.has_suffix(3));

    // Verify data before drop
    assert_eq!(sidecar.get(0), Some(b"hello".as_slice()));
    assert_eq!(sidecar.get(1), Some(b"world".as_slice()));

    // Drop happens here - Miri verifies no leaks
}

/// Test sidecar creation and drop with external overflow allocation.
/// This exercises the Drop implementation that frees the external [`SuffixBag`].
/// Miri will detect any memory leaks or double-frees.
#[test]
fn test_sidecar_drop_with_external() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // Force external allocation (infallible - aborts on OOM)
    // SAFETY: Test-only, no concurrent access
    let external_ptr = unsafe { sidecar.ensure_external() };

    // Write to external bag
    // SAFETY: We have exclusive access in this test
    unsafe {
        (*external_ptr).assign(0, b"external_suffix_0");
        (*external_ptr).assign(1, b"external_suffix_1");
    }

    // Verify external is allocated
    assert!(!sidecar
        .external
        .load(std::sync::atomic::Ordering::Relaxed)
        .is_null());

    // Verify we can read from external
    assert_eq!(sidecar.get(0), Some(b"external_suffix_0".as_slice()));
    assert_eq!(sidecar.get(1), Some(b"external_suffix_1".as_slice()));

    // Drop happens here - Miri verifies external bag is freed without leaks
}

/// Test sidecar with both inline and external data.
/// Exercises the full Drop path.
#[test]
fn test_sidecar_drop_mixed_inline_external() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // Add inline suffixes
    sidecar.inline.try_assign(0, b"inline_0");
    sidecar.inline.try_assign(1, b"inline_1");

    // Force external allocation (infallible - aborts on OOM)
    // SAFETY: Test-only, no concurrent access
    let external_ptr = unsafe { sidecar.ensure_external() };

    // Write to external bag for different slots
    // SAFETY: We have exclusive access in this test
    unsafe {
        (*external_ptr).assign(5, b"external_5");
        (*external_ptr).assign(10, b"external_10");
    }

    // Verify inline takes precedence for slots 0, 1
    assert_eq!(sidecar.get(0), Some(b"inline_0".as_slice()));
    assert_eq!(sidecar.get(1), Some(b"inline_1".as_slice()));

    // Verify external is used for slots 5, 10
    assert_eq!(sidecar.get(5), Some(b"external_5".as_slice()));
    assert_eq!(sidecar.get(10), Some(b"external_10".as_slice()));

    // Drop happens here - Miri verifies both inline and external cleanup
}

/// Test sidecar default and new are equivalent.
#[test]
fn test_sidecar_default() {
    let s1: SuffixSidecar = SuffixSidecar::new();
    let s2: SuffixSidecar = SuffixSidecar::default();

    assert!(s1
        .external
        .load(std::sync::atomic::Ordering::Relaxed)
        .is_null());
    assert!(s2
        .external
        .load(std::sync::atomic::Ordering::Relaxed)
        .is_null());
    assert_eq!(s1.inline.count(), 0);
    assert_eq!(s2.inline.count(), 0);
}

/// Test `ensure_external` is idempotent (returns same pointer on repeated calls).
#[test]
fn test_sidecar_ensure_external_idempotent() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // SAFETY: Test-only, no concurrent access. Allocation is infallible.
    let ptr1 = unsafe { sidecar.ensure_external() };
    let ptr2 = unsafe { sidecar.ensure_external() };
    let ptr3 = unsafe { sidecar.ensure_external() };

    // All calls should return the same pointer
    assert_eq!(ptr1, ptr2);
    assert_eq!(ptr2, ptr3);

    // Only one allocation happened
    assert!(!ptr1.is_null());

    // Drop frees exactly one allocation - Miri verifies no double-free
}
