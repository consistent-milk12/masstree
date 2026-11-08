use std::cmp::Ordering;
use std::mem as StdMem;

use super::{INITIAL_CAPACITY, InlineSuffixBag, SuffixBag, SuffixSidecar};
use crate::permuter::Permuter15;
// Note: AllocError removed - allocations are now infallible

#[test]
fn raw_buffer_header_has_no_duplicate_owner() {
    static_assertions::assert_not_impl_any!(SuffixBag: Sync);
    static_assertions::assert_impl_all!(SuffixBag: Send);
    assert_eq!(
        StdMem::size_of::<SuffixBag>(),
        15 * 8 + 3 * StdMem::size_of::<usize>()
    );
}

#[test]
fn append_preserves_borrowed_bytes_and_fails_without_growth() {
    let bag = SuffixBag::with_capacity(10);
    assert!(bag.try_assign_append_only(0, b"hello"));
    let old = bag.get(0).unwrap();
    assert!(bag.try_assign_append_only(0, b"world"));
    assert_eq!(old, b"hello");
    assert_eq!(bag.get(0), Some(b"world".as_slice()));
    assert!(!bag.try_assign_append_only(1, b"!"));
    assert_eq!(bag.used(), 10);
    assert_eq!(bag.get(1), None);
}

#[test]
fn zero_capacity_growth_and_failed_reserve_preserve_ownership() {
    let mut bag = SuffixBag::from_vec(Vec::new());
    bag.assign(5, b"");
    assert_eq!(bag.get(5), Some(b"".as_slice()));
    bag.assign(0, b"abc");
    bag.reserve(1_024);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bag.reserve(usize::MAX)));
    assert!(failed.is_err());
    bag.assign(6, b"zz");
    assert_eq!(bag.get(5), Some(b"".as_slice()));
    assert_eq!(bag.get(0), Some(b"abc".as_slice()));
    assert_eq!(bag.get(6), Some(b"zz".as_slice()));
}

#[test]
fn concurrent_external_append_keeps_reader_slices_valid() {
    use std::sync::Barrier;

    struct Shared(super::SuffixBagCell);
    // SAFETY: This test has one append-only writer, a fixed-capacity bag, and
    // readers of immutable published bytes. Scoped threads join before drop.
    unsafe impl Sync for Shared {}

    let shared = Shared(super::SuffixBagCell::from_bag(SuffixBag::with_capacity(
        4_096,
    )));
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let shared = &shared;
        let barrier = &barrier;
        scope.spawn(move || {
            // SAFETY: This is the only writer, and it never grows the buffer.
            let bag = unsafe { shared.0.as_writer() };
            for byte in 1..=32 {
                barrier.wait();
                assert!(bag.try_assign_append_only(0, &[byte; 64]));
                barrier.wait();
            }
        });
        // SAFETY: The cell is alive until the writer joins; metadata is atomic
        // and published byte ranges are never overwritten.
        let bag = unsafe { shared.0.as_ref() };
        for byte in 1..=32 {
            let previous = bag.get(0);
            barrier.wait();
            if let Some(bytes) = bag.get(0) {
                assert!(bytes.iter().all(|&b| b == bytes[0]));
            }
            if let Some(bytes) = previous {
                assert_eq!(bytes, &[byte - 1; 64]);
            }
            barrier.wait();
            assert_eq!(bag.get(0), Some([byte; 64].as_slice()));
        }
    });
}

// ========================================================================
//  Basic Tests
// ========================================================================

#[test]
fn test_new_suffix_bag() {
    let bag: SuffixBag = SuffixBag::new();

    assert_eq!((0..15).filter(|&slot| bag.has_suffix(slot)).count(), 0);
    assert!(bag.capacity() >= INITIAL_CAPACITY);
    assert_eq!(bag.used(), 0);
}

#[test]
fn test_with_capacity() {
    let bag: SuffixBag = SuffixBag::with_capacity(256);

    assert!(bag.capacity() >= 256);
    assert_eq!((0..15).filter(|&slot| bag.has_suffix(slot)).count(), 0);
}

#[test]
fn test_default() {
    let bag: SuffixBag = SuffixBag::default();

    assert_eq!((0..15).filter(|&slot| bag.has_suffix(slot)).count(), 0);
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
    assert_eq!((0..15).filter(|&slot| bag.has_suffix(slot)).count(), 3);
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
//  InlineSuffixBag Tests
// ========================================================================

#[test]
fn test_inline_new() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    #[cfg(not(feature = "small-suffix-capacity"))]
    {
        assert_eq!(bag.capacity(), 512);
        assert_eq!(bag.remaining(), 512);
    }

    #[cfg(feature = "small-suffix-capacity")]
    {
        assert_eq!(bag.capacity(), 256);
        assert_eq!(bag.remaining(), 256);
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
    #[cfg(not(feature = "small-suffix-capacity"))]
    assert_eq!(StdMem::size_of::<InlineSuffixBag>(), 576);

    #[cfg(feature = "small-suffix-capacity")]
    assert_eq!(StdMem::size_of::<InlineSuffixBag>(), 320);
}

// ============================================================================
//  Tests for suffix_count maintenance
// ============================================================================

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

#[test]
fn test_inline_drain_to_external_with_vec_reserves_enough() {
    let bag: InlineSuffixBag = InlineSuffixBag::new();

    // Fill a few slots.
    assert!(bag.try_assign(0, b"suffix0"));
    assert!(bag.try_assign(1, b"suffix1_long"));
    assert!(bag.try_assign(2, b"suffix2"));

    let perm: Permuter15 = Permuter15::make_sorted(3);

    let new_slot: usize = 3;
    let new_suffix: &[u8] = b"new_suffix";

    // Required capacity = sum(existing) + new.
    let required_capacity: usize =
        b"suffix0".len() + b"suffix1_long".len() + b"suffix2".len() + new_suffix.len();

    // Deliberately pick a non-zero but insufficient capacity to catch the
    // "reserve vs capacity" bug via the debug_assert in the implementation.
    let buffer: Vec<u8> = Vec::with_capacity(16);

    let external: SuffixBag = bag.drain_to_external_with_vec(&perm, new_slot, new_suffix, buffer);

    assert!(external.capacity() >= required_capacity);
    assert_eq!(external.get(0), Some(b"suffix0".as_slice()));
    assert_eq!(external.get(1), Some(b"suffix1_long".as_slice()));
    assert_eq!(external.get(2), Some(b"suffix2".as_slice()));
    assert_eq!(external.get(3), Some(new_suffix));
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
/// This exercises the Drop implementation that frees the external `SuffixBagCell`.
/// Miri will detect any memory leaks or double-frees.
#[test]
fn test_sidecar_drop_with_external() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // Force external allocation (infallible - aborts on OOM)
    // SAFETY: Test-only, no concurrent access
    let external_bag = unsafe { sidecar.ensure_external() };

    // Write to external bag (we have exclusive access in this test)
    external_bag.assign(0, b"external_suffix_0");
    external_bag.assign(1, b"external_suffix_1");

    // Verify external is allocated
    assert!(
        !sidecar
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            .is_null()
    );

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
    let external_bag = unsafe { sidecar.ensure_external() };

    // Write to external bag for different slots (we have exclusive access)
    external_bag.assign(5, b"external_5");
    external_bag.assign(10, b"external_10");

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

    assert!(
        s1.external
            .load(std::sync::atomic::Ordering::Relaxed)
            .is_null()
    );
    assert!(
        s2.external
            .load(std::sync::atomic::Ordering::Relaxed)
            .is_null()
    );
    assert_eq!(s1.inline.count(), 0);
    assert_eq!(s2.inline.count(), 0);
}

/// Test `ensure_external` is idempotent (returns same backing allocation).
#[test]
fn test_sidecar_ensure_external_idempotent() {
    let sidecar: SuffixSidecar = SuffixSidecar::new();

    // SAFETY: Test-only, no concurrent access. Allocation is infallible.
    let ptr1: *const SuffixBag = unsafe { sidecar.ensure_external() };
    let ptr2: *const SuffixBag = unsafe { sidecar.ensure_external() };
    let ptr3: *const SuffixBag = unsafe { sidecar.ensure_external() };

    // All calls should return references into the same allocation
    assert_eq!(ptr1, ptr2);
    assert_eq!(ptr2, ptr3);
    assert!(!ptr1.is_null());

    // Drop frees exactly one allocation - Miri verifies no double-free
}
