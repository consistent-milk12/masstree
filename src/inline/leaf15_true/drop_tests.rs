use super::{LeafNode15TrueInline, SuffixBag, WIDTH_15};
use std::sync::atomic::Ordering as AtomicOrdering;

#[test]
fn inline_get_unchecked_returns_correct_value() {
    let leaf = LeafNode15TrueInline::<u64>::new_with_root(false);

    // Set a value at slot 0
    leaf.inline_set(0, 42u64);

    // Safe version should return the value
    assert_eq!(leaf.inline_get(0), Some(42u64));

    // Unchecked version should return the same value
    // SAFETY: We just set an inline value at slot 0
    let unchecked_value = unsafe { leaf.inline_get_unchecked(0) };
    assert_eq!(unchecked_value, 42u64);
}

#[test]
fn inline_get_unchecked_multiple_slots() {
    let leaf = LeafNode15TrueInline::<u64>::new_with_root(false);

    // Set values at multiple slots
    for slot in 0..5 {
        leaf.inline_set(slot, (slot * 100) as u64);
    }

    // Verify all slots via unchecked access
    for slot in 0..5 {
        // SAFETY: We just set inline values at these slots
        let value = unsafe { leaf.inline_get_unchecked(slot) };
        assert_eq!(value, (slot * 100) as u64);
    }
}

#[test]
fn inline_get_unchecked_after_swap() {
    let leaf = LeafNode15TrueInline::<u64>::new_with_root(false);

    // Set initial value
    leaf.inline_set(3, 100u64);

    // Swap to new value
    let old = leaf.inline_swap(3, 200u64);
    assert_eq!(old, 100u64);

    // Unchecked get should return swapped value
    // SAFETY: Slot 3 still contains an inline value after swap
    let value = unsafe { leaf.inline_get_unchecked(3) };
    assert_eq!(value, 200u64);
}

#[test]
fn external_suffix_bag_freed_on_drop() {
    // Craete a leaf
    let leaf = LeafNode15TrueInline::<u64>::new_with_root(false);

    // Manually set external_ksuf to simulate allocation
    let bag: Box<SuffixBag<WIDTH_15>> = Box::default();
    let ptr = Box::into_raw(bag);
    leaf.external_ksuf.store(ptr, AtomicOrdering::Relaxed);

    // Drop should free the bag
    drop(leaf);

    // Miri will catch if ptr is leaked.
}

#[test]
fn null_external_ksuf_is_safe() {
    // Ensure Drop handles null correctly
    let leaf = LeafNode15TrueInline::<u64>::new_with_root(false);

    assert!(leaf.external_ksuf_ptr().is_null());

    drop(leaf); // Should not panic or crash
}
