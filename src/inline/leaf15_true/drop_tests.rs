use super::*;
use std::sync::atomic::Ordering as AtomicOrdering;

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
