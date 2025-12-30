use super::*;
use std::ptr as StdPtr;

// ------------------------------------------------------------------------
//  LeafValue<V> (Arc mode) Tests
// ------------------------------------------------------------------------

#[test]
fn arc_mode_into_output_allocates_once() {
    let output1: Arc<u64> = LeafValue::<u64>::into_output(42);
    let output2: Arc<u64> = Arc::clone(&output1);

    // Both point to same allocation
    assert_eq!(Arc::strong_count(&output1), 2);
    assert_eq!(*output1, 42);
    assert_eq!(*output2, 42);
}

#[test]
fn arc_mode_from_output_no_realloc() {
    let output: Arc<u64> = Arc::new(42);
    let slot: LeafValue<u64> = LeafValue::from_output(Arc::clone(&output));

    // Slot shares the same Arc (refcount = 2)
    assert!(slot.is_value());
    assert_eq!(Arc::strong_count(&output), 2);
}

#[test]
fn arc_mode_swap_output() {
    let mut slot: LeafValue<u64> = LeafValue::from_output(Arc::new(100));

    let old: Option<Arc<u64>> = slot.swap_output(Arc::new(200));
    assert_eq!(*old.unwrap(), 100);
    assert_eq!(*slot.try_get().unwrap(), 200);
}

#[test]
fn arc_mode_swap_output_empty_returns_none() {
    let mut slot: LeafValue<u64> = LeafValue::Empty;

    let old: Option<Arc<u64>> = slot.swap_output(Arc::new(42));
    assert!(old.is_none());
    assert_eq!(*slot.try_get().unwrap(), 42);
}

#[test]
fn arc_mode_predicates() {
    let empty: LeafValue<u64> = LeafValue::Empty;
    assert!(empty.is_empty());
    assert!(!empty.is_value());
    assert!(!empty.is_layer());

    let value: LeafValue<u64> = LeafValue::from_output(Arc::new(42));
    assert!(!value.is_empty());
    assert!(value.is_value());
    assert!(!value.is_layer());

    let mut layer: LeafValue<u64> = LeafValue::Empty;
    let mut dummy: u64 = 0;
    layer.set_layer(StdPtr::addr_of_mut!(dummy).cast());

    assert!(!layer.is_empty());
    assert!(!layer.is_value());
    assert!(layer.is_layer());
}

#[test]
fn arc_mode_take() {
    let mut slot: LeafValue<u64> = LeafValue::from_output(Arc::new(42));
    let taken: LeafValue<u64> = slot.take();

    assert!(slot.is_empty());
    assert!(taken.is_value());
    assert_eq!(*taken.try_get().unwrap(), 42);
}

// ------------------------------------------------------------------------
//  LeafValueIndex<V: Copy> (Inline mode) Tests
// ------------------------------------------------------------------------

#[test]
fn inline_mode_into_output_no_allocation() {
    // into_output is identity for Copy types - no allocation!
    let output: u64 = LeafValueIndex::<u64>::into_output(42);
    assert_eq!(output, 42);
}

#[test]
fn inline_mode_from_output() {
    let slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
    assert!(slot.is_value());
    assert_eq!(slot.try_get(), Some(42));
}

#[test]
fn inline_mode_swap_output() {
    let mut slot: LeafValueIndex<u64> = LeafValueIndex::from_output(100);

    let old: Option<u64> = slot.swap_output(200);
    assert_eq!(old, Some(100));
    assert_eq!(slot.try_get(), Some(200));
}

#[test]
fn inline_mode_predicates() {
    let empty: LeafValueIndex<u64> = LeafValueIndex::Empty;
    assert!(empty.is_empty());
    assert!(!empty.is_value());
    assert!(!empty.is_layer());

    let value: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
    assert!(!value.is_empty());
    assert!(value.is_value());
    assert!(!value.is_layer());
}

#[test]
fn inline_mode_is_copy() {
    let slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
    let copied: LeafValueIndex<u64> = slot; // Copy, not move

    assert_eq!(slot.try_get(), Some(42));
    assert_eq!(copied.try_get(), Some(42));
}

#[test]
fn inline_mode_take() {
    let mut slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
    let taken: LeafValueIndex<u64> = slot.take();

    assert!(slot.is_empty());
    assert_eq!(taken.try_get(), Some(42));
}

#[test]
fn inline_mode_layer() {
    let mut slot: LeafValueIndex<u64> = LeafValueIndex::Empty;
    let mut dummy: u64 = 0;
    let ptr: *mut u8 = StdPtr::addr_of_mut!(dummy).cast();

    slot.set_layer(ptr);

    assert!(slot.is_layer());
    assert_eq!(slot.try_layer(), Some(ptr));
    assert!(slot.try_get().is_none());
}

// ------------------------------------------------------------------------
//  Cross-mode comparison tests
// ------------------------------------------------------------------------

#[test]
fn both_modes_layer_works_same() {
    let mut arc_slot: LeafValue<u64> = LeafValue::Empty;
    let mut inline_slot: LeafValueIndex<u64> = LeafValueIndex::Empty;

    let mut dummy: u64 = 0;
    let ptr: *mut u8 = StdPtr::addr_of_mut!(dummy).cast();

    arc_slot.set_layer(ptr);
    inline_slot.set_layer(ptr);

    assert_eq!(arc_slot.try_layer(), Some(ptr));
    assert_eq!(inline_slot.try_layer(), Some(ptr));
}
