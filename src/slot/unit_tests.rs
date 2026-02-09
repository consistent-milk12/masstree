use super::{Arc, LeafValue, ValueSlot};
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
