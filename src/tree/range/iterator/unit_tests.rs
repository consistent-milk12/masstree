use super::{RangeBound, ScanEntry};

#[test]
fn test_range_bound_contains() {
    // Unbounded contains everything
    assert!(RangeBound::Unbounded.contains(b"anything"));
    assert!(RangeBound::Unbounded.contains(b""));

    // Included: key <= bound
    let included = RangeBound::Included(b"middle");
    assert!(included.contains(b"aaa"));
    assert!(included.contains(b"middle"));
    assert!(!included.contains(b"zzz"));

    // Excluded: key < bound
    let excluded = RangeBound::Excluded(b"middle");
    assert!(excluded.contains(b"aaa"));
    assert!(!excluded.contains(b"middle"));
    assert!(!excluded.contains(b"zzz"));
}

#[test]
fn test_range_bound_to_start_params() {
    let (key, emit) = RangeBound::Unbounded.to_start_params();
    assert_eq!(key, b"");
    assert!(emit);

    let (key, emit) = RangeBound::Included(b"start").to_start_params();
    assert_eq!(key, b"start");
    assert!(emit);

    let (key, emit) = RangeBound::Excluded(b"start").to_start_params();
    assert_eq!(key, b"start");
    assert!(!emit);
}

#[test]
fn test_range_bound_from_std_bound() {
    use std::ops::Bound;

    let rb: RangeBound = Bound::Unbounded.into();
    assert!(matches!(rb, RangeBound::Unbounded));

    let rb: RangeBound = Bound::Included(b"key".as_slice()).into();
    assert!(matches!(rb, RangeBound::Included(k) if k == b"key"));

    let rb: RangeBound = Bound::Excluded(b"key".as_slice()).into();
    assert!(matches!(rb, RangeBound::Excluded(k) if k == b"key"));
}

#[test]
fn test_scan_entry() {
    let entry = ScanEntry::new(b"key".to_vec(), 42u64);

    assert_eq!(entry.key(), b"key");
    assert_eq!(*entry.value(), 42);

    let (key, value) = entry.into_parts();
    assert_eq!(key, b"key");
    assert_eq!(value, 42);
}

#[test]
fn test_range_bound_is_unbounded() {
    assert!(RangeBound::Unbounded.is_unbounded());
    assert!(!RangeBound::Included(b"key").is_unbounded());
    assert!(!RangeBound::Excluded(b"key").is_unbounded());
}

#[test]
fn test_range_bound_key() {
    assert!(RangeBound::Unbounded.key().is_none());
    assert_eq!(RangeBound::Included(b"key").key(), Some(b"key".as_slice()));
    assert_eq!(RangeBound::Excluded(b"key").key(), Some(b"key".as_slice()));
}
