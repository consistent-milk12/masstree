use super::*;

#[test]
fn sentinel_is_consistent() {
    let ptr1: *mut u8 = InlineSentinel::inline_sentinel_ptr();
    let ptr2: *mut u8 = InlineSentinel::inline_sentinel_ptr();

    assert!(StdPtr::eq(ptr1, ptr2));
}

#[test]
fn sentinel_is_non_null() {
    let ptr: *mut u8 = InlineSentinel::inline_sentinel_ptr();

    assert!(!ptr.is_null());
}

#[test]
fn sentinel_detection_works() {
    let sentinel: *mut u8 = InlineSentinel::inline_sentinel_ptr();
    assert!(InlineSentinel::is_inline_sentinel(sentinel));

    let non_sentinel: &u8 = &42u8;
    assert!(!InlineSentinel::is_inline_sentinel(non_sentinel));

    let null: *const u8 = StdPtr::null();
    assert!(!InlineSentinel::is_inline_sentinel(null));
}
