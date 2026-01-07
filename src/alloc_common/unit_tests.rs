use super::{AllocKind, BoxAllocator, GenericAllocator};
use core::ptr as CorePtr;

#[test]
fn test_try_alloc_success() {
    let ptr: *mut u64 = GenericAllocator::try_alloc::<u64>().expect("allocation should succeed");
    assert!(!ptr.is_null());

    // Clean up
    unsafe {
        CorePtr::write(ptr, 42);
        GenericAllocator::dealloc(ptr);
    }
}

#[test]
fn test_try_alloc_zeroed_success() {
    let ptr: *mut [u8; 64] =
        GenericAllocator::try_alloc_zeroed::<[u8; 64]>().expect("allocation should succeed");

    // Verify zeroed
    unsafe {
        let arr: &[u8; 64] = &*ptr;
        assert!(arr.iter().all(|b: &u8| *b == 0));

        GenericAllocator::dealloc(ptr);
    }
}

#[test]
fn test_try_box_success() {
    let boxed: Box<u64> = BoxAllocator::try_box(42u64).expect("allocation should succeed");
    assert_eq!(*boxed, 42);
}

#[test]
fn test_try_box_struct() {
    #[derive(Debug, PartialEq)]
    struct TestStruct {
        a: u64,
        b: String,
    }

    let value: TestStruct = TestStruct {
        a: 123,
        b: "hello".to_string(),
    };

    let boxed: Box<TestStruct> = BoxAllocator::try_box(value).expect("allocation should succeed");
    assert_eq!(boxed.a, 123);
    assert_eq!(boxed.b, "hello");
}

#[test]
fn test_alloc_with_kind() {
    let ptr: *mut u64 = GenericAllocator::try_alloc_with_kind::<u64>(AllocKind::Value)
        .expect("allocation should succeed");

    assert!(!ptr.is_null());

    unsafe {
        GenericAllocator::dealloc(ptr);
    }
}
