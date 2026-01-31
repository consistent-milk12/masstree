use super::{BoxAllocator, GenericAllocator};
use core::ptr as CorePtr;

#[test]
fn test_alloc_success() {
    let ptr: *mut u64 = GenericAllocator::alloc::<u64>();
    assert!(!ptr.is_null());

    // Clean up
    unsafe {
        CorePtr::write(ptr, 42);
        GenericAllocator::dealloc(ptr);
    }
}

#[test]
fn test_alloc_zeroed_success() {
    let ptr: *mut [u8; 64] = GenericAllocator::alloc_zeroed::<[u8; 64]>();

    // Verify zeroed
    unsafe {
        let arr: &[u8; 64] = &*ptr;
        assert!(arr.iter().all(|b: &u8| *b == 0));

        GenericAllocator::dealloc(ptr);
    }
}

#[test]
fn test_boxed_success() {
    let boxed: Box<u64> = BoxAllocator::boxed(42u64);
    assert_eq!(*boxed, 42);
}

#[test]
fn test_boxed_struct() {
    #[derive(Debug, PartialEq)]
    struct TestStruct {
        a: u64,
        b: String,
    }

    let value: TestStruct = TestStruct {
        a: 123,
        b: "hello".to_string(),
    };

    let boxed: Box<TestStruct> = BoxAllocator::boxed(value);
    assert_eq!(boxed.a, 123);
    assert_eq!(boxed.b, "hello");
}
