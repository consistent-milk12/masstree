use super::{Layout, pool_alloc, pool_dealloc, size_class};

#[test]
fn test_size_class_computation() {
    // 64 bytes = 1 cache line
    let layout1: Layout = Layout::from_size_align(64, 8).unwrap();
    assert_eq!(size_class(layout1), Some(1));

    // 65 bytes = 2 cache lines
    let layout2: Layout = Layout::from_size_align(65, 8).unwrap();
    assert_eq!(size_class(layout2), Some(2));

    // 768 bytes = 12 cache lines
    let layout12: Layout = Layout::from_size_align(768, 64).unwrap();
    assert_eq!(size_class(layout12), Some(12));

    // 1280 bytes = 20 cache lines (max)
    let layout20: Layout = Layout::from_size_align(1280, 64).unwrap();
    assert_eq!(size_class(layout20), Some(20));

    // 1281 bytes = too large
    let layout_big: Layout = Layout::from_size_align(1281, 64).unwrap();
    assert_eq!(size_class(layout_big), None);
}

#[test]
fn test_alloc_dealloc_roundtrip() {
    let layout: Layout = Layout::from_size_align(128, 64).unwrap();

    let ptr: *mut u8 = pool_alloc(layout);
    assert!(!ptr.is_null());

    // Write to verify it's usable
    unsafe { ptr.write(0xAB) };

    unsafe { pool_dealloc(ptr, layout) };

    // Allocate again - should get the same pointer back (from pool)
    let ptr2: *mut u8 = pool_alloc(layout);
    assert_eq!(ptr, ptr2);

    unsafe { pool_dealloc(ptr2, layout) };
}

#[test]
fn test_different_layouts_same_class() {
    // Both fit in 2 cache lines
    let layout_a: Layout = Layout::from_size_align(100, 8).unwrap();
    let layout_b: Layout = Layout::from_size_align(120, 8).unwrap();

    assert_eq!(size_class(layout_a), size_class(layout_b));

    let ptr_a: *mut u8 = pool_alloc(layout_a);
    unsafe { pool_dealloc(ptr_a, layout_a) };

    // Should reuse from same bucket
    let ptr_b: *mut u8 = pool_alloc(layout_b);
    assert_eq!(ptr_a, ptr_b);

    unsafe { pool_dealloc(ptr_b, layout_b) };
}
