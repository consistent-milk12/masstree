//! Allocation regression for boxed primitive updates through insert_with_guard.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use masstree::MassTree15;

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: All allocation and deallocation requests are forwarded unchanged to
// System; the atomic counter neither accesses nor changes allocation contents.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The caller supplies the valid layout required by GlobalAlloc.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from this System-backed allocator;
        // the caller transfers the allocation back exactly once.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() {
    let tree = MassTree15::<u64>::new();
    let guard = tree.guard();
    let key = [b'x'; 208];
    tree.insert_with_guard(&key, 0, &guard);
    assert_eq!(tree.get(&key), Some(0));
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for value in 1..=260 {
        assert!(tree.insert_with_guard(&key, value, &guard).is_some());
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    println!("260 guarded updates: {allocations} allocations");
    assert_eq!(allocations, 0);
    assert_eq!(tree.get(&key), Some(260));
    assert_eq!(tree.len(), 1);
}
