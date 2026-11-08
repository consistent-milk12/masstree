//! Count allocator calls and time the first insert on fresh threads.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use masstree::{BoxPolicy, LeafNode15, MassTree15};

struct Counting;
static LEAF_ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: All requests are forwarded to System with unchanged layout and
// ownership. The counter only observes the layout of boxed-policy leaf blocks.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() == 64 && layout.size() == size_of::<LeafNode15<BoxPolicy<u64>>>() {
            LEAF_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: GlobalAlloc's caller supplies the valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The caller returns its System allocation with the same layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() {
    let mut times = Vec::new();
    let mut counts = Vec::new();
    for _ in 0..7 {
        let (count, elapsed) = std::thread::spawn(|| {
            LEAF_ALLOCS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            let tree = MassTree15::<u64>::new();
            tree.insert(b"first", 1);
            let elapsed = start.elapsed();
            let count = LEAF_ALLOCS.load(Ordering::Relaxed);
            assert_eq!(tree.get(b"first"), Some(1));
            // Exercise growth beyond the maximum refill, outside the timer.
            for i in 0..5000u64 {
                tree.insert(&i.to_be_bytes(), i);
            }
            for i in 0..5000u64 {
                assert_eq!(tree.get(&i.to_be_bytes()), Some(i));
            }
            (count, elapsed)
        })
        .join()
        .unwrap();
        times.push(elapsed);
        counts.push(count);
    }
    times.sort();
    println!(
        "cold first insert: {:?} median; leaf allocations: {counts:?}",
        times[3]
    );
    assert_eq!(counts, vec![16; 7]);
}
