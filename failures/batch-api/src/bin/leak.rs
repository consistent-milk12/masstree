use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn live() -> isize { LIVE.load(Ordering::Relaxed) }

use masstree::MassTree15;

fn main() {
    let n = 4000u64;
    let base_entries: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i)).collect();

    // Control: `insert_batch` creates its own guard; no leaked outputs expected.
    {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert_batch(base_entries.clone());
        let start = live();
        for round in 1..=20u64 {
            let entries: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i + round)).collect();
            let r = tree.insert_batch(entries);
            assert_eq!(r.updated, n as usize);
            drop(r);
        }
        drop(tree);
        println!("control insert_batch: start={start} end={} delta={}", live(), live() - start);
    }

    // Test: `insert_batch_with_guard` on write-through BoxPolicy<u64>.
    {
        let tree: MassTree15<u64> = MassTree15::new();
        tree.insert_batch(base_entries.clone());
        let start = live();
        {
            let g = tree.guard();
            for round in 1..=20u64 {
                let entries: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i + round)).collect();
                let r = tree.insert_batch_with_guard(entries, &g);
                assert_eq!(r.updated, n as usize);
                drop(r);
            }
            drop(g);
        }
        drop(tree);
        println!("test insert_batch_with_guard: start={start} end={} delta={}", live(), live() - start);
        println!("expected leak if each old value Box is 8B: {}", 20 * n * 8);
    }
}
