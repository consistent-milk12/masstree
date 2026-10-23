use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static LIVE_ALLOCS: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(l.size() as isize, Ordering::Relaxed);
        LIVE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE_BYTES.fetch_sub(l.size() as isize, Ordering::Relaxed);
        LIVE_ALLOCS.fetch_sub(1, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE_BYTES.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(l.size() as isize, Ordering::Relaxed);
        LIVE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

use masstree::{MassTree15, MassTree15Inline};

fn bytes() -> isize { LIVE_BYTES.load(Ordering::Relaxed) }
fn allocs() -> isize { LIVE_ALLOCS.load(Ordering::Relaxed) }

fn main() {
    let n: u64 = std::env::args().nth(2).and_then(|x| x.parse().ok()).unwrap_or(1000);
    let rounds: u64 = std::env::args().nth(3).and_then(|x| x.parse().ok()).unwrap_or(50);
    let base: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i)).collect();
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "control" => {
            let tree: MassTree15<u64> = MassTree15::new();
            let _ = tree.insert_batch(base);
            let (b0, a0) = (bytes(), allocs());
            for round in 1..=rounds {
                let e: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i + round)).collect();
                let _ = tree.insert_batch(e);
            }
            drop(tree);
            println!("control insert_batch: dbytes={} dallocs={}", bytes()-b0, allocs()-a0);
        }
        "wt_guard" => {
            let tree: MassTree15<u64> = MassTree15::new();
            let _ = tree.insert_batch(base);
            let g = tree.guard();
            let (b0, a0) = (bytes(), allocs());
            for round in 1..=rounds {
                let e: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i + round)).collect();
                let r = tree.insert_batch_with_guard(e, &g);
                drop(r);
            }
            drop(g);
            drop(tree);
            println!("wt insert_batch_with_guard: dbytes={} dallocs={} (updates={})", bytes()-b0, allocs()-a0, rounds*n);
        }
        "inline_guard" => {
            let tree: MassTree15Inline<u64> = MassTree15Inline::new();
            let _ = tree.insert_batch(base);
            let g = tree.guard();
            let (b0, a0) = (bytes(), allocs());
            for round in 1..=rounds {
                let e: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (i.to_be_bytes().to_vec(), i + round)).collect();
                let r = tree.insert_batch_with_guard(e, &g);
                drop(r);
            }
            drop(g);
            drop(tree);
            println!("inline insert_batch_with_guard: dbytes={} dallocs={}", bytes()-b0, allocs()-a0);
        }
        _ => println!("usage: leak2 control|wt_guard|inline_guard"),
    }
}
