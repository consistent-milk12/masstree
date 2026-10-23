use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
struct C; static A_: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for C {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { A_.fetch_add(1, Ordering::Relaxed); unsafe { System.alloc(l) } }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { A_.fetch_sub(1, Ordering::Relaxed); unsafe { System.dealloc(p, l) } }
}
#[global_allocator] static G: C = C;
use masstree::{BatchEntry, MassTree15};

fn main() {
    // 1. Dropping a public BatchEntry leaks its output box.
    let before = A_.load(Ordering::Relaxed);
    for i in 0..1000u64 {
        let e = BatchEntry::<masstree::BoxPolicy<String>>::new(vec![1], format!("value-{i}"));
        drop(e);
    }
    println!("BatchEntry drop leak: {} allocations still live", A_.load(Ordering::Relaxed) - before);

    // 2. Panic from an over-long key after a smaller key was applied: partial batch + leaked pending outputs.
    let tree: MassTree15<String> = MassTree15::new();
    let before = A_.load(Ordering::Relaxed);
    let entries: Vec<(Vec<u8>, String)> = vec![
        (vec![], "applied".to_string()),
        (vec![b'a'; 257], "too-long".to_string()),
    ];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tree.insert_batch(entries)));
    println!("panic caught: {} tree.len()={} get(empty)={:?}", r.is_err(), tree.len(), tree.get(&[]));
    println!("alloc delta after panic (leak) = {}", A_.load(Ordering::Relaxed) - before);
    drop(tree);
    println!("alloc delta after dropping tree = {}", A_.load(Ordering::Relaxed) - before);
}
