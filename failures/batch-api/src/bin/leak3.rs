use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
struct C;
static A_: AtomicIsize = AtomicIsize::new(0);
static TOTAL: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for C {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        A_.fetch_add(1, Ordering::Relaxed);
        TOTAL.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        A_.fetch_sub(1, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static G: C = C;
use masstree::{BatchEntry, MassTree15};

fn main() {
    // 1. Dropping a public BatchEntry leaks its output box.
    let before = A_.load(Ordering::Relaxed);
    for i in 0..1000u64 {
        let e = BatchEntry::<masstree::BoxPolicy<String>>::new(vec![1], format!("value-{i}"));
        drop(e);
    }
    assert_eq!(A_.load(Ordering::Relaxed) - before, 0);
    println!(
        "BatchEntry drop leak: {} allocations still live",
        A_.load(Ordering::Relaxed) - before
    );

    // 2. Panic from an over-long key after a smaller key was applied: partial batch + leaked pending outputs.
    let tree: MassTree15<String> = MassTree15::new();
    let before = A_.load(Ordering::Relaxed);
    let entries: Vec<(Vec<u8>, String)> = vec![
        (vec![], "applied".to_string()),
        (vec![b'a'; 257], "too-long".to_string()),
    ];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tree.insert_batch(entries)));
    assert!(r.is_err());
    assert!(tree.is_empty());
    println!(
        "panic caught: {} tree.len()={} get(empty)={:?}",
        r.is_err(),
        tree.len(),
        tree.get(&[])
    );
    println!(
        "alloc delta after panic (leak) = {}",
        A_.load(Ordering::Relaxed) - before
    );
    drop(tree);
    println!(
        "alloc delta after dropping tree = {}",
        A_.load(Ordering::Relaxed) - before
    );

    // 3. A preconstructed write-through update batch allocates only its old
    // value result (three Vec growths for 15 results), not another entry vector
    // or a speculative suffix buffer.
    let tree = MassTree15::<u64>::new();
    let guard = tree.guard();
    let entries: Vec<_> = (0..15u64)
        .map(|i| {
            let mut key = i.to_be_bytes().to_vec();
            key.extend_from_slice(&[b'x'; 200]);
            tree.insert_with_guard(&key, i, &guard);
            BatchEntry::<masstree::BoxPolicy<u64>>::new(key, i + 100)
        })
        .collect();
    let before = TOTAL.load(Ordering::Relaxed);
    let result = tree.insert_batch_entries(entries, &guard);
    let allocations = TOTAL.load(Ordering::Relaxed) - before;
    println!("preconstructed update batch: {allocations} allocations");
    assert!(
        allocations <= 3,
        "only the old-value result should allocate"
    );
    assert_eq!(result.updated, 15);
    assert_eq!(result.inserted, 0);
    for (i, old) in result.old_values.iter().enumerate() {
        assert_eq!(**old, i as u64);
    }
}
