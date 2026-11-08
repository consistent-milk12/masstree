use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct C;
static A: AtomicUsize = AtomicUsize::new(0);
static B: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for C {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        A.fetch_add(1, Ordering::Relaxed);
        B.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        A.fetch_add(1, Ordering::Relaxed);
        B.fetch_add(n, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static G: C = C;
use masstree::MassTree15Inline;
fn main() {
    // (1) update path, 13 keys x 40B suffix in one leaf
    let t: MassTree15Inline<u64> = MassTree15Inline::new();
    let g = t.guard();
    let mut ks = Vec::new();
    for i in 0u64..13 {
        let mut k = i.to_be_bytes().to_vec();
        k.extend(std::iter::repeat_n(b'x', 40));
        ks.push(k);
    }
    for (i, k) in ks.iter().enumerate() {
        let _ = t.insert_with_guard(k, i as u64, &g);
    }
    let (a0, b0) = (A.load(Ordering::Relaxed), B.load(Ordering::Relaxed));
    for _ in 0..20 {
        for (i, k) in ks.iter().enumerate() {
            let _ = t.insert_with_guard(k, 1000 + i as u64, &g);
        }
    }
    let (a1, b1) = (A.load(Ordering::Relaxed), B.load(Ordering::Relaxed));
    println!(
        "UPDATE n=260 allocs={} bytes={} per={:.2}/{:.1}",
        a1 - a0,
        b1 - b0,
        (a1 - a0) as f64 / 260.0,
        (b1 - b0) as f64 / 260.0
    );
    assert_eq!(
        a1 - a0,
        0,
        "value-only updates must not allocate suffix storage"
    );
    // (2) churn full leaf, 15 keys x 200B suffix
    let t2: MassTree15Inline<u64> = MassTree15Inline::new();
    let g2 = t2.guard();
    let mut ks2 = Vec::new();
    for i in 0u64..15 {
        let mut k = i.to_be_bytes().to_vec();
        k.extend(std::iter::repeat_n(b'q', 200));
        ks2.push(k);
    }
    for (i, k) in ks2.iter().enumerate() {
        let _ = t2.insert_with_guard(k, i as u64, &g2);
    }
    let (a0, b0) = (A.load(Ordering::Relaxed), B.load(Ordering::Relaxed));
    for _ in 0..20 {
        t2.remove_with_guard(&ks2[0], &g2).unwrap();
        let _ = t2.insert_with_guard(&ks2[0], 0, &g2);
    }
    let (a1, b1) = (A.load(Ordering::Relaxed), B.load(Ordering::Relaxed));
    println!(
        "CHURN n=20 allocs={} bytes={} per={:.2}/{:.1}",
        a1 - a0,
        b1 - b0,
        (a1 - a0) as f64 / 20.0,
        (b1 - b0) as f64 / 20.0
    );
    assert!(
        a1 - a0 <= 8,
        "suffix churn must amortize rebuilds: {} allocations",
        a1 - a0
    );
    assert!(
        b1 - b0 <= 25_000,
        "suffix churn allocated {} bytes",
        b1 - b0
    );
    assert_eq!(t2.len(), 15);
    for (i, k) in ks2.iter().enumerate() {
        assert_eq!(t2.get(k), Some(i as u64), "suffix lost at slot {i}");
    }
}
