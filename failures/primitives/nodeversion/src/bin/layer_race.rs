use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use masstree::MassTree15;

fn key_a(i: u64) -> [u8; 9] {
    let mut k = [0u8; 9];
    k[..8].copy_from_slice(&i.to_be_bytes());
    k[8] = b'a';
    k
}
fn key_b(i: u64) -> [u8; 9] {
    let mut k = [0u8; 9];
    k[..8].copy_from_slice(&i.to_be_bytes());
    k[8] = b'b';
    k
}

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let skip_b = std::env::args().any(|a| a == "--skip-b");
    let tree: Arc<MassTree15<u64>> = Arc::new(MassTree15::new());
    let current = Arc::new(AtomicUsize::new(usize::MAX));
    let stop = Arc::new(AtomicBool::new(false));
    let bad = Arc::new(AtomicUsize::new(0));
    let gets = Arc::new(AtomicUsize::new(0));

    let mut readers = Vec::new();
    for _ in 0..2 {
        let tree = Arc::clone(&tree);
        let current = Arc::clone(&current);
        let stop = Arc::clone(&stop);
        let bad = Arc::clone(&bad);
        let gets = Arc::clone(&gets);
        readers.push(thread::spawn(move || {
            let g = tree.guard();
            while !stop.load(Ordering::Relaxed) {
                let i = current.load(Ordering::Acquire);
                if i == usize::MAX { continue; }
                let k = key_a(i as u64);
                let r = tree.get_with_guard(&k, &g);
                gets.fetch_add(1, Ordering::Relaxed);
                if r.is_none() {
                    let again = tree.get_with_guard(&k, &g);
                    bad.fetch_add(1, Ordering::Relaxed);
                    if again.is_none() {
                        eprintln!("PERSISTENT MISS for key A at i={i}");
                    }
                }
            }
        }));
    }

    let g = tree.guard();
    for i in 0..n {
        tree.insert_with_guard(&key_a(i), 1, &g);
        current.store(i as usize, Ordering::Release);
        if !skip_b { tree.insert_with_guard(&key_b(i), 2, &g); } else { std::hint::spin_loop(); }
        if i % 50_000 == 0 { eprintln!("i={i} gets={}", gets.load(Ordering::Relaxed)); }
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers { r.join().unwrap(); }
    let bad = bad.load(Ordering::Relaxed);
    println!("done n={n} gets={} transient_misses={bad}", gets.load(Ordering::Relaxed));
    if bad == 0 { eprintln!("no race observed this run"); }
}
