use masstree::MassTree15;
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SHORT: &[u8] = b"ABCDEFGH";
const LONG: &[u8] = b"ABCDEFGH01234567";

fn main() {
    let threads: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let tree = Arc::new(MassTree15::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let bad = Arc::new(AtomicBool::new(false));
    let wrong = Arc::new(AtomicUsize::new(0));
    let mut hs = Vec::new();
    for _ in 0..threads {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let bad = Arc::clone(&bad);
        let wrong = Arc::clone(&wrong);
        hs.push(std::thread::spawn(move || {
            let guard = tree.guard();
            barrier.wait();
            for _ in 0..iters {
                if bad.load(Ordering::Relaxed) { break; }
                tree.insert_with_guard(SHORT, 1, &guard);
                tree.insert_with_guard(LONG, 2, &guard);
                match tree.remove_with_guard(SHORT, &guard).unwrap() {
                    Some(v) if *v == 2 => {
                        wrong.fetch_add(1, Ordering::Relaxed);
                        bad.store(true, Ordering::Relaxed);
                        eprintln!("WRONG KEY REMOVED: remove(short) returned long's value 2");
                    }
                    _ => {}
                }
            }
        }));
    }
    for h in hs { h.join().unwrap(); }
    println!("wrong={} bad={}", wrong.load(Ordering::Relaxed), bad.load(Ordering::Relaxed));
}
