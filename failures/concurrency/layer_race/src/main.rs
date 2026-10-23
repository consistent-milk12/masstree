use masstree::MassTree15;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const A: &[u8] = b"PREFIX00aaaaaaaa";
const B: &[u8] = b"PREFIX00bbbbbbbb";

fn main() {
    let cycles: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let nreaders: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let tree = Arc::new(MassTree15::<u64>::new());
    let phase = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let genr = Arc::new(AtomicUsize::new(0));
    let none_count = Arc::new(AtomicUsize::new(0));
    let stable_none = Arc::new(AtomicUsize::new(0));
    let second_none = Arc::new(AtomicUsize::new(0));
    let b_done = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..nreaders {
        let tree = Arc::clone(&tree);
        let phase = Arc::clone(&phase);
        let stop = Arc::clone(&stop);
        let genr = Arc::clone(&genr);
        let none_count = Arc::clone(&none_count);
        let stable_none = Arc::clone(&stable_none);
        let second_none = Arc::clone(&second_none);
        let b_done = Arc::clone(&b_done);
        readers.push(std::thread::spawn(move || {
            let guard = tree.guard();
            loop {
                if stop.load(Ordering::Relaxed) { break; }
                while phase.load(Ordering::Acquire) != 1 {
                    if stop.load(Ordering::Relaxed) { return; }
                    std::hint::spin_loop();
                }
                // A must be present whenever phase == 1 for this generation.
                let g = genr.load(Ordering::Acquire);
                if tree.get_ref(A, &guard).is_none() {
                    none_count.fetch_add(1, Ordering::Relaxed);
                    // Still same cycle and writer has not started removal: genuine wrong None.
                    if phase.load(Ordering::Acquire) == 1 && genr.load(Ordering::Acquire) == g {
                        stable_none.fetch_add(1, Ordering::Relaxed);
                        let bd = b_done.load(Ordering::Acquire);
                        if tree.get_ref(A, &guard).is_none() {
                            second_none.fetch_add(1, Ordering::Relaxed);
                            if !bd { eprintln!("None BEFORE B insert"); }
                        }
                    }
                }
            }
        }));
    }
    // Remover thread: hammers remove(A) while B insertion creates/moves a layer.
    let r_tree = Arc::clone(&tree);
    let r_phase = Arc::clone(&phase);
    let r_stop = Arc::clone(&stop);
    let remover = std::thread::spawn(move || {
        let guard = r_tree.guard();
        while !r_stop.load(Ordering::Relaxed) {
            if r_phase.load(Ordering::Acquire) == 1 {
                let _ = r_tree.remove_with_guard(A, &guard);
            }
        }
    });
    let guard = tree.guard();
    for _ in 0..cycles {
        tree.insert_with_guard(A, 1, &guard);
        genr.fetch_add(1, Ordering::Release);
        b_done.store(false, Ordering::Release);
        phase.store(1, Ordering::Release);
        tree.insert_with_guard(B, 2, &guard);
        b_done.store(true, Ordering::Release);
        // keep readers running in the post-insert window
        let until = Instant::now() + Duration::from_micros(2);
        while Instant::now() < until { std::hint::spin_loop(); }
        phase.store(0, Ordering::Release);
        let _ = tree.remove_with_guard(A, &guard).unwrap();
        let _ = tree.remove_with_guard(B, &guard).unwrap();
        let _ = tree.process_coalesce(&guard);
    }
    stop.store(true, Ordering::Release);
    for r in readers { r.join().unwrap(); }
    remover.join().unwrap();
    println!("none_count={} stable_none={} second_none={}", none_count.load(Ordering::Relaxed), stable_none.load(Ordering::Relaxed), second_none.load(Ordering::Relaxed));
}
