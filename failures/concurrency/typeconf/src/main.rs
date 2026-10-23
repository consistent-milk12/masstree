use masstree::MassTree15;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const A: &[u8] = b"PREFIX00aaaaaaaa";
const B: &[u8] = b"PREFIX00bbbbbbbb";

fn main() {
    let cycles: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30000);
    let nreaders: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let tree = Arc::new(MassTree15::<String>::new());
    let phase = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let genr = Arc::new(AtomicUsize::new(0));
    let bad = Arc::new(AtomicUsize::new(0));
    let nones = Arc::new(AtomicUsize::new(0));
    let mut readers = Vec::new();
    for _ in 0..nreaders {
        let tree = Arc::clone(&tree);
        let phase = Arc::clone(&phase);
        let stop = Arc::clone(&stop);
        let genr = Arc::clone(&genr);
        let bad = Arc::clone(&bad);
        let nones = Arc::clone(&nones);
        readers.push(std::thread::spawn(move || {
            let guard = tree.guard();
            loop {
                if stop.load(Ordering::Relaxed) { break; }
                while phase.load(Ordering::Acquire) != 1 {
                    if stop.load(Ordering::Relaxed) { return; }
                    std::hint::spin_loop();
                }
                let g = genr.load(Ordering::Acquire);
                if let Some(v) = tree.get_ref(A, &guard) {
                    // If the value is a misinterpreted layer pointer, this can
                    // read invalid String metadata.
                    if v.as_str() != "A"
                        && phase.load(Ordering::Acquire) == 1
                        && genr.load(Ordering::Acquire) == g
                    {
                        bad.fetch_add(1, Ordering::Relaxed);
                        eprintln!("BOGUS VALUE {:?}", v.as_str());
                    }
                } else if phase.load(Ordering::Acquire) == 1
                    && genr.load(Ordering::Acquire) == g
                {
                    nones.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    let guard = tree.guard();
    for _ in 0..cycles {
        tree.insert_with_guard(A, "A".to_string(), &guard);
        genr.fetch_add(1, Ordering::Release);
        phase.store(1, Ordering::Release);
        tree.insert_with_guard(B, "B".to_string(), &guard);
        let until = Instant::now() + Duration::from_micros(2);
        while Instant::now() < until { std::hint::spin_loop(); }
        phase.store(0, Ordering::Release);
        let _ = tree.remove_with_guard(A, &guard).unwrap();
        let _ = tree.remove_with_guard(B, &guard).unwrap();
        let _ = tree.process_coalesce(&guard);
    }
    stop.store(true, Ordering::Release);
    for r in readers { r.join().unwrap(); }
    println!("bad={} nones={}", bad.load(Ordering::Relaxed), nones.load(Ordering::Relaxed));
}
