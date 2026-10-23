use masstree::MassTree;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

fn key(k: u64) -> [u8; 8] {
    k.to_be_bytes()
}

fn main() {
    let tree = Arc::new(MassTree::<u64>::new());

    for k in 0..15u64 {
        let _ = tree.insert(&key(k), k);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let rounds_done = Arc::new(AtomicU64::new(0));

    let bg_tree = Arc::clone(&tree);
    let bg_stop = Arc::clone(&stop);
    let bg = thread::spawn(move || {
        while !bg_stop.load(Ordering::Relaxed) {
            let guard = bg_tree.guard();
            let _ = bg_tree.process_coalesce(&guard);
        }
    });

    let rp = Arc::clone(&rounds_done);
    let mon = thread::spawn(move || {
        let mut last = 0;
        let mut stalls = 0;
        loop {
            thread::sleep(Duration::from_secs(1));
            let r = rp.load(Ordering::Relaxed);
            if r == last {
                stalls += 1;
                eprintln!("monitor: rounds={r} stalled {stalls}s");
                if stalls >= 20 {
                    eprintln!("monitor: HANG at round {r}");
                    std::process::abort();
                }
            } else {
                stalls = 0;
                last = r;
                eprintln!("monitor: rounds={r}");
            }
        }
    });

    let mut k = 1000u64;
    loop {
        let _ = tree.insert(&key(k), k);
        let _ = tree.remove(&key(k));
        let _ = tree.insert(&key(k), k);
        k += 1;
        rounds_done.store(k - 1000, Ordering::Relaxed);
    }
}
