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

    // Build a leftmost full leaf (keys 0..14), then repeatedly create a new
    // rightmost leaf via an append split, empty it, and race an insert against
    // coalesce of the empty rightmost leaf.
    for k in 0..15u64 {
        let _ = tree.insert(&key(k), k);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));

    // Background: keep processing the coalesce queue.
    let bg_tree = Arc::clone(&tree);
    let bg_stop = Arc::clone(&stop);
    let bg_progress = Arc::clone(&progress);
    let bg = thread::spawn(move || {
        while !bg_stop.load(Ordering::Relaxed) {
            let guard = bg_tree.guard();
            let n = bg_tree.process_coalesce(&guard);
            if n > 0 {
                bg_progress.fetch_add(n as u64, Ordering::Relaxed);
            }
            drop(guard);
        }
    });

    // Watchdog: if the main thread livelocks, abort loudly.
    let wd_progress = Arc::clone(&progress);
    let wd = thread::spawn(move || {
        let mut last = 0u64;
        let mut stalls = 0u32;
        loop {
            thread::sleep(Duration::from_millis(500));
            let p = wd_progress.load(Ordering::Relaxed);
            if p == last {
                stalls += 1;
                if stalls > 20 {
                    eprintln!("WATCHDOG: no coalesce progress for 10s; likely livelock");
                    std::process::abort();
                }
            } else {
                stalls = 0;
                last = p;
            }
        }
    });

    let mut k = 1000u64;
    let mut rounds = 0u64;
    loop {
        // This insert splits the rightmost leaf (append case), creating a new
        // rightmost leaf holding only `k`.
        let _ = tree.insert(&key(k), k);
        // Empty that rightmost leaf and queue it for coalesce.
        let _ = tree.remove(&key(k));

        // Race: this insert targets the now-empty rightmost leaf while the
        // background thread may be coalescing it.
        let _ = tree.insert(&key(k), k);

        rounds += 1;
        if rounds % 10_000 == 0 {
            eprintln!("round {rounds} k={k} len={}", tree.len());
        }
        // Keep the main thread from finishing too early; watchdog catches hangs.
        if rounds > 200_000 {
            break;
        }
        k += 1;
    }

    stop.store(true, Ordering::Relaxed);
    let _ = bg.join();
    drop(wd);

    eprintln!("completed without hang after {rounds} rounds");
}
