//! Deterministic repro: two *immediate* chain entries for the same leaf.
//!
//! No concurrency is needed. The leaf is emptied, re-filled, and emptied again
//! without any maintenance call, so `chains` holds E1 and E2 pointing at the
//! same live leaf. A single `process_coalesce` pops E1, coalesces/retires the
//! leaf, then pops E2 and re-runs `unlink_from_chain` on the already-unlinked,
//! already-deleted leaf. `lock_next` spins forever because the first unlink
//! left `next` marked (and `wait_for_split` returns immediately on a deleted
//! leaf), so the maintenance thread wedges.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use masstree::MassTree15Inline;

fn k(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

#[test]
fn two_immediate_entries_same_leaf() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());

    // Build several leaves; keep key 0 (leftmost) and key 20.
    for i in 0..40u64 {
        tree.insert(&k(i), i);
    }
    for i in 0..40u64 {
        if i != 0 && i != 20 {
            let _ = tree.remove(&k(i));
        }
    }
    {
        let g = tree.guard();
        for _ in 0..50 {
            tree.process_coalesce(&g);
        }
    }
    assert_eq!(tree.get(&k(20)), Some(20));
    assert_eq!(tree.pending_coalesce(), 0, "setup queue must be empty");

    // E1: empty the leaf.
    assert_eq!(tree.remove(&k(20)).unwrap(), Some(20));
    assert_eq!(tree.pending_coalesce(), 1, "E1 queued");

    // Re-fill the same leaf (clears its queued bit).
    tree.insert(&k(20), 20);
    assert_eq!(tree.get(&k(20)), Some(20));

    // E2: empty it again while E1 is still in the immediate queue.
    assert_eq!(tree.remove(&k(20)).unwrap(), Some(20));
    assert_eq!(
        tree.pending_coalesce(),
        2,
        "E1 and E2 queued for the same leaf"
    );

    // Watchdog turns the expected wedge into an observable exit.
    let done = Arc::new(AtomicBool::new(false));
    let watchdog_done = Arc::clone(&done);
    let watchdog = thread::spawn(move || {
        thread::sleep(Duration::from_secs(3));
        if !watchdog_done.load(Ordering::Relaxed) {
            eprintln!("WEDGED: process_coalesce did not return");
            std::process::exit(42);
        }
    });

    let g = tree.guard();
    let processed = tree.process_coalesce(&g);
    done.store(true, Ordering::Relaxed);
    watchdog.join().unwrap();
    println!("processed={processed}");
}
