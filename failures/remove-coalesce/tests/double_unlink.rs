//! Repro attempt: immediate chain entries are raw pointers that can outlive
//! the leaf they point to when an older *deferred* entry for the same leaf is
//! processed first.
//!
//! 1. remove(K) empties leaf L and schedules immediate entry E1.
//! 2. Maintenance pops E1 while a churner holds L's lock -> try_lock fails,
//!    E1 is requeued as deferred entry D (ikey_bound only).
//! 3. A churner reuses L (queued bit cleared), then empties it again ->
//!    schedules a new immediate entry E2 for L.
//! 4. Maintenance processes deferred-first: D re-traverses, finds L empty and
//!    queued (E2's bit), coalesces/retires L.
//! 5. E2 is popped later, dereferences the retired leaf, re-runs
//!    mark_deleted + unlink_from_chain on an already-unlinked leaf.
//!
//! `unlink_from_chain` leaves `self.next` marked; a second call spins forever
//! in `lock_next` (wait_for_split sees is_deleted and returns, loop retries).
//! A watchdog turns that hang into a process exit so the test is observable.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use masstree::MassTree15Inline;

fn k(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

#[test]
fn immediate_entry_outlives_leaf() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());

    // Build several leaves, then leave only two keys so the churn leaf has a
    // non-null prev (so chain coalesce is allowed to unlink it).
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
        for _ in 0..20 {
            tree.process_coalesce(&g);
        }
    }
    assert_eq!(tree.get(&k(20)), Some(20));

    let stop = Arc::new(AtomicBool::new(false));
    let ops = Arc::new(AtomicU64::new(0));

    // Watchdog: if the workload wedges, exit loudly.
    let watchdog_stop = Arc::clone(&stop);
    let watchdog = thread::spawn(move || {
        let mut last = 0u64;
        let mut stuck = 0u32;
        while !watchdog_stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(500));
            let now = OPS.with(|c| c.load(Ordering::Relaxed));
            if now == last {
                stuck += 1;
                if stuck >= 8 {
                    eprintln!("WATCHDOG: no progress for 4s - maintenance/remove is wedged");
                    if std::env::var_os("WATCHDOG_HOLD").is_some() {
                        loop {
                            thread::sleep(Duration::from_secs(1));
                        }
                    }
                    std::process::exit(42);
                }
            } else {
                stuck = 0;
                last = now;
            }
        }
    });

    thread_local! {
        static OPS: AtomicU64 = const { AtomicU64::new(0) };
    }

    let mut handles = Vec::new();
    for _ in 0..4 {
        let tree = Arc::clone(&tree);
        let stop = Arc::clone(&stop);
        let ops = Arc::clone(&ops);
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = tree.remove(&k(20));
                let _ = tree.insert(&k(20), 20);
                ops.fetch_add(1, Ordering::Relaxed);
                OPS.with(|c| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
            }
        }));
    }
    for _ in 0..2 {
        let tree = Arc::clone(&tree);
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let g = tree.guard();
                let _ = tree.process_coalesce_batch(&g, 8);
                OPS.with(|c| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
            }
        }));
    }

    thread::sleep(Duration::from_secs(6));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
    watchdog.join().unwrap();

    println!(
        "completed: churn ops={}, pending={}, deferred={}, abandoned={}",
        ops.load(Ordering::Relaxed),
        tree.pending_coalesce(),
        tree.coalesce_deferred(),
        tree.coalesce_abandoned()
    );
}
