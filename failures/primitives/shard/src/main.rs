mod shard_counter;

use shard_counter::ShardedCounter;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn main() {
    let c = Arc::new(ShardedCounter::new());
    let created = Arc::new(AtomicUsize::new(0));
    let removed = Arc::new(AtomicUsize::new(0));
    let n: usize = 400_000;

    let ca = Arc::clone(&c); let cr = Arc::clone(&created);
    let ta = thread::spawn(move || {
        for i in 1..=n {
            ca.increment();
            cr.store(i, Ordering::Release);
        }
    });
    let cb = Arc::clone(&c); let cr = Arc::clone(&created); let cd = Arc::clone(&removed);
    let tb = thread::spawn(move || {
        while cd.load(Ordering::Relaxed) < n {
            if cr.load(Ordering::Acquire) > cd.load(Ordering::Relaxed) {
                cb.decrement();
                cd.fetch_add(1, Ordering::Relaxed);
            } else {
                std::hint::spin_loop();
            }
        }
    });

    // Keep loading while the pair runs. A negative sum panics in `load`'s
    // debug_assert even though the logical count is never negative.
    let mut loads = 0usize;
    while created.load(Ordering::Acquire) < n && removed.load(Ordering::Relaxed) < n {
        let _ = c.load();
        loads += 1;
    }
    ta.join().unwrap();
    tb.join().unwrap();
    println!("loads={loads} final={}", c.load());
}
