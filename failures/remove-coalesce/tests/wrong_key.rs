//! Repro: same-ikey short keys can be removed in place of the requested key.
//!
//! All keys below share the same 8-byte ikey (all-zero) and are legal,
//! distinct keys in one leaf (keylenx 0..8). `search_for_remove` identifies
//! the victim by logical index `ki`; `lock_and_verify_for_remove` re-reads
//! `perm.get(ki)` under the leaf lock but only compares the *ikey*. Any
//! concurrent insert/remove of another same-ikey key shifts `ki`, so the
//! cursor removes a different key and returns its value.

use masstree::MassTree15Inline;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

const N: usize = 9;

fn key(i: usize) -> Vec<u8> {
    vec![0u8; i]
}

fn val(i: usize) -> u64 {
    100 + i as u64
}

#[test]
fn wrong_key_removal_same_ikey() {
    let tree = MassTree15Inline::<u64>::new();
    for i in 0..N {
        tree.insert(&key(i), val(i));
    }
    for i in 0..N {
        assert_eq!(tree.get(&key(i)), Some(val(i)), "setup key {i}");
    }

    let tree = std::sync::Arc::new(tree);
    let threads = 8usize;
    let barrier = std::sync::Arc::new(Barrier::new(threads + 1));
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let ops = std::sync::Arc::new(AtomicU64::new(0));
    let errors = std::sync::Arc::new(AtomicU64::new(0));
    let first_error = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));

    let mut handles = Vec::new();
    for t in 0..threads {
        let tree = std::sync::Arc::clone(&tree);
        let barrier = std::sync::Arc::clone(&barrier);
        let stop = std::sync::Arc::clone(&stop);
        let ops = std::sync::Arc::clone(&ops);
        let errors = std::sync::Arc::clone(&errors);
        let first_error = std::sync::Arc::clone(&first_error);
        handles.push(thread::spawn(move || {
            let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64);
            barrier.wait();
            while !stop.load(Ordering::Relaxed) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let i = (state as usize) % N;
                if state & 0x100 != 0 {
                    let r = tree.remove(&key(i)).unwrap();
                    if let Some(v) = r {
                        if v != val(i) {
                            errors.fetch_add(1, Ordering::Relaxed);
                            let mut slot = first_error.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(format!(
                                    "remove(key len {i}) returned value {v} (expected {}); \
                                     value belongs to key len {}",
                                    val(i),
                                    v - 100
                                ));
                            }
                        }
                    }
                } else {
                    let _ = tree.insert(&key(i), val(i));
                }
                ops.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    barrier.wait();
    thread::sleep(std::time::Duration::from_secs(3));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    let e = errors.load(Ordering::Relaxed);
    let o = ops.load(Ordering::Relaxed);
    println!("wrong-key churn: {o} ops, {e} wrong-key removals");
    assert_eq!(
        e,
        0,
        "{} wrong-key removals in {} ops; first: {:?}",
        e,
        o,
        first_error.lock().unwrap()
    );
}
