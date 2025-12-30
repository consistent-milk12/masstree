//! WD2: Complex write/delete pattern with status tracking.
//!
//! Port of `kvtest_wd2` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Multiple threads maintain status and main keys
//! - Random removes on other threads' keys
//! - Tests complex concurrent insert/delete interaction

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const SEED: u64 = 31949;
const SEP: usize = 26; // Number of separator keys (A-Z)
const P_REMOVE: u32 = 1000;
const P_PUT2: u32 = 10000;
const P_REMOVE2: u32 = 20000;

fn make_status_key(client_id: usize, suffix: Option<char>) -> Vec<u8> {
    let mut key = format!("s{:03}", client_id).into_bytes();
    if let Some(c) = suffix {
        key.push(c as u8);
    }
    key
}

fn make_main_key(client_id: usize, suffix: Option<char>) -> Vec<u8> {
    let mut key = format!("k{:03}", client_id).into_bytes();
    if let Some(c) = suffix {
        key.push(c as u8);
    }
    key
}

#[test]
fn wd2_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));
    let num_threads = 4;

    // Store thread count
    {
        let guard = tree.guard();
        tree.insert_with_guard(b"n", num_threads as u64, &guard)
            .unwrap();
    }

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);
                let mut x = 0u64;

                // Set up status keys (s{tid}A, s{tid}B, ..., s{tid}Z)
                for i in 0..SEP {
                    let key = make_status_key(tid, Some((b'A' + i as u8) as char));
                    let _ = tree.insert_with_guard(&key, 0, &guard);
                }
                // Main status key s{tid}
                let status_key = make_status_key(tid, None);
                let _ = tree.insert_with_guard(&status_key, x, &guard);

                // Set up main keys (k{tid}A, k{tid}B, ..., k{tid}Z)
                for i in 0..SEP {
                    let key = make_main_key(tid, Some((b'A' + i as u8) as char));
                    let _ = tree.insert_with_guard(&key, 0, &guard);
                }
                // Main key k{tid}
                let main_key = make_main_key(tid, None);
                let _ = tree.insert_with_guard(&main_key, 0, &guard);

                let next_tid = (tid + 1) % num_threads;

                // Main loop
                let mut nrounds = 0u64;
                while !done.load(Ordering::Relaxed) {
                    nrounds += 1;

                    // Put to main key
                    let _ = tree.insert_with_guard(&main_key, x, &guard);

                    // Maybe remove next thread's main key
                    if rng.random_range(0..65536u32) < P_REMOVE {
                        let next_key = make_main_key(next_tid, None);
                        let _ = tree.remove_with_guard(&next_key, &guard);
                    }

                    let rand_val = rng.random_range(0..65536u32);
                    if rand_val < P_PUT2 {
                        // Put separator keys for next thread (reverse order)
                        for i in (0..SEP).rev() {
                            let key = make_main_key(next_tid, Some((b'A' + i as u8) as char));
                            let _ = tree.insert_with_guard(&key, 0, &guard);
                        }
                    } else if rand_val < P_REMOVE2 {
                        // Remove separator keys for next thread (reverse order)
                        for i in (0..SEP).rev() {
                            let key = make_main_key(next_tid, Some((b'A' + i as u8) as char));
                            let _ = tree.remove_with_guard(&key, &guard);
                        }
                    }

                    // Increment counter
                    x += 1;
                    let _ = tree.insert_with_guard(&status_key, x, &guard);
                }
            })
        })
        .collect();

    // Let it run
    thread::sleep(Duration::from_millis(500));
    done.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn wd2_simple() {
    // Simplified version: one writer, one deleter on same keys
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));

    let tree_w = Arc::clone(&tree);
    let done_w = Arc::clone(&done);
    let writer = thread::spawn(move || {
        let guard = tree_w.guard();
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut n = 0u64;

        while !done_w.load(Ordering::Relaxed) {
            // Insert keys with separators
            for i in 0..SEP {
                let key = format!("key{:04}{}", n, (b'A' + i as u8) as char);
                let _ = tree_w.insert_with_guard(key.as_bytes(), n, &guard);
            }
            n += 1;

            // Random remove
            if n > 10 && rng.random::<bool>() {
                let target = rng.random_range(0..n);
                for i in 0..SEP {
                    let key = format!("key{:04}{}", target, (b'A' + i as u8) as char);
                    let _ = tree_w.remove_with_guard(key.as_bytes(), &guard);
                }
            }
        }
    });

    let tree_d = Arc::clone(&tree);
    let done_d = Arc::clone(&done);
    let deleter = thread::spawn(move || {
        let guard = tree_d.guard();
        let mut rng = StdRng::seed_from_u64(SEED + 1);

        while !done_d.load(Ordering::Relaxed) {
            // Random delete attempts
            let target = rng.random_range(0..1000u64);
            for i in 0..SEP {
                let key = format!("key{:04}{}", target, (b'A' + i as u8) as char);
                let _ = tree_d.remove_with_guard(key.as_bytes(), &guard);
            }
            thread::yield_now();
        }
    });

    thread::sleep(Duration::from_millis(300));
    done.store(true, Ordering::Relaxed);

    writer.join().unwrap();
    deleter.join().unwrap();
}
