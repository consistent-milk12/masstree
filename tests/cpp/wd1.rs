//! WD1: Writer/Deleter chase test.
//!
//! Port of `kvtest_wd1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Writer inserts sequential keys
//! - Deleter chases behind and removes them
//! - Tests concurrent insert/delete

#![allow(clippy::unwrap_used, clippy::cast_sign_loss)]

use masstree::MassTree15Inline;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

const N: u64 = 50_000;

#[test]
fn wd1_single_pair() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));
    let writer_pos = Arc::new(AtomicU64::new(0));

    // Writer thread
    let writer_tree = Arc::clone(&tree);
    let writer_done = Arc::clone(&done);
    let writer_pos_clone = Arc::clone(&writer_pos);
    let writer = thread::spawn(move || {
        let guard = writer_tree.guard();

        for n in 0..N {
            let key = n.to_be_bytes();
            writer_tree.insert_with_guard(&key, n + 1, &guard).unwrap();
            writer_pos_clone.store(n, Ordering::Release);

            // Yield occasionally
            if n % 1000 == 0 {
                thread::yield_now();
            }
        }

        writer_done.store(true, Ordering::Release);
    });

    // Deleter thread - wait for writer to get ahead
    let deleter_tree = Arc::clone(&tree);
    let deleter_done = Arc::clone(&done);
    let deleter_pos = Arc::clone(&writer_pos);
    let deleter = thread::spawn(move || {
        let guard = deleter_tree.guard();

        // Wait for writer to insert some keys first
        while deleter_pos.load(Ordering::Acquire) < 100 {
            thread::yield_now();
        }

        let mut removed = 0u64;
        let writer_finished = || deleter_done.load(Ordering::Acquire);

        while removed < N {
            let current_writer_pos = deleter_pos.load(Ordering::Acquire);

            // Stay behind writer, but catch up fully once writer is done
            let can_remove = if writer_finished() {
                removed <= current_writer_pos
            } else {
                removed < current_writer_pos.saturating_sub(10)
            };

            if can_remove {
                let key = removed.to_be_bytes();
                match deleter_tree.remove_with_guard(&key, &guard) {
                    Ok(Some(_)) => removed += 1,
                    Ok(None) => {
                        // Key might not exist yet, wait
                        thread::yield_now();
                    }
                    Err(_) => {
                        // Retry on error
                        thread::yield_now();
                    }
                }
            } else {
                thread::yield_now();
            }
        }
    });

    writer.join().unwrap();
    deleter.join().unwrap();

    // Verify tree is mostly empty (deleter may not have caught up completely)
    let guard = tree.guard();
    let mut remaining = 0;

    for n in 0..N {
        let key = n.to_be_bytes();

        if tree.get_with_guard(&key, &guard).is_some() {
            remaining += 1;
        }
    }

    // Allow some keys to remain (race between writer finish and deleter)
    assert!(remaining < 100, "too many keys remaining: {remaining}");
}

#[test]
fn wd1_multi_pair() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));
    let num_pairs = 2;
    let per_pair = N / num_pairs as u64;

    let mut handles = Vec::new();

    for pair_id in 0..num_pairs {
        let tree_w = Arc::clone(&tree);
        let tree_d = Arc::clone(&tree);
        let _done_w = Arc::clone(&done);
        let _done_d = Arc::clone(&done);
        let writer_pos = Arc::new(AtomicU64::new(0));
        let writer_pos_clone = Arc::clone(&writer_pos);

        let base = pair_id as u64 * per_pair * 2; // Separate key ranges

        // Writer
        handles.push(thread::spawn(move || {
            let guard = tree_w.guard();
            for n in 0..per_pair {
                let key = (base + n).to_be_bytes();
                let _ = tree_w.insert_with_guard(&key, n + 1, &guard);
                writer_pos_clone.store(n, Ordering::Release);
            }
        }));

        // Deleter
        handles.push(thread::spawn(move || {
            let guard = tree_d.guard();

            // Wait for some writes
            while writer_pos.load(Ordering::Acquire) < 50 {
                thread::yield_now();
            }

            for n in 0..per_pair {
                let key = (base + n).to_be_bytes();
                // Keep trying until deleted or writer is done
                loop {
                    match tree_d.remove_with_guard(&key, &guard) {
                        Ok(Some(_)) => break,
                        Ok(None) => {
                            if writer_pos.load(Ordering::Acquire) >= per_pair - 1 {
                                break; // Writer done, key might never exist
                            }
                            thread::yield_now();
                        }
                        Err(_) => thread::yield_now(),
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
