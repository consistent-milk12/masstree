//! Repro: `remove` can return `None` for a present key when a concurrent
//! insert splits the leaf and moves the key to the new right sibling.
//!
//! `remove_concurrent_generic` never calls `advance_to_key_by_bound_generic`
//! (unlike insert) and never follows `next`, so once its root load / `reach`
//! has targeted the pre-split left leaf, `search_for_remove` searches only
//! that leaf. If the key moved right, the search returns NotFound -> Ok(None)
//! even though the key is still in the tree.

use masstree::MassTree15Inline;
use std::sync::{Arc, Barrier};
use std::thread;

fn k(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

#[test]
fn lost_remove_during_split() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let mut detected = 0usize;
    let attempts: usize = std::env::var("ATTEMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300usize);

    for attempt in 0..attempts {
        eprintln!("ATTEMPT {attempt} start");
        // Reset to exactly one full leaf: keys 0,10,...,140.
        let g = tree.guard();
        for i in 0..15u64 {
            let key = i * 10;
            for _ in 0..4 {
                let _ = tree.remove_with_guard(&k(key), &g);
                if tree.get_with_guard(&k(key), &g).is_none() {
                    break;
                }
            }
        }
        for _ in 0..4 {
            let _ = tree.remove_with_guard(&k(75), &g);
            if tree.get_with_guard(&k(75), &g).is_none() {
                break;
            }
        }
        tree.process_coalesce(&g);
        drop(g);

        for i in 0..15u64 {
            tree.insert(&k(i * 10), i * 10);
        }
        for i in 0..15u64 {
            assert_eq!(tree.get(&k(i * 10)), Some(i * 10), "reset failed for {i}");
        }

        // One inserter forces a real mid-leaf split (key 75 sorts in the
        // middle), moving keys 70..140 to the new right sibling. Removers
        // target keys that move right.
        let barrier = Arc::new(Barrier::new(9));
        let inserter = {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                eprintln!("I insert start");
                tree.insert(&k(75), 75);
                eprintln!("I insert done");
            })
        };
        let mut removers = Vec::new();
        for i in 7..=14u64 {
            let key = i * 10;
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            removers.push(thread::spawn(move || {
                barrier.wait();
                eprintln!("R {key} start");
                let r = tree.remove(&k(key)).unwrap().is_some();
                eprintln!("R {key} done {r}");
                (key, r)
            }));
        }
        inserter.join().unwrap();
        let mut results = Vec::new();
        for h in removers {
            results.push(h.join().unwrap());
        }

        eprintln!("ATTEMPT {attempt} joined");
        for (key, removed) in results {
            if !removed {
                // It returned None. If the key is still present, the remove was
                // lost (it existed when the call started and nobody else
                // removes this key).
                if tree.get(&k(key)).is_some() {
                    detected += 1;
                    if detected <= 5 {
                        println!(
                            "attempt {attempt}: remove({key}) returned None but key is present"
                        );
                    }
                }
            }
        }

        // Remove the split-trigger key for the reset.
        let _ = tree.remove(&k(75));
    }

    println!("lost removes detected: {detected} / {} attempts", attempts);
    assert_eq!(detected, 0, "remove() lost present keys during splits");
}
