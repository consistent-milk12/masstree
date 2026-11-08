//! Bounded forward/reverse scan regression with concurrent slot reuse and splits.
//! Barriers start each writer/reader round; channel timeouts bound hangs.
use masstree::{LeafPolicy, MassTree, MassTree15, MassTreeGeneric, RangeBound, SeizeAllocator};
use std::{
    sync::{Arc, Barrier, mpsc},
    thread,
    time::Duration,
};

fn key(value: u64, depth: usize) -> Vec<u8> {
    let mut key = vec![b'p'; depth * 8];
    key.extend_from_slice(&value.to_be_bytes());
    key
}

fn check_order(previous: &mut Option<u64>, value: u64, reverse: bool, round: usize) {
    if let Some(last) = *previous {
        assert!(
            if reverse { value < last } else { value > last },
            "round {round}: out of order {last} -> {value}, reverse={reverse}"
        );
    }
    *previous = Some(value);
}

fn exercise<P: LeafPolicy<Value = u64>>(tree: MassTreeGeneric<P, SeizeAllocator<P>>, depth: usize) {
    let tree = Arc::new(tree);
    let size = if cfg!(miri) { 20u64 } else { 48u64 };
    let rounds = if cfg!(miri) { 4 } else { 128 };
    for i in 0..size {
        tree.insert(&key(i, depth), i);
    }
    let gate = Arc::new(Barrier::new(2));
    let (done, receive) = mpsc::channel();
    let writer_tree = Arc::clone(&tree);
    let writer_gate = Arc::clone(&gate);
    let writer_done = done.clone();
    let writer = thread::spawn(move || {
        let guard = writer_tree.guard();
        for round in 0..rounds {
            writer_gate.wait();
            for step in 0..if cfg!(miri) { 6 } else { size } {
                let value = (round as u64 * 7 + step) % size;
                let key = key(value, depth);
                writer_tree.remove_with_guard(&key, &guard).unwrap();
                writer_tree.insert_with_guard(&key, value, &guard);
            }
            writer_gate.wait();
        }
        writer_done.send(()).unwrap();
    });
    let reader = thread::spawn(move || {
        let guard = tree.guard();
        for round in 0..rounds {
            gate.wait();
            let mut previous = None;
            let mut visits = 0;
            match round % 4 {
                0 | 3 => {
                    let mut iter = tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard);
                    let reverse = round % 4 == 0;
                    while let Some(entry) = if reverse {
                        iter.next_back()
                    } else {
                        iter.next()
                    } {
                        // SAFETY: The output was validated by the iterator and this
                        // tree's guard remains held through the atomic/owned copy.
                        let value = unsafe { P::clone_value_from_output(&entry.value) };
                        assert_eq!(
                            entry.key,
                            key(value, depth),
                            "round {round}: wrong key/value"
                        );
                        check_order(&mut previous, value, reverse, round);
                        visits += 1;
                        assert!(visits <= size, "round {round}: duplicate emission");
                    }
                }
                1 => {
                    let count = tree
                        .range(RangeBound::Unbounded, RangeBound::Unbounded, &guard)
                        .rev_for_each_intra_leaf_batch(|k, output| {
                            // SAFETY: The callback's output remains protected by guard.
                            let value = unsafe { P::clone_value_from_output(&output) };
                            assert_eq!(
                                k,
                                key(value, depth),
                                "round {round}: wrong batch key/value"
                            );
                            check_order(&mut previous, value, true, round);
                            visits += 1;
                            assert!(visits <= size, "round {round}: duplicate batch emission");
                            true
                        });
                    assert_eq!(count as u64, visits);
                }
                _ => {
                    let count = tree.scan_values_rev(
                        RangeBound::Unbounded,
                        RangeBound::Unbounded,
                        |output| {
                            // SAFETY: The callback's output remains protected by guard.
                            let value = unsafe { P::clone_value_from_output(&output) };
                            check_order(&mut previous, value, true, round);
                            visits += 1;
                            assert!(visits <= size, "round {round}: duplicate values emission");
                            true
                        },
                        &guard,
                    );
                    assert_eq!(count as u64, visits);
                }
            }
            gate.wait();
        }
        done.send(()).unwrap();
    });
    for _ in 0..2 {
        receive
            .recv_timeout(Duration::from_secs(30))
            .expect("scan/writer watchdog expired");
    }
    writer.join().unwrap();
    reader.join().unwrap();
}

fn run() {
    for depth in [0, 1, 31] {
        exercise(MassTree::<u64>::new(), depth);
        exercise(MassTree15::<u64>::new(), depth);
    }
}

#[cfg(not(test))]
fn main() {
    run();
    println!("bounded forward/reverse races passed");
}

#[test]
fn concurrent_scans_preserve_order_and_key_value_identity() {
    run();
}
