//! Debug reproduction for the "insert then get loses key" flaky test.
//!
//! This reproduces the bug where:
//! - Thread inserts a key, insert returns Ok
//! - Immediate get returns None
//! - But the key IS findable via leaf chain scan
//!
//! Run with: `cargo run --example debug_lost_key --release`
//!
//! The example matches the exact parameters of the failing unit test:
//! `concurrent_insert_then_get_does_not_lose_key`

#![expect(clippy::unwrap_used)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use masstree::{InternodeNode, LeafNode15, LeafValue, MassTree, NodeVersion};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// Match the failing unit test parameters exactly
const NUM_THREADS: usize = 4;
const OPS_PER_THREAD: usize = 500;

fn main() {
    println!("Debug: Reproducing lost key bug");
    println!("Threads: {NUM_THREADS}, Ops/thread: {OPS_PER_THREAD}");
    println!("Key pattern: t * 10_000 + i (matches unit test exactly)\n");

    let start = Instant::now();
    let mut run = 0u64;

    loop {
        run += 1;

        if let Some(failure) = run_single_iteration(run) {
            let elapsed = start.elapsed();
            println!("\n============================================================");
            println!("=== BUG REPRODUCED at run {run} ===");
            println!("============================================================");
            print_failure(&failure);
            diagnose_failure(&failure);

            println!(
                "\nFound after {} runs in {:.2}s ({:.0} runs/sec)",
                run,
                elapsed.as_secs_f64(),
                run as f64 / elapsed.as_secs_f64()
            );
            return;
        }

        if run % 100 == 0 {
            let elapsed = start.elapsed();
            println!(
                "Completed {} runs in {:.2}s ({:.0} runs/sec)",
                run,
                elapsed.as_secs_f64(),
                run as f64 / elapsed.as_secs_f64()
            );
        }
    }
}

struct Failure {
    run: u64,
    thread_id: usize,
    iteration: usize,
    key_val: u64,
    tree_len: usize,
    in_leaf_chain: bool,
    tree: Arc<MassTree<u64>>,
}

fn run_single_iteration(run: u64) -> Option<Failure> {
    let tree = Arc::new(MassTree::<u64>::new());
    let missing_key = Arc::new(AtomicU64::new(u64::MAX));
    let missing_thread = Arc::new(AtomicUsize::new(usize::MAX));
    let missing_i = Arc::new(AtomicUsize::new(usize::MAX));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let missing_key = Arc::clone(&missing_key);
            let missing_thread = Arc::clone(&missing_thread);
            let missing_i = Arc::clone(&missing_i);

            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..OPS_PER_THREAD {
                    // Key computation matches unit test exactly
                    let key_val: u64 = (t * 10_000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        let _ = missing_key.compare_exchange(
                            u64::MAX,
                            key_val,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        );
                        missing_thread.store(t, Ordering::SeqCst);
                        missing_i.store(i, Ordering::SeqCst);
                        break;
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let key_val = missing_key.load(Ordering::SeqCst);
    if key_val != u64::MAX {
        let thread_id = missing_thread.load(Ordering::SeqCst);
        let iteration = missing_i.load(Ordering::SeqCst);
        let in_leaf_chain = scan_leaf_chain_contains(&tree, key_val);

        Some(Failure {
            run,
            thread_id,
            iteration,
            key_val,
            tree_len: tree.len(),
            in_leaf_chain,
            tree,
        })
    } else {
        None
    }
}

/// Scan the leaf chain directly (bypasses interior node routing).
/// This is the same logic as the unit test's `scan_leaf_chain_contains`.
fn scan_leaf_chain_contains(tree: &MassTree<u64>, target: u64) -> bool {
    let ikey = target; // For 8-byte keys, ikey == key value
    let guard = tree.guard();

    // Descend to the leftmost leaf from the current root
    let mut node: *mut u8 = tree.debug_root_ptr(&guard);
    let mut depth = 0;

    while !node.is_null() {
        depth += 1;
        if depth > 128 {
            println!("WARNING: Unexpected depth while descending");
            break;
        }

        // SAFETY: node is a tree node protected by the guard
        let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };
        let _ = version.stable();

        if version.is_leaf() {
            break;
        }

        // SAFETY: Not a leaf, so this is an internode
        let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };
        node = unsafe { inode.child_unguarded(0) };
    }

    if node.is_null() {
        return false;
    }

    // Now scan the leaf chain
    let mut leaf_ptr: *mut LeafNode15<LeafValue<u64>> = node.cast();
    let mut steps = 0;

    while !leaf_ptr.is_null() {
        steps += 1;
        if steps > 100_000 {
            println!("WARNING: Possible leaf chain cycle detected");
            break;
        }

        // SAFETY: leaf_ptr is protected by the guard
        let leaf: &LeafNode15<LeafValue<u64>> = unsafe { &*leaf_ptr };
        let _ = leaf.version().stable();

        let perm = leaf.permutation();
        for i in 0..perm.size() {
            let slot = perm.get(i);
            if leaf.ikey(slot) == ikey && leaf.keylenx(slot) == 8 {
                let ptr = leaf.leaf_value_ptr(slot);
                if !ptr.is_null() {
                    return true;
                }
            }
        }

        // SAFETY: Single-threaded at this point (after join)
        leaf_ptr = unsafe { leaf.safe_next_unguarded() };
    }

    false
}

fn print_failure(failure: &Failure) {
    println!("Run:        {}", failure.run);
    println!("Thread:     {}", failure.thread_id);
    println!("Iteration:  {}", failure.iteration);
    println!(
        "Key:        {} (0x{:016x})",
        failure.key_val, failure.key_val
    );
    println!("Tree len:   {}", failure.tree_len);
    println!("In leaf chain: {}", failure.in_leaf_chain);

    if failure.in_leaf_chain {
        println!("\n*** ROUTING BUG: Key exists in leaf chain but get() can't find it ***");
        println!("This indicates interior node routing is broken after a split.");
    } else {
        println!("\n*** INSERT BUG: Key not in leaf chain - insertion failed silently ***");
    }
}

fn diagnose_failure(failure: &Failure) {
    let guard = failure.tree.guard();
    let key = failure.key_val.to_be_bytes();

    println!("\n--- Post-failure diagnosis ---");

    // Try get() again now that threads are joined
    let get_now = failure.tree.get_with_guard(&key, &guard);
    println!("get() after join: {:?}", get_now);

    // Check neighboring keys
    println!("\nNeighboring keys via get():");
    for offset in -5i64..=5 {
        let check_val = (failure.key_val as i64 + offset) as u64;
        let check_key = check_val.to_be_bytes();
        let result = failure.tree.get_with_guard(&check_key, &guard);
        let marker = if offset == 0 { " <-- MISSING" } else { "" };
        let status = if result.is_some() { "found" } else { "MISS" };
        println!(
            "  0x{:016x} ({:>6}): {}{marker}",
            check_val, check_val, status
        );
    }

    // Check what keys are around this in the same thread's sequence
    println!("\nThread {}'s nearby keys:", failure.thread_id);
    let t = failure.thread_id;
    for i_offset in -3i64..=3 {
        let i = (failure.iteration as i64 + i_offset) as usize;
        if i < OPS_PER_THREAD {
            let kv = (t * 10_000 + i) as u64;
            let k = kv.to_be_bytes();
            let result = failure.tree.get_with_guard(&k, &guard);
            let marker = if i_offset == 0 { " <-- MISSING" } else { "" };
            let status = if result.is_some() { "found" } else { "MISS" };
            println!("  t={t} i={i:>3} key={kv:>6}: {status}{marker}");
        }
    }
}
