//! Stress: concurrent chain coalesce of many adjacent empty leaves.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use masstree::MassTree15Inline;

fn k(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

#[test]
fn concurrent_chain_coalesce_adjacent() {
    for round in 0..30u64 {
        eprintln!("ROUND {round} start");
        let tree = Arc::new(MassTree15Inline::<u64>::new());
        for i in 0..300u64 {
            tree.insert(&k(i), i);
        }
        for i in 1..300u64 {
            let r = tree.remove(&k(i)).unwrap();
            assert_eq!(r, Some(i), "round {round}: remove {i}");
        }
        eprintln!("ROUND {round} removed, pending={}", tree.pending_coalesce());
        assert!(tree.pending_coalesce() > 0);

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let watchdog = thread::spawn(move || {
            if rx.recv_timeout(Duration::from_secs(5)).is_err() {
                eprintln!("WEDGED: concurrent coalesce round {round} did not finish");
                std::process::exit(42);
            }
        });

        let barrier = Arc::new(std::sync::Barrier::new(4));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                loop {
                    let g = tree.guard();
                    let n = tree.process_coalesce_batch(&g, 4);
                    if n == 0 {
                        break;
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        eprintln!("ROUND {round} coalesced");
        tx.send(()).unwrap();
        watchdog.join().unwrap();

        assert_eq!(tree.get(&k(0)), Some(0), "round {round}: key 0 survived");
        assert_eq!(tree.len(), 1, "round {round}");
        // Every empty leaf should be gone except the leftmost root leaf.
        assert_eq!(
            tree.pending_coalesce(),
            0,
            "round {round}: pending={} deferred={} abandoned={}",
            tree.pending_coalesce(),
            tree.coalesce_deferred(),
            tree.coalesce_abandoned()
        );
    }
}
