use std::sync::{Arc, Barrier};
use masstree::MassTree;

fn main() {
    let threads = 8usize;
    let iters = 200_000usize;
    let tree: Arc<MassTree<u64>> = Arc::new(MassTree::new());
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::new();
    for t in 0..threads {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            let key = vec![b'a'; t + 1];
            let g = tree.guard();
            barrier.wait();
            for i in 0..iters {
                let v = ((t as u64) << 32) | i as u64;
                tree.insert_with_guard(&key, v, &g);
                match tree.remove_with_guard(&key, &g) {
                    Ok(Some(old)) if old == v => {}
                    other => {
                        eprintln!("MISMATCH thread={t} iter={i} key_len={} expected={v} got={other:?}", key.len());
                        std::process::exit(2);
                    }
                }
            }
        }));
    }
    barrier.wait();
    for h in handles { h.join().unwrap(); }
    println!("len={} expected=0", tree.len());
}
