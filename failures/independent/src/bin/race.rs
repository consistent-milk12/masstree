use std::sync::{Arc, Barrier};
use masstree::MassTree15;

fn main() {
    let tree: Arc<MassTree15<u64>> = Arc::new(MassTree15::new());
    {
        let g = tree.guard();
        tree.insert_with_guard(b"k", 1, &g);
    }
    let barrier = Arc::new(Barrier::new(2));
    let t2 = tree.clone();
    let b2 = barrier.clone();
    let writer = std::thread::spawn(move || {
        
        b2.wait();
        for i in 0..10_000u64 {
            t2.insert(b"k", i);
        }
    });
    let g = tree.guard();
    barrier.wait();
    let ptr = tree.get_with_guard(b"k", &g).unwrap();
    let mut sum = 0u64;
    for _ in 0..10_000 {
        sum = sum.wrapping_add(*ptr);
    }
    writer.join().unwrap();
    println!("{sum}");
}
