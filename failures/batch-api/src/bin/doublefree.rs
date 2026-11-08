// SAFE-03 regression (formerly a double-free).
//
// Each `BatchEntry` owns a distinct value; removal retires each allocation
// exactly once.
use masstree::{BatchEntry, BoxPolicy, MassTree15};
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
struct D(u64);

impl Drop for D {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let tree: MassTree15<D> = MassTree15::new();
    tree.insert(b"a", D(1));

    {
        let g = tree.guard();
        let entry: BatchEntry<BoxPolicy<D>> = BatchEntry::new(b"b".to_vec(), D(2));
        let r = tree.insert_batch_entries(vec![entry], &g);
        assert_eq!(r.inserted, 1);

        let _ = tree.remove_with_guard(b"a", &g).unwrap();
        let _ = tree.remove_with_guard(b"b", &g).unwrap();
    }

    drop(tree);

    let drops = DROPS.load(Ordering::SeqCst);
    assert_eq!(drops, 2, "each value must be dropped exactly once");
    println!("drops={drops}");
}
