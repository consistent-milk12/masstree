use masstree::{BatchEntry, BoxPolicy, MassTree15};
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
struct D(u64);
impl Drop for D {
    fn drop(&mut self) {
        let n = DROPS.fetch_add(1, Ordering::SeqCst) + 1;
        println!("DROP #{} of value {}", n, self.0);
    }
}

fn main() {
    let tree: MassTree15<D> = MassTree15::new();
    tree.insert(b"a", D(1));
    let g = tree.guard();
    let p = tree.get_with_guard(b"a", &g).unwrap();
    println!("got ptr for key a");

    // Safe API: wrap an existing tree-owned output in a new batch entry for a new key.
    let e: BatchEntry<BoxPolicy<D>> = BatchEntry::from_output(b"b".to_vec(), p);
    let r = tree.insert_batch_entries(vec![e], &g);
    println!("batch inserted={} updated={} total={}", r.inserted, r.updated, r.total());
    drop(r);
    drop(g);

    println!("about to remove a");
    let _ = tree.remove_with_guard(b"a", &tree.guard()).unwrap();
    println!("about to remove b");
    let _ = tree.remove_with_guard(b"b", &tree.guard()).unwrap();
    println!("removes done; drops so far={}", DROPS.load(Ordering::SeqCst));
    drop(tree);
    println!("after tree drop; drops={}", DROPS.load(Ordering::SeqCst));
}
