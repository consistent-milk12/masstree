// SAFE-03 regression (formerly cross-key aliasing + double-free).
//
// `BatchEntry` now owns a real user value and its fields are private. There is
// no safe way to wrap a tree-owned output handle in an entry, so inserting a
// key cannot alias a value stored under another key.
use masstree::{BatchEntry, BoxPolicy, MassTree15};

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"a", 1);

    let g = tree.guard();

    let entry: BatchEntry<BoxPolicy<u64>> = BatchEntry::new(b"b".to_vec(), 2);
    let r = tree.insert_batch_entries(vec![entry], &g);
    assert_eq!(r.inserted, 1);
    assert_eq!(r.updated, 0);

    assert_eq!(*tree.get_with_guard(b"a", &g).unwrap(), 1);
    assert_eq!(*tree.get_with_guard(b"b", &g).unwrap(), 2);

    tree.insert_value_with_guard(b"a", 3, &g);
    assert_eq!(*tree.get_with_guard(b"a", &g).unwrap(), 3);
    assert_eq!(*tree.get_with_guard(b"b", &g).unwrap(), 2);

    println!("no cross-key aliasing");
}
