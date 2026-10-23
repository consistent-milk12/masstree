use masstree::{BatchEntry, BoxPolicy, MassTree15};

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"a", 1);

    let g = tree.guard();
    let p = tree.get_with_guard(b"a", &g).unwrap();
    let e: BatchEntry<BoxPolicy<u64>> = BatchEntry::from_output(b"b".to_vec(), p);
    let r = tree.insert_batch_entries(vec![e], &g);
    println!("batch inserted={} updated={}", r.inserted, r.updated);
    drop(r);
    println!("before: get(a)={:?} get(b)={:?}", *tree.get_with_guard(b"a", &g).unwrap(), *tree.get_with_guard(b"b", &g).unwrap());

    tree.insert_value_with_guard(b"a", 2, &g);
    println!("after updating a: get(a)={:?} get(b)={:?}",
        *tree.get_with_guard(b"a", &g).unwrap(), *tree.get_with_guard(b"b", &g).unwrap());
}
