use masstree::MassTree;

#[test]
fn prepend_split_lost_key_consequences() {
    let tree: MassTree<usize> = MassTree::new();
    let a = b"AAAAAAAA".as_slice();
    let a_long = b"AAAAAAAAA".as_slice();

    tree.insert(a, 1usize);
    tree.insert(a_long, 2usize);
    for (i, k) in [b"B", b"C", b"D", b"E", b"F", b"G", b"H", b"I", b"J", b"K", b"L", b"M", b"N"]
        .iter()
        .enumerate()
    {
        tree.insert(*k, 100 + i);
    }
    tree.insert(b"AAA", 16usize);

    let dump = |label: &str| {
        let guard = tree.guard();
        let mut v = Vec::new();
        for e in tree.iter(&guard) {
            v.push((e.key.to_vec(), e.value));
        }
        drop(guard);
        eprintln!("--- {label}: {} entries, len_counter={}", v.len(), tree.len());
        for (k, val) in v {
            eprintln!("  {k:?} => {val}");
        }
    };

    dump("after split");
    eprintln!("get original = {:?}", tree.get(a));
    let old = tree.insert(a, 77usize);
    eprintln!("second insert old = {old:?}");
    dump("after duplicate insert");
    eprintln!("remove -> {:?}", tree.remove(a));
    dump("after remove");
}
