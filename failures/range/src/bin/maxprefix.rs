use masstree::MassTree;

fn main() {
    let t: MassTree<u64> = MassTree::new();
    let a255 = vec![b'a'; 255];
    let a256 = vec![b'a'; 256];
    let a255b = {
        let mut k = a255.clone();
        k.push(b'b');
        k
    };
    t.insert(&a255, 1);
    t.insert(&a256, 2);
    t.insert(&a255b, 3);
    let g = t.guard();
    let mut keys = Vec::new();
    t.scan_prefix(&a256, |k, _| {
        keys.push(k.to_vec());
        true
    }, &g);
    let mut vals = Vec::new();
    t.scan_prefix_values(&a256, |v| {
        vals.push(v);
        true
    }, &g);
    println!("keys={keys:?} vals={vals:?}");
    assert_eq!(keys.len(), 1);
    assert_eq!(vals.len(), 1);
}
