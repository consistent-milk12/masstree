use masstree::MassTree15;

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    let g = tree.guard();
    tree.insert_with_guard(b"a", 1, &g);
    let p = tree.get_with_guard(b"a", &g).unwrap();
    println!("p before = {}", *p);
    tree.insert_value_with_guard(b"a", 2, &g);
    println!("p after write-through update = {} (same ValuePtr)", *p);
    // A concurrent writer doing the same while another thread derefs `p`
    // is a non-atomic read racing an atomic in-place store.
}
