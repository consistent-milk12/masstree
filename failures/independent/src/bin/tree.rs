use masstree::MassTree;

fn main() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    for i in 0..20u64 {
        let mut k = vec![b'p'; 8];
        k.push(i as u8);
        tree.insert_with_guard(&k, i, &guard);
    }
    println!("{}", tree.print_tree());
    for i in [0u8, 14, 15, 16, 19] {
        let start = { let mut k = vec![b'p'; 8]; k.push(i); k };
        let got: Vec<u8> = tree.range(masstree::RangeBound::Included(&start), masstree::RangeBound::Unbounded, &guard).map(|e| e.key()[8]).collect();
        println!("start={i} -> {got:?}");
    }}
