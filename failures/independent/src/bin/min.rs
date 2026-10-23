use masstree::{MassTree, RangeBound};

fn dump(tree: &MassTree<u64>, guard: &seize::LocalGuard<'_>, start: RangeBound<'_>, end: RangeBound<'_>) -> Vec<Vec<u8>> {
    tree.range(start, end, guard).map(|e| e.key().to_vec()).collect()
}

fn main() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    let keys: Vec<Vec<u8>> = vec![
        b"shared00A".to_vec(),
        b"shared00B".to_vec(),
        b"shared00C".to_vec(),
        b"shared00AA".to_vec(),
        b"shared00AB".to_vec(),
        b"shared00B0".to_vec(),
        b"shared00BA".to_vec(),
    ];
    for (i, k) in keys.iter().enumerate() {
        tree.insert_with_guard(k, i as u64, &guard);
    }
    for (start, end) in [
        (RangeBound::Included(&b"shared00A"[..]), RangeBound::Unbounded),
        (RangeBound::Included(&b"shared00B"[..]), RangeBound::Included(&b"shared00B"[..])),
        (RangeBound::Included(&b"shared00AA"[..]), RangeBound::Unbounded),
        (RangeBound::Unbounded, RangeBound::Included(&b"shared00B"[..])),
        (RangeBound::Unbounded, RangeBound::Excluded(&b"shared00B"[..])),
    ] {
        let fwd = dump(&tree, &guard, start, end);
        let rev: Vec<Vec<u8>> = tree.range(start, end, &guard).rev().map(|e| e.key().to_vec()).collect();
        let mut expected_rev = fwd.clone();
        expected_rev.reverse();
        println!("{start:?}..{end:?}\n  fwd={fwd:?}\n  rev={rev:?}\n  exp_rev={expected_rev:?}");
    }
}
