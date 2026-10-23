use masstree::{MassTree, RangeBound};

fn main() {
    let t: MassTree<u64> = MassTree::new();
    for k in [b"a".as_slice(), b"b", b"c"] {
        t.insert(k, k[0] as u64);
    }
    let g = t.guard();

    let mut it = t.range(
        RangeBound::Included(b"a".as_slice()),
        RangeBound::Included(b"c".as_slice()),
        &g,
    );
    let mut forward = Vec::new();
    for _ in 0..4 {
        forward.push(it.next().map(|e| e.key().to_vec()));
    }
    let mut backward_after_exhaustion = Vec::new();
    while let Some(e) = it.next_back() {
        backward_after_exhaustion.push(e.key().to_vec());
    }
    println!("forward={forward:?}");
    println!("backward_after_exhaustion={backward_after_exhaustion:?}");
    assert!(backward_after_exhaustion.is_empty());
}
