use masstree::{MassTree, RangeBound};
fn main() {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for (i, k) in [&b""[..], b"\0", b"\0\0", b"a", b"aa", b"b"].iter().enumerate() { t.insert_with_guard(k, i as u64, &g); }
    let cases: Vec<(RangeBound, RangeBound)> = vec![
        (RangeBound::Unbounded, RangeBound::Excluded(b"".as_slice())),
        (RangeBound::Unbounded, RangeBound::Excluded(b"\0".as_slice())),
        (RangeBound::Excluded(b"".as_slice()), RangeBound::Unbounded),
        (RangeBound::Excluded(b"\0".as_slice()), RangeBound::Unbounded),
        (RangeBound::Included(b"a".as_slice()), RangeBound::Included(b"a".as_slice())),
        (RangeBound::Included(b"aa".as_slice()), RangeBound::Included(b"aa".as_slice())),
        (RangeBound::Excluded(b"a".as_slice()), RangeBound::Included(b"aa".as_slice())),
        (RangeBound::Included(b"aa".as_slice()), RangeBound::Excluded(b"b".as_slice())),
    ];
    for (s,e) in cases {
        let f: Vec<Vec<u8>> = t.range(s,e,&g).map(|x| x.key().to_vec()).collect();
        let r: Vec<Vec<u8>> = t.range(s,e,&g).rev().map(|x| x.key().to_vec()).collect();
        println!("{s:?}..{e:?} fwd={f:?} rev={r:?}");
    }
}
