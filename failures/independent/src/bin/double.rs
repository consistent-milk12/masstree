use masstree::{MassTree, RangeBound};
fn main() {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for i in 0..10u64 { let k = format!("k{i:04}"); t.insert_with_guard(k.as_bytes(), i, &g); }
    let mut it = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g);
    let mut out = Vec::new();
    for turn in 0..12 {
        let x = if turn % 2 == 0 { it.next() } else { it.next_back() };
        match x { Some(e) => out.push(*e.value()), None => { out.push(999); } }
    }
    println!("alternating={out:?}");

    let mut it = t.range(RangeBound::Included(b"k0003".as_slice()), RangeBound::Excluded(b"k0007".as_slice()), &g);
    let mut out = Vec::new();
    loop { match it.next() { Some(e) => out.push(*e.value()), None => break } }
    println!("forward={out:?}");
    let mut it = t.range(RangeBound::Included(b"k0003".as_slice()), RangeBound::Excluded(b"k0007".as_slice()), &g);
    let mut out = Vec::new();
    loop { match it.next_back() { Some(e) => out.push(*e.value()), None => break } }
    println!("backward={out:?}");
}
