use masstree::{MassTree, RangeBound};

fn main() {
    // 1. Forward Included bound includes a shorter prefix with the same ikey.
    {
        let t: MassTree<u64> = MassTree::new();
        let g = t.guard();
        t.insert_with_guard(b"", 0, &g);
        t.insert_with_guard(b"\0", 1, &g);
        let got: Vec<Vec<u8>> = t.range(RangeBound::Included(b"\0".as_slice()), RangeBound::Unbounded, &g).map(|e| e.key().to_vec()).collect();
        println!("A Included([0]) -> {:?}", got);
        let mut p = Vec::new(); t.scan_prefix(b"\0", |k, _| { p.push(k.to_vec()); true }, &g);
        println!("A prefix([0]) -> {:?}", p);
    }
    // 2. Forward Excluded bound fails to exclude same-ikey keys.
    {
        let t: MassTree<u64> = MassTree::new();
        let g = t.guard();
        for (i, k) in [&b""[..], b"\0", b"\0\0", b"\0\0\0"].iter().enumerate() { t.insert_with_guard(k, i as u64, &g); }
        let got: Vec<Vec<u8>> = t.range(RangeBound::Excluded(b"\0\0".as_slice()), RangeBound::Unbounded, &g).map(|e| e.key().to_vec()).collect();
        println!("B Excluded([0,0]) -> {:?}", got);
    }
    // 3. Forward Included bound in a later leaf of a leaf-rooted layer skips the bound.
    {
        let t: MassTree<u64> = MassTree::new();
        let g = t.guard();
        for i in 0..20u8 { let mut k = vec![b'p'; 8]; k.push(i); t.insert_with_guard(&k, i as u64, &g); }
        let mut start = vec![b'p'; 8]; start.push(15);
        let got: Vec<u8> = t.range(RangeBound::Included(&start), RangeBound::Unbounded, &g).map(|e| e.key()[8]).collect();
        println!("C layer start=15 -> {:?}", got);
        let mut p = Vec::new(); t.scan_prefix(&start, |k, _| { p.push(k.to_vec()); true }, &g);
        println!("C prefix exact -> {:?}", p);
        println!("C last -> {:?}", t.last().map(|e| e.key));
    }
    // 4. Reverse full scan/last after rightmost leaf becomes empty.
    {
        let t: MassTree<u64> = MassTree::new();
        let g = t.guard();
        for i in 0..20u64 { let k = format!("k{i:04}"); t.insert_with_guard(k.as_bytes(), i, &g); }
        for i in 15..20u64 { let k = format!("k{i:04}"); t.remove_with_guard(k.as_bytes(), &g).unwrap(); }
        let rev: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| String::from_utf8(e.key().to_vec()).unwrap()).collect();
        println!("D len={} last={:?} rev={:?}", t.len(), t.last().map(|e| e.key), rev);
    }
    // 5. Reverse upper bound loses a key sharing ikey.
    {
        let t: MassTree<u64> = MassTree::new();
        let g = t.guard();
        for (i, k) in [&b""[..], b"\0", b"\0\0"].iter().enumerate() { t.insert_with_guard(k, i as u64, &g); }
        let fwd: Vec<Vec<u8>> = t.range(RangeBound::Unbounded, RangeBound::Included(b"\0".as_slice()), &g).map(|e| e.key().to_vec()).collect();
        let rev: Vec<Vec<u8>> = t.range(RangeBound::Unbounded, RangeBound::Included(b"\0".as_slice()), &g).rev().map(|e| e.key().to_vec()).collect();
        println!("E fwd={:?} rev={:?}", fwd, rev);
    }
}
