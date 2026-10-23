use masstree::{MassTree, RangeBound, ScanEntry};
fn k(it: impl Iterator<Item=ScanEntry<u64>>) -> Vec<Vec<u8>> { it.map(|e|e.key).collect() }
fn main(){
    let keys: Vec<(Vec<u8>,u64)> = vec![
        (vec![0,2,97,97],10),(vec![0,98,97,1],18),(vec![1,254],5),(vec![1,254,0,255],19),
        (vec![49,50,51,52,53,54,55,56],6),
        (b"PREFIX00OTHER000MORE0000\0".to_vec(),1),
        (b"PREFIX00OTHER000MORE0000\x07".to_vec(),12),
        (b"PREFIX00OTHER000MORE0000\x08".to_vec(),3),
        (b"PREFIX00OTHER000MORE0000\x0b".to_vec(),21),
        (b"PREFIX00OTHER000MORE0000\x10".to_vec(),11),
        (b"PREFIX00SUFFIX00\x03".to_vec(),8),
        (b"SAMEAAAAcSAMEB2".to_vec(),2),
        (b"abcdefgh".to_vec(),17),(vec![254],4),(vec![255,97],0),
        (b"\xff\xff\xff\xff\xff\xff\xff\xffX".to_vec(),20),
    ];
    let t: MassTree<u64> = MassTree::new();
    for (key,v) in &keys { t.insert(key,*v); }
    let g=t.guard();
    println!("all={:?}", k(t.iter(&g)));
    let mut vv=vec![]; let n=t.scan_values(RangeBound::Unbounded, RangeBound::Unbounded, |v|{vv.push(v);true}, &g);
    println!("unbounded scan_values n={n} vv={vv:?}");
    for (s,e,label) in [
        (RangeBound::Included(b"123".as_slice()), RangeBound::Included(b"123\xff".as_slice()), "123..123ff"),
        (RangeBound::Included(b"123".as_slice()), RangeBound::Excluded(b"124".as_slice()), "123..124"),
        (RangeBound::Included(b"123".as_slice()), RangeBound::Unbounded, "123..inf"),
        (RangeBound::Unbounded, RangeBound::Included(b"123\xff".as_slice()), "..123ff"),
        (RangeBound::Included(b"P".as_slice()), RangeBound::Included(b"Q".as_slice()), "P..Q"),
    ] {
        let mut vv=vec![]; let n=t.scan_values(s,e,|v|{vv.push(v);true}, &g);
        let mut kk=vec![]; t.scan_intra_leaf_batch(s,e,|key,_|{kk.push(key.to_vec());true}, &g);
        println!("{label}: n={n} vv={vv:?} keyed={kk:?}");
    }
    // print tree via debug
    
    t.print_tree();
}
