use masstree::{MassTree, RangeBound};
fn main(){
 let t:MassTree<u64>=MassTree::new();
 t.insert(b"a",1); t.insert(b"zzzzzzzzX",2); t.insert(b"zzzzzzzzY",3);
 let g=t.guard();
 for (s,e,l) in [
  (RangeBound::Included(b"a".as_slice()),RangeBound::Excluded(b"b".as_slice()),"a..b"),
  (RangeBound::Included(b"a".as_slice()),RangeBound::Included(b"a".as_slice()),"a..a"),
  (RangeBound::Included(b"b".as_slice()),RangeBound::Included(b"c".as_slice()),"b..c empty-ish"),
 ] {
  let mut v=vec![]; let n=t.scan_values(s,e,|x|{v.push(x);true},&g);
  let mut kv=vec![]; let m=t.scan_intra_leaf_batch(s,e,|k,_|{kv.push(k.to_vec());true},&g);
  println!("{l}: vals n={n} {v:?} | keyed n={m} {kv:?}");
 }
}
