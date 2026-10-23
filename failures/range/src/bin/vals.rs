use masstree::{MassTree, RangeBound};
fn main(){
 let t:MassTree<u64>=MassTree::new();
 t.insert(b"a",1); t.insert(b"b",2); t.insert(b"c",3);
 let g=t.guard();
 for (s,e,l) in [
  (RangeBound::Included(b"b".as_slice()),RangeBound::Included(b"a".as_slice()),"b..a empty"),
  (RangeBound::Excluded(b"c".as_slice()),RangeBound::Included(b"a".as_slice()),"exc c..a empty"),
  (RangeBound::Included(b"b".as_slice()),RangeBound::Excluded(b"b".as_slice()),"b..exc b empty"),
  (RangeBound::Included(b"c".as_slice()),RangeBound::Included(b"b".as_slice()),"c..b empty"),
 ] {
  let mut v=vec![]; let n=t.scan_values(s,e,|x|{v.push(x);true},&g);
  let mut r=vec![]; let m=t.scan_values_rev(s,e,|x|{r.push(x);true},&g);
  println!("{l}: fwd n={n} {v:?} | rev n={m} {r:?}");
 }
 // reverse batch valid range where end is suffix and start is suffix
 let t2:MassTree<u64>=MassTree::new();
 t2.insert(b"abc",1); t2.insert(b"abd",2); t2.insert(b"abe",3);
 let g2=t2.guard();
 for (s,e,l) in [
  (RangeBound::Included(b"abd".as_slice()),RangeBound::Included(b"abd".as_slice()),"eq abd"),
  (RangeBound::Included(b"abc".as_slice()),RangeBound::Included(b"abd".as_slice()),"abc..abd"),
  (RangeBound::Excluded(b"abc".as_slice()),RangeBound::Included(b"abd".as_slice()),"excabc..abd"),
 ] {
  let mut v=vec![]; let n=t2.scan_rev_batch(s,e,|k,_|{v.push(k.to_vec());true},&g2);
  println!("revbatch {l}: n={n} {v:?}");
 }
}
