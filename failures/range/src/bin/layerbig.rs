use std::collections::BTreeMap;
use masstree::{MassTree, RangeBound};
fn main(){
 for (label, depth) in [("one",1usize),("two",2),("three",3)] {
  let mut m:BTreeMap<Vec<u8>,u64>=BTreeMap::new();
  let t:MassTree<u64>=MassTree::new();
  for i in 0..600u64 {
   let mut k=Vec::new();
   for d in 0..depth { k.extend_from_slice(format!("CHUNK{d:03}").as_bytes()); }
   k.extend_from_slice(&i.to_be_bytes());
   m.insert(k.clone(),i); t.insert(&k,i);
  }
  let g=t.guard();
  let all:Vec<Vec<u8>>=t.iter(&g).map(|x|x.key).collect();
  let exp:Vec<Vec<u8>>=m.keys().cloned().collect();
  println!("{label}: all_ok={} n={}",all==exp,all.len());
  if all!=exp {for (i,(a,b)) in all.iter().zip(exp.iter()).enumerate(){if a!=b{println!(" first diff at {i}: got={a:?} exp={b:?}");break}}}
  let mut rev:Vec<Vec<u8>>=t.iter(&g).rev().map(|x|x.key).collect(); rev.reverse();
  println!("{label}: rev_ok={}",rev==exp);
  if rev!=exp { let gs:std::collections::BTreeSet<_>=rev.iter().cloned().collect(); let es:std::collections::BTreeSet<_>=exp.iter().cloned().collect(); println!(" missing={:?} extra={:?}", es.difference(&gs).take(3).collect::<Vec<_>>(), gs.difference(&es).take(3).collect::<Vec<_>>()); }
  // range around a middle key with suffix bound
  let mid=exp[300].clone(); let end=exp[450].clone();
  let e2:Vec<Vec<u8>>=m.range(mid.clone()..end.clone()).map(|(k,_)|k.clone()).collect();
  let a2:Vec<Vec<u8>>=t.range(RangeBound::Included(&mid),RangeBound::Excluded(&end),&g).map(|x|x.key).collect();
  let mut r2:Vec<Vec<u8>>=t.range(RangeBound::Included(&mid),RangeBound::Excluded(&end),&g).rev().map(|x|x.key).collect(); r2.reverse();
  let mut b2:Vec<Vec<u8>>=Vec::new(); t.scan_intra_leaf_batch(RangeBound::Included(&mid),RangeBound::Excluded(&end),|k,_|{b2.push(k.to_vec());true},&g);
  let mut rb2:Vec<Vec<u8>>=Vec::new(); t.scan_rev_batch(RangeBound::Included(&mid),RangeBound::Excluded(&end),|k,_|{rb2.push(k.to_vec());true},&g); rb2.reverse();
  println!("{label}: range fwd={} rev={} batch={} revbatch={} expN={}",a2==e2,r2==e2,b2==e2,rb2==e2,e2.len());
  if a2!=e2 {println!(" fwd first/last got {:?} {:?} exp {:?} {:?}",a2.first(),a2.last(),e2.first(),e2.last());}
  if r2!=e2 {println!(" rev first/last got {:?} {:?}",r2.first(),r2.last());}
  let pfx:Vec<u8>=exp[0].iter().take(depth*8).cloned().collect();
  let pexp:Vec<Vec<u8>>=exp.iter().filter(|k|k.starts_with(&pfx)).cloned().collect();
  let mut pg=Vec::new(); t.scan_prefix(&pfx,|k,_|{pg.push(k.to_vec());true},&g);
  let mut pv=0; t.scan_prefix_values(&pfx,|_|{pv+=1;true},&g);
  println!("{label}: pfx_ok={} vals={}/{}",pg==pexp,pv,pexp.len());
  if pg!=pexp {println!(" pfx first/last got {:?} {:?} exp {:?} {:?}",pg.first(),pg.last(),pexp.first(),pexp.last());}
 }
}
