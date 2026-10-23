use masstree::{MassTree, RangeBound};
fn main(){
 let t:MassTree<u64>=MassTree::new();
 let mut keys=vec![];
 for i in 0..40u64 { let mut k=b"PREFIX00".to_vec(); k.extend_from_slice(&i.to_be_bytes()); keys.push(k); t.insert(&keys[i as usize],i); }
 let g=t.guard();
 println!("all={}", t.iter(&g).count());
 for idx in [0usize,5,15,16,20,30,39] {
  let s=keys[idx].clone();
  let exp:Vec<Vec<u8>>=keys[idx..].to_vec();
  let got:Vec<Vec<u8>>=t.range(RangeBound::Included(&s),RangeBound::Unbounded,&g).map(|x|x.key).collect();
  let got_exc:Vec<Vec<u8>>=t.range(RangeBound::Excluded(&s),RangeBound::Unbounded,&g).map(|x|x.key).collect();
  let exp_exc:Vec<Vec<u8>>=keys[idx+1..].to_vec();
  println!("start idx {idx}: inc_ok={} n={} exc_ok={}",got==exp,got.len(),got_exc==exp_exc);
  if got!=exp { println!("  inc first={:?} expfirst={:?}",got.first(),exp.first()); }
  if got_exc!=exp_exc { println!("  exc first={:?} expfirst={:?}",got_exc.first(),exp_exc.first()); }
  // prefix full key
  let mut pk=vec![]; t.scan_prefix(&s,|k,_|{pk.push(k.to_vec());true},&g);
  println!("  pfx full: got={} exp=1 ok={}",pk.len(),pk==vec![s.clone()]);
  // reverse
  let mut rv:Vec<Vec<u8>>=t.range(RangeBound::Included(&s),RangeBound::Unbounded,&g).rev().map(|x|x.key).collect(); rv.reverse();
  println!("  rev_ok={}",rv==exp);
 }
}
