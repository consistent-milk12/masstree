use masstree::{MassTree, RangeBound};
fn r(t:&MassTree<u64>,s:RangeBound<'_>,e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();let mut v:Vec<_>=t.range(s,e,&g).rev().map(|x|x.key).collect();v.reverse();v}
fn main(){
 for (label,keys) in [
  ("small", vec![b"abcdefghX".to_vec(),b"abcdefghY".to_vec(),b"abcdefghZ".to_vec()]),
  ("big", (0..40u64).map(|i|{let mut k=b"PREFIX00".to_vec();k.extend_from_slice(&i.to_be_bytes());k}).collect::<Vec<_>>()),
 ] {
  let t:MassTree<u64>=MassTree::new();
  for (i,k) in keys.iter().enumerate(){t.insert(k,i as u64);}
  let p=&keys[0][..8];
  for (s,e,l) in [
   (RangeBound::Unbounded,RangeBound::Included(p),"end inc 8pfx"),
   (RangeBound::Unbounded,RangeBound::Excluded(p),"end exc 8pfx"),
   (RangeBound::Included(p),RangeBound::Unbounded,"start inc 8pfx"),
  ] {
   let got=r(&t,s,e);
   let exp:Vec<Vec<u8>>=keys.iter().filter(|k| (match s {RangeBound::Unbounded=>true,RangeBound::Included(b)=>k.as_slice()>=b,RangeBound::Excluded(b)=>k.as_slice()>b}) && (match e {RangeBound::Unbounded=>true,RangeBound::Included(b)=>k.as_slice()<=b,RangeBound::Excluded(b)=>k.as_slice()<b})).cloned().collect();
   let g=t.guard(); let mut revb=vec![]; t.scan_rev_batch(s,e,|k,_|{revb.push(k.to_vec());true},&g); revb.reverse();
   println!("{label} {l}: rev_ok={} n={} exp={} revbatch_n={} revbatch_ok={}",got==exp,got.len(),exp.len(),revb.len(),revb==exp);
   if got!=exp {println!("  got first/last {:?} {:?}",got.first(),got.last());}
  }
 }
}
