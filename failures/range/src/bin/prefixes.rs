use masstree::{MassTree, RangeBound};
fn main(){
 let mut keys:Vec<Vec<u8>>=vec![vec![],vec![0xff;255],vec![0xff;256],vec![b'a';255],vec![b'a';256],vec![b'a';255].into_iter().chain(vec![b'b']).collect(),vec![0;8],vec![0;9],vec![0;16],vec![0u8;7],vec![0u8;7].into_iter().chain(vec![1]).collect()];
 keys.sort(); keys.dedup();
 let t:MassTree<u64>=MassTree::new();
 for (i,k) in keys.iter().enumerate(){t.insert(k,i as u64);}
 let g=t.guard();
 for p in [vec![],vec![0xff;255],vec![0xff;256],vec![b'a';255],vec![b'a';256],vec![b'a';254],vec![0;7],vec![0;8],vec![0;15],vec![0;16],vec![0u8;7].into_iter().chain(vec![0]).collect::<Vec<_>>()] {
  let exp:Vec<Vec<u8>>=keys.iter().filter(|k|k.starts_with(&p)).cloned().collect();
  let mut got=vec![]; t.scan_prefix(&p,|k,_|{got.push(k.to_vec());true},&g);
  let mut vals=0; t.scan_prefix_values(&p,|_|{vals+=1;true},&g);
  println!("pfx_len={} pfx_tail={:?}: exp={} got={} vals={} ok={} val_ok={}", p.len(), &p[p.len().saturating_sub(4)..], exp.len(), got.len(), vals, got==exp, vals==exp.len());
  if got!=exp { println!(" exp={exp:?}\n got={got:?}"); }
  if vals!=exp.len() { println!(" VALS MISMATCH"); }
 }
 // ranges on max keys
 for (s,e,l) in [
   (RangeBound::Included(vec![0xff;255].as_slice()),RangeBound::Unbounded,"inc 255ff"),
   (RangeBound::Included(vec![0xff;256].as_slice()),RangeBound::Unbounded,"inc 256ff"),
   (RangeBound::Unbounded,RangeBound::Included(vec![0xff;255].as_slice()),"end inc 255ff"),
   (RangeBound::Unbounded,RangeBound::Included(vec![0xff;256].as_slice()),"end inc 256ff"),
   (RangeBound::Included(vec![b'a';255].as_slice()),RangeBound::Included(vec![b'a';256].as_slice()),"a255..a256"),
 ] {
  let exp:Vec<Vec<u8>>=keys.iter().filter(|k| (match s {RangeBound::Unbounded=>true,RangeBound::Included(b)=>k.as_slice()>=b,RangeBound::Excluded(b)=>k.as_slice()>b}) && (match e {RangeBound::Unbounded=>true,RangeBound::Included(b)=>k.as_slice()<=b,RangeBound::Excluded(b)=>k.as_slice()<b})).cloned().collect();
  let mut got:Vec<Vec<u8>>=t.range(s,e,&g).map(|x|x.key).collect();
  let mut rev:Vec<Vec<u8>>=t.range(s,e,&g).rev().map(|x|x.key).collect(); rev.reverse();
  println!("{l}: fwd_ok={} rev_ok={} exp={:?} fwd={:?} rev={:?}", got==exp, rev==exp, exp, got, rev);
 }
}
