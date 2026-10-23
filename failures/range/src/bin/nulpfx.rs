use masstree::{MassTree};
fn main(){
 let t:MassTree<u64>=MassTree::new(); t.insert(b"",1); t.insert(b"\0",2); t.insert(b"\0\0",3); t.insert(b"a",4);
 let g=t.guard();
 for p in [b"".as_slice(),b"\0",b"\0\0"] {
  let mut k=vec![]; t.scan_prefix(p,|x,_|{k.push(x.to_vec());true},&g);
  let mut v=vec![]; t.scan_prefix_values(p,|x|{v.push(x);true},&g);
  println!("pfx {p:?}: keys={k:?} vals={v:?}");
 }
}
