use masstree::{MassTree};
fn main(){
 let t:MassTree<u64>=MassTree::new();
 t.insert(&vec![b'a';255],255); t.insert(&vec![b'a';256],256);
 let g=t.guard();
 let mut k=vec![]; t.scan_prefix(&vec![b'a';256],|x,_|{k.push((x.len(),*x.last().unwrap()));true},&g);
 let mut v=vec![]; t.scan_prefix_values(&vec![b'a';256],|x|{v.push(x);true},&g);
 println!("keys={k:?} vals={v:?}");
 let mut k2=vec![]; t.scan_prefix(&vec![b'a';255],|x,_|{k2.push((x.len(),*x.last().unwrap()));true},&g);
 let mut v2=vec![]; t.scan_prefix_values(&vec![b'a';255],|x|{v2.push(x);true},&g);
 println!("pfx255 keys={k2:?} vals={v2:?}");
}
