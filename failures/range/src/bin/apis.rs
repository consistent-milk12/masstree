use masstree::{MassTree15, RangeBound};
fn fwd(t:&MassTree15<u64>, s:RangeBound<'_>, e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();t.range(s,e,&g).map(|x|x.key).collect()}
fn rv(t:&MassTree15<u64>, s:RangeBound<'_>, e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();let mut v:Vec<_>=t.range(s,e,&g).rev().map(|x|x.key).collect();v.reverse();v}
fn main(){
 let t:MassTree15<u64>=MassTree15::new();
 t.insert(b"abcdefgh",1); t.insert(b"abcdefghX",2); t.insert(b"abcdefghY",3); t.insert(b"abcdefghX0",4);
 let g=t.guard();
 println!("all fwd={:?}", fwd(&t,RangeBound::Unbounded,RangeBound::Unbounded));
 println!("all rev={:?}", rv(&t,RangeBound::Unbounded,RangeBound::Unbounded));
 for (s,e,lbl) in [
  (RangeBound::Included(b"abcdefghX".as_slice()),RangeBound::Unbounded,"inc X"),
  (RangeBound::Excluded(b"abcdefghY".as_slice()),RangeBound::Unbounded,"exc Y"),
  (RangeBound::Included(b"abcdefghY".as_slice()),RangeBound::Unbounded,"inc Y"),
  (RangeBound::Excluded(b"abcdefgh".as_slice()),RangeBound::Unbounded,"exc exact"),
 ] { println!("fwd {lbl}: {:?}", fwd(&t,s,e)); }
 for (s,e,lbl) in [
  (RangeBound::Unbounded,RangeBound::Included(b"abcdefghY".as_slice()),"end inc Y"),
  (RangeBound::Unbounded,RangeBound::Included(b"abcdefghX".as_slice()),"end inc X"),
  (RangeBound::Unbounded,RangeBound::Excluded(b"abcdefghX".as_slice()),"end exc X"),
  (RangeBound::Unbounded,RangeBound::Excluded(b"abcdefghY".as_slice()),"end exc Y"),
  (RangeBound::Included(b"abcdefgh".as_slice()),RangeBound::Included(b"abcdefgh".as_slice()),"eq exact"),
  (RangeBound::Included(b"abcdefghX".as_slice()),RangeBound::Included(b"abcdefghX".as_slice()),"eq X"),
 ] { println!("rev {lbl}: {:?}", rv(&t,s,e)); }
 // prefix
 for p in [b"abcdefgh".as_slice(),b"abcdefghX",b"abcdefghY",b"abcdefghZ",b"abcdefghX0"] {
  let mut kv=vec![]; t.scan_prefix(p,|k,_|{kv.push(k.to_vec());true},&g);
  let mut vv=vec![]; t.scan_prefix_values(p,|v|{vv.push(v);true},&g);
  println!("prefix {p:?}: keys={kv:?} vals={vv:?}");
 }
 // API variants on bad range inc X .. inc Y (expected keys X,X0,Y)
 let s=RangeBound::Included(b"abcdefghX".as_slice()); let e=RangeBound::Included(b"abcdefghY".as_slice());
 let mut v=vec![]; t.scan_ref(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_ref={v:?}");
 let mut v=vec![]; t.scan_batch_ref(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_batch_ref={v:?}");
 let mut v=vec![]; t.scan_intra_leaf_batch_ref(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_intra_leaf_batch_ref={v:?}");
 let mut v=vec![]; t.scan(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan={v:?}");
 let mut v=vec![]; t.scan_intra_leaf_batch(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_intra_leaf_batch={v:?}");
 let mut v=vec![]; t.scan_values(s,e,|x|{v.push(x);true},&g); println!("scan_values={v:?}");
 let mut v=vec![]; t.scan_values_rev(s,e,|x|{v.push(x);true},&g); println!("scan_values_rev={v:?}");
 let mut v=vec![]; t.scan_rev_batch(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_rev_batch={v:?}");
 let mut v=vec![]; t.scan_rev_batch_ref(s,e,|k,_|{v.push(k.to_vec());true},&g); println!("scan_rev_batch_ref={v:?}");
 // empty range values
 let mut v=vec![]; t.scan_values(RangeBound::Included(b"z"),RangeBound::Included(b"a"),|x|{v.push(x);true},&g); println!("empty fwd values={v:?}");
 let mut v=vec![]; t.scan_values_rev(RangeBound::Included(b"z"),RangeBound::Included(b"a"),|x|{v.push(x);true},&g); println!("empty rev values={v:?}");
 let mut v=vec![]; t.scan_values(RangeBound::Unbounded,RangeBound::Excluded(b"a"),|x|{v.push(x);true},&g); println!("end<min values={v:?}");
 // keys/values adapters + last
 println!("keys={:?}", t.keys(&g).collect::<Vec<_>>());
 println!("last={:?}", t.last_with_guard(&g).map(|x|x.key));
 println!("first={:?}", t.first_with_guard(&g).map(|x|x.key));
}
