use masstree::{MassTree, RangeBound};
fn f(t:&MassTree<u64>,s:RangeBound<'_>,e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();t.range(s,e,&g).map(|x|x.key).collect()}
fn r(t:&MassTree<u64>,s:RangeBound<'_>,e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();let mut v:Vec<_>=t.range(s,e,&g).rev().map(|x|x.key).collect();v.reverse();v}
fn main(){
 let t:MassTree<u64>=MassTree::new();
 t.insert(b"",1); t.insert(b"\0",2); t.insert(b"\0\0",3); t.insert(b"a",4);
 println!("all={:?}",f(&t,RangeBound::Unbounded,RangeBound::Unbounded));
 for q in [b"".as_slice(),b"\0",b"\0\0"] {
  println!("start inc {:?}: {:?}",q,f(&t,RangeBound::Included(q),RangeBound::Unbounded));
  println!("start exc {:?}: {:?}",q,f(&t,RangeBound::Excluded(q),RangeBound::Unbounded));
  println!("end inc {:?}: {:?}",q,r(&t,RangeBound::Unbounded,RangeBound::Included(q)));
  println!("end exc {:?}: {:?}",q,r(&t,RangeBound::Unbounded,RangeBound::Excluded(q)));
 }
}
