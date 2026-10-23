use masstree::{MassTree, RangeBound};
fn fwd(t:&MassTree<u64>,s:RangeBound<'_>,e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();t.range(s,e,&g).map(|x|x.key).collect()}
fn rv(t:&MassTree<u64>,s:RangeBound<'_>,e:RangeBound<'_>)->Vec<Vec<u8>>{let g=t.guard();let mut v:Vec<_>=t.range(s,e,&g).rev().map(|x|x.key).collect();v.reverse();v}
fn main(){
 let t:MassTree<u64>=MassTree::new();
 for (i,k) in [b"abcdefghX".as_slice(),b"abcdefghY",b"abcdefghZ",b"abcdefghX0"].iter().enumerate(){t.insert(k,i as u64);}
 println!("all={:?}",fwd(&t,RangeBound::Unbounded,RangeBound::Unbounded));
 for q in [b"abcdefgh".as_slice(),b"abcdefghW",b"abcdefghX",b"abcdefghX0",b"abcdefghY",b"abcdefghZ",b"abcdefghZZ"] {
  let f=fwd(&t,RangeBound::Included(q),RangeBound::Unbounded);
  let fr=fwd(&t,RangeBound::Excluded(q),RangeBound::Unbounded);
  let r=rv(&t,RangeBound::Unbounded,RangeBound::Included(q));
  let re=rv(&t,RangeBound::Unbounded,RangeBound::Excluded(q));
  println!("q={:?}\n  f_inc={:?}\n  f_exc={:?}\n  r_inc={:?}\n  r_exc={:?}", String::from_utf8_lossy(q), f, fr, r, re);
 }
}
