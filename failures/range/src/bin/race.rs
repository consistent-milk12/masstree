use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::Duration;
use masstree::{MassTree, RangeBound};
fn main(){
 for round in 0..20 {
  let tree:Arc<MassTree<u64>>=Arc::new(MassTree::new());
  for i in 0..30u64 { tree.insert(&i.to_be_bytes(), i); }
  let stop=Arc::new(AtomicBool::new(false));
  let bad=Arc::new(AtomicU64::new(0));
  let mut hs=vec![];
  for tid in 0..3u64 {
   let t=tree.clone(); let st=stop.clone();
   hs.push(thread::spawn(move||{
    let g=t.guard(); let mut x=tid.wrapping_mul(0x9E3779B97F4A7C15)|1;
    while !st.load(Ordering::Relaxed){
     x^=x<<13; x^=x>>7; x^=x<<17;
     let k=x%30;
     if (x>>8)&1==0 { let _=t.remove_with_guard(&k.to_be_bytes(),&g); } else { t.insert_with_guard(&k.to_be_bytes(),x,&g); }
    }
   }));
  }
  let mut ss=vec![];
  for _ in 0..2 {
   let t=tree.clone(); let st=stop.clone(); let bad=bad.clone();
   ss.push(thread::spawn(move||{
    let g=t.guard();
    while !st.load(Ordering::Relaxed){
     let res=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
      let mut prev:Option<Vec<u8>>=None;
      for e in t.range(RangeBound::Included(&0u64.to_be_bytes()),RangeBound::Unbounded,&g){
       if let Some(p)=&prev { if p.as_slice()>=e.key.as_slice(){ bad.fetch_add(1,Ordering::Relaxed); } }
       prev=Some(e.key);
      }
     }));
     if res.is_err() { bad.fetch_add(1_000_000,Ordering::Relaxed); }
    }
   }));
  }
  thread::sleep(Duration::from_millis(200));
  stop.store(true,Ordering::Relaxed);
  for h in hs {h.join().unwrap();} for h in ss {h.join().unwrap();}
  let b=bad.load(Ordering::Relaxed);
  if b>0 { println!("round {round}: BAD={b}"); return; }
 }
 println!("race: no violations observed");
}
