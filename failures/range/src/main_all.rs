use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use masstree::{MassTree, RangeBound};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x }
    fn below(&mut self, n: usize) -> usize { (self.next() as usize) % n.max(1) }
    fn byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
}
fn expected_range<'a>(m: &'a BTreeMap<Vec<u8>, u64>, s: RangeBound<'a>, e: RangeBound<'a>) -> Vec<Vec<u8>> {
    if let (Some(sk), Some(ek)) = (s.key(), e.key()) {
        if sk > ek || (sk == ek && !(matches!(s, RangeBound::Included(_)) && matches!(e, RangeBound::Included(_)))) { return Vec::new(); }
    }
    let sb = match s { RangeBound::Unbounded=>Bound::Unbounded, RangeBound::Included(k)=>Bound::Included(k), RangeBound::Excluded(k)=>Bound::Excluded(k) };
    let eb = match e { RangeBound::Unbounded=>Bound::Unbounded, RangeBound::Included(k)=>Bound::Included(k), RangeBound::Excluded(k)=>Bound::Excluded(k) };
    m.range::<[u8], _>((sb,eb)).map(|(k,_)| k.clone()).collect()
}
fn keys_of(it: impl Iterator<Item=masstree::ScanEntry<u64>>) -> Vec<Vec<u8>> { it.map(|e| e.key).collect() }

#[derive(Default)]
struct Fails { fwd: usize, rev: usize, batch: usize, revbatch: usize, mixed: usize, prefix: usize, val: usize }

fn run(mode: &str, seed: u64, rounds: usize, pool: &[Vec<u8>], f: &mut Fails) {
    let mut rng = Rng(seed);
    for iter in 0..rounds {
        let mut m: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        let t: MassTree<u64> = MassTree::new();
        let ops = 20 + rng.below(100);
        for op in 0..ops {
            let k = pool[rng.below(pool.len())].clone();
            if rng.below(100) < 60 {
                let val = (iter*1000+op) as u64;
                m.insert(k.clone(), val); t.insert(&k, val);
            } else {
                let a=m.remove(&k); let b=t.remove(&k).unwrap();
                if a!=b { println!("[{mode}] REMOVE DIV iter={iter} key={k:?}"); return; }
            }
        }
        let g=t.guard();
        let exp: Vec<Vec<u8>> = m.keys().cloned().collect();
        let full = keys_of(t.iter(&g));
        if full != exp { println!("[{mode}] FULL DIV iter={iter} exp={exp:?} act={full:?}"); return; }
        // values-only unbounded should be exactly all
        let mut vv=vec![]; t.scan_values(RangeBound::Unbounded, RangeBound::Unbounded, |v|{vv.push(v);true}, &g);
        let mut ve: Vec<u64> = m.values().copied().collect();
        let mut vs=vv.clone(); vs.sort(); ve.sort();
        if vs!=ve { println!("[{mode}] VAL FULL DIV iter={iter} got {} want {}", vv.len(), ve.len()); f.val+=1; }
        for _bi in 0..50 {
            let pick = |rng:&mut Rng| -> Vec<u8> {
                match rng.below(6) {
                    0=>pool[rng.below(pool.len())].clone(),
                    1=>{let mut k=pool[rng.below(pool.len())].clone(); k.push(rng.byte()); k},
                    2=>{let mut k=pool[rng.below(pool.len())].clone(); if !k.is_empty(){let n=k.len()-1; k[n]=k[n].wrapping_add(1);} k},
                    3=>vec![rng.byte(); rng.below(10)],
                    4=>m.keys().nth(rng.below(m.len().max(1))).cloned().unwrap_or_default(),
                    _=>Vec::new(),
                }
            };
            let mk = |rng:&mut Rng| -> (RangeBound<'static>, Option<Vec<u8>>) {
                if rng.below(4)==0 { (RangeBound::Unbounded,None) } else {
                    let k=pick(rng); let leaked: &'static [u8]=Box::leak(k.clone().into_boxed_slice());
                    if rng.below(2)==0 {(RangeBound::Included(leaked),Some(k))} else {(RangeBound::Excluded(leaked),Some(k))}
                }
            };
            let (s,sk)=mk(&mut rng); let (e,ek)=mk(&mut rng);
            let exp2=expected_range(&m,s,e);
            let act=keys_of(t.range(s,e,&g));
            if act!=exp2 { f.fwd+=1; if f.fwd<=3 {println!("[{mode}] FWD BOUND iter={iter} start={sk:?} end={ek:?}\n exp={exp2:?}\n act={act:?}");} }
            let mut actr=keys_of(t.range(s,e,&g).rev()); actr.reverse();
            if actr!=exp2 { f.rev+=1; if f.rev<=3 {println!("[{mode}] REV BOUND iter={iter} start={sk:?} end={ek:?}\n exp={exp2:?}\n act={actr:?}");} }
            let mut actb=vec![]; t.scan_intra_leaf_batch(s,e,|k,_|{actb.push(k.to_vec());true},&g);
            if actb!=exp2 { f.batch+=1; if f.batch<=2 {println!("[{mode}] FWD BATCH iter={iter} start={sk:?} end={ek:?}\n exp={exp2:?}\n act={actb:?}");} }
            let mut actrb=vec![]; t.scan_rev_batch(s,e,|k,_|{actrb.push(k.to_vec());true},&g); actrb.reverse();
            if actrb!=exp2 { f.revbatch+=1; if f.revbatch<=2 {println!("[{mode}] REV BATCH iter={iter} start={sk:?} end={ek:?}\n exp={exp2:?}\n act={actrb:?}");} }
            // values-only: every emitted value must map to a key inside range, or share boundary ikey (documented approx)
            let mut vv=vec![]; t.scan_values(s,e,|v|{vv.push(v);true},&g);
            let byval: std::collections::HashMap<u64,Vec<u8>> = m.iter().map(|(k,v)|(*v,k.clone())).collect();
            let mut gross=false;
            for v in &vv {
                if let Some(k)=byval.get(v) {
                    let inrange = (match s { RangeBound::Unbounded=>true, RangeBound::Included(b)=>k.as_slice()>=b, RangeBound::Excluded(b)=>k.as_slice()>b })
                        && (match e { RangeBound::Unbounded=>true, RangeBound::Included(b)=>k.as_slice()<=b, RangeBound::Excluded(b)=>k.as_slice()<b });
                    if !inrange {
                        let ikey=|x:&[u8]| {let mut a=[0u8;8]; let n=x.len().min(8); a[..n].copy_from_slice(&x[..n]); u64::from_be_bytes(a)};
                        let sb=s.key().map(ikey); let eb=e.key().map(ikey);
                        if Some(ikey(k))!=sb && Some(ikey(k))!=eb { gross=true; }
                    }
                } else { gross=true; }
            }
            let ins: BTreeSet<&u64> = m.iter().filter(|(k,_)| {
                (match s { RangeBound::Unbounded=>true, RangeBound::Included(b)=>k.as_slice()>=b, RangeBound::Excluded(b)=>k.as_slice()>b })
                && (match e { RangeBound::Unbounded=>true, RangeBound::Included(b)=>k.as_slice()<=b, RangeBound::Excluded(b)=>k.as_slice()<b }) }).map(|(_,v)|v).collect();
            let got: BTreeSet<&u64> = vv.iter().collect();
            if gross || !ins.is_subset(&got) { f.val+=1; if f.val<=3 {println!("[{mode}] VAL BOUND iter={iter} start={sk:?} end={ek:?} emitted={} expected={} gross={gross} missing={:?}", vv.len(), ins.len(), ins.difference(&got).take(4).collect::<Vec<_>>());} }
            // reverse values-only
            let mut rv=vec![]; t.scan_values_rev(s,e,|v|{rv.push(v);true},&g);
            let rgot: BTreeSet<&u64> = rv.iter().collect();
            if !ins.is_subset(&rgot) || rv.len()>ins.len()+2 { f.val+=1; if f.val<=3 {println!("[{mode}] RVAL BOUND iter={iter} start={sk:?} end={ek:?} emitted={} expected={}", rv.len(), ins.len());} }
            // mixed de
            let mut it=t.range(s,e,&g); let mut got=vec![];
            let mut fd=false; let mut bd=false;
            for _ in 0..(exp2.len()+3) {
                if rng.below(2)==0 { if !fd { if let Some(x)=it.next(){got.push(x.key)}else{fd=true} } }
                else if !bd { if let Some(x)=it.next_back(){got.push(x.key)}else{bd=true} }
            }
            while let Some(x)=it.next(){got.push(x.key)}
            while let Some(x)=it.next_back(){got.push(x.key)}
            let gs:BTreeSet<_>=got.iter().cloned().collect(); let es:BTreeSet<_>=exp2.iter().cloned().collect();
            if got.len()!=exp2.len() || gs!=es { f.mixed+=1; if f.mixed<=2 {println!("[{mode}] MIXED iter={iter} start={sk:?} end={ek:?} expN={} gotN={} missing={:?} extra={:?}", exp2.len(), got.len(), es.difference(&gs).take(3).collect::<Vec<_>>(), gs.difference(&es).take(3).collect::<Vec<_>>());} }
            if let Some(k)=sk.clone() {
                let pfx=&k[..rng.below(k.len()+1)];
                let exp3:Vec<Vec<u8>>=m.range::<[u8],_>((Bound::Included(pfx),Bound::Unbounded)).take_while(|(kk,_)|kk.starts_with(pfx)).map(|(kk,_)|kk.clone()).collect();
                let mut got3=vec![]; t.scan_prefix(pfx,|kk,_|{got3.push(kk.to_vec());true},&g);
                if got3!=exp3 { f.prefix+=1; if f.prefix<=2 {println!("[{mode}] PREFIX iter={iter} pfx={pfx:?}\n exp={exp3:?}\n act={got3:?}");} }
            }
        }
    }
}

fn main(){
    let mut a:Vec<Vec<u8>>=vec![vec![]];
    let alpha=[0u8,1,b'a',b'b',b'c',0xfe,0xff];
    let mut rng=Rng(12345);
    for i in 0..300usize { let len=rng.below(9); a.push((0..len).map(|_|alpha[rng.below(alpha.len())]).collect()); if i<50 {a.push(vec![b'k';len]);} }
    a.push(vec![0xff;8]);
    let mut f=Fails::default();
    run("short", 999, 80, &a, &mut f);
    println!("SHORT fails fwd={} rev={} batch={} revbatch={} mixed={} prefix={} val={}", f.fwd,f.rev,f.batch,f.revbatch,f.mixed,f.prefix,f.val);

    // clean short keys: no trailing-NUL / no shared ikeys
    let mut c:Vec<Vec<u8>>=Vec::new();
    let alpha2=[b'a',b'b',b'c',0xfe,0xff];
    let mut rng2=Rng(555);
    for _ in 0..300 { let len=1+rng2.below(8); c.push((0..len).map(|_|alpha2[rng2.below(alpha2.len())]).collect()); }
    let mut f3=Fails::default();
    run("cleanshort", 444, 80, &c, &mut f3);
    println!("CLEANSHORT fails fwd={} rev={} batch={} revbatch={} mixed={} prefix={} val={}", f3.fwd,f3.rev,f3.batch,f3.revbatch,f3.mixed,f3.prefix,f3.val);

    // mode: suffix keys with distinct 8-byte prefixes
    let mut b:Vec<Vec<u8>>=Vec::new();
    for i in 0..200u16 { let mut k=format!("K{:07}", i).into_bytes(); k.push(b'a'+(i%5) as u8); b.push(k); }
    for i in 0..50u16 { let mut k=format!("K{:07}", i).into_bytes(); b.push(k); } // exact 8-byte terminals
    b.push(vec![]);
    let mut f2=Fails::default();
    run("suffix", 777, 60, &b, &mut f2);
    println!("SUFFIX fails fwd={} rev={} batch={} revbatch={} mixed={} prefix={} val={}", f2.fwd,f2.rev,f2.batch,f2.revbatch,f2.mixed,f2.prefix,f2.val);
}
