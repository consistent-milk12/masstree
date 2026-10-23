use std::collections::BTreeMap;
use std::ops::Bound;

use masstree::{MassTree, MassTree15, RangeBound};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn corpus() -> Vec<Vec<u8>> {
    let mut keys: Vec<Vec<u8>> = Vec::new();
    let bases: [&[u8]; 6] = [b"", b"a", b"aaaaaaaa", b"aaaaaaaaBBBBBBBB", b"prefix00", b"\x00\xff\x00\xff"];
    for b in bases {
        for extra in 0..=17usize {
            let mut k = b.to_vec();
            k.extend(std::iter::repeat_n(b'x', extra));
            keys.push(k);
        }
    }
    for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 255, 256] {
        let mut k = Vec::with_capacity(len);
        for i in 0..len { k.push(((i as u8).wrapping_mul(37)).wrapping_add(11)); }
        keys.push(k);
    }
    for i in 0..64u8 {
        let mut k = vec![b'p'; 24]; k.push(i); keys.push(k);
    }
    for i in 0..32u8 {
        let mut k = b"shared00".to_vec();
        k.extend_from_slice(&[b'0' + (i % 10), b'A' + (i / 10)]);
        k.push(i); keys.push(k);
    }
    for n in 0..=8usize { keys.push(vec![0xff; n]); keys.push(vec![0xfe; n]); }
    keys.sort(); keys.dedup(); keys
}

fn bound_pair<'a>(b: &Bound<&'a [u8]>) -> RangeBound<'a> {
    match *b {
        Bound::Unbounded => RangeBound::Unbounded,
        Bound::Included(k) => RangeBound::Included(k),
        Bound::Excluded(k) => RangeBound::Excluded(k),
    }
}
fn std_bound<'a>(b: &Bound<&'a [u8]>) -> Bound<&'a [u8]> {
    match *b {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(k) => Bound::Included(k),
        Bound::Excluded(k) => Bound::Excluded(k),
    }
}

fn check_get(tree: &MassTree<u64>, oracle: &BTreeMap<Vec<u8>, u64>, guard: &seize::LocalGuard<'_>) {
    for (k, v) in oracle {
        assert_eq!(tree.get_with_guard(k, guard), Some(*v), "get mismatch for {k:?}");
    }
}

#[derive(Default)]
struct Mismatches {
    forward: Vec<String>,
    reverse: Vec<String>,
    scan: Vec<String>,
    batch: Vec<String>,
    revbatch: Vec<String>,
}

fn check_range(
    tree: &MassTree<u64>, oracle: &BTreeMap<Vec<u8>, u64>,
    start: &Bound<&[u8]>, end: &Bound<&[u8]>, guard: &seize::LocalGuard<'_>,
    out: &mut Mismatches,
) {
    let expected: Vec<(Vec<u8>, u64)> = oracle.range::<[u8], _>((std_bound(start), std_bound(end))).map(|(k, v)| (k.clone(), *v)).collect();
    let mut expected_rev = expected.clone(); expected_rev.reverse();
    let describe = || format!("start={start:?} end={end:?} exp_len={} got_len", expected.len());

    let got: Vec<(Vec<u8>, u64)> = tree.range(bound_pair(start), bound_pair(end), guard).map(|e| (e.key().to_vec(), *e.value())).collect();
    if got != expected && out.forward.len() < 20 { out.forward.push(format!("{} got={:?} first_exp={:?} first_got={:?}", describe(), got.len(), expected.first(), got.first())); }
    let got_rev: Vec<(Vec<u8>, u64)> = tree.range(bound_pair(start), bound_pair(end), guard).rev().map(|e| (e.key().to_vec(), *e.value())).collect();
    if got_rev != expected_rev && out.reverse.len() < 20 { out.reverse.push(format!("{} got={:?} first_exp={:?} first_got={:?}", describe(), got_rev.len(), expected_rev.first(), got_rev.first())); }
    let mut scanned: Vec<(Vec<u8>, u64)> = Vec::new();
    tree.scan(bound_pair(start), bound_pair(end), |k, v| { scanned.push((k.to_vec(), v)); true }, guard);
    if scanned != expected && out.scan.len() < 20 { out.scan.push(describe()); }
    let mut batch: Vec<(Vec<u8>, u64)> = Vec::new();
    tree.scan_intra_leaf_batch(bound_pair(start), bound_pair(end), |k, v| { batch.push((k.to_vec(), v)); true }, guard);
    if batch != expected && out.batch.len() < 20 { out.batch.push(describe()); }
    let mut revbatch: Vec<(Vec<u8>, u64)> = Vec::new();
    tree.scan_rev_batch(bound_pair(start), bound_pair(end), |k, v| { revbatch.push((k.to_vec(), v)); true }, guard);
    if revbatch != expected_rev && out.revbatch.len() < 20 { out.revbatch.push(describe()); }
}

fn test_inline_sequential(seed: u64, report: &mut Mismatches) {
    let keys = corpus();
    let mut rng = Rng::new(seed);
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for step in 0..3000u64 {
        let k = keys[rng.range(keys.len())].clone();
        match rng.range(10) {
            0..=5 => { let v = rng.next(); let old = tree.insert_with_guard(&k, v, &guard); assert_eq!(old, oracle.insert(k.clone(), v), "insert old step={step} key={k:?}"); }
            6..=8 => { let old = tree.remove_with_guard(&k, &guard).unwrap(); assert_eq!(old, oracle.remove(&k), "remove step={step} key={k:?}"); }
            _ => { assert_eq!(tree.get_with_guard(&k, &guard), oracle.get(&k).copied(), "get step={step} key={k:?}"); }
        }
        assert_eq!(tree.len(), oracle.len(), "len step={step}");
    }
    check_get(&tree, &oracle, &guard);
    let mut all_keys: Vec<&Vec<u8>> = keys.iter().collect();
    all_keys.extend(oracle.keys()); all_keys.sort(); all_keys.dedup();
    for _ in 0..80 {
        let a = all_keys[rng.range(all_keys.len())].as_slice();
        let b = all_keys[rng.range(all_keys.len())].as_slice();
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        for start in [Bound::Included(lo), Bound::Excluded(lo), Bound::Unbounded] {
            for end in [Bound::Included(hi), Bound::Excluded(hi), Bound::Unbounded] {
                if lo == hi && matches!(start, Bound::Excluded(_)) && matches!(end, Bound::Excluded(_)) { continue; }
                check_range(&tree, &oracle, &start, &end, &guard, report);
            }
        }
    }
}

fn test_boxed_batch(seed: u64) {
    let keys = corpus();
    let mut rng = Rng::new(seed ^ 0xdead_beef);
    let tree: MassTree15<String> = MassTree15::new();
    let guard = tree.guard();
    let mut oracle: BTreeMap<Vec<u8>, String> = BTreeMap::new();
    for round in 0..40u64 {
        let n = 1 + rng.range(30);
        let mut batch: Vec<(Vec<u8>, String)> = Vec::new();
        for _ in 0..n {
            let k = keys[rng.range(keys.len())].clone();
            let v = format!("v{round}-{}", rng.next());
            batch.push((k, v));
        }
        let result = tree.insert_batch_with_guard(batch.clone(), &guard);
        assert_eq!(result.inserted + result.updated, batch.len(), "batch count round={round}");
        for (k, v) in batch { oracle.insert(k, v); }
        for (k, v) in &oracle {
            let got = tree.get_with_guard(k, &guard).map(|p| p.to_string());
            assert_eq!(got.as_deref(), Some(v.as_str()), "boxed get {k:?}");
        }
        for _ in 0..10 {
            let k = keys[rng.range(keys.len())].clone();
            let old = tree.remove_with_guard(&k, &guard).unwrap();
            assert_eq!(old.map(|p| p.to_string()), oracle.remove(&k), "boxed remove round={round} key={k:?}");
        }
        assert_eq!(tree.len(), oracle.len(), "boxed len round={round}");
    }
    let expected: Vec<(Vec<u8>, String)> = oracle.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let got: Vec<(Vec<u8>, String)> = tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard).map(|e| (e.key().to_vec(), e.value().to_string())).collect();
    assert_eq!(got, expected, "boxed full scan");
    let got_rev: Vec<(Vec<u8>, String)> = tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard).rev().map(|e| (e.key().to_vec(), e.value().to_string())).collect();
    let mut expected_rev = expected.clone(); expected_rev.reverse();
    assert_eq!(got_rev, expected_rev, "boxed reverse full");
}

fn test_prefix_edges(report: &mut Mismatches) {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    let keys: Vec<Vec<u8>> = vec![
        vec![], vec![0], vec![0xff], vec![0xff, 0xff], vec![0xff, 0xff, 0xff],
        b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(), b"abcd".to_vec(),
        vec![b'a'; 8], vec![b'a'; 9], vec![b'a'; 16], vec![b'a'; 17],
        vec![0xff; 8], vec![0xff; 9], vec![0xff; 16], vec![0xff; 17], vec![0xff; 256],
    ];
    for (i, k) in keys.iter().enumerate() { tree.insert_with_guard(k, i as u64, &guard); oracle.insert(k.clone(), i as u64); }
    for p in &keys {
        let mut got: Vec<Vec<u8>> = Vec::new();
        tree.scan_prefix(p, |k, _v| { got.push(k.to_vec()); true }, &guard);
        let expected: Vec<Vec<u8>> = oracle.keys().filter(|k| k.starts_with(p)).cloned().collect();
        if got != expected && report.scan.len() < 20 { report.scan.push(format!("prefix={p:?} exp={expected:?} got={got:?}")); }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let what = args.get(1).map(String::as_str).unwrap_or("all");
    match what {
        "seq" => {
            let mut m = Mismatches::default();
            for seed in 1..=5 { eprintln!("seq seed {seed}"); test_inline_sequential(seed, &mut m); }
            println!("forward={} reverse={} scan={} batch={} revbatch={}", m.forward.len(), m.reverse.len(), m.scan.len(), m.batch.len(), m.revbatch.len());
            for (label, v) in [("FWD", &m.forward), ("REV", &m.reverse), ("SCAN", &m.scan), ("BATCH", &m.batch), ("REVBATCH", &m.revbatch)] {
                for s in v { println!("{label}: {s}"); }
            }
        }
        "batch" => { for seed in 1..=10 { eprintln!("batch seed {seed}"); test_boxed_batch(seed); } }
        "prefix" => { let mut m = Mismatches::default(); test_prefix_edges(&mut m); for s in &m.scan { println!("{s}"); } }
        _ => panic!("unknown"),
    }
}
