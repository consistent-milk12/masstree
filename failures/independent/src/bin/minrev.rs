use masstree::{MassTree, MassTree15, RangeBound};

fn rev_inline(n: usize, remove_range: std::ops::Range<usize>) {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for i in 0..n {
        let k = format!("k{i:04}");
        t.insert_with_guard(k.as_bytes(), i as u64, &g);
    }
    for i in remove_range {
        let k = format!("k{i:04}");
        t.remove_with_guard(k.as_bytes(), &g).unwrap();
    }
    let fwd: Vec<u64> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).map(|e| *e.value()).collect();
    let rev: Vec<u64> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| *e.value()).collect();
    println!("inline n={n} removed -> fwd={fwd:?} rev={rev:?} pending={}", t.pending_coalesce());
}

fn rev_boxed(n: usize, remove_range: std::ops::Range<usize>) {
    let t: MassTree15<String> = MassTree15::new();
    let g = t.guard();
    for i in 0..n {
        let k = format!("k{i:04}");
        t.insert_with_guard(k.as_bytes(), i.to_string(), &g);
    }
    for i in remove_range {
        let k = format!("k{i:04}");
        t.remove_with_guard(k.as_bytes(), &g).unwrap();
    }
    let fwd: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).map(|e| t.get(e.key()).expect("value")).collect();
    let rev: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| t.get(e.key()).expect("value")).collect();
    println!("boxed n={n} removed -> fwd_len={} rev={rev:?} pending={}", fwd.len(), t.pending_coalesce());
}

fn main() {
    rev_inline(20, 15..20);
    rev_inline(20, 14..20);
    rev_inline(20, 0..20);
    rev_boxed(20, 15..20);
    rev_boxed(20, 14..20);
    rev_boxed(20, 0..20);
    for n in [15, 16, 20, 30, 40] {
        rev_inline(n, 0..n);
        rev_boxed(n, 0..n);
    }
}
