use masstree::{MassTree15, MassTree, RangeBound};

fn main() {
    let t: MassTree15<String> = MassTree15::new();
    let g = t.guard();
    for (k, v) in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")] {
        t.insert_with_guard(k.as_bytes(), v.to_string(), &g);
    }
    let fwd: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).map(|e| e.value().to_string()).collect();
    let rev: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| e.value().to_string()).collect();
    println!("boxed fwd={fwd:?} rev={rev:?}");
    let rev2: Vec<String> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| e.value().to_string()).collect();
    println!("boxed rev2={rev2:?}");

    let ti: MassTree<u64> = MassTree::new();
    let gi = ti.guard();
    for (k, v) in [("a", 1u64), ("b", 2), ("c", 3), ("d", 4), ("e", 5)] {
        ti.insert_with_guard(k.as_bytes(), v, &gi);
    }
    let rev: Vec<u64> = ti.range(RangeBound::Unbounded, RangeBound::Unbounded, &gi).rev().map(|e| *e.value()).collect();
    println!("inline rev={rev:?}");
}
