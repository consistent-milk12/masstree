use masstree::{MassTree15, RangeBound};
fn main(){
    let e;
    {
        let tree: MassTree15<String> = MassTree15::new();
        tree.insert(b"key", "a-long-string-value-to-allocate".to_string());
        let guard = tree.guard();
        e = tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard).next().unwrap();
        // guard drops here
    }
    // tree dropped while e.value points into it
    println!("key={:?} value={:?}", e.key, &*e.value);
}
