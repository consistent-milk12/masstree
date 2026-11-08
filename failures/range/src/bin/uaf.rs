// SAFE-01 range variant (formerly a Miri use-after-free).
//
// `ScanEntry` no longer exposes a safely-dereferenceable value handle. The
// old program moved the entry out of the guard/tree scope and dereferenced
// `e.value`; that is now a borrow-check error. Read guard-scoped values with
// `get_ref` or collect owned values through the tree APIs instead.
use masstree::{MassTree15, RangeBound};

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"key", "a-long-string-value-to-allocate".to_string());

    {
        let guard = tree.guard();
        let entry = tree
            .range(RangeBound::Unbounded, RangeBound::Unbounded, &guard)
            .next()
            .unwrap();

        assert_eq!(entry.key, b"key");

        let value = tree.get_ref(b"key", &guard).unwrap();
        println!("key={:?} value={:?}", entry.key, &*value);
    }

    assert_eq!(
        tree.get(b"key").as_deref(),
        Some("a-long-string-value-to-allocate")
    );
    println!("range entry stayed inside its guard");
}
