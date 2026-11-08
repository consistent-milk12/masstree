// SAFE-01 regression (formerly a Miri use-after-free).
//
// A guard-scoped `ValueRef` cannot escape its guard. Read the value while the
// guard is alive; returning it from the block is a borrow-check error.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"k", "hello".to_string());

    {
        let guard = tree.guard();
        let value = tree.get_ref(b"k", &guard).expect("value");
        assert_eq!(&*value, "hello");
    }

    assert_eq!(tree.get(b"k").as_deref(), Some("hello"));
    println!("guard-scoped value did not escape");
}
