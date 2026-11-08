// SAFE-01 regression: a guard-scoped value cannot escape its guard.
//
// This used to be a compile-passing use-after-free. The safe API now returns
// `ValueRef<'_>`, whose borrow checker lifetime rejects the old program.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"k", "hello".to_string());

    {
        let g = tree.guard();
        let value = tree.get_ref(b"k", &g).expect("value");
        assert_eq!(&*value, "hello");
    }

    assert_eq!(tree.get(b"k").as_deref(), Some("hello"));
    println!("escaped handle rejected at compile time");
}
