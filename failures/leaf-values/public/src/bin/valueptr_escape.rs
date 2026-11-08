// SAFE-01 regression (formerly a Miri use-after-free).
//
// `ValuePtr` no longer has a safe accessor, and public read APIs return a
// guard-scoped `ValueRef`. Returning the value from the block below is a
// borrow-check error; this program reads it while the guard is alive and
// verifies the tree still works afterwards.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"k", "a value long enough to be heap allocated".to_string());

    {
        let guard = tree.guard();
        let value = tree.get_ref(b"k", &guard).expect("value should exist");
        println!("len={}", value.len());
    } // `value` cannot outlive `guard`.

    assert_eq!(
        tree.get(b"k").as_deref(),
        Some("a value long enough to be heap allocated")
    );
    println!("guard-scoped value did not escape");
}
