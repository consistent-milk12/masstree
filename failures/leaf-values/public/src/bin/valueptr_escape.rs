// Miri: `ValuePtr` is Copy, has no lifetime, and dereferences safely.
// Expected: use-after-free after the guard and tree are dropped.
use masstree::MassTree15;

fn main() {
    let leaked = {
        let tree: MassTree15<String> = MassTree15::new();
        tree.insert(b"k", "a value long enough to be heap allocated".to_string());
        let guard = tree.guard();
        tree.get_with_guard(b"k", &guard).unwrap()
        // guard and tree are dropped here
    };
    println!("len={}", leaked.len());
}
