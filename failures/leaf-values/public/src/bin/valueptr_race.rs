// SAFE-02 regression (formerly a Miri data race).
//
// `get_with_guard` now returns a guard-scoped `ValueRef`. For a write-through
// value it is an atomic snapshot (`ValueRef::Owned`), so the reader never forms
// a `&V` into a Box that another thread updates in place.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"k", 1u64);
    let guard = tree.guard();
    let value = tree.get_with_guard(b"k", &guard).unwrap();

    let mut sum = 0u64;
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for v in 0..64u64 {
                let _ = tree.insert(b"k", v);
            }
        });
        for _ in 0..64 {
            sum = sum.wrapping_add(*value);
        }
    });
    println!("sum={sum}");
}
