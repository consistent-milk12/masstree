// Miri: non-atomic `ValuePtr::deref` while a write-through insert performs an
// atomic store to the same Box. Expected: data race.
use masstree::MassTree15;

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"k", 1u64);
    let guard = tree.guard();
    let ptr = tree.get_with_guard(b"k", &guard).unwrap();

    let mut sum = 0u64;
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for v in 0..64u64 {
                let _ = tree.insert(b"k", v);
            }
        });
        for _ in 0..64 {
            sum = sum.wrapping_add(*ptr);
        }
    });
    println!("sum={sum}");
}
