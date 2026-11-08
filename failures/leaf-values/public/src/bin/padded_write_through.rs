// SAFE-05 regression: a type with uninitialized padding must not qualify for
// write-through. Run under Miri; a clean run means no uninitialized read.
use masstree::{BoxPolicy, LeafPolicy, MassTree15};

#[repr(align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Padded {
    a: u32,
    // four bytes of padding
}

fn main() {
    assert!(!<BoxPolicy<Padded> as LeafPolicy>::can_write_through());
    let tree: MassTree15<Padded> = MassTree15::new();
    tree.insert(b"k", Padded { a: 7 });
    let got = tree.get(b"k").unwrap();
    println!("{got:?}");
}
