// Miri: `CAN_WRITE_THROUGH` admits a type with uninitialized padding.
// Expected: uninitialized-memory read through the AtomicU64 load.
use masstree::MassTree15;

#[repr(align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Padded {
    a: u32,
    // four bytes of padding
}

fn main() {
    let tree: MassTree15<Padded> = MassTree15::new();
    tree.insert(b"k", Padded { a: 7 });
    let got = tree.get(b"k").unwrap();
    println!("{got:?}");
}
