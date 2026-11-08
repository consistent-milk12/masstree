use masstree::SuffixBag;

fn main() {
    let mut b = SuffixBag::with_capacity(4);
    b.assign(5, b"");
    b.assign(0, b"abc");
    // The next assignment cannot fit, so the bag compacts. The zero-length
    // suffix at offset 0 shares an offset with the later "abc".
    let grew = b.try_assign(6, b"zz");
    println!("grew={grew} get5={:?} get0={:?} get6={:?}", b.get(5), b.get(0), b.get(6));
}
