use masstree::SuffixBag;

fn main() {
    let mut b = SuffixBag::with_capacity(4);
    b.assign(5, b"");
    b.assign(0, b"abc");
    // The unused compaction API was removed. Its zero-length-suffix case
    // remains covered across growth of the production append-only buffer.
    b.assign(6, b"zz");
    assert_eq!(b.get(5), Some(b"".as_slice()));
    assert_eq!(b.get(0), Some(b"abc".as_slice()));
    assert_eq!(b.get(6), Some(b"zz".as_slice()));
}
