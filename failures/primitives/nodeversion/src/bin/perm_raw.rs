use masstree::Permuter;
fn main() {
    let mut p: Permuter<7> = Permuter::make_sorted(5);
    println!("before raw={:#x} size={} slots={:?}", p.value(), p.size(), (0..7).map(|i| p.get(i)).collect::<Vec<_>>());
    p.remove_to_back(0);
    let mask: u64 = (1u64 << ((7 + 1) * 4)) - 1;
    println!("after  raw={:#x} size={} slots={:?} bits_above_width={:#x}", p.value(), p.size(), (0..7).map(|i| p.get(i)).collect::<Vec<_>>(), p.value() & !mask);
    assert_ne!(p.value() & !mask, 0);
}
