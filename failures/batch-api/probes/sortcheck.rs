fn main() {
    for n in [4usize, 16, 17, 32, 100, 1000, 10000] {
        let mut v: Vec<(u64,u64)> = (0..n as u64).map(|i| (0u64, i)).collect();
        v.sort_unstable_by_key(|x| x.0);
        println!("n={n} first={} last={}", v[0].1, v[n-1].1);
    }
}
