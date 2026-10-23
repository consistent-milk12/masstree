fn check(n: usize, groups: u64, label: &str) {
    let mut v: Vec<(u64,u64)> = (0..n as u64).map(|i| (i % groups, i)).collect();
    v.sort_unstable_by_key(|x| x.0);
    let mut ok = true;
    for g in 0..groups {
        let seq: Vec<u64> = v.iter().filter(|x| x.0==g).map(|x| x.1).collect();
        if seq.windows(2).any(|w| w[0] > w[1]) { ok = false; }
    }
    println!("{label} n={n} groups={groups} stable={ok} last={}", v[n-1].1);
}
fn main() {
    for n in [100usize, 1000, 10000, 100000] {
        for g in [2u64, 5, 100] { check(n,g,"mod"); }
    }
    // duplicate exact key among others: all same key value in middle
    let mut v: Vec<(u64,u64)> = (0..1000).map(|i| if i%10==0 {(0u64, i)} else {(1u64,i)}).collect();
    v.sort_unstable_by_key(|x| x.0);
    let seq: Vec<u64> = v.iter().filter(|x| x.0==0).map(|x| x.1).collect();
    println!("sparse equal: {} {:?}", seq.windows(2).all(|w| w[0]<=w[1]), &seq[..seq.len().min(10)]);
}
