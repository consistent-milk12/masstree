use masstree::MassTree;
use std::collections::BTreeMap;

// Tiny deterministic PRNG.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn main() {
    let alphabet: [u8; 4] = [0, b'A', b'B', b'C'];
    let mut fail = 0usize;
    for seed in 1..=200u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let tree: MassTree<u64> = MassTree::new();
        let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for step in 0..3000u64 {
            let op = rng.below(10);
            let klen = 1 + rng.below(9) as usize;
            let mut k = Vec::with_capacity(klen);
            for _ in 0..klen {
                k.push(alphabet[rng.below(4) as usize]);
            }
            if op < 7 {
                let v = (seed << 32) | step;
                let old = tree.insert(&k, v);
                let model_old = model.insert(k.clone(), v);
                if old != model_old {
                    eprintln!("seed {seed} step {step}: insert {k:?} old {old:?} expected {model_old:?}");
                    fail += 1;
                    break;
                }
            } else if op < 9 {
                let old = tree.remove(&k).expect("remove");
                let model_old = model.remove(&k);
                if old != model_old {
                    eprintln!("seed {seed} step {step}: remove {k:?} -> {old:?} expected {model_old:?}");
                    fail += 1;
                    break;
                }
            } else {
                let got = tree.get(&k);
                let want = model.get(&k).copied();
                if got != want {
                    eprintln!("seed {seed} step {step}: get {k:?} -> {got:?} expected {want:?}");
                    fail += 1;
                    break;
                }
            }

            if step % 97 == 0 {
                if tree.len() != model.len() {
                    eprintln!("seed {seed} step {step}: len {} expected {}", tree.len(), model.len());
                    fail += 1;
                    break;
                }
                // Verify every model key is retrievable.
                for (mk, mv) in &model {
                    match tree.get(mk) {
                        Some(v) if v == *mv => {}
                        other => {
                            eprintln!("seed {seed} step {step}: lost key {mk:?} got {other:?} expected {mv}");
                            fail += 1;
                            break;
                        }
                    }
                }
                if fail > 0 {
                    break;
                }
            }
        }
        if fail > 0 {
            break;
        }
    }
    if fail == 0 {
        eprintln!("random oracle test passed");
    } else {
        eprintln!("FAILURES: {fail}");
        std::process::exit(1);
    }
}
