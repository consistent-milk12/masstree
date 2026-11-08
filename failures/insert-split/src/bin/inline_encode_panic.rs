//! INS-04: encoder panics during split or twig construction preserve the tree.
//! Run in a disposable process with a timeout; Miri also checks scratch cleanup.
use masstree::{InlineBits, MassTree};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Encoded(u64);

impl InlineBits for Encoded {
    fn to_bits(self) -> u64 {
        assert_ne!(self.0, 999, "intentional encoder panic");
        self.0
    }

    fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

fn main() {
    let tree = MassTree::<Encoded>::new();
    for key in (0u64..30).step_by(2) {
        tree.insert(&key.to_be_bytes(), Encoded(key));
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        tree.insert(&15u64.to_be_bytes(), Encoded(999));
    }));
    assert!(result.is_err(), "probe did not reach the encoder panic");
    eprintln!("caught encoder panic; checking existing key");
    assert_eq!(tree.get(&0u64.to_be_bytes()), Some(Encoded(0)));
    eprintln!("existing tree remains usable");

    let first = [b'x'; 256];
    let mut second = first;
    second[255] += 1;
    let tree = MassTree::<Encoded>::new();
    tree.insert(&first, Encoded(1));
    let result = catch_unwind(AssertUnwindSafe(|| {
        tree.insert(&second, Encoded(999));
    }));
    assert!(result.is_err());
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.get(&first), Some(Encoded(1)));
    assert_eq!(tree.get(&second), None);
    tree.insert(&second, Encoded(2));
    assert_eq!(tree.get(&first), Some(Encoded(1)));
    assert_eq!(tree.get(&second), Some(Encoded(2)));
    assert_eq!(tree.len(), 2);
    eprintln!("maximum-depth twig panic leaves the tree usable");
}
