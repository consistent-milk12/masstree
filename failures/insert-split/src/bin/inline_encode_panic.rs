//! INS-04: an encoder panic during split must leave the original tree usable.
//! Run in a disposable process with a timeout; current code hangs after the sentinel.
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
}
