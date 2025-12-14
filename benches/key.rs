//! Fast benchmarks for `Key` using Divan.
//!
//! Run with: `cargo bench --bench key`

use divan::{Bencher, black_box};
use madtree::key::Key;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
#[expect(clippy::cast_possible_truncation)]
mod construction {
    use super::{Bencher, Key, black_box};

    #[divan::bench(args = [0, 1, 4, 8, 12, 16, 32, 64, 128, 256])]
    fn new(bencher: Bencher, len: usize) {
        let data: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
        bencher.bench_local(|| Key::new(black_box(&data)));
    }

    #[divan::bench]
    const fn from_ikey_zero() -> Key<'static> {
        Key::from_ikey(black_box(0))
    }

    #[divan::bench]
    const fn from_ikey_full() -> Key<'static> {
        Key::from_ikey(black_box(u64::from_be_bytes(*b"12345678")))
    }
}

// =============================================================================
// read_ikey (hot path)
// =============================================================================

#[divan::bench_group]
mod read_ikey {
    use super::{Key, black_box};

    // Fast path: 8+ bytes available
    #[divan::bench]
    fn fast_8b() -> u64 {
        Key::read_ikey(black_box(b"12345678"), black_box(0))
    }

    #[divan::bench]
    fn fast_16b_start() -> u64 {
        Key::read_ikey(black_box(b"1234567890abcdef"), black_box(0))
    }

    #[divan::bench]
    fn fast_16b_mid() -> u64 {
        Key::read_ikey(black_box(b"1234567890abcdef"), black_box(4))
    }

    // Slow path: < 8 bytes
    #[divan::bench]
    fn slow_1b() -> u64 {
        Key::read_ikey(black_box(b"x"), black_box(0))
    }

    #[divan::bench]
    fn slow_4b() -> u64 {
        Key::read_ikey(black_box(b"1234"), black_box(0))
    }

    #[divan::bench]
    fn slow_7b() -> u64 {
        Key::read_ikey(black_box(b"1234567"), black_box(0))
    }

    // Edge cases
    #[divan::bench]
    fn empty() -> u64 {
        Key::read_ikey(black_box(b""), black_box(0))
    }

    #[divan::bench]
    fn out_of_bounds() -> u64 {
        Key::read_ikey(black_box(b"12345678"), black_box(100))
    }
}

// =============================================================================
// Accessors
// =============================================================================

#[divan::bench_group]
mod accessors {
    use super::{Bencher, Key, black_box};

    const DATA: &[u8] = b"hello world! this is a longer key";

    #[divan::bench]
    fn ikey(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).ikey());
    }

    #[divan::bench]
    fn len(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).len());
    }

    #[divan::bench]
    fn current_len(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).current_len());
    }

    #[divan::bench]
    fn has_suffix(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).has_suffix());
    }

    #[divan::bench]
    fn suffix(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).suffix());
    }

    #[divan::bench]
    fn suffix_len(bencher: Bencher) {
        let key = Key::new(DATA);
        bencher.bench_local(|| black_box(&key).suffix_len());
    }
}

// =============================================================================
// Shift/Unshift
// =============================================================================

#[divan::bench_group]
#[expect(clippy::cast_possible_truncation)]
mod shift {
    use super::{Bencher, Key};

    #[divan::bench]
    fn single_shift(bencher: Bencher) {
        bencher
            .with_inputs(|| Key::new(b"hello world! this is a test key"))
            .bench_local_values(|mut key| {
                key.shift();
                key
            });
    }

    #[divan::bench(args = [2, 3, 4])]
    fn multi_shift(bencher: Bencher, n: usize) {
        let data: Vec<u8> = (0..(n + 1) * 8).map(|i| (i & 0xFF) as u8).collect();
        bencher
            .with_inputs(|| Key::new(&data))
            .bench_local_values(|mut key| {
                for _ in 0..n {
                    key.shift();
                }
                key
            });
    }

    #[divan::bench]
    fn single_unshift(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut key = Key::new(b"hello world! this is a test key");
                key.shift();
                key
            })
            .bench_local_values(|mut key| {
                key.unshift();
                key
            });
    }

    #[divan::bench(args = [1, 2, 3, 4])]
    fn unshift_all(bencher: Bencher, n: usize) {
        let data: Vec<u8> = (0..(n + 1) * 8).map(|i| (i & 0xFF) as u8).collect();
        bencher
            .with_inputs(|| {
                let mut key = Key::new(&data);
                for _ in 0..n {
                    key.shift();
                }
                key
            })
            .bench_local_values(|mut key| {
                key.unshift_all();
                key
            });
    }
}

// =============================================================================
// Comparison
// =============================================================================

#[divan::bench_group]
mod compare {
    use super::{Key, black_box};
    use std::cmp::Ordering;

    #[divan::bench]
    fn equal_short() -> Ordering {
        let key = Key::new(b"hello");
        let stored = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);
        black_box(&key).compare(black_box(stored), black_box(5))
    }

    #[divan::bench]
    fn less_by_ikey() -> Ordering {
        let key = Key::new(b"apple");
        let stored = u64::from_be_bytes([b'z', b'e', b'b', b'r', b'a', 0, 0, 0]);
        black_box(&key).compare(black_box(stored), black_box(5))
    }

    #[divan::bench]
    fn greater_by_ikey() -> Ordering {
        let key = Key::new(b"zebra");
        let stored = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);
        black_box(&key).compare(black_box(stored), black_box(5))
    }

    #[divan::bench]
    fn suffix_vs_no_suffix() -> Ordering {
        let key = Key::new(b"hello world!");
        let stored = u64::from_be_bytes(*b"hello wo");
        black_box(&key).compare(black_box(stored), black_box(8))
    }

    #[divan::bench]
    const fn compare_ikey_less() -> Ordering {
        Key::compare_ikey(black_box(100), black_box(200))
    }

    #[divan::bench]
    const fn compare_ikey_equal() -> Ordering {
        Key::compare_ikey(black_box(100), black_box(100))
    }
}

// =============================================================================
// Realistic Workloads
// =============================================================================

#[divan::bench_group]
mod workload {
    use super::{Bencher, Key, black_box};

    #[divan::bench]
    fn traverse_3_layers(bencher: Bencher) {
        let data = b"username:session:token!";
        bencher.bench_local(|| {
            let mut key = Key::new(black_box(data));
            let ikey0 = key.ikey();
            key.shift();
            let ikey1 = key.ikey();
            key.shift();
            let ikey2 = key.ikey();
            black_box((ikey0, ikey1, ikey2))
        });
    }

    #[divan::bench]
    fn lookup_with_suffix(bencher: Bencher) {
        let data = b"user:session:id=123";
        let stored_l0 = u64::from_be_bytes(*b"user:ses");

        let stored_l1 = u64::from_be_bytes(*b"sion:id=");
        bencher.bench_local(|| {
            let mut key = Key::new(black_box(data));
            let cmp0 = key.compare(black_box(stored_l0), 20);
            key.shift();
            let cmp1 = key.compare(black_box(stored_l1), 12);
            black_box((cmp0, cmp1))
        });
    }

    #[divan::bench]
    fn sequential_4_ikeys() {
        let data = b"abcdefghijklmnopqrstuvwxyz012345";
        let d = black_box(data);
        black_box(Key::read_ikey(d, 0));
        black_box(Key::read_ikey(d, 8));
        black_box(Key::read_ikey(d, 16));
        black_box(Key::read_ikey(d, 24));
    }
}
