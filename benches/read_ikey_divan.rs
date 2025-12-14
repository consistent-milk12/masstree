//! Benchmarks for `read_ikey` using divan.
//!
//! Run with: `cargo bench --bench read_ikey_divan`

use divan::{black_box, Bencher};
use masstree::key::Key;

fn main() {
    divan::main();
}

// =============================================================================
// Fast path benchmarks (8+ bytes available)
// =============================================================================

mod fast_path {
    use super::{black_box, Bencher, Key};

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_8b(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"12345678".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_16b_start(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"1234567890abcdef".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_16b_mid(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"1234567890abcdef".as_slice(), 4usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_64b(bencher: Bencher<'_, '_>) {
        const DATA: &[u8; 64] = &[b'x'; 64];
        bencher
            .with_inputs(|| (DATA.as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }
}

// =============================================================================
// Slow path benchmarks (< 8 bytes available)
// =============================================================================

mod slow_path {
    use super::{black_box, Bencher, Key};

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_1b(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"x".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_4b(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"1234".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_7b(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"1234567".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_tail_3b(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"1234567890".as_slice(), 7usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }
}

// =============================================================================
// Edge cases (empty, out of bounds)
// =============================================================================

mod edge_cases {
    use super::{black_box, Bencher, Key};

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_empty(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"".as_slice(), 0usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }

    #[divan::bench(sample_count = 200, sample_size = 10000)]
    fn read_oob(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| (b"12345678".as_slice(), 100usize))
            .bench_local_values(|(data, offset)| black_box(Key::read_ikey(data, offset)));
    }
}

// =============================================================================
// Sequential reads (realistic workload)
// =============================================================================

mod sequential {
    use super::{black_box, Bencher, Key};

    const DATA: &[u8] = b"abcdefghijklmnopqrstuvwxyz012345";

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_4_ikeys(bencher: Bencher<'_, '_>) {
        bencher.with_inputs(|| DATA).bench_local_values(|data| {
            black_box(Key::read_ikey(data, 0));
            black_box(Key::read_ikey(data, 8));
            black_box(Key::read_ikey(data, 16));
            black_box(Key::read_ikey(data, 24));
        });
    }
}

// =============================================================================
// Mixed lengths (real-world distribution)
// =============================================================================

mod mixed {
    use super::{black_box, Bencher, Key};

    const KEYS: &[&[u8]] = &[
        b"a",                               // 1 byte
        b"hello",                           // 5 bytes
        b"username",                        // 8 bytes exact
        b"longer_key_here",                 // 15 bytes
        b"this_is_a_32_byte_key_exactly!!", // 32 bytes
    ];

    #[divan::bench(sample_count = 200, sample_size = 5000)]
    fn read_varied_keys(bencher: Bencher<'_, '_>) {
        bencher.with_inputs(|| KEYS).bench_local_values(|keys| {
            for key in keys {
                black_box(Key::read_ikey(key, 0));
            }
        });
    }
}
