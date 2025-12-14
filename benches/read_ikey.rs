//! Benchmarks for `read_ikey` using criterion.
//!
//! Run with: `cargo bench --bench read_ikey`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use masstree::key::Key;

/// Custom criterion configuration for sub-nanosecond measurements.
fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(500)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1))
        .noise_threshold(0.02)
}

/// Benchmark the fast path (8+ bytes available).
fn bench_fast_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_ikey/fast_path");

    let test_cases: &[(&str, &[u8], usize)] = &[
        ("8B_exact", b"12345678", 0),
        ("16B_start", b"1234567890abcdef", 0),
        ("16B_mid", b"1234567890abcdef", 4),
        ("64B_start", &[b'x'; 64], 0),
        ("64B_mid", &[b'x'; 64], 32),
    ];

    for (name, data, offset) in test_cases {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(name), offset, |b, &off| {
            b.iter(|| black_box(Key::read_ikey(black_box(*data), black_box(off))));
        });
    }

    group.finish();
}

/// Benchmark the slow path (< 8 bytes available).
fn bench_slow_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_ikey/slow_path");

    let test_cases: &[(&str, &[u8], usize)] = &[
        ("1B", b"x", 0),
        ("4B", b"1234", 0),
        ("7B", b"1234567", 0),
        ("tail_3B", b"1234567890", 7),
    ];

    for (name, data, offset) in test_cases {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(name), offset, |b, &off| {
            b.iter(|| black_box(Key::read_ikey(black_box(*data), black_box(off))));
        });
    }

    group.finish();
}

/// Benchmark edge cases (empty, out of bounds).
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_ikey/edge_cases");

    let test_cases: &[(&str, &[u8], usize)] = &[
        ("empty", b"", 0),
        ("oob_small", b"1234", 10),
        ("oob_large", b"12345678", 100),
    ];

    for (name, data, offset) in test_cases {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(name), offset, |b, &off| {
            b.iter(|| black_box(Key::read_ikey(black_box(*data), black_box(off))));
        });
    }

    group.finish();
}

/// Benchmark realistic workload: sequential reads through a key.
fn bench_sequential_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_ikey/sequential");

    let data: &[u8] = b"abcdefghijklmnopqrstuvwxyz012345";
    group.throughput(Throughput::Elements(4));

    group.bench_function("4_ikeys", |b| {
        b.iter(|| {
            let d = black_box(data);
            black_box(Key::read_ikey(d, 0));
            black_box(Key::read_ikey(d, 8));
            black_box(Key::read_ikey(d, 16));
            black_box(Key::read_ikey(d, 24));
        });
    });

    group.finish();
}

/// Benchmark with varying key lengths to simulate real-world distribution.
fn bench_mixed_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_ikey/mixed");

    let keys: &[&[u8]] = &[
        b"a",                               // 1 byte
        b"hello",                           // 5 bytes
        b"username",                        // 8 bytes exact
        b"longer_key_here",                 // 15 bytes
        b"this_is_a_32_byte_key_exactly!!", // 32 bytes
    ];

    group.throughput(Throughput::Elements(keys.len() as u64));

    group.bench_function("varied_keys", |b| {
        b.iter(|| {
            for key in black_box(keys) {
                black_box(Key::read_ikey(key, 0));
            }
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_fast_path, bench_slow_path, bench_edge_cases, bench_sequential_reads, bench_mixed_lengths
}

criterion_main!(benches);
