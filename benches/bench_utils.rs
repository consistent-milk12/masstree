//! Shared helpers for benchmarks.
//!
//! Goals:
//! - Avoid per-op heap allocation for keys (use fixed-size arrays where possible).
//! - Keep key generation deterministic across benches.

#![allow(dead_code, unfulfilled_lint_expectations)]
#![expect(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::items_after_statements,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

/// Deterministically generate fixed-size byte-array keys.
///
/// - `K` must be a multiple of 8, between 8 and 32 (inclusive).
/// - Keys are built from 8-byte chunks derived from `i` with different multipliers.
#[must_use]
pub fn keys<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=32).contains(&K), "key size must be 8..=32");

    const MULTIPLIERS: [u64; 4] = [
        1,
        0x517c_c1b7_2722_0a95,
        0x9e37_79b9_7f4a_7c15,
        0xbf58_476d_1ce4_e5b9,
    ];

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut key = [0u8; K];

        for c in 0..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let bytes = v.to_be_bytes();
            let start = c * 8;

            key[start..start + 8].copy_from_slice(&bytes);
        }

        out.push(key);
    }

    out
}

/// Deterministically generate fixed-size keys where the first 8 bytes are drawn
/// from a small bucketed prefix space to force shared prefixes (ikey collisions).
///
/// This is useful for benchmarking Masstree behavior when many distinct keys
/// share the same initial 8-byte chunk and must be disambiguated by deeper
/// layers.
///
/// - `K` must be a multiple of 8, between 16 and 32 (inclusive).
/// - `prefix_buckets` must be > 0. Smaller values increase collisions.
#[must_use]
pub fn keys_shared_prefix<const K: usize>(n: usize, prefix_buckets: u64) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!(
        (16..=32).contains(&K),
        "key size must be 16..=32 for shared-prefix keys"
    );
    assert!(prefix_buckets > 0, "prefix_buckets must be > 0");

    const MULTIPLIERS: [u64; 4] = [
        1,
        0x517c_c1b7_2722_0a95,
        0x9e37_79b9_7f4a_7c15,
        0xbf58_476d_1ce4_e5b9,
    ];

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];

        let prefix = ((i as u64) % prefix_buckets).to_be_bytes();
        key[0..8].copy_from_slice(&prefix);

        for c in 1..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let bytes = v.to_be_bytes();
            let start = c * 8;
            key[start..start + 8].copy_from_slice(&bytes);
        }

        out.push(key);
    }
    out
}

/// Like [`keys_shared_prefix`], but forces collisions across the first `prefix_chunks`
/// 8-byte chunks (not just the first one).
///
/// This is a harder Masstree workload when `prefix_chunks` is large and
/// `prefix_buckets` is small (e.g. `prefix_chunks=3`, `prefix_buckets=1` for 32B),
/// because many distinct keys share the same prefixes for multiple layers.
///
/// Requirements:
/// - `K` must be a multiple of 8, between 16 and 32 (inclusive).
/// - `prefix_chunks` must be in `1..chunks` (must leave at least one unique chunk).
/// - `prefix_buckets` must be > 0.
#[must_use]
pub fn keys_shared_prefix_chunks<const K: usize>(
    n: usize,
    prefix_chunks: usize,
    prefix_buckets: u64,
) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((16..=32).contains(&K), "key size must be 16..=32");
    assert!(prefix_buckets > 0, "prefix_buckets must be > 0");

    let chunks = K / 8;
    assert!(
        (1..chunks).contains(&prefix_chunks),
        "prefix_chunks must be in 1..chunks"
    );

    const MULTIPLIERS: [u64; 4] = [
        1,
        0x517c_c1b7_2722_0a95,
        0x9e37_79b9_7f4a_7c15,
        0xbf58_476d_1ce4_e5b9,
    ];

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];

        for c in 0..chunks {
            let v = if c < prefix_chunks {
                // Keep each prefix chunk in a small bucket-space.
                // Using a per-chunk multiplier helps avoid "all prefix chunks identical"
                // when prefix_buckets > 1, while still keeping collisions high.
                ((i as u64) % prefix_buckets).wrapping_mul(MULTIPLIERS[c])
            } else {
                // Ensure remaining chunks vary with `i` so keys remain distinct.
                (i as u64).wrapping_mul(MULTIPLIERS[c])
            };
            let bytes = v.to_be_bytes();
            let start = c * 8;
            key[start..start + 8].copy_from_slice(&bytes);
        }

        out.push(key);
    }

    out
}

/// Generate Zipfian-distributed indices (hot keys accessed more frequently).
/// Uses s=1.0 (standard Zipf), approximated via rejection sampling.
#[must_use]
pub fn zipfian_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;
        let idx = ((n as f64).powf(1.0 - u) - 1.0).max(0.0) as usize;
        indices.push(idx.min(n - 1));
    }
    indices
}

/// Uniform random indices.
#[must_use]
pub fn uniform_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        indices.push((state as usize) % n);
    }
    indices
}

/// Shuffle a slice in-place using Fisher-Yates algorithm.
/// Matches the C++ Masstree benchmark pattern.
pub fn shuffle<T>(slice: &mut [T], seed: u64) {
    let n = slice.len();
    if n <= 1 {
        return;
    }

    let mut state = seed;
    for i in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (state as usize) % n;
        slice.swap(i, j);
    }
}

/// Generate random i32 values (like C++ Masstree rw1 test).
/// Returns (keys, values) where value[i] = key[i] + 1.
#[must_use]
pub fn rw1_keys(n: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut keys = Vec::with_capacity(n);
    let mut state = seed;

    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let key = state as i32;
        keys.push(key);
    }

    let values: Vec<i32> = keys.iter().map(|k| k.wrapping_add(1)).collect();
    (keys, values)
}

/// Generate shuffled lookup order for rw1-style benchmarks.
/// Returns indices into the keys array, shuffled randomly.
#[must_use]
pub fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);
    indices
}
