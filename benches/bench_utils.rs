//! Shared helpers for benchmarks.

#![allow(dead_code, unfulfilled_lint_expectations, missing_docs)]
#![expect(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::items_after_statements,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used
)]

use std::sync::atomic::{Ordering as AtomicOrdering, fence};

use arrayvec::ArrayVec;
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Zipf};

/// Multipliers for deterministic key chunk generation.
/// Each chunk uses a different multiplier to ensure variation across chunks.
/// Extended to 16 entries to support keys up to 128 bytes.
const MULTIPLIERS: [u64; 16] = [
    1,
    0x517c_c1b7_2722_0a95,
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x6c8e_9448_1e2f_3d4b,
    0xa5c2_f831_7d6e_4a9f,
    0x3b7d_c4e6_2a8f_5c1d,
    0xd92e_8b5a_4f7c_3e6d,
    0x1f4a_9c3b_8e7d_2a5f,
    0xe8b3_6d4c_a2f5_7e9b,
    0x4c9f_2e7a_b5d3_8c6f,
    0x7a5e_c9d4_3f8b_6a2e,
    0xb3d7_4a8f_c2e5_9b3d,
    0x2e6c_b8a3_d5f4_7c9e,
    0x9f3b_5e7c_a4d6_8b2f,
    0x5d8a_c3f6_b9e2_4a7d,
];

/// Deterministically generate fixed-size byte-array keys.
///
/// - `K` must be a multiple of 8, between 8 and 128 (inclusive).
/// - Keys are built from 8-byte chunks derived from `i` with different multipliers.
#[must_use]
pub fn keys<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let chunks: usize = K / 8;
    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];

        for c in 0..chunks {
            let v: u64 = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
            let bytes: [u8; _] = v.to_be_bytes();
            let start: usize = c * 8;

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
/// - `K` must be a multiple of 8, between 16 and 128 (inclusive).
/// - `prefix_buckets` must be > 0. **Smaller values = harder test for Masstree**:
///   - `prefix_buckets=1`: All keys share the SAME prefix (maximum trie depth stress)
///   - `prefix_buckets=10`: ~n/10 keys per prefix bucket (moderate collision)
///   - `prefix_buckets=n`: Each key has unique prefix (no benefit vs standard keys)
#[must_use]
pub fn keys_shared_prefix<const K: usize>(n: usize, prefix_buckets: u64) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!(
        (16..=128).contains(&K),
        "key size must be 16..=128 for shared-prefix keys"
    );
    assert!(prefix_buckets > 0, "prefix_buckets must be > 0");

    let chunks: usize = K / 8;
    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);
    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];

        let prefix: [u8; _] = ((i as u64) % prefix_buckets).to_be_bytes();
        key[0..8].copy_from_slice(&prefix);

        for c in 1..chunks {
            let v: u64 = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
            let bytes: [u8; _] = v.to_be_bytes();
            let start: usize = c * 8;

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
/// - `K` must be a multiple of 8, between 16 and 128 (inclusive).
/// - `prefix_chunks` must be in `1..chunks` (must leave at least one unique chunk).
/// - `prefix_buckets` must be > 0.
#[must_use]
pub fn keys_shared_prefix_chunks<const K: usize>(
    n: usize,
    prefix_chunks: usize,
    prefix_buckets: u64,
) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((16..=128).contains(&K), "key size must be 16..=128");
    assert!(prefix_buckets > 0, "prefix_buckets must be > 0");

    let chunks: usize = K / 8;
    assert!(
        (1..chunks).contains(&prefix_chunks),
        "prefix_chunks must be in 1..chunks"
    );

    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];

        for c in 0..chunks {
            let v: u64 = if c < prefix_chunks {
                // Keep each prefix chunk in a small bucket-space.
                // Using a per-chunk multiplier helps avoid "all prefix chunks identical"
                // when prefix_buckets > 1, while still keeping collisions high.
                ((i as u64) % prefix_buckets).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()])
            } else {
                // Ensure remaining chunks vary with `i` so keys remain distinct.
                (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()])
            };

            let bytes: [u8; _] = v.to_be_bytes();
            let start: usize = c * 8;

            key[start..start + 8].copy_from_slice(&bytes);
        }

        out.push(key);
    }

    out
}

/// Fast power-law skewed indices (not true Zipfian but similar shape).
///
/// Uses `n^(1-u) - 1` which produces a skewed distribution where lower
/// indices are accessed more frequently. This is faster than true Zipfian
/// (no precomputation) but doesn't match the exact Zipfian PMF.
///
/// For true Zipfian distribution, use [`zipfian_indices`] instead.
#[must_use]
pub fn skewed_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    assert!(n > 0, "n must be > 0");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_idx: f64 = (n - 1) as f64;

    (0..count)
        .map(|_| {
            let u: f64 = rng.random();
            ((n as f64).powf(1.0 - u) - 1.0).clamp(0.0, max_idx) as usize
        })
        .collect()
}

/// Generate true Zipfian-distributed indices using `rand_distr::Zipf`.
///
/// The skew parameter `s` controls the distribution shape:
/// - `s = 1.0`: Standard Zipf's law (top 20% keys get ~80% accesses)
/// - `s > 1.0`: More skewed toward lower indices
/// - `s < 1.0`: Flatter distribution
///
/// Uses the well-tested `rand_distr` crate for correct Zipfian sampling.
#[must_use]
pub fn zipfian_indices(n: usize, count: usize, s: f64, seed: u64) -> Vec<usize> {
    assert!(n > 0, "n must be > 0");
    assert!(s > 0.0, "s must be > 0 for Zipf distribution");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let zipf: Zipf<f64> = Zipf::new(n as f64, s).expect("invalid Zipf parameters");

    (0..count)
        .map(|_| {
            // Zipf returns 1-indexed values in [1, n], convert to 0-indexed
            let sample: f64 = zipf.sample(&mut rng);
            (sample as usize).saturating_sub(1).min(n - 1)
        })
        .collect()
}

/// Uniform random indices.
#[must_use]
pub fn uniform_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    assert!(n > 0, "n must be > 0");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| rng.random_range(0..n)).collect()
}

/// Shuffle a slice in-place using Fisher-Yates algorithm.
pub fn shuffle<T>(slice: &mut [T], seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

/// Generate random start indices for range scan benchmarks.
///
/// Returns a vector of indices into a key array, allowing scans to start
/// from random positions rather than always from the beginning.
/// This provides more realistic benchmarks by:
/// - Avoiding always scanning the first few leaves (cache-hot)
/// - Testing different parts of the tree structure
/// - Reducing measurement bias from predictable access patterns
///
/// The indices are pre-generated to avoid RNG overhead during measurement.
#[must_use]
pub fn random_start_indices(key_count: usize, ops_count: usize, seed: u64) -> Vec<usize> {
    // Reserve room at the end so scans don't immediately hit end-of-tree
    let max_start: usize = key_count.saturating_sub(100).max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..ops_count)
        .map(|_| rng.random_range(0..max_start))
        .collect()
}

/// Generate random i32 values (like C++ Masstree rw1 test).
/// Returns (keys, values) where value[i] = key[i] + 1.
#[must_use]
pub fn rw1_keys(n: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let keys: Vec<i32> = (0..n).map(|_| rng.random()).collect();
    let values: Vec<i32> = keys.iter().map(|k: &i32| k.wrapping_add(1)).collect();

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

// =============================================================================
// General Patterns
// =============================================================================

/// Generate sequential (sorted ascending) keys.
///
/// Best-case for range scans and sequential access patterns.
/// Keys are simply `0, 1, 2, ...` in big-endian format.
#[must_use]
pub fn keys_sequential<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];
        let bytes: [u8; _] = (i as u64).to_be_bytes();

        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }

    out
}

/// Generate reverse-sorted (descending) keys.
///
/// Stress test for insertion (worst-case for some B-tree implementations).
/// Also useful for reverse range scan benchmarks.
#[must_use]
pub fn keys_reverse<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];
        let val: u64 = (n - 1 - i) as u64;
        let bytes: [u8; _] = val.to_be_bytes();

        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }
    out
}

/// Generate clustered keys with hot ranges and gaps.
///
/// Simulates real-world access patterns where certain key ranges
/// are accessed more frequently (e.g., recent timestamps, popular users).
///
/// - `clusters`: Number of hot clusters
/// - `keys_per_cluster`: Keys in each cluster
/// - `gap_size`: Gap between clusters
#[must_use]
pub fn keys_clustered<const K: usize>(
    clusters: usize,
    keys_per_cluster: usize,
    gap_size: u64,
) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let mut out: Vec<[u8; K]> = Vec::with_capacity(clusters * keys_per_cluster);

    for cluster in 0..clusters {
        // Use saturating arithmetic to prevent overflow
        let cluster_stride: u64 = (keys_per_cluster as u64).saturating_add(gap_size);
        let base: u64 = (cluster as u64).saturating_mul(cluster_stride);

        for i in 0..keys_per_cluster {
            let mut key: [u8; K] = [0u8; K];
            let val: u64 = base.saturating_add(i as u64);
            let bytes: [u8; _] = val.to_be_bytes();

            key[K - 8..].copy_from_slice(&bytes);
            out.push(key);
        }
    }

    out
}

/// Generate sparse keys with wide gaps.
///
/// Stress test for cache misses and memory access patterns.
/// Keys are spread across a large keyspace.
///
/// - `spacing`: Gap between consecutive keys
#[must_use]
pub fn keys_sparse<const K: usize>(n: usize, spacing: u64) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");
    assert!(spacing > 0, "spacing must be > 0");

    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];
        // Use saturating_mul to prevent overflow for large n * spacing
        let val: u64 = (i as u64).saturating_mul(spacing);
        let bytes: [u8; _] = val.to_be_bytes();

        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }

    out
}

// =============================================================================
// MassTree-Optimized Patterns
// =============================================================================

/// Generate keys where only the last 8-byte chunk differs.
///
/// **`MassTree` advantage**: All keys share the same prefix through all layers
/// except the final one. The suffix mechanism handles this efficiently.
///
/// - `K` must be >= 16 (need at least 2 chunks)
#[must_use]
pub fn keys_suffix_only_differ<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((16..=128).contains(&K), "key size must be 16..=128");

    let chunks: usize = K / 8;
    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    // Fixed prefix for all keys
    let prefix_value: u64 = 0xDEAD_BEEF_CAFE_BABE;

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];

        // All prefix chunks are identical
        for c in 0..(chunks - 1) {
            let bytes: [u8; _] = prefix_value.to_be_bytes();
            let start: usize = c * 8;

            key[start..start + 8].copy_from_slice(&bytes);
        }

        // Only the last chunk varies
        let suffix: [u8; _] = (i as u64).to_be_bytes();

        key[K - 8..].copy_from_slice(&suffix);
        out.push(key);
    }
    out
}

/// Generate hierarchical namespace-style keys.
///
/// **`MassTree` advantage**: Trie structure naturally shares common prefixes.
/// Keys look like: `namespace:category:subcategory:id`
///
/// - `namespaces`: Number of top-level namespaces
/// - `categories_per_ns`: Categories within each namespace
/// - `items_per_cat`: Items within each category
#[must_use]
pub fn keys_hierarchical<const K: usize>(
    namespaces: usize,
    categories_per_ns: usize,
    items_per_cat: usize,
) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!(
        (24..=128).contains(&K),
        "key size must be 24..=128 for hierarchical keys"
    );

    let mut out: Vec<[u8; K]> = Vec::with_capacity(namespaces * categories_per_ns * items_per_cat);

    for ns in 0..namespaces {
        for cat in 0..categories_per_ns {
            for item in 0..items_per_cat {
                let mut key: [u8; K] = [0u8; K];

                // Chunk 0: namespace
                let ns_bytes: [u8; _] = (ns as u64).to_be_bytes();
                key[0..8].copy_from_slice(&ns_bytes);

                // Chunk 1: category
                let cat_bytes: [u8; _] = (cat as u64).to_be_bytes();
                key[8..16].copy_from_slice(&cat_bytes);

                // Chunk 2: item ID
                let item_bytes: [u8; _] = (item as u64).to_be_bytes();
                key[16..24].copy_from_slice(&item_bytes);

                // Remaining chunks: padding with combined hash
                let combined: u64 = ((ns as u64) << 32) | ((cat as u64) << 16) | (item as u64);

                for c in 3..(K / 8) {
                    let v: u64 = combined.wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
                    let start: usize = c * 8;
                    key[start..start + 8].copy_from_slice(&v.to_be_bytes());
                }

                out.push(key);
            }
        }
    }
    out
}

/// Container for variable-length keys using inline storage.
///
/// Uses `ArrayVec<u8, 32>` to store keys inline (up to 32 bytes) without
/// heap allocation per key, eliminating millions of small allocations.
#[expect(missing_docs)]
#[derive(Clone, Debug)]
pub struct VariableLengthKeys {
    pub keys: Vec<ArrayVec<u8, 32>>,
    pub length_distribution: [usize; 4], // Count of 8B, 16B, 24B, 32B keys
}

/// Generate keys with varying lengths (8, 16, 24, 32 bytes).
///
/// **`MassTree` advantage**: Handles variable-length keys naturally through
/// the trie structure without padding overhead.
///
/// Distribution can be uniform or weighted toward certain sizes.
/// Keys are stored inline using `ArrayVec` to avoid per-key heap allocations.
#[must_use]
pub fn keys_variable_length(n: usize, seed: u64) -> VariableLengthKeys {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut keys: Vec<ArrayVec<u8, 32>> = Vec::with_capacity(n);
    let mut distribution: [usize; 4] = [0usize; 4];

    let sizes: [usize; 4] = [8, 16, 24, 32];

    for i in 0..n {
        let size_idx: usize = rng.random_range(0..4);
        let size: usize = sizes[size_idx];
        distribution[size_idx] += 1;

        let mut key = ArrayVec::<u8, 32>::new();
        // Pre-fill with zeros to the target size
        key.extend(std::iter::repeat_n(0u8, size));
        let chunks: usize = size / 8;

        for c in 0..chunks {
            let v: u64 = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let start: usize = c * 8;

            key[start..start + 8].copy_from_slice(&v.to_be_bytes());
        }

        keys.push(key);
    }

    VariableLengthKeys {
        keys,
        length_distribution: distribution,
    }
}

/// Generate URL-like string keys.
///
/// **`MassTree` advantage**: URLs often share common prefixes (domain, path segments).
/// The trie structure efficiently shares these prefixes.
///
/// Format: `https://domain{i%domains}.com/path/{category}/{id}`
#[must_use]
pub fn string_keys_urls(n: usize, domains: usize) -> Vec<String> {
    let mut keys: Vec<String> = Vec::with_capacity(n);

    let categories: [&str; 6] = ["products", "users", "orders", "api", "static", "images"];

    for i in 0..n {
        let domain: usize = i % domains;
        let category: &str = categories[i % categories.len()];
        let id: usize = i / domains;

        keys.push(format!(
            "https://example{domain:04}.com/{category}/item{id:08x}"
        ));
    }

    keys
}

/// Generate file path-like string keys.
///
/// **`MassTree` advantage**: File paths have natural hierarchy with shared prefixes.
///
/// Format: `/home/user{u}/projects/proj{p}/src/module{m}/file{f}.rs`
#[must_use]
pub fn string_keys_paths(n: usize, users: usize, projects_per_user: usize) -> Vec<String> {
    let mut keys: Vec<String> = Vec::with_capacity(n);

    let modules: [&str; 8] = [
        "core", "util", "api", "db", "cache", "net", "auth", "config",
    ];
    let extensions: [&str; 5] = ["rs", "toml", "md", "json", "yaml"];

    for i in 0..n {
        let user: usize = i % users;
        let project: usize = (i / users) % projects_per_user;
        let module: &str = modules[i % modules.len()];
        let file_id: usize = i / (users * projects_per_user);
        let ext: &str = extensions[i % extensions.len()];

        keys.push(format!(
            "/home/user{user:03}/projects/project{project:02}/src/{module}/file{file_id:06}.{ext}"
        ));
    }

    keys
}

// =============================================================================
// Stress Patterns (worst-case scenarios)
// =============================================================================

/// Generate keys designed to force maximum leaf splits.
///
/// Creates sequential keys that will fill leaves and force splits,
/// then inserts keys between existing ones to cause cascading splits.
#[must_use]
pub fn keys_adversarial_splits<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    // Phase 1: Sequential keys to fill initial structure
    let phase1_count: usize = n / 2;

    for i in 0..phase1_count {
        let mut key: [u8; K] = [0u8; K];

        // Multiply by 2 to leave gaps
        let val: u64 = (i as u64) * 2;
        let bytes: [u8; _] = val.to_be_bytes();

        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }

    // Phase 2: Insert keys in the gaps to force splits
    let phase2_count: usize = n - phase1_count;

    for i in 0..phase2_count {
        let mut key: [u8; K] = [0u8; K];

        // Insert odd numbers to land between existing keys
        let val: u64 = (i as u64) * 2 + 1;
        let bytes: [u8; _] = val.to_be_bytes();

        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }

    out
}

/// Generate interleaved hot/cold ranges for cache thrashing.
///
/// Creates alternating bands of keys that, when accessed in order,
/// will thrash the cache by jumping between distant memory regions.
///
/// - `hot_ranges`: Number of hot ranges
/// - `keys_per_range`: Keys in each range
/// - `cold_gap`: Gap between hot ranges (cold region)
#[must_use]
pub fn keys_interleaved_ranges<const K: usize>(
    hot_ranges: usize,
    keys_per_range: usize,
    cold_gap: u64,
) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    let total_keys: usize = hot_ranges * keys_per_range;
    let mut out: Vec<[u8; K]> = Vec::with_capacity(total_keys);

    // Generate keys but interleave them: key 0 from range 0, key 0 from range 1, ...
    for key_idx in 0..keys_per_range {
        for range_idx in 0..hot_ranges {
            let mut key: [u8; K] = [0u8; K];
            let base: u64 = (range_idx as u64) * (keys_per_range as u64 + cold_gap);
            let val: u64 = base + key_idx as u64;
            let bytes: [u8; _] = val.to_be_bytes();

            key[K - 8..].copy_from_slice(&bytes);
            out.push(key);
        }
    }

    out
}

/// Generate keys with random lengths for layer transition stress.
///
/// Unlike `keys_variable_length`, this returns fixed-size arrays but with
/// "effective" variable lengths by using different multipliers to make
/// later chunks "less unique" (simulating shorter keys padded with zeros).
#[must_use]
pub fn keys_random_length_simulation<const K: usize>(n: usize, seed: u64) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((16..=128).contains(&K), "key size must be 16..=128");

    let chunks: usize = K / 8;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        // Determine "effective length" for this key (1 to chunks)
        let effective_chunks: usize = rng.random_range(1..=chunks);

        let mut key: [u8; K] = [0u8; K];

        for c in 0..chunks {
            if c < effective_chunks {
                let v: u64 = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
                let start: usize = c * 8;

                key[start..start + 8].copy_from_slice(&v.to_be_bytes());
            }
        }

        out.push(key);
    }

    out
}

/// Generate keys that stress B-link pointer following during scans.
///
/// Uses bit-reversal permutation to create maximum fragmentation:
/// - Keys are logically sequential (0, 1, 2, ...)
/// - But insertion order causes splits at every level
/// - Range scans must follow many B-link pointers
#[must_use]
pub fn keys_blink_stress<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");

    // Early return for trivial cases (avoids log2(0) = -inf issue)
    if n == 0 {
        return Vec::new();
    }

    if n == 1 {
        let mut key: [u8; K] = [0u8; K];
        key[K - 8..].copy_from_slice(&0u64.to_be_bytes());
        return vec![key];
    }

    // Generate sequential keys
    let mut keys: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; K] = [0u8; K];

        key[K - 8..].copy_from_slice(&(i as u64).to_be_bytes());
        keys.push(key);
    }

    // Reorder using bit-reversal permutation for maximum split stress
    // This interleaves keys so that consecutive insertions land in different leaves
    // Use integer arithmetic to avoid precision loss for large n
    let bits: u32 = usize::BITS - n.saturating_sub(1).leading_zeros();
    let mut reordered: Vec<[u8; K]> = Vec::with_capacity(n);

    for i in 0..n {
        let reversed: u32 = (i as u32).reverse_bits() >> (32 - bits);
        let idx: usize = (reversed as usize).min(n - 1);
        reordered.push(keys[idx]);
    }

    // Deduplicate (bit reversal can produce duplicates for non-power-of-2 n)
    reordered.sort_unstable();
    reordered.dedup();

    // If we lost keys due to dedup, fill with remaining sequential keys
    let mut next_key = n as u64;
    while reordered.len() < n {
        let mut key = [0u8; K];
        key[K - 8..].copy_from_slice(&next_key.to_be_bytes());
        reordered.push(key);
        next_key += 1;
    }

    reordered
}

/// Generate keys optimized for testing concurrent range scans.
///
/// Creates multiple distinct ranges that can be scanned in parallel
/// without overlap, plus some overlapping ranges for contention testing.
///
/// Returns: (`non_overlapping_ranges`, `overlapping_ranges`)
/// Each range is a (`start_key`, `end_key`) pair.
#[must_use]
#[expect(clippy::type_complexity)]
pub fn scan_ranges<const K: usize>(
    total_keys: usize,
    num_ranges: usize,
) -> (Vec<([u8; K], [u8; K])>, Vec<([u8; K], [u8; K])>) {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=128).contains(&K), "key size must be 8..=128");
    assert!(num_ranges > 0, "num_ranges must be > 0");
    assert!(
        total_keys >= num_ranges,
        "total_keys must be >= num_ranges to have non-empty ranges"
    );

    let range_size: u64 = (total_keys / num_ranges) as u64;

    // Non-overlapping ranges
    let mut non_overlapping: Vec<([u8; K], [u8; K])> = Vec::with_capacity(num_ranges);

    for r in 0..num_ranges {
        // Cast before multiply to prevent overflow
        let start_val: u64 = (r as u64) * range_size;
        let end_val: u64 = ((r + 1) as u64) * range_size - 1;

        let mut start_key: [u8; K] = [0u8; K];
        let mut end_key: [u8; K] = [0u8; K];

        start_key[K - 8..].copy_from_slice(&start_val.to_be_bytes());
        end_key[K - 8..].copy_from_slice(&end_val.to_be_bytes());

        non_overlapping.push((start_key, end_key));
    }

    // Overlapping ranges (each overlaps with neighbor by 50%)
    let mut overlapping: Vec<([u8; K], [u8; K])> = Vec::with_capacity(num_ranges);
    for r in 0..num_ranges {
        // Cast before multiply to prevent overflow
        let start_val: u64 = (r as u64) * (range_size / 2);
        let end_val: u64 = start_val + range_size;

        let mut start_key: [u8; K] = [0u8; K];
        let mut end_key: [u8; K] = [0u8; K];

        start_key[K - 8..].copy_from_slice(&start_val.to_be_bytes());
        end_key[K - 8..].copy_from_slice(&end_val.to_be_bytes());

        overlapping.push((start_key, end_key));
    }

    (non_overlapping, overlapping)
}

/// Generate prefix ranges for prefix scan benchmarks.
///
/// Returns prefixes that match different numbers of keys.
/// Includes 4 additional prefixes that won't match anything (for miss testing).
#[must_use]
pub fn scan_prefixes(prefix_buckets: u64) -> Vec<Vec<u8>> {
    let mut prefixes: Vec<Vec<u8>> = Vec::with_capacity(prefix_buckets as usize + 4);

    for bucket in 0..prefix_buckets {
        let prefix: Vec<u8> = bucket.to_be_bytes().to_vec();
        prefixes.push(prefix);
    }

    // Also add some prefixes that won't match anything
    for i in 0..4 {
        let no_match: Vec<u8> = (prefix_buckets + i + 1000).to_be_bytes().to_vec();
        prefixes.push(no_match);
    }

    prefixes
}

// =============================================================================
// Benchmark Warmup & Methodology Helpers
// =============================================================================

/// Number of warmup iterations to run before actual measurement.
/// This ensures CPU caches, branch predictors, and JIT are stable.
pub const DEFAULT_WARMUP_ITERS: usize = 1000;

/// Perform explicit warmup by running a closure multiple times.
#[inline(never)]
pub fn warmup<F: FnMut()>(mut f: F, iterations: usize) {
    for _ in 0..iterations {
        f();
    }

    // Memory barrier to ensure warmup effects are visible
    fence(AtomicOrdering::SeqCst);
}

/// Warmup a data structure by accessing random elements.
///
/// Generic helper that warms up any data structure with a get-like operation.
/// Uses the provided indices to touch memory locations.
#[inline(never)]
pub fn warmup_with_indices<F: FnMut(usize)>(
    mut access_fn: F,
    indices: &[usize],
    iterations: usize,
) {
    let n = indices.len();
    if n == 0 {
        return;
    }
    for i in 0..iterations {
        let idx = indices[i % n];
        access_fn(idx);
    }
    fence(AtomicOrdering::SeqCst);
}

/// Memory barrier before measurement.
///
/// Prevents both compiler and CPU from reordering measured code before this point.
/// Uses full `fence` for cross-platform correctness (ARM, RISC-V need hardware fence).
#[inline]
pub fn pre_measurement_barrier() {
    fence(AtomicOrdering::SeqCst);
}

/// Memory barrier after measurement.
///
/// Prevents both compiler and CPU from reordering measured code after this point.
/// Uses full `fence` for cross-platform correctness (ARM, RISC-V need hardware fence).
#[inline]
pub fn post_measurement_barrier() {
    fence(AtomicOrdering::SeqCst);
}

/// Helper to run a benchmark with proper setup.
///
/// This provides a standard pattern for benchmarks:
/// 1. Pin thread (if possible)
/// 2. Run warmup iterations
/// 3. Insert memory barrier
/// 4. Run measured iterations
/// 5. Insert memory barrier
#[derive(Debug)]
pub struct BenchmarkRunner {
    pub warmup_iters: usize,

    pub pin_core: Option<usize>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            warmup_iters: DEFAULT_WARMUP_ITERS,
            pin_core: None,
        }
    }
}

impl BenchmarkRunner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_warmup(mut self, iters: usize) -> Self {
        self.warmup_iters = iters;
        self
    }

    #[must_use]
    pub const fn with_pin(mut self, core: usize) -> Self {
        self.pin_core = Some(core);
        self
    }

    /// Execute warmup and setup, then run the benchmark closure.
    pub fn run<W, B, R>(&self, warmup_fn: W, bench_fn: B) -> R
    where
        W: FnMut(),
        B: FnOnce() -> R,
    {
        // Warmup
        warmup(warmup_fn, self.warmup_iters);

        // Barrier before measurement
        pre_measurement_barrier();

        // Run benchmark
        let result: R = bench_fn();

        // Barrier after measurement
        post_measurement_barrier();

        result
    }
}

// =============================================================================
// Concurrent Benchmark Harness
// =============================================================================

/// Context provided to each thread in a concurrent benchmark.
///
/// Controls the warmup → synchronize → measure → finish protocol.
pub struct ThreadContext<'a> {
    /// Thread index (0-based).
    pub tid: usize,
    /// Effective warmup ops (proportional to workload size).
    pub warmup_ops: usize,
    warmup_done: &'a std::sync::Barrier,
    start: &'a std::sync::Barrier,
}

impl ThreadContext<'_> {
    /// Signal that this thread has completed its warmup phase.
    /// Blocks until all threads have finished warmup.
    pub fn finish_warmup(&self) {
        self.warmup_done.wait();
    }

    /// Synchronize all threads and begin the measurement phase.
    /// Inserts a memory barrier to prevent reordering, then waits for all threads.
    pub fn begin_measurement(&self) {
        pre_measurement_barrier();
        self.start.wait();
    }

    /// End the measurement phase with a memory barrier.
    pub fn end_measurement(&self) {
        post_measurement_barrier();
    }
}

/// Check if core pinning is requested via environment variable.
fn should_pin_cores() -> bool {
    std::env::var("BENCH_PIN_CORES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Run a concurrent benchmark with proper thread management.
///
/// Uses `std::thread::scope` for zero-cost borrowing (no Arc needed).
///
/// Features:
/// - Proportional warmup: `min(warmup_ops, ops_per_thread / 5)`
/// - Optional core pinning via `BENCH_PIN_CORES=1` environment variable
/// - Two-barrier synchronization: warmup completion + measurement start
///
/// The `per_thread` closure receives a [`ThreadContext`] and should follow
/// this protocol:
/// 1. Perform warmup using `ctx.warmup_ops` iterations
/// 2. Call `ctx.finish_warmup()`
/// 3. Call `ctx.begin_measurement()`
/// 4. Execute measured operations
/// 5. Call `ctx.end_measurement()`
pub fn run_concurrent<F>(threads: usize, warmup_ops: usize, ops_per_thread: usize, per_thread: F)
where
    F: Fn(&ThreadContext) + Send + Sync,
{
    let effective_warmup = warmup_ops.min(ops_per_thread / 5);
    let warmup_done = std::sync::Barrier::new(threads);
    let start = std::sync::Barrier::new(threads);
    let pin = should_pin_cores();
    let core_ids = if pin {
        core_affinity::get_core_ids().unwrap_or_default()
    } else {
        vec![]
    };

    std::thread::scope(|s| {
        for tid in 0..threads {
            let warmup_done = &warmup_done;
            let start = &start;
            let core_ids = &core_ids;
            let per_thread = &per_thread;
            s.spawn(move || {
                if !core_ids.is_empty() {
                    let core_id = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core_id);
                }
                let ctx = ThreadContext {
                    tid,
                    warmup_ops: effective_warmup,
                    warmup_done,
                    start,
                };
                per_thread(&ctx);
            });
        }
    });
}
