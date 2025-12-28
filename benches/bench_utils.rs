//! Shared helpers for benchmarks.
//!
//! Goals:
//! - Avoid per-op heap allocation for keys (use fixed-size arrays where possible).
//! - Keep key generation deterministic across benches.
//!
//! ## Key Generation Functions
//!
//! ### General Patterns
//!
//! | Function | Pattern | Use Case |
//! |----------|---------|----------|
//! | `keys` | Deterministic, varied chunks | General benchmarks |
//! | `keys_sequential` | Sorted ascending | Sequential scan, best-case B-tree |
//! | `keys_reverse` | Sorted descending | Reverse scan, insertion stress |
//! | `keys_clustered` | Hot ranges with gaps | Real-world access patterns |
//! | `keys_sparse` | Wide gaps between keys | Cache miss stress |
//!
//! ### MassTree-Optimized Patterns (where [`MassTree`] should excel)
//!
//! | Function | Pattern | Why `MassTree` Wins |
//! |----------|---------|-------------------|
//! | `keys_shared_prefix` | First 8B identical | Trie shares prefix traversal |
//! | `keys_shared_prefix_chunks` | Multiple 8B chunks identical | Multi-layer prefix sharing |
//! | `keys_suffix_only_differ` | Only last chunk differs | Suffix mechanism optimized |
//! | `keys_hierarchical` | Nested namespace keys | Trie naturally handles hierarchy |
//! | `keys_variable_length` | Mix of 8B, 16B, 24B, 32B | `MassTree` handles any length |
//! | `string_keys_urls` | URL-like strings | Long keys with shared domains |
//! | `string_keys_paths` | File path strings | Hierarchical shared prefixes |
//!
//! ### Stress Patterns (worst-case scenarios)
//!
//! | Function | Pattern | Stress Target |
//! |----------|---------|---------------|
//! | `keys_adversarial_splits` | Forces maximum splits | Split propagation |
//! | `keys_interleaved_ranges` | Alternating hot/cold | Cache thrashing |
//! | `keys_random_length_simulation` | Unpredictable sizes | Layer transition overhead |
//! | `keys_blink_stress` | Bit-reversal order | B-link pointer following |
//!
//! ### Scan Helpers
//!
//! | Function | Purpose |
//! |----------|---------|
//! | `scan_ranges` | Generate overlapping/non-overlapping range pairs |
//! | `scan_prefixes` | Generate prefix list for prefix scan benchmarks |

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

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut key = [0u8; K];

        for c in 0..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
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
/// - `K` must be a multiple of 8, between 16 and 128 (inclusive).
/// - `prefix_buckets` must be > 0. Smaller values increase collisions.
#[must_use]
pub fn keys_shared_prefix<const K: usize>(n: usize, prefix_buckets: u64) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!(
        (16..=128).contains(&K),
        "key size must be 16..=128 for shared-prefix keys"
    );
    assert!(prefix_buckets > 0, "prefix_buckets must be > 0");

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];

        let prefix = ((i as u64) % prefix_buckets).to_be_bytes();
        key[0..8].copy_from_slice(&prefix);

        for c in 1..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
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

    let chunks = K / 8;
    assert!(
        (1..chunks).contains(&prefix_chunks),
        "prefix_chunks must be in 1..chunks"
    );

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];

        for c in 0..chunks {
            let v = if c < prefix_chunks {
                // Keep each prefix chunk in a small bucket-space.
                // Using a per-chunk multiplier helps avoid "all prefix chunks identical"
                // when prefix_buckets > 1, while still keeping collisions high.
                ((i as u64) % prefix_buckets).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()])
            } else {
                // Ensure remaining chunks vary with `i` so keys remain distinct.
                (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()])
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

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];
        // Put the index in the last 8 bytes for proper sorting
        let bytes = (i as u64).to_be_bytes();
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

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];
        let val = (n - 1 - i) as u64;
        let bytes = val.to_be_bytes();
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

    let mut out = Vec::with_capacity(clusters * keys_per_cluster);
    for cluster in 0..clusters {
        let base = (cluster as u64) * (keys_per_cluster as u64 + gap_size);
        for i in 0..keys_per_cluster {
            let mut key = [0u8; K];
            let val = base + i as u64;
            let bytes = val.to_be_bytes();
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

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];
        let val = (i as u64) * spacing;
        let bytes = val.to_be_bytes();
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

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);

    // Fixed prefix for all keys
    let prefix_value: u64 = 0xDEAD_BEEF_CAFE_BABE;

    for i in 0..n {
        let mut key = [0u8; K];

        // All prefix chunks are identical
        for c in 0..(chunks - 1) {
            let bytes = prefix_value.to_be_bytes();
            let start = c * 8;
            key[start..start + 8].copy_from_slice(&bytes);
        }

        // Only the last chunk varies
        let suffix = (i as u64).to_be_bytes();
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

    let mut out = Vec::with_capacity(namespaces * categories_per_ns * items_per_cat);

    for ns in 0..namespaces {
        for cat in 0..categories_per_ns {
            for item in 0..items_per_cat {
                let mut key = [0u8; K];

                // Chunk 0: namespace
                let ns_bytes = (ns as u64).to_be_bytes();
                key[0..8].copy_from_slice(&ns_bytes);

                // Chunk 1: category
                let cat_bytes = (cat as u64).to_be_bytes();
                key[8..16].copy_from_slice(&cat_bytes);

                // Chunk 2: item ID
                let item_bytes = (item as u64).to_be_bytes();
                key[16..24].copy_from_slice(&item_bytes);

                // Remaining chunks: padding with combined hash
                let combined = ((ns as u64) << 32) | ((cat as u64) << 16) | (item as u64);
                for c in 3..(K / 8) {
                    let v = combined.wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
                    let start = c * 8;
                    key[start..start + 8].copy_from_slice(&v.to_be_bytes());
                }

                out.push(key);
            }
        }
    }
    out
}

/// Container for variable-length keys (as `Vec<u8>`).
#[expect(missing_docs)]
#[derive(Clone, Debug)]
pub struct VariableLengthKeys {
    pub keys: Vec<Vec<u8>>,
    pub length_distribution: [usize; 4], // Count of 8B, 16B, 24B, 32B keys
}

/// Generate keys with varying lengths (8, 16, 24, 32 bytes).
///
/// **`MassTree` advantage**: Handles variable-length keys naturally through
/// the trie structure without padding overhead.
///
/// Distribution can be uniform or weighted toward certain sizes.
#[must_use]
pub fn keys_variable_length(n: usize, seed: u64) -> VariableLengthKeys {
    let mut keys = Vec::with_capacity(n);
    let mut distribution = [0usize; 4];
    let mut state = seed;

    let sizes = [8, 16, 24, 32];

    for i in 0..n {
        // Pseudo-random size selection
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let size_idx = ((state >> 62) as usize) % 4;
        let size = sizes[size_idx];
        distribution[size_idx] += 1;

        let mut key = vec![0u8; size];
        let chunks = size / 8;

        for c in 0..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let start = c * 8;
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
    let mut keys = Vec::with_capacity(n);

    let categories = ["products", "users", "orders", "api", "static", "images"];

    for i in 0..n {
        let domain = i % domains;
        let category = categories[i % categories.len()];
        let id = i / domains;

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
    let mut keys = Vec::with_capacity(n);

    let modules = [
        "core", "util", "api", "db", "cache", "net", "auth", "config",
    ];
    let extensions = ["rs", "toml", "md", "json", "yaml"];

    for i in 0..n {
        let user = i % users;
        let project = (i / users) % projects_per_user;
        let module = modules[i % modules.len()];
        let file_id = i / (users * projects_per_user);
        let ext = extensions[i % extensions.len()];

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

    let mut out = Vec::with_capacity(n);

    // Phase 1: Sequential keys to fill initial structure
    let phase1_count = n / 2;
    for i in 0..phase1_count {
        let mut key = [0u8; K];
        // Multiply by 2 to leave gaps
        let val = (i as u64) * 2;
        let bytes = val.to_be_bytes();
        key[K - 8..].copy_from_slice(&bytes);
        out.push(key);
    }

    // Phase 2: Insert keys in the gaps to force splits
    let phase2_count = n - phase1_count;
    for i in 0..phase2_count {
        let mut key = [0u8; K];
        // Insert odd numbers to land between existing keys
        let val = (i as u64) * 2 + 1;
        let bytes = val.to_be_bytes();
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

    let total_keys = hot_ranges * keys_per_range;
    let mut out = Vec::with_capacity(total_keys);

    // Generate keys but interleave them: key 0 from range 0, key 0 from range 1, ...
    for key_idx in 0..keys_per_range {
        for range_idx in 0..hot_ranges {
            let mut key = [0u8; K];
            let base = (range_idx as u64) * (keys_per_range as u64 + cold_gap);
            let val = base + key_idx as u64;
            let bytes = val.to_be_bytes();
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

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;

    for i in 0..n {
        // Determine "effective length" for this key (1 to chunks)
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let effective_chunks = 1 + ((state >> 60) as usize % chunks);

        let mut key = [0u8; K];

        for c in 0..chunks {
            if c < effective_chunks {
                let v = (i as u64).wrapping_mul(MULTIPLIERS[c % MULTIPLIERS.len()]);
                let start = c * 8;
                key[start..start + 8].copy_from_slice(&v.to_be_bytes());
            }
            // else: leave as zeros (simulating shorter key)
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

    // Generate sequential keys
    let mut keys: Vec<[u8; K]> = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = [0u8; K];
        key[K - 8..].copy_from_slice(&(i as u64).to_be_bytes());
        keys.push(key);
    }

    // Reorder using bit-reversal permutation for maximum split stress
    // This interleaves keys so that consecutive insertions land in different leaves
    let bits = (n as f64).log2().ceil() as u32;
    let mut reordered = Vec::with_capacity(n);

    for i in 0..n {
        let reversed = (i as u32).reverse_bits() >> (32 - bits);
        let idx = (reversed as usize).min(n - 1);
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

    let range_size = total_keys / num_ranges;

    // Non-overlapping ranges
    let mut non_overlapping = Vec::with_capacity(num_ranges);
    for r in 0..num_ranges {
        let start_val = (r * range_size) as u64;
        let end_val = ((r + 1) * range_size - 1) as u64;

        let mut start_key = [0u8; K];
        let mut end_key = [0u8; K];
        start_key[K - 8..].copy_from_slice(&start_val.to_be_bytes());
        end_key[K - 8..].copy_from_slice(&end_val.to_be_bytes());

        non_overlapping.push((start_key, end_key));
    }

    // Overlapping ranges (each overlaps with neighbor by 50%)
    let mut overlapping = Vec::with_capacity(num_ranges);
    for r in 0..num_ranges {
        let start_val = (r * range_size / 2) as u64;
        let end_val = start_val + range_size as u64;

        let mut start_key = [0u8; K];
        let mut end_key = [0u8; K];
        start_key[K - 8..].copy_from_slice(&start_val.to_be_bytes());
        end_key[K - 8..].copy_from_slice(&end_val.to_be_bytes());

        overlapping.push((start_key, end_key));
    }

    (non_overlapping, overlapping)
}

/// Generate prefix ranges for prefix scan benchmarks.
///
/// Returns prefixes that match different numbers of keys.
/// Includes some prefixes that won't match anything (for miss testing).
#[must_use]
pub fn scan_prefixes(prefix_buckets: u64) -> Vec<Vec<u8>> {
    let mut prefixes = Vec::with_capacity(prefix_buckets as usize);

    for bucket in 0..prefix_buckets {
        let prefix = bucket.to_be_bytes().to_vec();
        prefixes.push(prefix);
    }

    // Also add some prefixes that won't match anything
    for i in 0..4 {
        let no_match = (prefix_buckets + i + 1000).to_be_bytes().to_vec();
        prefixes.push(no_match);
    }

    prefixes
}
