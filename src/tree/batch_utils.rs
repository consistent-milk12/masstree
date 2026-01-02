//! ========================================================================
//!  Batch Insert Utilities
//! ========================================================================
//!
//! Utility functions for preparing and analyzing batch insert operations.
//!
//! # Note
//!
//! The `insert_batch()` method is available directly on all tree types
//! (`MassTree24`, `MassTree15`, `MassTree24Inline`, `MassTree15Inline`)
//! because they are type aliases for `MassTreeGeneric`.

#![allow(clippy::indexing_slicing)]

use std::collections::BTreeSet;

// ============================================================================
//  Entry Preparation Utilities
// ============================================================================

/// Prepare a batch of entries from parallel iterators.
///
/// Combines separate key and value iterators into batch entries.
///
/// # Arguments
///
/// * `keys` - Iterator of key bytes (anything that converts to `Vec<u8>`)
/// * `values` - Iterator of values
///
/// # Returns
///
/// Vector of `(Vec<u8>, V)` tuples ready for `insert_batch`.
///
/// # Example
///
/// ```rust,ignore
/// use masstree::batch::zip_into_entries;
///
/// let keys = vec![b"a".to_vec(), b"b".to_vec()];
/// let values = vec![1u64, 2u64];
///
/// let entries = zip_into_entries(keys, values);
/// tree.insert_batch(entries)?;
/// ```
#[inline]
#[must_use]
pub fn zip_into_entries<K, V, IK, IV>(keys: IK, values: IV) -> Vec<(Vec<u8>, V)>
where
    K: Into<Vec<u8>>,
    IK: IntoIterator<Item = K>,
    IV: IntoIterator<Item = V>,
{
    keys.into_iter()
        .zip(values)
        .map(|(k, v)| (k.into(), v))
        .collect()
}

/// Prepare a batch of entries from a map-like iterator.
///
/// # Arguments
///
/// * `iter` - Iterator of (key, value) pairs where keys convert to bytes
///
/// # Returns
///
/// Vector of `(Vec<u8>, V)` tuples ready for `insert_batch`.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use masstree::batch::from_iter;
///
/// let mut map = HashMap::new();
/// map.insert("key1", 1u64);
/// map.insert("key2", 2u64);
///
/// let entries = from_iter(map.into_iter().map(|(k, v)| (k.as_bytes().to_vec(), v)));
/// tree.insert_batch(entries)?;
/// ```
#[inline]
#[must_use]
pub fn from_iter<K, V, I>(iter: I) -> Vec<(Vec<u8>, V)>
where
    K: Into<Vec<u8>>,
    I: IntoIterator<Item = (K, V)>,
{
    iter.into_iter().map(|(k, v)| (k.into(), v)).collect()
}

/// Generate sequential byte keys with a common prefix.
///
/// Useful for benchmarking and testing batch insert performance.
///
/// # Arguments
///
/// * `prefix` - Common prefix for all keys
/// * `start` - Starting number (inclusive)
/// * `end` - Ending number (exclusive)
/// * `width` - Zero-padding width for numbers
///
/// # Returns
///
/// Vector of byte keys.
///
/// # Example
///
/// ```rust,ignore
/// use masstree::batch::sequential_keys;
///
/// let keys = sequential_keys(b"user:", 0, 1000, 6);
/// // Produces: ["user:000000", "user:000001", ..., "user:000999"]
/// ```
#[must_use]
pub fn sequential_keys(prefix: &[u8], start: usize, end: usize, width: usize) -> Vec<Vec<u8>> {
    (start..end)
        .map(|i| {
            let mut key = prefix.to_vec();
            key.extend(format!("{i:0width$}").as_bytes());
            key
        })
        .collect()
}

/// Generate sequential 8-byte keys (u64 big-endian).
///
/// These keys have optimal locality for batch insert.
///
/// # Arguments
///
/// * `start` - Starting value (inclusive)
/// * `end` - Ending value (exclusive)
///
/// # Example
///
/// ```rust,ignore
/// use masstree::batch::sequential_u64_keys;
///
/// let keys = sequential_u64_keys(0, 1000);
/// let entries: Vec<_> = keys.into_iter().zip(0u64..1000).collect();
/// tree.insert_batch(entries)?;
/// ```
#[must_use]
#[inline(always)]
pub fn sequential_u64_keys(start: u64, end: u64) -> Vec<Vec<u8>> {
    (start..end).map(|i| i.to_be_bytes().to_vec()).collect()
}

// ============================================================================
//  Batch Analysis
// ============================================================================

/// Statistics about a batch of entries before insertion.
///
/// Use this to estimate batch performance and optimize entry ordering.
#[must_use]
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of entries.
    pub count: usize,

    /// Number of unique ikey prefixes (first 8 bytes).
    ///
    /// Lower values indicate better locality (entries cluster into fewer leaves).
    pub unique_ikeys: usize,

    /// Average key length in bytes.
    pub avg_key_len: f64,

    /// Number of keys with suffixes (> 8 bytes).
    pub keys_with_suffix: usize,

    /// Number of keys that are single-layer (≤ 8 bytes).
    pub single_layer_keys: usize,
}

impl BatchStats {
    /// Analyze a batch of entries and compute statistics.
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of (key, value) pairs
    ///
    /// # Returns
    ///
    /// Statistics about the batch.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use masstree::batch::BatchStats;
    ///
    /// let entries = vec![
    ///     (b"short".to_vec(), 1),
    ///     (b"this_is_a_longer_key".to_vec(), 2),
    /// ];
    ///
    /// let stats = BatchStats::analyze(&entries);
    /// println!("Unique prefixes: {}", stats.unique_ikeys);
    /// println!("Keys with suffix: {}", stats.keys_with_suffix);
    /// ```
    pub fn analyze<V>(entries: &[(Vec<u8>, V)]) -> Self {
        let count = entries.len();

        if count == 0 {
            return Self::default();
        }

        // Use BTreeSet instead of HashSet to avoid allocation overhead
        // and provide deterministic iteration order
        let mut unique_ikeys = BTreeSet::new();
        let mut total_key_len: usize = 0;
        let mut keys_with_suffix = 0;
        let mut single_layer_keys = 0;

        for (key, _) in entries {
            // Compute ikey
            let mut buf = [0u8; 8];
            let len = key.len().min(8);
            buf[..len].copy_from_slice(&key[..len]);
            let ikey = u64::from_be_bytes(buf);

            unique_ikeys.insert(ikey);
            total_key_len += key.len();

            if key.len() > 8 {
                keys_with_suffix += 1;
            } else {
                single_layer_keys += 1;
            }
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "Statistical averages don't need full usize precision"
        )]
        let avg_key_len = total_key_len as f64 / count as f64;

        Self {
            count,
            unique_ikeys: unique_ikeys.len(),
            avg_key_len,
            keys_with_suffix,
            single_layer_keys,
        }
    }

    /// Estimate the locality factor (0.0 to 1.0).
    ///
    /// Higher values indicate better locality:
    /// - 1.0 = All entries have the same ikey prefix (best case)
    /// - 0.0 = All entries have unique ikey prefixes (worst case)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = BatchStats::analyze(&entries);
    /// let locality = stats.locality_factor();
    ///
    /// if locality > 0.5 {
    ///     println!("Good locality - batch insert recommended");
    /// } else {
    ///     println!("Poor locality - consider sorting keys differently");
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Locality ratio doesn't need full usize precision"
    )]
    pub fn locality_factor(&self) -> f64 {
        if self.count == 0 || self.unique_ikeys == 0 {
            return 0.0;
        }
        // Clamp to [0.0, 1.0] to handle edge cases
        (1.0 - (self.unique_ikeys as f64 / self.count as f64)).clamp(0.0, 1.0)
    }

    /// Estimate the number of leaves that will be touched.
    ///
    /// This is a rough estimate based on unique ikey prefixes.
    /// Actual leaf count may be higher if leaves split during insertion.
    #[must_use]
    #[inline(always)]
    pub const fn estimated_leaves(&self) -> usize {
        self.unique_ikeys
    }

    /// Check if single-layer optimization applies to all keys.
    #[must_use]
    #[inline(always)]
    pub const fn all_single_layer(&self) -> bool {
        self.keys_with_suffix == 0
    }
}
