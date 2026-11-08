//! Filepath: `src/tree/range/iterator/range_bound.rs`
//!
//! Range bound type for scan operations.

use std::cmp::Ordering;
use std::ops::Bound;

use crate::key::IKEY_SIZE;
use crate::tree::range::cursor_key::CursorKey;

/// How a bound applies to the sublayer the cursor is currently in.
///
/// A bound's next 8 bytes can only order slots inside a sublayer when the
/// sublayer's prefix equals the bound's prefix. Otherwise the entire sublayer
/// is either wholly inside or wholly outside the bound, and per-slot checks are
/// both unnecessary and wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorBound {
    /// No per-slot check is needed: the bound is unbounded, or every key in
    /// this sublayer is inside it.
    Inside,

    /// Compare slot ikeys against this value (the bound's bytes at the
    /// current offset).
    At(u64),

    /// Every key in this sublayer is outside the bound.
    Outside,
}

/// Range bound for scan operations.
///
/// Specifies the start or end of a key range for scanning.
///
/// # Variants
///
/// - [`Unbounded`](RangeBound::Unbounded): No bound (all keys)
/// - [`Included`](RangeBound::Included): Include the specified key
/// - [`Excluded`](RangeBound::Excluded): Exclude the specified key
///
/// # Example
///
/// ```rust,ignore
/// // Scan from "aaa" (inclusive) to "zzz" (exclusive)
/// let start = RangeBound::Included(b"aaa");
/// let end = RangeBound::Excluded(b"zzz");
///
/// for entry in tree.range(start, end, &guard) {
///     // entry.key will be >= "aaa" and < "zzz"
/// }
///```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeBound<'a> {
    /// No bound - start from minimum or continue to maximum.
    Unbounded,

    /// Inclusive bound - include the specified key.
    Included(&'a [u8]),

    /// Exclusive bound - exclude the specified key.
    Excluded(&'a [u8]),
}

impl<'a> RangeBound<'a> {
    /// Check if a key is within this bound (for end bound checking).
    ///
    /// For end bounds:
    /// - `Unbounded`: all keys are within
    /// - `Included(k)`: keys <= k are within
    /// - `Excluded(k)`: keys < k are within
    ///
    /// # Arguments
    ///
    /// - `key`: The key to check
    ///
    /// # Returns
    ///
    /// `true` if the key is within the bound.
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, key: &[u8]) -> bool {
        match self {
            RangeBound::Unbounded => true,

            RangeBound::Included(bound) => key <= *bound,

            RangeBound::Excluded(bound) => key < *bound,
        }
    }

    /// Convert to `(start_key, emit_equal)` parameters for `find_initial`.
    ///
    /// For start bounds:
    /// - `Unbounded`: empty key, `emit_equal = true`
    /// - `Included(k)`: key k, `emit_equal = true`
    /// - `Excluded(k)`: key k, `emit_equal = false`
    ///
    /// # Returns
    ///
    /// Tuple of (key bytes, `emit_equal` flag).
    #[must_use]
    #[inline(always)]
    pub const fn to_start_params(&self) -> (&'a [u8], bool) {
        match self {
            RangeBound::Unbounded => (&[], true),

            RangeBound::Included(k) => (*k, true),

            RangeBound::Excluded(k) => (*k, false),
        }
    }

    /// Check if this is an unbounded bound.
    #[must_use]
    #[inline(always)]
    pub const fn is_unbounded(&self) -> bool {
        matches!(self, RangeBound::Unbounded)
    }

    /// Get the bound key if this is a bounded bound.
    ///
    /// # Note
    ///
    /// This method is provided for API completeness and may be useful for
    /// external callers who need to inspect bounds programmatically.
    #[must_use]
    #[inline(always)]
    pub const fn key(&self) -> Option<&'a [u8]> {
        match self {
            RangeBound::Unbounded => None,

            RangeBound::Included(k) | RangeBound::Excluded(k) => Some(*k),
        }
    }

    /// Check if a key is within this bound for reverse iteration (start bound).
    ///
    /// For start bounds used in reverse iteration:
    /// - `Unbounded`: all keys are within
    /// - `Included(k)`: keys >= k are within
    /// - `Excluded(k)`: keys > k are within
    ///
    /// This is the reverse of `contains()` which is for end bounds.
    #[must_use]
    #[inline(always)]
    pub fn contains_reverse(&self, key: &[u8]) -> bool {
        match self {
            RangeBound::Unbounded => true,

            RangeBound::Included(bound) => key >= *bound,

            RangeBound::Excluded(bound) => key > *bound,
        }
    }

    /// Extract the ikey from a bound for fast comparison.
    ///
    /// Returns `None` for `Unbounded`, otherwise extracts the first 8 bytes
    /// as a big-endian u64 (zero-padded if key is shorter).
    ///
    /// Scans use this for fast interior-key checks; equal ikeys require the
    /// full key and exact bound comparison.
    #[must_use]
    #[inline]
    pub fn extract_ikey(&self) -> Option<u64> {
        self.extract_ikey_at(0)
    }

    /// Extract the ikey at a specific key offset for fast layer-aligned checks.
    ///
    /// `offset` is the number of key bytes already consumed by trie descent.
    /// Returns `None` for `Unbounded`.
    #[must_use]
    #[inline]
    pub fn extract_ikey_at(&self, offset: usize) -> Option<u64> {
        match self {
            RangeBound::Unbounded => None,

            RangeBound::Included(bound) | RangeBound::Excluded(bound) => {
                Some(Self::key_to_ikey_at(bound, offset))
            }
        }
    }

    /// Resolve how this bound applies inside the cursor's current sublayer.
    ///
    /// `forward` selects end-bound semantics (`contains`) when true and
    /// start-bound semantics (`contains_reverse`) when false.
    ///
    /// The next 8 bound bytes are meaningful only when the sublayer's prefix
    /// matches the bound's prefix. If the sublayer's prefix is lexicographically
    /// before the bound, a forward scan may still enter it but a reverse scan
    /// never can (and vice versa), so the whole sublayer is classified without
    /// per-slot work.
    ///
    /// # Panics
    ///
    /// Panics only on an internal invariant violation: a bounded key must
    /// yield an ikey.
    #[must_use]
    #[inline]
    pub fn cursor_bound(&self, cursor: &CursorKey, forward: bool) -> CursorBound {
        let Some(bound) = self.key() else {
            return CursorBound::Inside;
        };

        let offset: usize = cursor.offset();

        if offset == 0 {
            // SAFETY: a bounded key always yields an ikey at offset 0.
            return CursorBound::At(self.extract_ikey_at(0).expect("bounded key has an ikey"));
        }

        let consumed: &[u8] = &cursor.full_key()[..offset];
        let common: usize = offset.min(bound.len());

        match consumed[..common].cmp(&bound[..common]) {
            // The sublayer prefix is below the bound.
            Ordering::Less => {
                if forward {
                    CursorBound::Inside
                } else {
                    CursorBound::Outside
                }
            }

            // The sublayer prefix is above the bound.
            Ordering::Greater => {
                if forward {
                    CursorBound::Outside
                } else {
                    CursorBound::Inside
                }
            }

            Ordering::Equal => {
                if bound.len() > offset {
                    // The bound continues inside this sublayer.
                    // SAFETY: bound has at least 8 bytes at `offset`
                    // only when `bound.len() - offset >= 8`; a shorter tail
                    // is zero-padded by key_to_ikey_at.
                    CursorBound::At(
                        self.extract_ikey_at(offset)
                            .expect("bounded key has an ikey"),
                    )
                } else if forward {
                    // The end bound ends at (or before) this prefix. Every key
                    // in the sublayer has this prefix plus more bytes, so all
                    // of them sort strictly after the bound.
                    CursorBound::Outside
                } else {
                    // The start bound ends at this prefix. Every key in the
                    // sublayer is strictly greater than it.
                    CursorBound::Inside
                }
            }
        }
    }

    /// Convert key bytes to ikey (first 8 bytes as big-endian u64).
    #[inline]
    #[expect(clippy::indexing_slicing, reason = "length is checked before indexing")]
    fn key_to_ikey_at(key: &[u8], offset: usize) -> u64 {
        if offset >= key.len() {
            return 0;
        }

        let remaining = key.len() - offset;
        if remaining >= IKEY_SIZE {
            // SAFETY: length checked above
            #[expect(clippy::expect_used, reason = "length is checked")]
            u64::from_be_bytes(
                key[offset..offset + IKEY_SIZE]
                    .try_into()
                    .expect("slice is 8 bytes"),
            )
        } else {
            // Zero-pad short keys
            let mut bytes = [0u8; IKEY_SIZE];
            bytes[..remaining].copy_from_slice(&key[offset..]);
            u64::from_be_bytes(bytes)
        }
    }
}

// Conversion from std::ops::Bound
impl<'a> From<Bound<&'a [u8]>> for RangeBound<'a> {
    fn from(bound: Bound<&'a [u8]>) -> Self {
        match bound {
            Bound::Unbounded => RangeBound::Unbounded,

            Bound::Included(k) => RangeBound::Included(k),

            Bound::Excluded(k) => RangeBound::Excluded(k),
        }
    }
}
