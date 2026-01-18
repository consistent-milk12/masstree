// ============================================================================
//  Fast Suffix Comparison (zero-overhead optimized)
// ============================================================================

use std::cmp::Ordering;

/// Threshold below which XOR-based comparison is used.
const WORD_COMPARE_THRESHOLD: usize = 8;

pub(super) struct CompareSuffix;

impl CompareSuffix {
    /// XOR-based equality for short byte slices (≤8 bytes).
    ///
    /// Uses safe copy_from_slice instead of unsafe copy_nonoverlapping.
    /// Matches C++ `equals_sloppy`.
    #[inline(always)]
    fn xor_eq_short(a: &[u8], b: &[u8], len: usize) -> bool {
        debug_assert!(len <= 8);
        debug_assert_eq!(a.len(), len);
        debug_assert_eq!(b.len(), len);

        if len == 0 {
            return true;
        }

        // Load both as padded u64 values
        // For partial reads, we pad with zeros
        let mut a_buf: [u8; 8] = [0u8; 8];
        let mut b_buf: [u8; 8] = [0u8; 8];

        // INVARIANT: len <= 8, both slices have exactly len bytes
        #[expect(clippy::indexing_slicing, reason = "len <= 8 by debug_assert")]
        {
            a_buf[..len].copy_from_slice(a);
            b_buf[..len].copy_from_slice(b);
        }

        let a_word = u64::from_ne_bytes(a_buf);
        let b_word = u64::from_ne_bytes(b_buf);

        // XOR gives 0 for matching bytes, non-zero for differences
        // Since we zero-padded, XOR result will have zeros in unused positions
        a_word == b_word
    }

    /// Read a u64 from a byte slice at the given offset.
    ///
    /// Zero-overhead: uses direct array conversion + from_ne_bytes.
    /// The compiler optimizes away the bounds check since caller guarantees size.
    #[inline(always)]
    fn read_u64_at(slice: &[u8], offset: usize) -> u64 {
        // INVARIANT: caller ensures offset + 8 <= slice.len()
        #[expect(clippy::indexing_slicing, reason = "caller ensures bounds")]
        let bytes: &[u8] = &slice[offset..offset + 8];

        // Zero-overhead conversion: TryInto for [u8; 8] is infallible here
        // since we sliced exactly 8 bytes. The unwrap is optimized away.
        let arr: [u8; 8] = bytes.try_into().expect("slice is exactly 8 bytes");
        u64::from_ne_bytes(arr)
    }

    /// Compare two byte slices for equality.
    ///
    /// - For slices ≤ 8 bytes: uses XOR-based comparison (single operation)
    /// - For slices > 8 bytes: compares 8 bytes at a time using word loads
    #[inline(always)]
    pub(super) fn fast_slice_eq(a: &[u8], b: &[u8]) -> bool {
        let len: usize = a.len();

        // Length mismatch is the most common failure case
        if crate::hints::unlikely(len != b.len()) {
            return false;
        }

        // Short path: XOR-based comparison for ≤8 bytes
        if len <= WORD_COMPARE_THRESHOLD {
            return Self::xor_eq_short(a, b, len);
        }

        // Word-aligned path for longer slices using word loads
        let chunks: usize = len / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let a_word = Self::read_u64_at(a, offset);
            let b_word = Self::read_u64_at(b, offset);

            if a_word != b_word {
                return false;
            }
        }

        // Compare remainder (0-7 bytes)
        let remainder_start: usize = chunks * 8;
        // INVARIANT: remainder_start = chunks * 8 <= len (both slices same length)
        #[expect(clippy::indexing_slicing, reason = "remainder_start <= len")]
        {
            a[remainder_start..] == b[remainder_start..]
        }
    }

    /// Lexicographic comparison of two byte slices.
    ///
    /// - For slices where `min_len` < 8: uses native `.cmp()` (LLVM memcmp)
    /// - For longer slices: compares 8 bytes at a time using word loads
    #[inline(always)]
    pub(super) fn fast_slice_cmp(a: &[u8], b: &[u8]) -> Ordering {
        let min_len: usize = a.len().min(b.len());

        // Short path: use native comparison
        if min_len < WORD_COMPARE_THRESHOLD {
            return a.cmp(b);
        }

        // Word-aligned path for longer slices using word loads
        let chunks: usize = min_len / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let a_word = Self::read_u64_at(a, offset);
            let b_word = Self::read_u64_at(b, offset);

            if a_word != b_word {
                // Convert to big-endian for correct lexicographic order
                let a_be: u64 = a_word.to_be();
                let b_be: u64 = b_word.to_be();

                return a_be.cmp(&b_be);
            }
        }

        // Compare remainder, then by length
        let remainder_start: usize = chunks * 8;

        // INVARIANT: remainder_start <= min_len <= both lengths
        #[expect(clippy::indexing_slicing, reason = "remainder_start <= min_len")]
        match a[remainder_start..min_len].cmp(&b[remainder_start..min_len]) {
            Ordering::Equal => a.len().cmp(&b.len()),
            other => other,
        }
    }
}
