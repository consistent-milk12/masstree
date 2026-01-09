// ============================================================================
//  Fast Suffix Comparison
// ============================================================================

use std::cmp::Ordering;

/// Threshold below which native comparison is faster than word-aligned.
/// LLVM's memcmp uses SIMD for any size; our manual loop only wins for longer slices.
const WORD_COMPARE_THRESHOLD: usize = 8;

pub(super) struct CompareSuffix;

impl CompareSuffix {
    /// Compare two byte slices for equality.
    ///
    /// - For slices < 8 bytes: uses native `==` (LLVM memcmp with SIMD)
    /// - For slices >= 8 bytes: compares 8 bytes at a time using word loads
    #[inline(always)]
    pub(super) fn fast_slice_eq(a: &[u8], b: &[u8]) -> bool {
        let len: usize = a.len();

        // Length mismatch is the most common failure case
        if len != b.len() {
            return false;
        }

        // Short path: use native comparison (LLVM optimizes this well)
        if len < WORD_COMPARE_THRESHOLD {
            return a == b;
        }

        // Word-aligned path for longer slices
        let a_ptr: *const u8 = a.as_ptr();
        let b_ptr: *const u8 = b.as_ptr();
        let chunks: usize = len / 8;

        for i in 0..chunks {
            // SAFETY: i * 8 + 8 <= chunks * 8 <= len, within bounds.
            // Unaligned reads are safe on all modern x86/ARM.
            let a_word: u64 = unsafe { a_ptr.add(i * 8).cast::<u64>().read_unaligned() };
            let b_word: u64 = unsafe { b_ptr.add(i * 8).cast::<u64>().read_unaligned() };

            if a_word != b_word {
                return false;
            }
        }

        // Compare remainder (0-7 bytes)
        let remainder_start: usize = chunks * 8;
        // SAFETY: remainder_start = chunks * 8 <= len (both slices same length)
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

        // Word-aligned path for longer slices
        let a_ptr: *const u8 = a.as_ptr();
        let b_ptr: *const u8 = b.as_ptr();
        let chunks: usize = min_len / 8;

        for i in 0..chunks {
            // SAFETY: i * 8 + 8 <= chunks * 8 <= min_len <= both lengths
            let a_word: u64 = unsafe { a_ptr.add(i * 8).cast::<u64>().read_unaligned() };
            let b_word: u64 = unsafe { b_ptr.add(i * 8).cast::<u64>().read_unaligned() };

            if a_word != b_word {
                // Convert to big-endian for correct lexicographic order
                let a_be: u64 = a_word.to_be();
                let b_be: u64 = b_word.to_be();
                return a_be.cmp(&b_be);
            }
        }

        // Compare remainder, then by length
        let remainder_start: usize = chunks * 8;
        // SAFETY: remainder_start <= min_len <= both lengths
        #[expect(clippy::indexing_slicing, reason = "remainder_start <= min_len")]
        match a[remainder_start..min_len].cmp(&b[remainder_start..min_len]) {
            Ordering::Equal => a.len().cmp(&b.len()),
            other => other,
        }
    }
}
