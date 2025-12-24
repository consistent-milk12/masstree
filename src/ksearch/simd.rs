//! SIMD-accelerated key comparison primitives.
//!
//! This module provides low-level SIMD operations for comparing multiple
//! 64-bit keys simultaneously. These primitives are used by the higher-level
//! search functions to accelerate key lookup.
//!
//! # Architecture Support
//!
//! - **`x86_64`** with SSE2: Compare 2 keys at a time (always available)
//! - **`x86_64`** with AVX2: Compare 4 keys at a time (runtime detection)
//! - **Other**: Falls back to scalar comparison
//!
//! # Design
//!
//! SIMD is most effective for:
//! - Linear scans for exact match (compare all keys in parallel)
//! - Small node upper bound (count keys ≤ target)
//!
//! Binary search cannot be parallelized with SIMD because each comparison
//! depends on the previous result. However, for small nodes (≤8 keys),
//! SIMD linear scan can be faster than binary search.

// ============================================================================
//  Core SIMD Primitives
// ============================================================================

/// Find the first index where `keys[i] == target`, or `None` if not found.
///
/// Uses SIMD when available to compare multiple keys simultaneously.
#[must_use]
#[inline(always)]
pub fn find_exact_u64(keys: &[u64], target: u64) -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return find_exact_u64_avx2(keys, target);
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        find_exact_u64_sse2(keys, target)
    }

    // Fallback for non-x86 architectures
    #[cfg(not(target_arch = "x86_64"))]
    {
        find_exact_u64_scalar(keys, target)
    }
}

/// Count how many keys are less than or equal to target.
///
/// This is useful for upper bound calculations: the count gives the
/// child index to follow in an internode.
#[must_use]
#[inline(always)]
pub fn count_le_u64(keys: &[u64], size: usize, target: u64) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return count_le_u64_avx2(keys, size, target);
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        count_le_u64_sse2(keys, size, target)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        count_le_u64_scalar(keys, size, target)
    }
}

// ============================================================================
//  Scalar Fallback (always available)
// ============================================================================

/// Scalar exact match search.
#[must_use]
#[inline(always)]
pub fn find_exact_u64_scalar(keys: &[u64], target: u64) -> Option<usize> {
    for (i, &key) in keys.iter().enumerate() {
        if key == target {
            return Some(i);
        }
    }
    None
}

/// Scalar count of keys ≤ target.
#[must_use]
#[inline(always)]
pub fn count_le_u64_scalar(keys: &[u64], size: usize, target: u64) -> usize {
    let mut count = 0;
    for key in keys.iter().take(size) {
        if *key <= target {
            count += 1;
        }
    }
    count
}

// ============================================================================
//  SSE2 Implementation (x86_64, always available)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(clippy::indexing_slicing)] // Bounds checked by loop condition
#[allow(clippy::cast_possible_wrap)] // SSE2 intrinsics use i64 for u64 comparison
mod sse2_impl {
    use std::arch::x86_64::{_mm_cmpeq_epi64, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi64x};

    /// SSE2 exact match: compare 2 keys at a time.
    #[inline]
    pub fn find_exact_u64_sse2(keys: &[u64], target: u64) -> Option<usize> {
        let len = keys.len();
        if len == 0 {
            return None;
        }

        // SAFETY: SSE2 is always available on x86_64
        unsafe {
            // Broadcast target to both lanes
            let target_vec = _mm_set1_epi64x(target as i64);
            let mut i = 0;

            // Process 2 keys at a time
            while i + 2 <= len {
                let keys_vec = _mm_loadu_si128(keys.as_ptr().add(i).cast());
                let cmp = _mm_cmpeq_epi64(keys_vec, target_vec);
                let mask = _mm_movemask_epi8(cmp);

                if mask != 0 {
                    // Found a match - determine which lane
                    // Lanes are 8 bytes each, so check bits 0-7 (first lane) vs 8-15 (second)
                    if mask & 0xFF != 0 {
                        return Some(i);
                    }
                    return Some(i + 1);
                }

                i += 2;
            }

            // Handle remainder
            while i < len {
                if keys[i] == target {
                    return Some(i);
                }
                i += 1;
            }

            None
        }
    }

    /// SSE2 count: uses scalar comparison.
    ///
    /// SSE2 doesn't have efficient 64-bit comparison, so we use scalar.
    /// SIMD benefits are mainly for exact match operations.
    #[inline]
    pub fn count_le_u64_sse2(keys: &[u64], size: usize, target: u64) -> usize {
        // SSE2 doesn't have efficient 64-bit LE comparison
        // Fall back to scalar which is already fast for small nodes
        super::count_le_u64_scalar(keys, size, target)
    }
}

#[cfg(target_arch = "x86_64")]
use sse2_impl::{count_le_u64_sse2, find_exact_u64_sse2};

// ============================================================================
//  AVX2 Implementation (x86_64, runtime detection)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    #[cfg(target_feature = "avx2")]
    use std::arch::x86_64::*;

    /// AVX2 exact match: compare 4 keys at a time.
    #[cfg(target_feature = "avx2")]
    #[inline]
    pub fn find_exact_u64_avx2(keys: &[u64], target: u64) -> Option<usize> {
        let len = keys.len();
        if len == 0 {
            return None;
        }

        // SAFETY: AVX2 is guaranteed by target_feature
        unsafe {
            let target_vec = _mm256_set1_epi64x(target as i64);
            let mut i = 0;

            // Process 4 keys at a time
            while i + 4 <= len {
                let keys_vec = _mm256_loadu_si256(keys.as_ptr().add(i).cast());
                let cmp = _mm256_cmpeq_epi64(keys_vec, target_vec);
                let mask = _mm256_movemask_epi8(cmp);

                if mask != 0 {
                    // Found a match - determine which lane (each lane is 8 bytes)
                    let lane_mask = mask as u32;
                    if lane_mask & 0x0000_00FF != 0 {
                        return Some(i);
                    }
                    if lane_mask & 0x0000_FF00 != 0 {
                        return Some(i + 1);
                    }
                    if lane_mask & 0x00FF_0000 != 0 {
                        return Some(i + 2);
                    }
                    return Some(i + 3);
                }

                i += 4;
            }

            // Handle remainder with SSE2 or scalar
            while i < len {
                if keys[i] == target {
                    return Some(i);
                }
                i += 1;
            }

            None
        }
    }

    /// AVX2 count: compare 4 keys at a time.
    #[cfg(target_feature = "avx2")]
    #[inline]
    pub fn count_le_u64_avx2(keys: &[u64], size: usize, target: u64) -> usize {
        if size == 0 {
            return 0;
        }

        // SAFETY: AVX2 is guaranteed by target_feature
        unsafe {
            let target_vec = _mm256_set1_epi64x(target as i64);
            let mut count = 0;
            let mut i = 0;

            // Process 4 keys at a time
            while i + 4 <= size {
                let keys_vec = _mm256_loadu_si256(keys.as_ptr().add(i).cast());

                // AVX2 has _mm256_cmpgt_epi64: gives 1s where key > target
                // We want key <= target, which is !(key > target)
                let gt = _mm256_cmpgt_epi64(keys_vec, target_vec);
                let mask = _mm256_movemask_epi8(gt);

                // Count lanes where key <= target (i.e., gt mask is 0)
                // Each lane is 8 bytes, so check each 8-bit group
                let lane_mask = mask as u32;
                if lane_mask & 0x0000_00FF == 0 {
                    count += 1;
                }
                if lane_mask & 0x0000_FF00 == 0 {
                    count += 1;
                }
                if lane_mask & 0x00FF_0000 == 0 {
                    count += 1;
                }
                if lane_mask & 0xFF00_0000 == 0 {
                    count += 1;
                }

                i += 4;
            }

            // Handle remainder
            while i < size {
                if keys[i] <= target {
                    count += 1;
                }
                i += 1;
            }

            count
        }
    }

    // Fallback when AVX2 is not enabled at compile time
    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    #[allow(dead_code)] // Used when AVX2 not available
    pub fn find_exact_u64_avx2(keys: &[u64], target: u64) -> Option<usize> {
        super::find_exact_u64_scalar(keys, target)
    }

    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    #[allow(dead_code)] // Used when AVX2 not available
    pub fn count_le_u64_avx2(keys: &[u64], size: usize, target: u64) -> usize {
        super::count_le_u64_scalar(keys, size, target)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)] // Used conditionally based on target_feature
use avx2_impl::{count_le_u64_avx2, find_exact_u64_avx2};

// ============================================================================
//  Bitmask Functions (for leaf search)
// ============================================================================

/// Find all indices where `keys[i] == target`, returning a bitmask.
///
/// Returns a u16 where bit `i` is set if `keys[i] == target`.
/// Only the first `len` keys are checked; bits beyond `len` are always 0.
///
/// This is useful for leaf search where we need to find all slots
/// with matching ikey, then filter by permutation membership.
#[inline]
#[must_use]
pub fn find_all_matches_u64(keys: &[u64], len: usize, target: u64) -> u16 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return find_all_matches_u64_avx2(keys, len, target);
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        find_all_matches_u64_sse2(keys, len, target)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        find_all_matches_u64_scalar(keys, len, target)
    }
}

/// Scalar implementation of `find_all_matches`.
#[inline]
#[must_use]
#[allow(dead_code)]
#[expect(clippy::indexing_slicing)]
pub fn find_all_matches_u64_scalar(keys: &[u64], len: usize, target: u64) -> u16 {
    let mut mask: u16 = 0;
    (0..len.min(16).min(keys.len())).for_each(|i| {
        if keys[i] == target {
            mask |= 1 << i;
        }
    });
    mask
}

#[cfg(target_arch = "x86_64")]
mod find_all_impl {
    use std::arch::x86_64::{_mm_cmpeq_epi64, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi64x};

    /// SSE2 implementation: compare 2 keys at a time, build bitmask.
    #[inline]
    #[expect(clippy::cast_possible_wrap, clippy::indexing_slicing)]
    pub fn find_all_matches_u64_sse2(keys: &[u64], len: usize, target: u64) -> u16 {
        if len == 0 {
            return 0;
        }

        let mut mask: u16 = 0;

        // SAFETY: SSE2 is always available on x86_64
        unsafe {
            let target_vec = _mm_set1_epi64x(target as i64);
            let mut i = 0;

            // Process 2 keys at a time
            while i + 2 <= len && i + 2 <= keys.len() {
                let keys_vec = _mm_loadu_si128(keys.as_ptr().add(i).cast());
                let cmp = _mm_cmpeq_epi64(keys_vec, target_vec);
                let byte_mask = _mm_movemask_epi8(cmp);

                // Each lane is 8 bytes
                if byte_mask & 0xFF != 0 {
                    mask |= 1 << i;
                }
                if byte_mask & 0xFF00 != 0 {
                    mask |= 1 << (i + 1);
                }

                i += 2;
            }

            // Handle remainder
            while i < len && i < keys.len() {
                if keys[i] == target {
                    mask |= 1 << i;
                }
                i += 1;
            }
        }

        mask
    }

    #[cfg(target_feature = "avx2")]
    use std::arch::x86_64::{
        _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi64x,
    };

    /// AVX2 implementation: compare 4 keys at a time, build bitmask.
    #[cfg(target_feature = "avx2")]
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub fn find_all_matches_u64_avx2(keys: &[u64], len: usize, target: u64) -> u16 {
        if len == 0 {
            return 0;
        }

        let mut mask: u16 = 0;

        // SAFETY: AVX2 is guaranteed by target_feature
        unsafe {
            let target_vec = _mm256_set1_epi64x(target as i64);
            let mut i = 0;

            // Process 4 keys at a time
            while i + 4 <= len && i + 4 <= keys.len() {
                let keys_vec = _mm256_loadu_si256(keys.as_ptr().add(i).cast());
                let cmp = _mm256_cmpeq_epi64(keys_vec, target_vec);
                let byte_mask = _mm256_movemask_epi8(cmp) as u32;

                // Each lane is 8 bytes
                if byte_mask & 0x0000_00FF != 0 {
                    mask |= 1 << i;
                }
                if byte_mask & 0x0000_FF00 != 0 {
                    mask |= 1 << (i + 1);
                }
                if byte_mask & 0x00FF_0000 != 0 {
                    mask |= 1 << (i + 2);
                }
                if byte_mask & 0xFF00_0000 != 0 {
                    mask |= 1 << (i + 3);
                }

                i += 4;
            }

            // Handle remainder with SSE2
            while i + 2 <= len && i + 2 <= keys.len() {
                let keys_vec = _mm_loadu_si128(keys.as_ptr().add(i).cast());
                let target_sse = _mm_set1_epi64x(target as i64);
                let cmp = _mm_cmpeq_epi64(keys_vec, target_sse);
                let byte_mask = _mm_movemask_epi8(cmp);

                if byte_mask & 0xFF != 0 {
                    mask |= 1 << i;
                }
                if byte_mask & 0xFF00 != 0 {
                    mask |= 1 << (i + 1);
                }

                i += 2;
            }

            // Handle final element
            while i < len && i < keys.len() {
                if keys[i] == target {
                    mask |= 1 << i;
                }
                i += 1;
            }
        }

        mask
    }

    // Fallback when AVX2 not enabled
    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    #[allow(dead_code)]
    pub fn find_all_matches_u64_avx2(keys: &[u64], len: usize, target: u64) -> u16 {
        super::find_all_matches_u64_scalar(keys, len, target)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use find_all_impl::{find_all_matches_u64_avx2, find_all_matches_u64_sse2};

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
#[expect(clippy::cast_sign_loss)]
mod tests {
    use super::*;

    // ========================================================================
    //  SIMD-dispatching tests (skip under Miri - no SIMD support)
    // ========================================================================

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact_empty() {
        let keys: [u64; 0] = [];
        assert_eq!(find_exact_u64(&keys, 42), None);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact_single_found() {
        let keys = [42u64];
        assert_eq!(find_exact_u64(&keys, 42), Some(0));
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact_single_not_found() {
        let keys = [42u64];
        assert_eq!(find_exact_u64(&keys, 43), None);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact_multiple() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(find_exact_u64(&keys, 10), Some(0));
        assert_eq!(find_exact_u64(&keys, 30), Some(2));
        assert_eq!(find_exact_u64(&keys, 50), Some(4));
        assert_eq!(find_exact_u64(&keys, 25), None);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_exact_alignment() {
        // Test with various sizes to exercise SIMD remainder handling
        for size in 1..=16 {
            let keys: Vec<u64> = (0..size).map(|i| i as u64 * 10).collect();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(find_exact_u64(&keys, key), Some(i), "size={size}, i={i}");
            }
            assert_eq!(find_exact_u64(&keys, 999), None, "size={size}");
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn test_count_le_empty() {
        let keys: [u64; 0] = [];
        assert_eq!(count_le_u64(&keys, 0, 42), 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_count_le_all() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(count_le_u64(&keys, 5, 100), 5);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_count_le_none() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(count_le_u64(&keys, 5, 5), 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_count_le_partial() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(count_le_u64(&keys, 5, 10), 1);
        assert_eq!(count_le_u64(&keys, 5, 15), 1);
        assert_eq!(count_le_u64(&keys, 5, 30), 3);
        assert_eq!(count_le_u64(&keys, 5, 35), 3);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_count_le_alignment() {
        // Test with various sizes
        for size in 1..=16 {
            let keys: Vec<u64> = (0..size).map(|i| (i as u64 + 1) * 10).collect();
            // Count keys <= 35 (should be 3 for size >= 3)
            let expected = size.min(3);
            assert_eq!(count_le_u64(&keys, size, 35), expected, "size={size}");
        }
    }

    // ========================================================================
    //  Scalar tests (safe under Miri)
    // ========================================================================

    #[test]
    fn test_scalar_find_exact() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(find_exact_u64_scalar(&keys, 30), Some(2));
        assert_eq!(find_exact_u64_scalar(&keys, 99), None);
    }

    #[test]
    fn test_scalar_count_le() {
        let keys = [10u64, 20, 30, 40, 50];
        assert_eq!(count_le_u64_scalar(&keys, 5, 25), 2);
    }

    // ========================================================================
    //  find_all_matches Tests (SIMD-dispatching, skip under Miri)
    // ========================================================================

    #[test]
    #[cfg(not(miri))]
    fn test_find_all_matches_empty() {
        let keys: [u64; 0] = [];
        assert_eq!(find_all_matches_u64(&keys, 0, 42), 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_all_matches_single() {
        let keys = [42u64];
        assert_eq!(find_all_matches_u64(&keys, 1, 42), 0b0001);
        assert_eq!(find_all_matches_u64(&keys, 1, 43), 0b0000);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_all_matches_multiple() {
        let keys = [10u64, 20, 30, 20, 50];
        // Find all 20s at indices 1 and 3
        assert_eq!(find_all_matches_u64(&keys, 5, 20), 0b01010);
        // Find 30 at index 2
        assert_eq!(find_all_matches_u64(&keys, 5, 30), 0b00100);
        // Find nothing
        assert_eq!(find_all_matches_u64(&keys, 5, 99), 0b00000);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_all_matches_all_same() {
        let keys = [42u64; 8];
        assert_eq!(find_all_matches_u64(&keys, 8, 42), 0b1111_1111);
        assert_eq!(find_all_matches_u64(&keys, 8, 43), 0b0000_0000);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_find_all_matches_various_sizes() {
        // Test with various sizes to exercise SIMD remainder handling
        for size in 1..=15 {
            let mut keys = [0u64; 16];
            (0..size).for_each(|i| {
                keys[i] = (i as u64) * 10;
            });
            // Add a duplicate at the end if possible
            if size > 1 {
                keys[size - 1] = 0; // Duplicate of keys[0]
            }

            // Find 0 - should match index 0 and possibly index size-1
            let mask = find_all_matches_u64(&keys, size, 0);
            assert!(mask & 1 != 0, "size={size}: index 0 should match");
            if size > 1 {
                let last_bit = 1u16 << (size - 1);
                assert!(mask & last_bit != 0, "size={size}: last index should match");
            }
        }
    }

    #[test]
    fn test_find_all_matches_scalar() {
        let keys = [10u64, 20, 30, 20, 50];
        assert_eq!(find_all_matches_u64_scalar(&keys, 5, 20), 0b01010);
        assert_eq!(find_all_matches_u64_scalar(&keys, 5, 99), 0b00000);
    }
}
