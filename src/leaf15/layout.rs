//! Compile-time layout assertions for [`LeafNode15`].
//!
//! This module verifies that [`LeafNode15`] maintains its intended cache-line
//! layout at compile time. Any refactoring that changes field offsets will
//! cause a build failure with a clear error message.
//!
//! # Cache Line Strategy (Suffix Sidecar)
//!
//! ```text
//! CL 0  (0-63):     version (4B) + modstate (1B) + _pad0 (55B) + 4B implicit padding
//! CL 1  (64-127):   permutation (8B) + _pad1 (56B)
//! CL 2  (128-191):  ikey0[0..=7] (8 keys, 64B)
//! CL 3  (192-255):  ikey0[8..=14] (7 keys, 56B) + keylenx[0..=7] (8B)
//! CL 4  (256-319):  keylenx[8..=14] (7B) + 1B pad + leaf_values[0..=6] (7 ptrs, 56B)
//! CL 5  (320-383):  leaf_values[7..=14] (8 ptrs, 64B)
//! CL 6  (384-447):  suffix_sidecar (8B) + next (8B) + prev (8B) + parent (8B) + 32B tail pad
//! ```
//!
//! # Memory Savings
//!
//! - Before (with `inline_ksuf` + `external_ksuf`): 768 bytes (12 cache lines)
//! - After (with `suffix_sidecar`): 448 bytes (7 cache lines)
//! - Savings: 320 bytes (42% reduction)
//!
//! # Hot Path (get) Cache Lines
//!
//! - CL 0: version (OCC validation)
//! - CL 1: permutation (slot ordering)
//! - CL 2-3: ikey0 (key comparison)
//! - CL 4-5: `leaf_values` (on match)

#![expect(clippy::items_after_statements, reason = "Compile time checks")]

use std::mem as StdMem;

use super::{LeafNode15, WIDTH_15};
use crate::nodeversion::NodeVersion;
use crate::permuter::AtomicPermuter15;
use crate::suffix::SuffixSidecar;
use crate::LeafValue;

// ============================================================================
//  Size and Alignment Assertions
// ============================================================================

/// Verify [`LeafNode15`] size and alignment.
///
/// Note: These assertions assume `target_pointer_width = 64`.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8};

    // LeafNode15 should be exactly 448 bytes (7 cache lines) with sidecar design
    assert!(StdMem::size_of::<LeafNode15<LeafValue<u64>>>() == 448);

    // Alignment should be 64 bytes (cache line)
    assert!(StdMem::align_of::<LeafNode15<LeafValue<u64>>>() == 64);

    // Component sizes
    assert!(StdMem::size_of::<NodeVersion>() == 4);
    assert!(StdMem::size_of::<AtomicPermuter15>() == 8);
    assert!(StdMem::size_of::<[AtomicU64; WIDTH_15]>() == 120);
    assert!(StdMem::size_of::<[AtomicU8; WIDTH_15]>() == 15);
    assert!(StdMem::size_of::<[AtomicPtr<u8>; WIDTH_15]>() == 120);
    assert!(StdMem::size_of::<AtomicPtr<SuffixSidecar<WIDTH_15>>>() == 8);
};

// ============================================================================
//  Field Offset Assertions
// ============================================================================

/// Verify critical field offsets for cache line optimization.
///
/// These assertions ensure the memory layout matches our cache line strategy:
/// - CL 0 (0-63): version + modstate isolated from hot fields
/// - CL 1 (64-127): permutation isolated (CAS-heavy)
/// - CL 2 (128-191): ikey0 starts at cache line boundary
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::offset_of;

    // Cache Line 0: Version and metadata (ends at offset 60, then 4B implicit padding)
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, version) == 0);
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, modstate) == 4);
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, _pad0) == 5);
    // _pad0 is 55 bytes, ending at offset 60. Permutation starts at 64 (4B implicit padding).

    // Cache Line 1: Permutation (isolated for CAS performance)
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, permutation) == 64);
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, _pad1) == 72);

    // Cache Line 2+: Keys start at offset 128 (cache line aligned)
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, ikey0) == 128);

    // keylenx follows ikey0: 128 + 120 = 248
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, keylenx) == 248);

    // leaf_values follows keylenx: 248 + 15 + 1 (padding) = 264
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, leaf_values) == 264);

    // suffix_sidecar follows leaf_values: 264 + 120 = 384 (cache line 6)
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, suffix_sidecar) == 384);

    // Linking pointers in cache line 6
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, next) == 392);
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, prev) == 400);
    assert!(offset_of!(LeafNode15<LeafValue<u64>>, parent) == 408);
};

// ============================================================================
//  Cache Line Boundary Assertions
// ============================================================================

/// Verify cache line boundaries are respected.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::offset_of;

    // Permutation must start exactly at cache line 1 (offset 64)
    const PERM_OFFSET: usize = offset_of!(LeafNode15<LeafValue<u64>>, permutation);
    assert!(PERM_OFFSET == 64);

    // ikey0 must start exactly at cache line 2 (offset 128)
    const IKEY_OFFSET: usize = offset_of!(LeafNode15<LeafValue<u64>>, ikey0);
    assert!(IKEY_OFFSET == 128);

    // First 8 ikeys (indices 0..=7) must fit in cache line 2 (offsets 128-191)
    const IKEY8_END: usize = IKEY_OFFSET + 8 * 8; // 8 keys * 8 bytes
    assert!(IKEY8_END == 192);
};
