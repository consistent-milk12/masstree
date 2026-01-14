//! Regression tests for split-related suffix handling bugs.
//!
//! These tests specifically cover:
//! - `BugHunt2`: Multi-layer key suffix corruption during split
//! - Suffix overflow from inline to external storage during split initialization
//! - Various edge cases for suffix handling across splits

#![allow(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use masstree::MassTree;

/// Regression test for `BugHunt2`: Multi-layer key suffix corruption during split.
///
/// This bug was caused by the `assign_ksuf` function not handling the case where
/// the permutation isn't set up yet during split initialization. When inline suffix
/// storage overflows to external storage, the code would iterate over an empty
/// permutation (size 0) and clear all inline suffixes without copying them.
///
/// The fix adds an `initializing` parameter to `assign_ksuf` that, when true,
/// iterates over slots 0..N directly instead of using the permutation.
///
/// Reference: C++ `masstree_struct.hh:736` `int n = initializing ? p : perm.size();`
#[test]
fn test_bughunt2_multilayer_keys_after_split() {
    // These are the exact keys from the bug reproduction
    let keys: &[&[u8]] = &[
        b"\x12\x5e\x0e\xc6\xb6\x0f\xdc\x35\x45\x7c\xe2\xf2\x63\x5a\x81\x6a\xa9\xc7\xb3\xf5\xf9\x0a\x58\xde\xbe\x80\x40\xfe\xbd\x85\x3f\x5c\x76\xa4\xe9\x6b\x8c\x72\x2b\x55\x6d\x26\x13\x32\x21\x35",
        b"\x3f\x3c\x1e\xe6\xa9\x36\x61\x27\x1a\xb2\x8b\xaa\x59\xd5\x9b\xec\x21\xec\x84\x24\x0f\xcc\x80\xa9\xdb\x5e\x2d\x3e\x42\x6f\x77\x85\xca\x74\x96\x90\xe0\x18\x78\x03\x06\x77\x63\x58\x48\x84\xab\x05\x31\xc7\x6d\x4e",
        b"\xde\xbe\xbd\x70\x93\xfc\x13\x3e\x7e\xec\xb9\x01\x7f\x61\xf1\x8f\x1f\xdc\xcc\xa4\xe9\xce\xb7\xd1\xbf\x75\x04\x4f\x9b\x91\x45\x37\xa4\xb5\x2f\x13\x1e\x16\xab\x43\x30\x73\xe7\x52\x46\x5e",
        b"\xab\x1d\x25\x1c\x32\xad\x10\xfc\x3e\x2d\x1e\x01\x0e\xea\x29\x6a\xb9\x2e\x38\x65\xec\x2b\x26\xd0\x08\xd5",
        b"\xe4\xaf\x38\xc9\x9e\x09\x43\x3b\xef\x97\x19\x2f\x7e\x37\xb0\x02\x19\x63\xb9\x27\x41\x2b\xd5\x49\x61\x3e\xdd\xb9\x96\x3a\x52\xda\xd6\x2e\x18\x41\x44\x11\x7a\x87\xbd\x78\x1b\x6b\x8a\x5f\xf9\x40\xcd\xc7\xbb\x5f\xf9\x2e\x2e\x55\x7e\x9a\xce\xca\xbe\x60\x0e\x9f",
        b"\x8f\x9b\x29\xa8\x6a\xcd\x90\xc6\xd3\x1e\xe8\x1b\x7f\x0f",
        b"\x13\x2b\x2d\x15\xd9\x5b\x75\x8a\x2e\x32",
        b"\x97\x8d\xb3\xc9\x0e\x29\x94\x0f\xf7\xa0",
        b"\xe8\xfa\x15\x29\x02\x18\xaf\x8b\xfb\x05",
        b"\xd0\x22\xcf\xb5\xa8\x85\x37\xac\x39\x9c\x65\x37\x21\x83\x61\xef\x44\xe9\xba\x7f\xb9\x68\xf9\xa9\x8b\x85",
        b"\x33\x3a\xfb\x7f\x3b\x7b\x97\x07\x20\x80\xbe\x84",
        b"\x91\x05\x1a\x47\x4c\x42\xb5\xd9\x15\x8e\x30\xe7\x51\x53\x54\xb9\xe7\x31\xcb\x48\x38\x90\xca\x3c\x47\xa9\x24\x94\x11\xac\x0f\x7b\x08\x6b\x1f\x7c\xeb\x57\x86\x0c\x99\xa2\xb7\x58\x09\x98\xa9\x58\xa0\xdb\x47\x6f\x45\xd7\x11\x62\x4a\x4f\x70",
        b"\x31\x57\x72\x56\xc6\x31\xea\xe6\xf9\x07\x45\x80\xd0\xe3\x94\xfc\x83\x18\xee\xe9",
        b"\x24\x2f\x82\xc5\x5b\xcd\x6a\x5e\x2d\x04\x16\x3e",
        b"\xd9\x8f\x04\x1e\x80\x63\x72\x99\x23\x22\x4f\x0b\x45\x12\x7c\xb9\x50\xdd\xc6\x21\x3f\x92\x36\xb9\xd1",
        b"\x60\xae\xcc\x3d\x49\xd9\x95\x86",
        b"\x79\xfa\x75\x0e\xfe\x31\x35\x09\x49\xf7\x01\xe9\xd5\xc7\x3e\x1a\x66\x7f\xbe\xd0\x61\xaa\x34\x3a\x0d\x3b\x36\x22\x78\xd6\x4a\x1c\x88\x27\x52\xad\x53\x7b\x1e\x14\xe5\x6a\xbb\xfa\x9e\x3b",
        b"\xc3\x53\x10\x67\xf3\x10\xc0\xaf\x35\xed\xc9\x60\x72\x1f\x89\xc6\xab\xfb\x07\x77\x86",
        b"\xee\x8d\xc5\x9a\xb3\x52\x7c\xcf\x87\xb3\xf7\x1c\x05\x5e\x86\x54\xf5\x4a\xa1\x34\x4e\xab\x4b\x3b\x43\xca\x1a\x4d\xdf\xc0\x1e\xb2\xc6\xc7\xbf\x12\x60\xf9\x03\xa5\x0d\x4e\x42\x78\xf6\xf7\xc4\xea\xd9\x18\xc3\xca\x8c\xe5\x0d\x2c",
        b"\xfb\xb6\x9d\xd7\x73\xb8\x7d\x3c\x6c\x1f\xb9\xa5\xa1\x96\xf9\x94\x24\xbf\xef\x15\x85\x07\x5a\x98\x3e\x4b\xc6\xb5\xc7\x07\x97\x69\xf7\x64\x44\xaa\xf3\x8b\xef\x44\xef\x1e\x35\x3a\x2f\x3a\xb9\x97\x57\xe9\x1b\xc9",
        b"\xe1\x68\xd5\x99\x99\xa0\x59\x56\x01\x55\x1a\x24\xe9\x94\xad\x5e\xaa\x9e\x24\x81\x4f\x8f\xfc\xb7\xd3\x39\xe9\xa9\x58\x72\xaf\xa7\xd5\xfe\xf9\xe7\x52\x9e\x58\xf2\xa0\x1e\x1c\x30\x19",
        b"\xd3\x9f\xa9\xc1\xda\x0e\x1b\x12\x55\xd8\x7b\xf5\xd4\xba\x04\xfb\xf9\xf9\x73\x2d\xc9\x05\xe7\xb5\xae\x75\x8f\x28\xe3\xb9\x82\x79\x5c\xb8\xd5\x73\x09\x49\x37\xb9\x95\x8f\x5a\xdf\xf2\xf3\xf5\xa5\x15\x8e\x2d\x18\xe9\x33\x10\xb9\x3e",
        b"\xcc\xd9\x49\x91\x24\x4e\xb5\xdf\x09\x50\x41\x26\xcc\x4f\x6a\x43\x1e\x1d\xca\x90\xaa\xd7\xaa\xe7\x17\x85\x3f\x75\xf0\x29\x76\x24\x30\x5f\x72\x14\x4c\x0d\x32\xb2\x80\x96\x7b\x62\x67\x97\x1c\xee\xeb\xaa\x5f\x0a\x00\xa8\xd9\x56\xbb\xc0\x5b\x5c\xf4\x3b\x8f\x60",
    ];

    let tree: MassTree<usize> = MassTree::new();

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key);
        assert!(
            result.is_some(),
            "Key {} (len={}) not found after split",
            i,
            key.len()
        );
        assert_eq!(
            *result.unwrap(),
            i,
            "Key {i} returned wrong value after split"
        );
    }

    // Verify via iteration
    let guard = tree.guard();
    let mut count = 0;
    for entry in tree.iter(&guard) {
        count += 1;
        let value = *entry.value;
        assert!(
            value < keys.len(),
            "Iteration returned invalid value {value}"
        );
    }
    assert_eq!(
        count,
        keys.len(),
        "Iteration count mismatch: expected {}, got {}",
        keys.len(),
        count
    );
}

/// Test that suffix keys with varying lengths are preserved across splits.
///
/// This tests the edge case where suffix lengths vary significantly,
/// potentially causing inline storage to overflow at different points.
#[test]
fn test_varying_suffix_lengths_across_split() {
    let tree: MassTree<usize> = MassTree::new();

    // Create keys with the same 8-byte prefix but different suffix lengths
    let prefix = b"AAAAAAAA";
    let mut keys: Vec<Vec<u8>> = Vec::new();

    // Create 30 keys with varying suffix lengths (will trigger split)
    for i in 0..30 {
        let mut key = prefix.to_vec();
        // Suffix length varies from 1 to 60 bytes
        let suffix_len = ((i * 7) % 60) + 1;
        for j in 0..suffix_len {
            key.push((i as u8).wrapping_add(j as u8));
        }
        keys.push(key);
    }

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key);
        assert!(
            result.is_some(),
            "Key {} (suffix_len={}) not found after split",
            i,
            key.len() - 8
        );
        assert_eq!(*result.unwrap(), i, "Key {i} returned wrong value");
    }
}

/// Test that inline suffix storage overflow during split is handled correctly.
///
/// This specifically tests the case where the cumulative suffix bytes exceed
/// the inline storage capacity (256 bytes), forcing a transition to external
/// storage during split initialization.
#[test]
fn test_inline_suffix_overflow_during_split() {
    let tree: MassTree<usize> = MassTree::new();

    // Create keys with different prefixes but large suffixes
    // Each suffix is about 40 bytes, so 8 keys = 320 bytes > 256 inline capacity
    let mut keys: Vec<Vec<u8>> = Vec::new();

    for i in 0u64..30 {
        let mut key = Vec::new();
        // 8-byte prefix (unique per key to avoid layer descent)
        key.extend_from_slice(&i.to_be_bytes());
        // ~40 byte suffix to ensure inline overflow
        for j in 0..40 {
            key.push((i as u8).wrapping_mul(7).wrapping_add(j));
        }
        keys.push(key);
    }

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key);
        assert!(
            result.is_some(),
            "Key {} (len={}) not found - possible suffix corruption during split",
            i,
            key.len()
        );
        assert_eq!(*result.unwrap(), i, "Key {i} returned wrong value");
    }

    // Additional check: verify tree length
    assert_eq!(
        tree.len(),
        keys.len(),
        "Tree length mismatch: expected {}, got {}",
        keys.len(),
        tree.len()
    );
}

/// Test that split correctly handles a mix of inline and suffix keys.
///
/// Some keys are short (inline), some have suffixes. The split should
/// preserve both types correctly.
#[test]
fn test_mixed_inline_and_suffix_keys_across_split() {
    let tree: MassTree<usize> = MassTree::new();

    let mut keys: Vec<Vec<u8>> = Vec::new();

    for i in 0u64..30 {
        let mut key = Vec::new();
        key.extend_from_slice(&i.to_be_bytes());

        // Alternate between inline keys (no suffix) and suffix keys
        if i % 2 == 0 {
            // Inline key: just the 8-byte ikey
        } else {
            // Suffix key: add 10-30 bytes of suffix
            let suffix_len = 10 + (i as usize % 20);
            for j in 0..suffix_len {
                key.push((i as u8).wrapping_add(j as u8));
            }
        }
        keys.push(key);
    }

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key);
        let key_type = if key.len() <= 8 { "inline" } else { "suffix" };
        assert!(
            result.is_some(),
            "Key {} ({}, len={}) not found after split",
            i,
            key_type,
            key.len()
        );
        assert_eq!(*result.unwrap(), i, "Key {i} returned wrong value");
    }
}

/// Test that multiple consecutive splits preserve all suffix data.
///
/// This tests the case where the tree grows large enough to require
/// multiple splits, ensuring suffix handling is correct at each split.
#[test]
fn test_multiple_splits_preserve_suffixes() {
    let tree: MassTree<usize> = MassTree::new();

    // Insert 100 keys to trigger multiple splits (WIDTH_24 = 24)
    let mut keys: Vec<Vec<u8>> = Vec::new();

    for i in 0u64..100 {
        let mut key = Vec::new();
        key.extend_from_slice(&i.to_be_bytes());
        // Add varying suffix
        let suffix_len = 5 + (i as usize % 30);
        for j in 0..suffix_len {
            key.push((i as u8).wrapping_mul(3).wrapping_add(j as u8));
        }
        keys.push(key);
    }

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    let mut missing = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if tree.get(key).is_none() {
            missing.push(i);
        }
    }

    assert!(
        missing.is_empty(),
        "Missing {} keys after multiple splits: {:?}",
        missing.len(),
        missing
    );

    // Verify values
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(*tree.get(key).unwrap(), i, "Key {i} returned wrong value");
    }
}

/// Test that layer keys (keys with same 8-byte prefix creating sublayers)
/// are handled correctly across splits.
#[test]
fn test_layer_keys_across_split() {
    let tree: MassTree<usize> = MassTree::new();

    // Create keys that share the same 8-byte prefix (will create layers)
    let prefix = b"SHAREDPX"; // 8 bytes
    let mut keys: Vec<Vec<u8>> = Vec::new();

    for i in 0..30 {
        let mut key = prefix.to_vec();
        // Second 8 bytes vary
        key.extend_from_slice(&(i as u64).to_be_bytes());
        // Add suffix
        for j in 0..10 {
            key.push((i as u8).wrapping_add(j));
        }
        keys.push(key);
    }

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i);
    }

    // Verify all keys are retrievable
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key);
        assert!(
            result.is_some(),
            "Layer key {} (len={}) not found after split",
            i,
            key.len()
        );
        assert_eq!(*result.unwrap(), i, "Key {i} returned wrong value");
    }
}
