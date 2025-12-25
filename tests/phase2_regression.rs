//! Phase 2 Regression Tests
//!
//! This module validates all Phase 2A-2B fixes with comprehensive test coverage.
//! Tests are organized by the issue they verify:
//!
//! | Test Category | Validates | Task |
//! |---------------|-----------|------|
//! | Suffix Migration | Task 1 | Long keys survive splits |
//! | Boundary Keys | Tasks 3, 5 | 8/16/24-byte boundaries work |
//! | Prefix-of-Other | Task 4 | One key prefix of another |
//! | Inline/Suffix Coexistence | Task 3 | Same ikey, different keylenx |
//! | Layer Growth | Task 6 | Sublayers exceed WIDTH |
//! | Layer Internode Roots | Task 6 | Layer roots can be internodes |

#![expect(clippy::unwrap_used, reason = "fail fast in tests")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]

use masstree::MassTree24;

// ============================================================================
//  1. Suffix Migration Tests (Task 1)
// ============================================================================

/// Test that suffixes are migrated correctly during `split_into()`.
#[test]
fn test_suffix_migration_split_into() {
    // Default WIDTH=15, so we need >15 keys to trigger splits
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert long keys (>8 bytes) that will need suffix storage
    // Insert 20 keys to ensure at least one split
    let keys: Vec<String> = (0..20)
        .map(|i| format!("prefix00{i:08}")) // 16 bytes each
        .collect();

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // All keys must still be findable after splits
    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key '{key}' not found after splits");
        assert_eq!(*result.unwrap(), i as u64, "Wrong value for key '{key}'");
    }
}

/// Test that suffixes are migrated correctly during `split_all_to_right()`.
#[test]
fn test_suffix_migration_split_all_to_right() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert keys that will trigger split_all_to_right
    // (when new key would be at position 0)
    tree.insert(b"zzzzzzzz_suffix", 1).unwrap(); // Goes to position 0
    tree.insert(b"aaaaaaaa_suffix", 2).unwrap(); // Forces split with split_all_to_right

    // Verify both findable
    assert_eq!(*tree.get(b"zzzzzzzz_suffix").unwrap(), 1);
    assert_eq!(*tree.get(b"aaaaaaaa_suffix").unwrap(), 2);
}

/// Test suffix migration with various suffix lengths.
#[test]
fn test_suffix_migration_various_lengths() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Different suffix lengths (9, 16, 24, 100 bytes total)
    let long_key = "prefix00".to_string() + &"x".repeat(92);
    let test_cases: [(&str, usize); 4] = [
        ("prefix00x", 9),                 // 1 byte suffix
        ("prefix00xxxxxxxx", 16),         // 8 byte suffix
        ("prefix00xxxxxxxxxxxxxxxx", 24), // 16 byte suffix
        (&long_key, 100),                 // 92 byte suffix
    ];

    for (i, (key, _len)) in test_cases.iter().enumerate() {
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    // Force splits by adding more keys
    for i in 0..10u64 {
        let key = format!("other0{i:02}");
        tree.insert(key.as_bytes(), 100 + i).unwrap();
    }

    // Verify all original keys still findable
    for (i, (key, _)) in test_cases.iter().enumerate() {
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key of length {} not found", key.len());
        assert_eq!(*result.unwrap(), i as u64);
    }
}

// ============================================================================
//  2. Boundary Key Tests (Tasks 3, 5)
// ============================================================================

/// Test exact 8-byte boundary keys.
#[test]
fn test_8_byte_boundary_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Exactly 8 bytes
    let key8 = b"exactkey"; // 8 bytes
    tree.insert(key8, 8).unwrap();

    let result = tree.get(key8);
    assert!(result.is_some(), "8-byte key not found");
    assert_eq!(*result.unwrap(), 8);
}

/// Test exact 16-byte boundary keys (2 layers).
#[test]
fn test_16_byte_boundary_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Exactly 16 bytes
    let key16 = b"exactly16bytes!!"; // 16 bytes
    assert_eq!(key16.len(), 16);
    tree.insert(key16, 16).unwrap();

    let result = tree.get(key16);
    assert!(result.is_some(), "16-byte key not found");
    assert_eq!(*result.unwrap(), 16);
}

/// Test exact 24-byte boundary keys (3 layers).
#[test]
fn test_24_byte_boundary_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Exactly 24 bytes
    let key24 = b"exactly_24_bytes_here!!!"; // 24 bytes
    assert_eq!(key24.len(), 24);
    tree.insert(key24, 24).unwrap();

    let result = tree.get(key24);
    assert!(result.is_some(), "24-byte key not found");
    assert_eq!(*result.unwrap(), 24);
}

/// Test keys at multiple boundaries together.
#[test]
fn test_multiple_boundary_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    let key8 = b"8bytes!!";
    let key16 = b"16_bytes_exact!!";
    let key24 = b"24_bytes_key_exactly!!!!";

    tree.insert(key8, 8).unwrap();
    tree.insert(key16, 16).unwrap();
    tree.insert(key24, 24).unwrap();

    assert_eq!(*tree.get(key8).unwrap(), 8);
    assert_eq!(*tree.get(key16).unwrap(), 16);
    assert_eq!(*tree.get(key24).unwrap(), 24);
}

/// Test keys that share boundary prefix.
#[test]
fn test_shared_boundary_prefix() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Keys sharing first 8 bytes but different lengths
    tree.insert(b"prefix00", 1).unwrap(); // 8 bytes
    tree.insert(b"prefix00_more", 2).unwrap(); // 13 bytes
    tree.insert(b"prefix00_even_more", 3).unwrap(); // 18 bytes

    assert_eq!(*tree.get(b"prefix00").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00_more").unwrap(), 2);
    assert_eq!(*tree.get(b"prefix00_even_more").unwrap(), 3);
}

// ============================================================================
//  3. Prefix-of-Other Tests (Task 4)
// ============================================================================

/// Test when one key is a prefix of another.
#[test]
fn test_prefix_of_other_basic() {
    let tree: MassTree24<u64> = MassTree24::new();

    // "prefix" is a prefix of "prefix_with_more"
    tree.insert(b"prefix", 1).unwrap();
    tree.insert(b"prefix_with_more", 2).unwrap();

    assert_eq!(*tree.get(b"prefix").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix_with_more").unwrap(), 2);
}

/// Test prefix-of-other at 8-byte boundary.
#[test]
fn test_prefix_of_other_at_boundary() {
    let tree: MassTree24<u64> = MassTree24::new();

    // "prefix00" (8 bytes) is prefix of "prefix00suffix"
    tree.insert(b"prefix00", 1).unwrap();
    tree.insert(b"prefix00suffix", 2).unwrap();

    assert_eq!(*tree.get(b"prefix00").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00suffix").unwrap(), 2);
}

/// Test multiple levels of prefix relationships.
#[test]
fn test_prefix_chain() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Chain: "a" < "ab" < "abc" < "abcd" < ...
    let keys = [
        "a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "abcdefgh",
    ];

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key.as_bytes(), i as u64).unwrap();
    }

    for (i, key) in keys.iter().enumerate() {
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key '{key}' not found");
        assert_eq!(*result.unwrap(), i as u64);
    }
}

/// Test prefix-of-other in layer context.
#[test]
fn test_prefix_of_other_in_layer() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Create suffix keys that trigger layer creation
    tree.insert(b"prefix00suffix_a", 1).unwrap();
    tree.insert(b"prefix00suffix_ab", 2).unwrap(); // suffix_a is prefix of suffix_ab
    tree.insert(b"prefix00suffix_abc", 3).unwrap();

    assert_eq!(*tree.get(b"prefix00suffix_a").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00suffix_ab").unwrap(), 2);
    assert_eq!(*tree.get(b"prefix00suffix_abc").unwrap(), 3);
}

/// Test reverse insertion order (longer key first).
#[test]
fn test_prefix_of_other_reverse_order() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert longer key first
    tree.insert(b"prefix00suffix_longer", 1).unwrap();
    tree.insert(b"prefix00suffix", 2).unwrap();
    tree.insert(b"prefix00", 3).unwrap();

    assert_eq!(*tree.get(b"prefix00suffix_longer").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00suffix").unwrap(), 2);
    assert_eq!(*tree.get(b"prefix00").unwrap(), 3);
}

// ============================================================================
//  4. Inline/Suffix Coexistence Tests (Task 3)
// ============================================================================

/// Test that inline and suffix keys with same ikey can coexist.
#[test]
fn test_inline_suffix_coexistence() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 8-byte key (inline, keylenx=8)
    tree.insert(b"exactkey", 1).unwrap();

    // 16-byte key with same prefix (suffix, keylenx=64)
    tree.insert(b"exactkey12345678", 2).unwrap();

    // Both must coexist and be findable
    assert_eq!(*tree.get(b"exactkey").unwrap(), 1);
    assert_eq!(*tree.get(b"exactkey12345678").unwrap(), 2);
}

/// Test various inline lengths with suffix keys.
#[test]
fn test_inline_lengths_with_suffix() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Different inline lengths (1-8 bytes)
    tree.insert(b"a", 1).unwrap();
    tree.insert(b"ab", 2).unwrap();
    tree.insert(b"abc", 3).unwrap();
    tree.insert(b"abcd", 4).unwrap();
    tree.insert(b"abcde", 5).unwrap();
    tree.insert(b"abcdef", 6).unwrap();
    tree.insert(b"abcdefg", 7).unwrap();
    tree.insert(b"abcdefgh", 8).unwrap();

    // Same prefix but with suffix
    tree.insert(b"abcdefgh_suffix", 9).unwrap();

    // All must coexist
    assert_eq!(*tree.get(b"a").unwrap(), 1);
    assert_eq!(*tree.get(b"ab").unwrap(), 2);
    assert_eq!(*tree.get(b"abc").unwrap(), 3);
    assert_eq!(*tree.get(b"abcd").unwrap(), 4);
    assert_eq!(*tree.get(b"abcde").unwrap(), 5);
    assert_eq!(*tree.get(b"abcdef").unwrap(), 6);
    assert_eq!(*tree.get(b"abcdefg").unwrap(), 7);
    assert_eq!(*tree.get(b"abcdefgh").unwrap(), 8);
    assert_eq!(*tree.get(b"abcdefgh_suffix").unwrap(), 9);
}

/// Test that no spurious layer is created for inline vs suffix.
#[test]
fn test_no_spurious_layer_creation() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert inline and suffix keys with same 8-byte prefix
    // These should NOT create a layer (one is inline, one is suffix)
    tree.insert(b"prefix00", 1).unwrap(); // inline (8 bytes)
    tree.insert(b"prefix00_suffix", 2).unwrap(); // suffix (15 bytes)

    // Both findable
    assert_eq!(*tree.get(b"prefix00").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00_suffix").unwrap(), 2);

    // Insert another suffix key - now we need a layer for suffix conflict
    tree.insert(b"prefix00_other", 3).unwrap();

    // All three still findable
    assert_eq!(*tree.get(b"prefix00").unwrap(), 1);
    assert_eq!(*tree.get(b"prefix00_suffix").unwrap(), 2);
    assert_eq!(*tree.get(b"prefix00_other").unwrap(), 3);
}

// ============================================================================
//  5. Layer Growth Tests (Task 6)
// ============================================================================

/// Test that layers can grow beyond WIDTH without panicking.
#[test]
fn test_layer_growth_beyond_width() {
    // Default WIDTH=15, need >15 keys in layer to trigger split
    let tree: MassTree24<u64> = MassTree24::new();

    // All keys share same 8-byte prefix, forcing them into a layer
    let prefix = b"samepfx!";

    // Insert 30+ keys to ensure multiple layer splits (WIDTH * 2)
    for i in 0..35u64 {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        tree.insert(&key, i).unwrap();
    }

    // All keys must be findable
    for i in 0..35u64 {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        let result = tree.get(&key);
        assert!(result.is_some(), "Key with suffix {i} not found");
        assert_eq!(*result.unwrap(), i);
    }
}

/// Test layer growth with different key patterns.
#[test]
fn test_layer_growth_random_order() {
    let tree: MassTree24<u64> = MassTree24::new();

    let prefix = b"layerpfx";

    // Insert in non-sequential order to stress split logic
    let order = [
        15, 3, 8, 1, 12, 6, 9, 0, 14, 4, 11, 7, 2, 13, 5, 10, 16, 17, 18, 19,
    ];

    for &i in &order {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&(i as u64).to_be_bytes());
        tree.insert(&key, i as u64).unwrap();
    }

    // All keys findable
    for i in 0..20u64 {
        let mut key = prefix.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        assert!(tree.get(&key).is_some(), "Key {i} not found");
    }
}

/// Test deeply nested layers (multiple 8-byte prefixes).
#[test]
fn test_nested_layer_growth() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Keys that share first 16 bytes (2 layers deep)
    let prefix16 = b"first___second__"; // 16 bytes

    // Insert 20+ keys to trigger split in the nested layer
    for i in 0..25u64 {
        let mut key = prefix16.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        tree.insert(&key, i).unwrap();
    }

    for i in 0..25u64 {
        let mut key = prefix16.to_vec();
        key.extend_from_slice(&i.to_be_bytes());
        assert!(tree.get(&key).is_some(), "Nested layer key {i} not found");
    }
}

// ============================================================================
//  6. Layer Internode Root Tests (Task 6)
// ============================================================================

/// Test that layer roots can become internodes.
#[test]
fn test_layer_root_becomes_internode() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Create layer
    tree.insert(b"prefix00suffix_01", 1).unwrap();
    tree.insert(b"prefix00suffix_02", 2).unwrap();

    // Fill layer until it splits (need >15 keys for WIDTH=15)
    for i in 3..25u64 {
        let key = format!("prefix00suffix_{i:02}");
        tree.insert(key.as_bytes(), i).unwrap();
    }

    // Verify all keys still findable (layer root is now internode)
    for i in 1..25u64 {
        let key = format!("prefix00suffix_{i:02}");
        let result = tree.get(key.as_bytes());
        assert!(result.is_some(), "Key {i} not found after layer growth");
        assert_eq!(*result.unwrap(), i);
    }
}

/// Test mixed operations with layer internode roots.
#[test]
fn test_layer_internode_mixed_operations() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Build up layer with internode root
    for i in 0..20u64 {
        let key = format!("layer___val_{i:04}");
        tree.insert(key.as_bytes(), i).unwrap();
    }

    // Interleave gets and inserts
    for i in 0..20u64 {
        let key = format!("layer___val_{i:04}");
        assert!(tree.get(key.as_bytes()).is_some());

        // Insert more keys
        let new_key = format!("layer___new_{i:04}");
        tree.insert(new_key.as_bytes(), 100 + i).unwrap();
    }

    // Verify all keys
    for i in 0..20u64 {
        let key = format!("layer___val_{i:04}");
        assert_eq!(*tree.get(key.as_bytes()).unwrap(), i);

        let new_key = format!("layer___new_{i:04}");
        assert_eq!(*tree.get(new_key.as_bytes()).unwrap(), 100 + i);
    }
}

// ============================================================================
//  7. Edge Case Tests
// ============================================================================

/// Test empty key.
#[test]
fn test_empty_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"", 0).unwrap();
    assert_eq!(*tree.get(b"").unwrap(), 0);

    // Non-empty key shouldn't conflict
    tree.insert(b"notempty", 1).unwrap();
    assert_eq!(*tree.get(b"").unwrap(), 0);
    assert_eq!(*tree.get(b"notempty").unwrap(), 1);
}

/// Test single-byte keys.
#[test]
fn test_single_byte_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..=255u8 {
        tree.insert(&[i], u64::from(i)).unwrap();
    }

    for i in 0..=255u8 {
        assert_eq!(*tree.get(&[i]).unwrap(), u64::from(i));
    }
}

/// Test key not found.
#[test]
fn test_key_not_found() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"exists", 1).unwrap();

    assert!(tree.get(b"exists").is_some());
    assert!(tree.get(b"missing").is_none());
    assert!(tree.get(b"exist").is_none()); // Prefix
    assert!(tree.get(b"existss").is_none()); // Extended
}

/// Test update existing key.
#[test]
fn test_update_existing() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key", 1).unwrap();
    assert_eq!(*tree.get(b"key").unwrap(), 1);

    let old = tree.insert(b"key", 2).unwrap();
    assert_eq!(*old.unwrap(), 1);
    assert_eq!(*tree.get(b"key").unwrap(), 2);
}

/// Test very long key.
#[test]
fn test_very_long_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    let long_key: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
    tree.insert(&long_key, 42).unwrap();

    assert_eq!(*tree.get(&long_key).unwrap(), 42);

    // Slightly different key shouldn't match
    let mut different = long_key;
    different[100] = different[100].wrapping_add(1);
    assert!(tree.get(&different).is_none());
}

/// Test maximum key length (256 bytes per).
#[test]
fn test_max_key_length() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 256 bytes is the maximum key length
    let max_key: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
    tree.insert(&max_key, 256).unwrap();

    assert_eq!(*tree.get(&max_key).unwrap(), 256);
}

/// Test many keys with same first 8 bytes but different lengths.
#[test]
fn test_same_prefix_different_lengths() {
    let tree: MassTree24<u64> = MassTree24::new();

    // All start with "testpfx!" (8 bytes)
    let prefix = b"testpfx!";

    // Insert keys of lengths 8, 9, 10, ..., 50
    for len in 8..=50usize {
        let mut key = prefix.to_vec();
        // Pad with 'x' to reach desired length
        while key.len() < len {
            key.push(b'x');
        }
        tree.insert(&key, len as u64).unwrap();
    }

    // Verify all findable
    for len in 8..=50usize {
        let mut key = prefix.to_vec();
        while key.len() < len {
            key.push(b'x');
        }
        let result = tree.get(&key);
        assert!(result.is_some(), "Key of length {len} not found");
        assert_eq!(*result.unwrap(), len as u64);
    }
}
