use crate::key::MAX_KEY_LENGTH;

use super::*;

#[test]
fn test_new_key() {
    let key: Key<'_> = Key::new(b"hello");

    assert_eq!(key.len(), 5);
    assert_eq!(key.shift_count(), 0);
    assert!(!key.is_shifted());
    assert!(!key.is_empty());
}

#[test]
fn test_empty_key() {
    let key: Key<'_> = Key::new(b"");

    assert!(key.is_empty());
    assert_eq!(key.ikey(), 0);
    assert_eq!(key.len(), 0);
}

#[test]
fn test_ikey_extraction() {
    let key: Key<'_> = Key::new(b"hello world!");
    let expected: u64 = u64::from_be_bytes(*b"hello wo");

    assert_eq!(key.ikey(), expected);
}

#[test]
fn test_short_key_padding() {
    let key: Key<'_> = Key::new(b"hi");
    let expected: u64 = u64::from_be_bytes([b'h', b'i', 0, 0, 0, 0, 0, 0]);

    assert_eq!(key.ikey(), expected);
}

#[test]
fn test_exact_8_bytes() {
    let key: Key<'_> = Key::new(b"12345678");

    assert_eq!(key.ikey(), u64::from_be_bytes(*b"12345678"));
    assert!(!key.has_suffix());
    assert_eq!(key.suffix_len(), 0);
}

#[test]
fn test_has_suffix() {
    let key: Key<'_> = Key::new(b"123456789"); // 9 bytes

    assert!(key.has_suffix());
    assert_eq!(key.suffix(), b"9");
    assert_eq!(key.suffix_len(), 1);
}

#[test]
fn test_shift() {
    let mut key: Key<'_> = Key::new(b"hello world!!!!!");
    assert_eq!(key.ikey(), u64::from_be_bytes(*b"hello wo"));
    assert_eq!(key.shift_count(), 0);

    key.shift();
    assert_eq!(key.ikey(), u64::from_be_bytes(*b"rld!!!!!"));
    assert_eq!(key.shift_count(), 1);
    assert!(key.is_shifted());
}

#[test]
fn test_shift_with_short_suffix() {
    let mut key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
    key.shift();

    // "rld!" padded with zeros
    let expected: u64 = u64::from_be_bytes([b'r', b'l', b'd', b'!', 0, 0, 0, 0]);
    assert_eq!(key.ikey(), expected);
}

#[test]
fn test_unshift() {
    let mut key: Key<'_> = Key::new(b"hello world!!!!!");
    let original_ikey: u64 = key.ikey();

    key.shift();
    assert_ne!(key.ikey(), original_ikey);

    key.unshift();
    assert_eq!(key.ikey(), original_ikey);
    assert_eq!(key.shift_count(), 0);
}

#[test]
fn test_unshift_all() {
    let mut key: Key<'_> = Key::new(b"hello world!!!!!!!!!!!!!!");
    let original_ikey: u64 = key.ikey();

    key.shift();
    key.shift();
    assert_eq!(key.shift_count(), 2);

    key.unshift_all();
    assert_eq!(key.ikey(), original_ikey);
    assert_eq!(key.shift_count(), 0);
}

#[test]
fn test_current_len() {
    let mut key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
    assert_eq!(key.current_len(), 12);

    key.shift();
    assert_eq!(key.current_len(), 4); // "rld!"
}

#[test]
fn test_compare_equal() {
    let key: Key<'_> = Key::new(b"hello");
    let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // Same ikey, same length -> Equal
    assert_eq!(key.compare(stored_ikey, 5), Ordering::Equal);
}

#[test]
fn test_compare_less_by_ikey() {
    let key: Key<'_> = Key::new(b"apple");
    let stored_ikey: u64 = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

    // ikey comparison: "apple" < "banana"
    assert_eq!(key.compare(stored_ikey, 6), Ordering::Less);
}

#[test]
fn test_compare_greater_by_ikey() {
    let key: Key<'_> = Key::new(b"zebra");
    let stored_ikey: u64 = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

    // ikey comparison: "zebra" > "apple"
    assert_eq!(key.compare(stored_ikey, 5), Ordering::Greater);
}

#[test]
fn test_compare_by_length() {
    // Same ikey, different lengths
    let key: Key<'_> = Key::new(b"hello");
    let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // Our key (5 bytes) vs stored key (3 bytes) -> Greater
    assert_eq!(key.compare(stored_ikey, 3), Ordering::Greater);

    // Our key (5 bytes) vs stored key (7 bytes) -> Less
    assert_eq!(key.compare(stored_ikey, 7), Ordering::Less);
}

#[test]
fn test_compare_with_suffix() {
    // Key with suffix (> 8 bytes)
    let key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
    let stored_ikey = u64::from_be_bytes(*b"hello wo");

    // Our key has suffix, stored key has no suffix (length 8) -> Greater
    assert_eq!(key.compare(stored_ikey, 8), Ordering::Greater);

    // Our key has suffix, stored key also has suffix (length > 8) -> Equal
    // (actual suffix comparison happens at leaf level)
    assert_eq!(key.compare(stored_ikey, 12), Ordering::Equal);
}

#[test]
fn test_from_ikey() {
    let ikey = u64::from_be_bytes(*b"test\0\0\0\0");
    let key: Key<'_> = Key::from_ikey(ikey);

    assert_eq!(key.ikey(), ikey);
    assert_eq!(key.suffix_start(), 4);
}

#[test]
fn test_lexicographic_ordering() {
    // Verify that u64 comparison equals lexicographic comparison
    let key_a: Key<'_> = Key::new(b"aaa");
    let key_b: Key<'_> = Key::new(b"aab");
    let key_c: Key<'_> = Key::new(b"baa");

    assert!(key_a.ikey() < key_b.ikey());
    assert!(key_b.ikey() < key_c.ikey());
}

#[test]
fn test_suffix_after_multiple_shifts() {
    let mut key: Key<'_> = Key::new(b"0123456789ABCDEF01234567"); // 24 bytes

    assert!(key.has_suffix());
    assert_eq!(key.suffix_len(), 16);

    key.shift();
    assert!(key.has_suffix());
    assert_eq!(key.suffix_len(), 8);

    key.shift();
    assert!(!key.has_suffix());
    assert_eq!(key.suffix_len(), 0);
}

#[test]
fn test_max_key_length() {
    // Exactly at the limit should succeed
    let max_key: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH];
    let key: Key<'_> = Key::new(&max_key);

    assert_eq!(key.len(), MAX_KEY_LENGTH);
}

#[test]
#[should_panic(expected = "key length 257 exceeds maximum 256")]
fn test_key_length_overflow() {
    let oversized: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH + 1];

    let _ = Key::new(&oversized);
}

#[test]
fn test_compare_ikey_fn() {
    assert_eq!(Key::compare_ikey(100, 200), Ordering::Less);
    assert_eq!(Key::compare_ikey(200, 100), Ordering::Greater);
    assert_eq!(Key::compare_ikey(100, 100), Ordering::Equal);
}

// ==================== Additional Edge Case Tests ====================

#[test]
fn test_full_data() {
    let key: Key<'_> = Key::new(b"hello world!");
    assert_eq!(key.full_data(), b"hello world!");
}

#[test]
fn test_full_data_after_shift() {
    let mut key: Key<'_> = Key::new(b"hello world!");
    key.shift();
    // full_data should still return the complete original data
    assert_eq!(key.full_data(), b"hello world!");
}

#[test]
fn test_compare_layer_keylenx() {
    // keylenx >= 128 indicates a layer pointer, should return Less
    let key: Key<'_> = Key::new(b"hello");
    let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // Layer indicator (keylenx = 128)
    assert_eq!(key.compare(stored_ikey, 128), Ordering::Less);

    // Higher layer values
    assert_eq!(key.compare(stored_ikey, 200), Ordering::Less);
    assert_eq!(key.compare(stored_ikey, 255), Ordering::Less);
}

#[test]
fn test_compare_ksuf_keylenx() {
    // keylenx == 64 indicates stored key has suffix
    let key: Key<'_> = Key::new(b"hello world!"); // 12 bytes, has suffix
    let stored_ikey = u64::from_be_bytes(*b"hello wo");

    // KSUF_KEYLENX = 64, both have suffixes -> Equal (needs suffix comparison)
    assert_eq!(key.compare(stored_ikey, 64), Ordering::Equal);

    // Key without suffix vs stored with suffix
    let short_key: Key<'_> = Key::new(b"hello"); // 5 bytes, no suffix
    assert_eq!(short_key.compare(stored_ikey, 64), Ordering::Less);
}

#[test]
fn test_ikey_byte_order() {
    // Verify big-endian byte order for lexicographic comparison
    let key: Key<'_> = Key::new(&[0x00, 0xFF]);
    let expected: u64 = 0x00FF_0000_0000_0000;
    assert_eq!(key.ikey(), expected);
}

#[test]
fn test_multi_shift_boundary() {
    // Test shifting exactly at 8-byte boundaries
    let mut key: Key<'_> = Key::new(b"12345678ABCDEFGH"); // 16 bytes exactly

    assert_eq!(key.ikey(), u64::from_be_bytes(*b"12345678"));
    assert!(key.has_suffix());
    assert_eq!(key.suffix_len(), 8);

    key.shift();
    assert_eq!(key.ikey(), u64::from_be_bytes(*b"ABCDEFGH"));
    assert!(!key.has_suffix());
    assert_eq!(key.suffix_len(), 0);
}

#[test]
fn test_from_ikey_full_bytes() {
    // ikey with all 8 bytes significant (no trailing zeros)
    let ikey: u64 = u64::from_be_bytes(*b"ABCDEFGH");
    let key: Key<'_> = Key::from_ikey(ikey);

    assert_eq!(key.ikey(), ikey);
    assert_eq!(key.suffix_start(), 8); // All 8 bytes are significant
}

#[test]
fn test_from_ikey_single_byte() {
    // ikey with only 1 significant byte
    let ikey: u64 = u64::from_be_bytes([b'A', 0, 0, 0, 0, 0, 0, 0]);
    let key: Key<'_> = Key::from_ikey(ikey);

    assert_eq!(key.ikey(), ikey);
    assert_eq!(key.suffix_start(), 1);
}

#[test]
fn test_from_ikey_zero() {
    // All-zero ikey
    let key: Key<'_> = Key::from_ikey(0);

    assert_eq!(key.ikey(), 0);
    assert_eq!(key.suffix_start(), 0);
}

#[test]
fn test_no_shift_when_no_suffix() {
    // Keys shorter than 8 bytes have no suffix, so shift() would panic
    let key: Key<'_> = Key::new(b"hi"); // 2 bytes
    assert!(!key.has_suffix());

    // Verify that attempting to shift would be invalid
    // (We don't actually call shift() because it would panic)
    assert_eq!(key.current_len(), 2);
}

#[test]
fn test_suffix_exact_boundary() {
    // Test suffix behavior at exact 8-byte boundary
    let key8: Key<'_> = Key::new(b"12345678");
    assert!(!key8.has_suffix());
    assert_eq!(key8.suffix(), b"");

    let key9: Key<'_> = Key::new(b"123456789");
    assert!(key9.has_suffix());
    assert_eq!(key9.suffix(), b"9");
}

#[test]
fn test_compare_empty_key() {
    let key: Key<'_> = Key::new(b"");

    // Empty key vs non-empty stored key
    let stored_ikey: u64 = u64::from_be_bytes([b'a', 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(key.compare(stored_ikey, 1), Ordering::Less);

    // Empty key vs empty stored key (keylenx = 0)
    assert_eq!(key.compare(0, 0), Ordering::Equal);
}

#[test]
fn test_binary_keys() {
    // Test with binary data containing null bytes
    let key: Key<'_> = Key::new(&[0x00, 0x01, 0x02, 0x00, 0xFF]);
    let expected: u64 = u64::from_be_bytes([0x00, 0x01, 0x02, 0x00, 0xFF, 0, 0, 0]);

    assert_eq!(key.ikey(), expected);
    assert_eq!(key.len(), 5);
}
