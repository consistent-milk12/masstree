use super::*;

#[test]
fn test_from_slice_basic() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello");

    assert_eq!(cursor.current_len(), 5);
    assert_eq!(cursor.offset(), 0);
    assert!(!cursor.has_suffix());
    assert!(cursor.is_at_root_layer());
}

#[test]
fn test_from_slice_with_suffix() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello world!");

    assert_eq!(cursor.current_len(), 12);
    assert!(cursor.has_suffix());
    assert_eq!(cursor.suffix(), b"rld!");
}

#[test]
fn test_empty_cursor() {
    let cursor: CursorKey = CursorKey::empty();

    assert_eq!(cursor.current_ikey(), 0);
    assert_eq!(cursor.current_len(), 0);
    assert_eq!(cursor.full_key(), b"");
    assert!(!cursor.has_suffix());
}

#[test]
fn test_ikey_extraction() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello world!");
    let expected: u64 = u64::from_be_bytes(*b"hello wo");

    assert_eq!(cursor.current_ikey(), expected);
}

#[test]
fn test_shift() {
    // "hello world!!!!!" is 16 bytes: "hello wo" (8) + "rld!!!!!" (8)
    let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!!");

    assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"hello wo"));
    assert_eq!(cursor.layer_depth(), 0);

    cursor.shift();

    assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"rld!!!!!"));
    assert_eq!(cursor.layer_depth(), 1);
    assert_eq!(cursor.offset(), 8);
}

#[test]
fn test_shift_clear() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"hello");

    cursor.shift_clear();

    assert_eq!(cursor.current_ikey(), 0);
    assert_eq!(cursor.current_len(), 0);
    assert_eq!(cursor.offset(), 8);
    assert!(!cursor.has_suffix());
}

#[test]
fn test_unshift() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");
    let original_ikey: u64 = cursor.current_ikey();

    cursor.shift();
    assert_ne!(cursor.current_ikey(), original_ikey);

    cursor.unshift();

    assert_eq!(cursor.current_ikey(), original_ikey);
    assert_eq!(cursor.offset(), 0);
    // len is set to 9 (sentinel)
    assert_eq!(cursor.current_len(), 9);
    assert!(cursor.has_suffix());
}

#[test]
fn test_unshift_sentinel() {
    // After unshift, len=9 ensures we compare >= layer pointers
    let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");

    cursor.shift();
    cursor.unshift();

    // len = 9, so has_suffix() is true
    assert!(cursor.has_suffix());
    assert_eq!(cursor.current_len(), 9);

    // Compare against a layer pointer (keylenx >= 128)
    // With len > 8, we should get Equal (both have "suffix")
    let ikey: u64 = cursor.current_ikey();
    assert_eq!(cursor.compare(ikey, 128), Ordering::Equal);
    assert_eq!(cursor.compare(ikey, 200), Ordering::Equal);
}

#[test]
fn test_assign_store_ikey() {
    let mut cursor: CursorKey = CursorKey::empty();
    let ikey: u64 = u64::from_be_bytes(*b"testkey\0");

    cursor.assign_store_ikey(ikey);

    assert_eq!(cursor.current_ikey(), ikey);
    assert_eq!(&cursor.buf[0..8], b"testkey\0");
}

#[test]
fn test_assign_store_suffix() {
    let mut cursor: CursorKey = CursorKey::empty();

    cursor.assign_store_ikey(u64::from_be_bytes(*b"hello wo"));
    let key_len: usize = cursor.assign_store_suffix(b"rld!");
    cursor.assign_store_length(key_len);

    assert_eq!(key_len, 12);
    assert_eq!(cursor.current_len(), 12);
    assert_eq!(cursor.suffix(), b"rld!");
    assert_eq!(cursor.full_key(), b"hello world!");
}

#[test]
fn test_compare_equal() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello");
    let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    assert_eq!(cursor.compare(stored_ikey, 5), Ordering::Equal);
}

#[test]
fn test_compare_less_by_ikey() {
    let cursor: CursorKey = CursorKey::from_slice(b"apple");
    let stored_ikey: u64 = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

    assert_eq!(cursor.compare(stored_ikey, 6), Ordering::Less);
}

#[test]
fn test_compare_greater_by_ikey() {
    let cursor: CursorKey = CursorKey::from_slice(b"zebra");
    let stored_ikey: u64 = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

    assert_eq!(cursor.compare(stored_ikey, 5), Ordering::Greater);
}

#[test]
fn test_compare_by_length() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello");
    let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // Our key (5 bytes) vs stored key (3 bytes) -> Greater
    assert_eq!(cursor.compare(stored_ikey, 3), Ordering::Greater);

    // Our key (5 bytes) vs stored key (7 bytes) -> Less
    assert_eq!(cursor.compare(stored_ikey, 7), Ordering::Less);
}

#[test]
fn test_compare_with_suffix() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello world!"); // 12 bytes
    let stored_ikey: u64 = u64::from_be_bytes(*b"hello wo");

    // Our key has suffix, stored key has no suffix (length 8) -> Greater
    assert_eq!(cursor.compare(stored_ikey, 8), Ordering::Greater);

    // Our key has suffix, stored key also has suffix -> Equal
    assert_eq!(cursor.compare(stored_ikey, 12), Ordering::Equal);
}

#[test]
fn test_compare_suffix_bytes() {
    let cursor: CursorKey = CursorKey::from_slice(b"hello world!");

    assert_eq!(cursor.compare_suffix(b"rld!"), Ordering::Equal);
    assert_eq!(cursor.compare_suffix(b"aaa!"), Ordering::Greater);
    assert_eq!(cursor.compare_suffix(b"zzz!"), Ordering::Less);
}

#[test]
fn test_full_key() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");

    assert_eq!(cursor.full_key(), b"hello world!!!!");

    cursor.shift();

    // After shift, full_key still includes all bytes up to offset + len
    assert_eq!(cursor.full_key(), b"hello world!!!!");
}

#[test]
fn test_layer_depth() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"0123456789ABCDEF01234567");

    assert_eq!(cursor.layer_depth(), 0);

    cursor.shift();
    assert_eq!(cursor.layer_depth(), 1);

    cursor.shift();
    assert_eq!(cursor.layer_depth(), 2);
}

#[test]
fn test_multiple_shift_unshift() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"0123456789ABCDEF01234567"); // 24 bytes
    let layer0_ikey: u64 = cursor.current_ikey();

    cursor.shift();
    let layer1_ikey: u64 = cursor.current_ikey();

    cursor.shift();

    // Now unshift back
    cursor.unshift();
    assert_eq!(cursor.current_ikey(), layer1_ikey);

    cursor.unshift();
    assert_eq!(cursor.current_ikey(), layer0_ikey);
}

#[test]
fn test_shift_clear_then_assign() {
    let mut cursor: CursorKey = CursorKey::from_slice(b"prefix!!");

    cursor.shift_clear();
    assert_eq!(cursor.current_ikey(), 0);

    // Now assign a new ikey at layer 1
    let new_ikey: u64 = u64::from_be_bytes(*b"sublayer");
    cursor.assign_store_ikey(new_ikey);
    cursor.assign_store_length(8);

    assert_eq!(cursor.current_ikey(), new_ikey);
    // full_key now includes the original prefix + new layer
    assert_eq!(&cursor.full_key()[8..16], b"sublayer");
}

#[test]
#[should_panic(expected = "key length")]
fn test_from_slice_overflow() {
    let oversized: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH + 1];
    let _ = CursorKey::from_slice(&oversized);
}

#[test]
fn test_clone() {
    let cursor1: CursorKey = CursorKey::from_slice(b"hello world!");
    let cursor2: CursorKey = cursor1.clone();

    assert_eq!(cursor1.current_ikey(), cursor2.current_ikey());
    assert_eq!(cursor1.current_len(), cursor2.current_len());
    assert_eq!(cursor1.full_key(), cursor2.full_key());
}

#[test]
fn test_debug_format() {
    let cursor: CursorKey = CursorKey::from_slice(b"test");
    let debug_str: String = format!("{cursor:?}");

    assert!(debug_str.contains("CursorKey"));
    assert!(debug_str.contains("offset"));
    assert!(debug_str.contains("len"));
}

#[test]
fn test_exact_8_bytes() {
    let cursor: CursorKey = CursorKey::from_slice(b"12345678");

    assert_eq!(cursor.current_len(), 8);
    assert!(!cursor.has_suffix());
    assert_eq!(cursor.suffix(), b"");
}

#[test]
fn test_9_bytes() {
    let cursor: CursorKey = CursorKey::from_slice(b"123456789");

    assert_eq!(cursor.current_len(), 9);
    assert!(cursor.has_suffix());
    assert_eq!(cursor.suffix(), b"9");
}
