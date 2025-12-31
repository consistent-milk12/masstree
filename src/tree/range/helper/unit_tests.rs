use super::*;

#[test]
fn test_key_indexed_position_not_found() {
    let pos = KeyIndexedPosition::not_found(5);
    assert_eq!(pos.i, 5);
    assert!(!pos.has_match());
    assert!(pos.p.is_none());
}

#[test]
fn test_key_indexed_position_found() {
    let pos = KeyIndexedPosition::found(3, 7);
    assert_eq!(pos.i, 3);
    assert!(pos.has_match());
    assert_eq!(pos.slot(), 7);
}

#[test]
fn test_forward_scan_helper_next() {
    assert_eq!(ForwardScanHelper::next(0), 1);
    assert_eq!(ForwardScanHelper::next(5), 6);
    assert_eq!(ForwardScanHelper::next(23), 24);
}

#[test]
fn test_initial_ksuf_match() {
    // Greater suffix always matches
    assert!(ForwardScanHelper::initial_ksuf_match(
        Ordering::Greater,
        true
    ));
    assert!(ForwardScanHelper::initial_ksuf_match(
        Ordering::Greater,
        false
    ));

    // Equal suffix matches only with emit_equal=true
    assert!(ForwardScanHelper::initial_ksuf_match(Ordering::Equal, true));
    assert!(!ForwardScanHelper::initial_ksuf_match(
        Ordering::Equal,
        false
    ));

    // Less suffix never matches
    assert!(!ForwardScanHelper::initial_ksuf_match(Ordering::Less, true));
    assert!(!ForwardScanHelper::initial_ksuf_match(
        Ordering::Less,
        false
    ));
}

#[test]
fn test_is_duplicate_less() {
    // Cursor key "apple" < slot "banana"
    let cursor = CursorKey::from_slice(b"apple");
    let slot_ikey = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

    // cursor < slot -> not a duplicate
    assert!(!ForwardScanHelper::is_duplicate(&cursor, slot_ikey, 6));
}

#[test]
fn test_is_duplicate_equal() {
    // Cursor key "hello" == slot "hello"
    let cursor = CursorKey::from_slice(b"hello");
    let slot_ikey = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // cursor == slot -> is a duplicate
    assert!(ForwardScanHelper::is_duplicate(&cursor, slot_ikey, 5));
}

#[test]
fn test_is_duplicate_greater() {
    // Cursor key "zebra" > slot "apple"
    let cursor = CursorKey::from_slice(b"zebra");
    let slot_ikey = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

    // cursor > slot -> is a duplicate
    assert!(ForwardScanHelper::is_duplicate(&cursor, slot_ikey, 5));
}

#[test]
fn test_keylenx_helpers() {
    assert!(!is_layer_keylenx(0));
    assert!(!is_layer_keylenx(8));
    assert!(!is_layer_keylenx(64));
    assert!(is_layer_keylenx(128));
    assert!(is_layer_keylenx(255));

    assert!(!has_suffix_keylenx(0));
    assert!(!has_suffix_keylenx(8));
    assert!(has_suffix_keylenx(64));
    assert!(!has_suffix_keylenx(128));

    assert_eq!(inline_key_len(0), 0);
    assert_eq!(inline_key_len(5), 5);
    assert_eq!(inline_key_len(8), 8);
    assert_eq!(inline_key_len(64), 8);
    assert_eq!(inline_key_len(128), 8);
}

#[test]
fn test_is_duplicate_with_suffix() {
    // Cursor suffix "xyz" > stored suffix "abc"
    let mut cursor = CursorKey::from_slice(b"hello world xyz");
    cursor.assign_store_ikey(u64::from_be_bytes(*b"hello wo"));
    let _ = cursor.assign_store_suffix(b"rld xyz");
    cursor.assign_store_length(15);

    // cursor.suffix() = "rld xyz"
    // stored_suffix = "rld abc"
    // "rld xyz" > "rld abc" -> is duplicate
    assert!(ForwardScanHelper::is_duplicate_with_suffix(
        &cursor, b"rld abc"
    ));

    // "rld xyz" < "rld zzz" -> not duplicate
    assert!(!ForwardScanHelper::is_duplicate_with_suffix(
        &cursor, b"rld zzz"
    ));

    // "rld xyz" == "rld xyz" -> is duplicate
    assert!(ForwardScanHelper::is_duplicate_with_suffix(
        &cursor, b"rld xyz"
    ));
}
