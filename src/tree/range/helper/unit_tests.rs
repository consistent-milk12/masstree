use super::{initial_ksuf_match, CursorKey, Ordering};

#[test]
fn test_initial_ksuf_match() {
    // Greater suffix always matches
    assert!(initial_ksuf_match(Ordering::Greater, true));
    assert!(initial_ksuf_match(Ordering::Greater, false));

    // Equal suffix matches only with emit_equal=true
    assert!(initial_ksuf_match(Ordering::Equal, true));
    assert!(!initial_ksuf_match(Ordering::Equal, false));

    // Less suffix never matches
    assert!(!initial_ksuf_match(Ordering::Less, true));
    assert!(!initial_ksuf_match(Ordering::Less, false));
}

#[test]
fn test_is_duplicate_less() {
    // Cursor key "apple" < slot "banana"
    let cursor = CursorKey::from_slice(b"apple");
    let slot_ikey = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

    // cursor < slot -> not a duplicate (cursor.compare != Less means duplicate)
    assert_eq!(cursor.compare(slot_ikey, 6), Ordering::Less);
}

#[test]
fn test_is_duplicate_equal() {
    // Cursor key "hello" == slot "hello"
    let cursor = CursorKey::from_slice(b"hello");
    let slot_ikey = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

    // cursor == slot -> is a duplicate
    assert_ne!(cursor.compare(slot_ikey, 5), Ordering::Less);
}

#[test]
fn test_is_duplicate_greater() {
    // Cursor key "zebra" > slot "apple"
    let cursor = CursorKey::from_slice(b"zebra");
    let slot_ikey = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

    // cursor > slot -> is a duplicate
    assert_ne!(cursor.compare(slot_ikey, 5), Ordering::Less);
}

#[test]
fn test_is_duplicate_with_suffix() {
    // Cursor key "hello world xyz"
    let mut cursor = CursorKey::from_slice(b"hello world xyz");
    cursor.assign_store_ikey(u64::from_be_bytes(*b"hello wo"));
    let _ = cursor.assign_store_suffix(b"rld xyz");
    cursor.assign_store_length(15);

    // cursor.suffix() = "rld xyz"
    // "rld xyz" > "rld abc" -> is duplicate (compare_suffix != Less)
    assert_ne!(cursor.compare_suffix(b"rld abc"), Ordering::Less);

    // "rld xyz" < "rld zzz" -> not duplicate
    assert_eq!(cursor.compare_suffix(b"rld zzz"), Ordering::Less);

    // "rld xyz" == "rld xyz" -> is duplicate
    assert_ne!(cursor.compare_suffix(b"rld xyz"), Ordering::Less);
}
