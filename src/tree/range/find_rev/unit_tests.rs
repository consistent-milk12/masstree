//! Unit tests for reverse scan cursor and helper operations.
//!
//! Tests cover:
//! - `handle_down_back` logic: Sublayer descent cursor setup
//! - `handle_up_back` logic: Layer ascent with cursor unshift
//! - `ReverseFlags` `upper_bound` state transitions
//! - `CursorKey` reverse operations

use crate::tree::range::cursor_key::CursorKey;
use crate::tree::range::helper::ReverseScanHelper;
use crate::tree::range::iterator::iter_flags::ReverseFlags;

/// Inline helper matching `ReverseScanCtx::handle_down_back` logic.
/// Shifts cursor into sublayer with MAX ikey and sets `upper_bound`.
fn handle_down_back(cursor_key: &mut CursorKey, flags: &mut ReverseFlags) {
    cursor_key.shift_clear_reverse();
    flags.set_upper_bound();
}

// ============================================================================
//  handle_down_back Tests
// ============================================================================

#[test]
fn test_handle_down_back_sets_cursor_to_max() {
    let mut cursor_key = CursorKey::from_slice(b"test1234suffix");
    let mut flags = ReverseFlags::new();

    // Before: cursor has specific key, upper_bound is false
    assert!(!flags.upper_bound());
    let original_offset = cursor_key.offset();

    handle_down_back(&mut cursor_key, &mut flags);

    // After: cursor at maximum, upper_bound set
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert_eq!(cursor_key.current_len(), 9); // IKEY_SIZE + 1
    assert_eq!(cursor_key.offset(), original_offset + 8); // shifted one layer
    assert!(flags.upper_bound());
}

#[test]
fn test_handle_down_back_from_root() {
    let mut cursor_key = CursorKey::from_slice(b"hello");
    let mut flags = ReverseFlags::new();

    assert_eq!(cursor_key.offset(), 0);
    assert!(!flags.upper_bound());

    handle_down_back(&mut cursor_key, &mut flags);

    // Descended one layer
    assert_eq!(cursor_key.offset(), 8);
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert!(flags.upper_bound());
}

#[test]
fn test_handle_down_back_multiple_descents() {
    let mut cursor_key = CursorKey::from_slice(b"a]");
    let mut flags = ReverseFlags::new();

    // First descent
    handle_down_back(&mut cursor_key, &mut flags);
    assert_eq!(cursor_key.offset(), 8);
    assert!(flags.upper_bound());

    // Reset upper_bound as would happen after emitting a value
    flags.clear_upper_bound();
    assert!(!flags.upper_bound());

    // Second descent
    handle_down_back(&mut cursor_key, &mut flags);
    assert_eq!(cursor_key.offset(), 16);
    assert!(flags.upper_bound());
}

// ============================================================================
//  CursorKey::is_at_empty_root Tests
// ============================================================================

#[test]
fn test_is_at_empty_root_false_normally() {
    let mut cursor_key = CursorKey::from_slice(b"hello world!!!!!");

    // Shift to layer 1, then unshift back
    cursor_key.shift();
    cursor_key.unshift();

    // Normal case: not empty because we have key content
    assert!(!cursor_key.is_at_empty_root());
}

#[test]
fn test_is_at_empty_root_at_root() {
    let cursor_key = CursorKey::empty();

    // Empty cursor at root layer
    assert_eq!(cursor_key.offset(), 0);
    assert_eq!(cursor_key.current_len(), 0);
    // is_at_empty_root checks offset==0 && len==0
    assert!(cursor_key.is_at_empty_root());
}

// ============================================================================
//  ReverseFlags upper_bound Tests
// ============================================================================

#[test]
fn test_flags_upper_bound_default_false() {
    let flags = ReverseFlags::new();
    assert!(!flags.upper_bound());
}

#[test]
fn test_flags_clear_upper_bound() {
    let mut flags = ReverseFlags::new();
    flags.set_upper_bound();

    flags.clear_upper_bound();

    assert!(!flags.upper_bound());
}

#[test]
fn test_helper_prev() {
    // prev() is a const fn that decrements index
    assert_eq!(ReverseScanHelper::prev(5), 4);
    assert_eq!(ReverseScanHelper::prev(1), 0);
    assert_eq!(ReverseScanHelper::prev(0), -1);

    // -1 is the "before first" sentinel
    assert_eq!(ReverseScanHelper::prev(-1), -2);
}

// ============================================================================
//  Cursor State After Reverse Operations
// ============================================================================

#[test]
fn test_shift_clear_reverse_sets_max_ikey() {
    let mut cursor_key = CursorKey::from_slice(b"test");

    cursor_key.shift_clear_reverse();

    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert_eq!(cursor_key.current_len(), 9); // IKEY_SIZE + 1
    assert_eq!(cursor_key.offset(), 8);
}

#[test]
fn test_unshift_sentinel_len() {
    let mut cursor_key = CursorKey::from_slice(b"hello world!!!!!");

    cursor_key.shift();
    assert_eq!(cursor_key.offset(), 8);

    cursor_key.unshift();
    assert_eq!(cursor_key.offset(), 0);
    assert_eq!(cursor_key.current_len(), 9); // Sentinel value
}

// ============================================================================
//  Integration-like Tests (no actual tree, but testing state flow)
// ============================================================================

#[test]
fn test_reverse_scan_cursor_flow() {
    // Simulate the cursor state through a reverse scan:
    // 1. Start at end bound
    // 2. Descend into sublayer (handle_down_back)
    // 3. Ascend back (unshift)

    let mut cursor_key = CursorKey::from_slice(b"prefix__suffix99");
    let mut flags = ReverseFlags::new();

    // Initial state
    assert_eq!(cursor_key.offset(), 0);
    let original_ikey = cursor_key.current_ikey();

    // Step 1: Descend into sublayer
    handle_down_back(&mut cursor_key, &mut flags);

    assert_eq!(cursor_key.offset(), 8);
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert!(flags.upper_bound());

    // Step 2: Mark key complete (would happen after emitting)
    flags.clear_upper_bound();
    assert!(!flags.upper_bound());

    // Step 3: Ascend back (unshift)
    cursor_key.unshift();

    assert_eq!(cursor_key.offset(), 0);
    assert_eq!(cursor_key.current_ikey(), original_ikey);
    assert_eq!(cursor_key.current_len(), 9); // Sentinel
}

#[test]
fn test_deep_descent_ascent_cycle() {
    // Simulate descending 3 layers and ascending back
    let mut cursor_key = CursorKey::from_slice(b"aaaaaaaa"); // exactly 8 bytes
    let mut flags = ReverseFlags::new();

    // Descend 3 times
    for expected_offset in [8, 16, 24] {
        handle_down_back(&mut cursor_key, &mut flags);
        assert_eq!(cursor_key.offset(), expected_offset);
        assert!(flags.upper_bound());
        flags.clear_upper_bound();
    }

    // Ascend 3 times
    for expected_offset in [16, 8, 0] {
        cursor_key.unshift();
        assert_eq!(cursor_key.offset(), expected_offset);
        assert_eq!(cursor_key.current_len(), 9); // Sentinel after each unshift
    }
}
