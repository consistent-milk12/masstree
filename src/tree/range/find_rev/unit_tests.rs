//! Unit tests for reverse scan operations.
//!
//! Tests cover:
//! - `handle_down_back`: Sublayer descent cursor setup
//! - `handle_up_back`: Layer ascent with empty stack handling
//! - State machine transitions

use super::{CursorKey, ReverseScan, ReverseScanHelper};

// ============================================================================
//  handle_down_back Tests
// ============================================================================

#[test]
fn test_handle_down_back_sets_cursor_to_max() {
    let mut cursor_key = CursorKey::from_slice(b"test1234suffix");
    let mut helper = ReverseScanHelper::new();

    // Before: cursor has specific key, upper_bound is false
    assert!(!helper.upper_bound);
    let original_offset = cursor_key.offset();

    ReverseScan::handle_down_back(&mut cursor_key, &mut helper);

    // After: cursor at maximum, upper_bound set
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert_eq!(cursor_key.current_len(), 9); // IKEY_SIZE + 1
    assert_eq!(cursor_key.offset(), original_offset + 8); // shifted one layer
    assert!(helper.upper_bound);
}

#[test]
fn test_handle_down_back_from_root() {
    let mut cursor_key = CursorKey::from_slice(b"hello");
    let mut helper = ReverseScanHelper::new();

    assert_eq!(cursor_key.offset(), 0);
    assert!(!helper.upper_bound);

    ReverseScan::handle_down_back(&mut cursor_key, &mut helper);

    // Descended one layer
    assert_eq!(cursor_key.offset(), 8);
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert!(helper.upper_bound);
}

#[test]
fn test_handle_down_back_multiple_descents() {
    let mut cursor_key = CursorKey::from_slice(b"a]");
    let mut helper = ReverseScanHelper::new();

    // First descent
    ReverseScan::handle_down_back(&mut cursor_key, &mut helper);
    assert_eq!(cursor_key.offset(), 8);
    assert!(helper.upper_bound);

    // Reset upper_bound as would happen after emitting a value
    helper.mark_key_complete();
    assert!(!helper.upper_bound);

    // Second descent
    ReverseScan::handle_down_back(&mut cursor_key, &mut helper);
    assert_eq!(cursor_key.offset(), 16);
    assert!(helper.upper_bound);
}

// ============================================================================
//  CursorKey::is_empty_after_unshift Tests
// ============================================================================

#[test]
fn test_is_empty_after_unshift_false_normally() {
    let mut cursor_key = CursorKey::from_slice(b"hello world!!!!!");

    // Shift to layer 1, then unshift back
    cursor_key.shift();
    cursor_key.unshift();

    // Normal case: not empty because we have key content
    assert!(!cursor_key.is_empty_after_unshift());
}

#[test]
fn test_is_empty_after_unshift_at_root() {
    let cursor_key = CursorKey::empty();

    // Empty cursor at root layer
    assert_eq!(cursor_key.offset(), 0);
    assert_eq!(cursor_key.current_len(), 0);
    // Note: is_empty_after_unshift checks offset==0 && len==0
    assert!(cursor_key.is_empty_after_unshift());
}

// ============================================================================
//  ReverseScanHelper Tests
// ============================================================================

#[test]
fn test_helper_new_state() {
    let helper = ReverseScanHelper::new();
    assert!(!helper.upper_bound);
}

#[test]
fn test_helper_at_upper_bound() {
    let helper = ReverseScanHelper::at_upper_bound();
    assert!(helper.upper_bound);
}

#[test]
fn test_helper_mark_key_complete_clears_upper_bound() {
    let mut helper = ReverseScanHelper::at_upper_bound();
    assert!(helper.upper_bound);

    helper.mark_key_complete();

    assert!(!helper.upper_bound);
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
    let mut helper = ReverseScanHelper::new();

    // Initial state
    assert_eq!(cursor_key.offset(), 0);
    let original_ikey = cursor_key.current_ikey();

    // Step 1: Descend into sublayer
    ReverseScan::handle_down_back(&mut cursor_key, &mut helper);

    assert_eq!(cursor_key.offset(), 8);
    assert_eq!(cursor_key.current_ikey(), u64::MAX);
    assert!(helper.upper_bound);

    // Step 2: Mark key complete (would happen after emitting)
    helper.mark_key_complete();
    assert!(!helper.upper_bound);

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
    let mut helper = ReverseScanHelper::new();

    // Descend 3 times
    for expected_offset in [8, 16, 24] {
        ReverseScan::handle_down_back(&mut cursor_key, &mut helper);
        assert_eq!(cursor_key.offset(), expected_offset);
        assert!(helper.upper_bound);
        helper.mark_key_complete();
    }

    // Ascend 3 times
    for expected_offset in [16, 8, 0] {
        cursor_key.unshift();
        assert_eq!(cursor_key.offset(), expected_offset);
        assert_eq!(cursor_key.current_len(), 9); // Sentinel after each unshift
    }
}
