use super::{LayerContext, LayerStack, NonNull, ScanSnapshotPtr, ScanState};
use arrayvec::ArrayVec;
use std::ptr as StdPtr;

// Note: These tests use mock types since we can't easily construct
// real LeafNode24 in unit tests. Integration tests  test with real types.

#[test]
fn test_scan_state_properties() {
    // Test is_emit
    assert!(ScanState::Emit.is_emit());
    assert!(!ScanState::FindNext.is_emit());
    assert!(!ScanState::Down.is_emit());
    assert!(!ScanState::Up.is_emit());
    assert!(!ScanState::Retry.is_emit());

    // Test is_layer_transition
    assert!(ScanState::Down.is_layer_transition());
    assert!(ScanState::Up.is_layer_transition());
    assert!(!ScanState::Emit.is_layer_transition());
    assert!(!ScanState::FindNext.is_layer_transition());
    assert!(!ScanState::Retry.is_layer_transition());
}

#[test]
fn test_scan_state_equality() {
    assert_eq!(ScanState::Emit, ScanState::Emit);
    assert_ne!(ScanState::Emit, ScanState::FindNext);
}

#[test]
fn test_scan_state_debug() {
    let state = ScanState::FindNext;
    let debug_str = format!("{state:?}");
    assert_eq!(debug_str, "FindNext");
}

#[test]
fn test_layer_context_creation() {
    let root: *const u8 = StdPtr::without_provenance(0x1000);
    let leaf: *mut u8 = StdPtr::without_provenance_mut(0x2000);

    let ctx: LayerContext<u8> = LayerContext::new(root, leaf);

    assert_eq!(ctx.root, root);
    assert_eq!(ctx.leaf_ptr(), leaf);
}

#[test]
#[expect(clippy::unwrap_used)]
fn test_layer_stack_operations() {
    let mut stack: LayerStack<u8> = ArrayVec::new();

    assert!(stack.is_empty());

    // Push some contexts
    stack.push(LayerContext::new(
        StdPtr::without_provenance(0x1000),
        StdPtr::without_provenance_mut(0x2000),
    ));
    stack.push(LayerContext::new(
        StdPtr::without_provenance(0x3000),
        StdPtr::without_provenance_mut(0x4000),
    ));

    assert_eq!(stack.len(), 2);

    // Pop
    let ctx = stack.pop().unwrap();
    assert_eq!(ctx.root, StdPtr::without_provenance(0x3000));
    assert_eq!(ctx.leaf_ptr(), StdPtr::without_provenance_mut(0x4000));

    assert_eq!(stack.len(), 1);

    // Pop again
    let ctx = stack.pop().unwrap();
    assert_eq!(ctx.root, StdPtr::without_provenance(0x1000));

    assert!(stack.is_empty());
    assert!(stack.pop().is_none());
}

#[test]
fn test_layer_stack_capacity() {
    let mut stack: LayerStack<u8> = ArrayVec::new();

    // ArrayVec has fixed capacity of 6
    assert_eq!(stack.capacity(), 6);
    assert!(stack.is_empty());

    // Push 6 elements (full capacity)
    for i in 1..=6 {
        stack.push(LayerContext::new(
            StdPtr::without_provenance(i * 0x1000),
            StdPtr::without_provenance_mut(i * 0x2000),
        ));
    }

    // Stack should be full but not panicking
    assert_eq!(stack.len(), 6);
    assert!(stack.is_full());

    // try_push should fail when full
    let result = stack.try_push(LayerContext::new(
        StdPtr::without_provenance(0x7000),
        StdPtr::without_provenance_mut(0x8000),
    ));
    assert!(result.is_err());
}

#[test]
#[expect(clippy::cast_ptr_alignment)]
fn test_scan_snapshot_ptr_generic() {
    // Test with u64
    let ptr: *const u64 = StdPtr::without_provenance(0x1000);
    let snap: ScanSnapshotPtr<u64> = ScanSnapshotPtr::new(ptr, 8);
    assert_eq!(snap.value_ptr, ptr);
    assert_eq!(snap.key_len, 8);

    // Test from_raw
    let raw: *const u8 = StdPtr::without_provenance(0x2000);
    let snap2: ScanSnapshotPtr<u64> = ScanSnapshotPtr::from_raw(raw, 4);
    assert_eq!(snap2.value_ptr, raw.cast::<u64>());
    assert_eq!(snap2.key_len, 4);
}

#[test]
fn test_nonnull_niche_optimization() {
    // Verify that Option<NonNull<L>> has the same size as *mut L
    // due to niche optimization
    use std::mem::size_of;
    assert_eq!(
        size_of::<Option<NonNull<u8>>>(),
        size_of::<*mut u8>(),
        "NonNull niche optimization should make Option<NonNull> same size as raw pointer"
    );
}
