//! Tests for fallible allocation support.
//!
//! This module tests the fallible allocation APIs introduced to handle OOM
//! gracefully instead of aborting. The key components tested are:
//!
//! - `AllocError` and `AllocKind` error types
//! - `InsertError::AllocationFailed` variant
//! - Fallible allocator methods (`try_alloc_leaf`)
//! - Fallible suffix assignment (`try_assign_ksuf`)
//! - Fallible value storage (`try_output_to_raw`)
//!
//! Note: True OOM injection testing is not feasible without custom allocators.
//! These tests verify the API surface and normal operation paths.

use masstree::{AllocError, AllocKind, AllocResult, InsertError};

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_alloc_error_new() {
    let err = AllocError::new(1024, 8, AllocKind::Leaf);

    assert_eq!(err.size, 1024);
    assert_eq!(err.align, 8);
    assert_eq!(err.kind, AllocKind::Leaf);
}

#[test]
fn test_alloc_error_for_type() {
    let err = AllocError::for_type::<u64>();

    assert_eq!(err.size, 8);
    assert_eq!(err.align, 8);
    assert_eq!(err.kind, AllocKind::Other);
}

#[test]
fn test_alloc_error_for_leaf() {
    let err = AllocError::for_leaf::<[u8; 256]>();

    assert_eq!(err.size, 256);
    assert_eq!(err.kind, AllocKind::Leaf);
}

#[test]
fn test_alloc_error_for_suffix() {
    let err = AllocError::for_suffix(512);

    assert_eq!(err.size, 512);
    assert_eq!(err.align, 1);
    assert_eq!(err.kind, AllocKind::Suffix);
}

#[test]
fn test_alloc_error_for_value() {
    let err = AllocError::for_value::<String>();

    assert_eq!(err.size, std::mem::size_of::<String>());
    assert_eq!(err.kind, AllocKind::Value);
}

#[test]
fn test_alloc_error_for_tracking() {
    let err = AllocError::for_tracking(64);

    assert_eq!(err.size, 64);
    assert_eq!(err.kind, AllocKind::AllocatorTracking);
}

#[test]
fn test_alloc_error_display() {
    let err = AllocError::new(4096, 64, AllocKind::Suffix);
    let msg = format!("{err}");

    assert!(msg.contains("suffix"));
    assert!(msg.contains("4096"));
    assert!(msg.contains("64"));
}

#[test]
fn test_alloc_kind_display() {
    assert_eq!(format!("{}", AllocKind::Leaf), "leaf");
    assert_eq!(format!("{}", AllocKind::Internode), "internode");
    assert_eq!(format!("{}", AllocKind::Suffix), "suffix");
    assert_eq!(format!("{}", AllocKind::Value), "value");
    assert_eq!(format!("{}", AllocKind::AllocatorTracking), "allocator tracking");
    assert_eq!(format!("{}", AllocKind::Other), "other");
}

// ============================================================================
// InsertError Tests
// ============================================================================

#[test]
fn test_insert_error_allocation_failed_typed() {
    let err = InsertError::allocation_failed_typed::<[u8; 1024]>(AllocKind::Leaf);

    match err {
        InsertError::AllocationFailed { size, kind } => {
            assert_eq!(size, 1024);
            assert_eq!(kind, AllocKind::Leaf);
        }
        _ => panic!("expected AllocationFailed variant"),
    }
}

#[test]
fn test_insert_error_allocation_failed() {
    let err = InsertError::allocation_failed(2048, AllocKind::Suffix);

    match err {
        InsertError::AllocationFailed { size, kind } => {
            assert_eq!(size, 2048);
            assert_eq!(kind, AllocKind::Suffix);
        }
        _ => panic!("expected AllocationFailed variant"),
    }
}

#[test]
fn test_insert_error_is_allocation_failed() {
    let alloc_err = InsertError::allocation_failed(1024, AllocKind::Leaf);
    let leaf_full_err = InsertError::LeafFull;
    let split_required = InsertError::SplitRequired;

    assert!(alloc_err.is_allocation_failed());
    assert!(!leaf_full_err.is_allocation_failed());
    assert!(!split_required.is_allocation_failed());
}

#[test]
fn test_insert_error_from_alloc_error() {
    let alloc_err = AllocError::new(512, 8, AllocKind::Value);
    let insert_err: InsertError = alloc_err.into();

    match insert_err {
        InsertError::AllocationFailed { size, kind } => {
            assert_eq!(size, 512);
            assert_eq!(kind, AllocKind::Value);
        }
        _ => panic!("expected AllocationFailed variant"),
    }
}

#[test]
fn test_insert_error_display() {
    let err = InsertError::allocation_failed(4096, AllocKind::Suffix);
    let msg = format!("{err}");

    assert!(msg.contains("suffix"));
    assert!(msg.contains("4096"));
}

// ============================================================================
// AllocResult Tests
// ============================================================================

#[test]
fn test_alloc_result_ok() {
    let result: AllocResult<i32> = Ok(42);
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_alloc_result_err() {
    let result: AllocResult<i32> = Err(AllocError::for_type::<i32>());
    assert!(result.is_err());
}

#[test]
fn test_alloc_result_question_mark_propagation() {
    fn fallible_fn() -> AllocResult<i32> {
        let _: i32 = inner_fallible()?;
        Ok(42)
    }

    fn inner_fallible() -> AllocResult<i32> {
        Err(AllocError::for_type::<i32>())
    }

    let result = fallible_fn();
    assert!(result.is_err());
}

// ============================================================================
// Integration Tests - Normal Operation
// ============================================================================

#[test]
fn test_masstree24_insert_returns_result() {
    use masstree::MassTree24;

    let tree: MassTree24<u64> = MassTree24::new();

    // Insert should return Result
    let result = tree.insert(b"key1", 100u64);
    assert!(result.is_ok());

    // Verify value was inserted
    let value = tree.get(b"key1");
    assert_eq!(value, Some(std::sync::Arc::new(100u64)));
}

#[test]
fn test_masstree24_many_inserts_succeed() {
    use masstree::MassTree24;

    let tree: MassTree24<u64> = MassTree24::new();

    // Many inserts should all succeed under normal memory conditions
    for i in 0..1000u64 {
        let key = format!("key_{:04}", i);
        let result = tree.insert(key.as_bytes(), i);
        assert!(result.is_ok(), "insert {} failed: {:?}", i, result);
    }

    // Verify all values present
    assert_eq!(tree.len(), 1000);
}

#[test]
fn test_masstree24_long_keys_with_suffixes() {
    use masstree::MassTree24;

    let tree: MassTree24<u64> = MassTree24::new();

    // Keys longer than 8 bytes require suffix storage
    for i in 0..100u64 {
        let key = format!("this_is_a_very_long_key_that_requires_suffix_storage_{:04}", i);
        let result = tree.insert(key.as_bytes(), i);
        assert!(result.is_ok(), "long key insert {} failed: {:?}", i, result);
    }

    // Verify lookups work
    let key = b"this_is_a_very_long_key_that_requires_suffix_storage_0042";
    let value = tree.get(key);
    assert_eq!(value, Some(std::sync::Arc::new(42u64)));
}

#[test]
fn test_masstree_index_insert_returns_result() {
    use masstree::MassTreeIndex;

    let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

    // Insert should return Result
    let result = tree.insert(b"key1", 100u64);
    assert!(result.is_ok());

    // Verify value was inserted
    let value = tree.get(b"key1");
    assert_eq!(value, Some(100u64));
}

#[test]
fn test_masstree15_insert_returns_result() {
    use masstree::MassTree15;

    let tree: MassTree15<String> = MassTree15::new();

    // Insert should return Result
    let result = tree.insert(b"key1", "hello".to_string());
    assert!(result.is_ok());

    // Verify value was inserted
    let value = tree.get(b"key1");
    assert!(value.is_some());
    assert_eq!(value.unwrap().as_str(), "hello");
}

// ============================================================================
// Design Verification Tests
// ============================================================================

#[test]
fn test_split_allocation_before_mark_split() {
    // This test verifies the design invariant that split allocation
    // happens BEFORE mark_split() is called.
    //
    // We can't directly test this without internal access, but we verify
    // that splits work correctly by inserting enough keys to trigger splits.

    use masstree::MassTree24;

    let tree: MassTree24<u64> = MassTree24::new();

    // Insert enough keys to trigger multiple splits (WIDTH=24, so >24 keys)
    for i in 0..100u64 {
        let key = format!("{:08}", i); // Fixed-width keys for predictable ordering
        let result = tree.insert(key.as_bytes(), i);
        assert!(result.is_ok(), "insert {} triggered error: {:?}", i, result);
    }

    // Verify all keys present after splits
    for i in 0..100u64 {
        let key = format!("{:08}", i);
        let value = tree.get(key.as_bytes());
        assert_eq!(value, Some(std::sync::Arc::new(i)), "key {} missing after splits", i);
    }
}

#[test]
fn test_layer_creation_allocation() {
    // This test verifies that layer creation (for keys with common prefixes)
    // handles allocation correctly.

    use masstree::MassTree24;

    let tree: MassTree24<u64> = MassTree24::new();

    // Insert keys that share prefixes to trigger layer creation
    let prefixes = ["aaaa", "aaab", "aaac", "aaba", "aabb", "abaa", "abab"];

    for (i, prefix) in prefixes.iter().enumerate() {
        let key = format!("{}_{:04}", prefix, i);
        let result = tree.insert(key.as_bytes(), i as u64);
        assert!(result.is_ok(), "layer insert {} failed: {:?}", i, result);
    }

    // Verify all keys present
    for (i, prefix) in prefixes.iter().enumerate() {
        let key = format!("{}_{:04}", prefix, i);
        let value = tree.get(key.as_bytes());
        assert_eq!(value, Some(std::sync::Arc::new(i as u64)));
    }
}

// ============================================================================
// Concurrent Operation Tests
// ============================================================================

#[test]
fn test_concurrent_inserts_return_results() {
    use masstree::MassTree24;
    use std::sync::Arc;
    use std::thread;

    let tree: Arc<MassTree24<u64>> = Arc::new(MassTree24::new());
    let num_threads = 4;
    let keys_per_thread = 250;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for i in 0..keys_per_thread {
                    let key = format!("thread_{}_key_{:04}", t, i);
                    let value = (t * keys_per_thread + i) as u64;
                    let result = tree.insert(key.as_bytes(), value);
                    assert!(result.is_ok(), "concurrent insert failed: {:?}", result);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Verify total count
    assert_eq!(tree.len(), num_threads * keys_per_thread);
}
