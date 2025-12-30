use super::*;
use crate::tree::MassTree24;
use std::sync::Arc;

#[test]
fn test_remove_single_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key1", 42).unwrap();
    assert_eq!(tree.len(), 1);

    let removed = tree.remove(b"key1").unwrap();
    assert_eq!(removed, Some(Arc::new(42)));
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_remove_nonexistent_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key1", 42).unwrap();

    let result = tree.remove(b"key2");
    assert!(matches!(result, Ok(None)));

    // Original key still exists
    assert_eq!(tree.get(b"key1"), Some(Arc::new(42)));
}

#[test]
fn test_remove_updates_count() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..10u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }
    assert_eq!(tree.len(), 10);

    for i in 0..5u64 {
        tree.remove(&i.to_be_bytes()).unwrap();
    }
    assert_eq!(tree.len(), 5);

    // Verify remaining keys
    for i in 5..10u64 {
        assert!(tree.get(&i.to_be_bytes()).is_some());
    }
    for i in 0..5u64 {
        assert!(tree.get(&i.to_be_bytes()).is_none());
    }
}

#[test]
fn test_remove_returns_old_value() {
    let tree: MassTree24<String> = MassTree24::new();

    tree.insert(b"key", "hello".to_string()).unwrap();
    tree.insert(b"key", "world".to_string()).unwrap();

    let removed = tree.remove(b"key").unwrap();
    assert_eq!(removed, Some(Arc::new("world".to_string())));
}

#[test]
fn test_remove_short_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 1-byte key
    tree.insert(&[42], 1).unwrap();
    assert_eq!(tree.remove(&[42]).unwrap(), Some(Arc::new(1)));

    // 8-byte key (max inline)
    let key8 = [1, 2, 3, 4, 5, 6, 7, 8];
    tree.insert(&key8, 8).unwrap();
    assert_eq!(tree.remove(&key8).unwrap(), Some(Arc::new(8)));
}

#[test]
fn test_remove_with_suffix() {
    let tree: MassTree24<u64> = MassTree24::new();

    // 16-byte key (requires suffix)
    let key16 = b"0123456789ABCDEF";
    tree.insert(key16, 16).unwrap();

    let removed = tree.remove(key16).unwrap();
    assert_eq!(removed, Some(Arc::new(16)));
    assert!(tree.get(key16).is_none());
}

#[test]
fn test_remove_all_keys_empties_tree() {
    let tree: MassTree24<u64> = MassTree24::new();

    let keys: Vec<_> = (0..100u64).map(|i| i.to_be_bytes()).collect();

    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64).unwrap();
    }
    assert_eq!(tree.len(), 100);

    for key in &keys {
        tree.remove(key).unwrap();
    }
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
}

#[test]
fn test_remove_in_reverse_order() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..50u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    // Remove in reverse order
    for i in (0..50u64).rev() {
        let removed = tree.remove(&i.to_be_bytes()).unwrap();
        assert_eq!(removed, Some(Arc::new(i)));
    }

    assert!(tree.is_empty());
}

#[test]
fn test_remove_alternating() {
    let tree: MassTree24<u64> = MassTree24::new();

    for i in 0..100u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    // Remove even keys
    for i in (0..100u64).step_by(2) {
        tree.remove(&i.to_be_bytes()).unwrap();
    }

    assert_eq!(tree.len(), 50);

    // Verify odd keys remain
    for i in (1..100u64).step_by(2) {
        assert!(tree.get(&i.to_be_bytes()).is_some());
    }
}

#[test]
fn test_remove_and_reinsert_same_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"key", 1).unwrap();
    tree.remove(b"key").unwrap();

    // Reinsert with different value
    tree.insert(b"key", 2).unwrap();
    assert_eq!(tree.get(b"key"), Some(Arc::new(2)));
}

#[test]
fn test_remove_reinsert_cycle() {
    let tree: MassTree24<u64> = MassTree24::new();
    let key = b"test_key";

    for i in 0..10u64 {
        tree.insert(key, i).unwrap();
        assert_eq!(tree.get(key), Some(Arc::new(i)));

        let removed = tree.remove(key).unwrap();
        assert_eq!(removed, Some(Arc::new(i)));
        assert!(tree.get(key).is_none());
    }
}

#[test]
fn test_remove_from_empty_tree() {
    let tree: MassTree24<u64> = MassTree24::new();
    let result = tree.remove(b"key");
    assert!(matches!(result, Ok(None)));
}

#[test]
fn test_remove_empty_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Empty key is valid
    tree.insert(&[], 0).unwrap();
    let removed = tree.remove(&[]).unwrap();
    assert_eq!(removed, Some(Arc::new(0)));
}

#[test]
fn test_remove_preserves_other_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    tree.insert(b"aaa", 1).unwrap();
    tree.insert(b"bbb", 2).unwrap();
    tree.insert(b"ccc", 3).unwrap();

    tree.remove(b"bbb").unwrap();

    assert_eq!(tree.get(b"aaa"), Some(Arc::new(1)));
    assert!(tree.get(b"bbb").is_none());
    assert_eq!(tree.get(b"ccc"), Some(Arc::new(3)));
}
