#![expect(clippy::unwrap_used)]
#![expect(clippy::panic)]

use crate::policy::ValuePtr;

use seize::LocalGuard;

use crate::{MassTree15, tree::generic::entry::Entry};

// ============================================================================
//  Entry API Tests
// ============================================================================

#[test]
fn test_entry_or_insert_vacant() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Key doesn't exist, should insert
    let value: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert(42);
    assert_eq!(*value, 42);
    assert_eq!(tree.len(), 1);

    // Verify it was inserted
    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 42);
}

#[test]
fn test_entry_or_insert_occupied() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Pre-insert a value
    tree.insert_with_guard(b"key", 100, &guard);

    // Key exists, should return existing value, not insert
    let value: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert(42);
    assert_eq!(*value, 100);
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_entry_or_insert() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Insert on vacant
    let result: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert(42);
    assert_eq!(*result, 42);

    // Insert on occupied returns cached value
    let result: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert(100);
    assert_eq!(*result, 42); // Original value, not 100
}

#[derive(PartialEq, Eq)]
enum Bool {
    True,
    False,
}

#[test]
fn test_entry_or_insert_with_lazy() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    let mut called: Bool = Bool::False;
    let value: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert_with(|| {
        called = Bool::True;
        99
    });

    assert!(called == Bool::True);
    assert_eq!(*value, 99);
}

#[test]
fn test_entry_or_insert_with_not_called_when_occupied() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    tree.insert_with_guard(b"key", 50, &guard);

    let mut called = false;
    let value: ValuePtr<u64> = tree.entry_with_guard(b"key", &guard).or_insert_with(|| {
        called = true;
        99
    });

    assert!(!called); // Should NOT be called
    assert_eq!(*value, 50);
}

#[test]
fn test_entry_or_insert_with_key() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Insert based on key length
    let value: ValuePtr<u64> = tree
        .entry_with_guard(b"hello", &guard)
        .or_insert_with_key(|k: &[u8]| k.len() as u64);

    assert_eq!(*value, 5); // "hello".len() == 5
}

#[test]
fn test_entry_and_modify_occupied() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"counter", 10, &guard);

    // Modify existing value (ValuePtr<u64>: dereference twice - &ValuePtr -> ValuePtr -> u64)
    let value: ValuePtr<u64> = tree
        .entry_with_guard(b"counter", &guard)
        .and_modify(|v: &ValuePtr<u64>| **v + 5)
        .or_insert(0);

    assert_eq!(*value, 15);
}

#[test]
fn test_entry_and_modify_vacant() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // and_modify should be skipped for vacant, or_insert should run
    let value = tree
        .entry_with_guard(b"counter", &guard)
        .and_modify(|v: &ValuePtr<u64>| **v + 100) // Should not be called
        .or_insert(0);

    assert_eq!(*value, 0);
}

#[test]
fn test_entry_try_and_modify() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"counter", 10, &guard);

    // and_modify
    let result = tree
        .entry_with_guard(b"counter", &guard)
        .and_modify(|v: &ValuePtr<u64>| **v + 5);

    match result {
        Entry::Occupied(o) => assert_eq!(**o.get(), 15),

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }
}

#[test]
fn test_entry_occupied_get() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"key", 42, &guard);

    match tree.entry_with_guard(b"key", &guard) {
        Entry::Occupied(o) => {
            assert_eq!(**o.get(), 42);
            assert_eq!(o.key(), b"key");
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }
}

#[test]
fn test_entry_occupied_insert_returns_old() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    tree.insert_with_guard(b"key", 10, &guard);

    match tree.entry_with_guard(b"key", &guard) {
        Entry::Occupied(mut o) => {
            let old = o.insert(20);
            assert_eq!(*old.unwrap(), 10); // Returns Some(old)
            assert_eq!(**o.get(), 20); // Cached value updated
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }

    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 20);
}

#[test]
fn test_entry_occupied_try_insert() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    tree.insert_with_guard(b"key", 10, &guard);

    match tree.entry_with_guard(b"key", &guard) {
        Entry::Occupied(mut o) => {
            let result = o.insert(20);
            assert_eq!(*result.unwrap(), 10);
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }
}

#[test]
fn test_entry_occupied_remove_returns_actual_value() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    tree.insert_with_guard(b"key", 42, &guard);
    assert_eq!(tree.len(), 1);

    match tree.entry_with_guard(b"key", &guard) {
        Entry::Occupied(o) => {
            // Remove returns the actually removed value
            let value = o.remove();
            assert_eq!(*value.unwrap(), 42);
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }

    assert_eq!(tree.len(), 0);
    assert!(tree.get_with_guard(b"key", &guard).is_none());
}

#[test]
fn test_entry_occupied_try_remove() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"key", 42, &guard);

    match tree.entry_with_guard(b"key", &guard) {
        Entry::Occupied(o) => {
            let result: Result<Option<ValuePtr<u64>>, crate::RemoveError> = o.try_remove();
            assert!(result.is_ok());
            assert_eq!(*result.unwrap().unwrap(), 42);
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }
}

#[test]
fn test_entry_occupied_remove_entry() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"mykey", 123, &guard);

    match tree.entry_with_guard(b"mykey", &guard) {
        Entry::Occupied(o) => {
            let result = o.remove_entry();
            assert!(result.is_some());

            let (key, value) = result.unwrap();
            assert_eq!(key, b"mykey");
            assert_eq!(*value, 123);
        }

        Entry::Vacant(_) => panic!("Expected Occupied"),
    }
}

#[test]
fn test_entry_vacant_key() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    match tree.entry_with_guard(b"newkey", &guard) {
        Entry::Occupied(_) => panic!("Expected Vacant"),

        Entry::Vacant(v) => {
            assert_eq!(v.key(), b"newkey");
        }
    }
}

#[test]
fn test_entry_vacant_into_key() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    match tree.entry_with_guard(b"newkey", &guard) {
        Entry::Occupied(_) => panic!("Expected Vacant"),

        Entry::Vacant(v) => {
            let key: Vec<u8> = v.into_key();
            assert_eq!(key, b"newkey");
        }
    }
}

#[test]
fn test_entry_vacant_insert() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    match tree.entry_with_guard(b"newkey", &guard) {
        Entry::Occupied(_) => panic!("Expected Vacant"),

        Entry::Vacant(v) => {
            let value = v.insert(999);
            assert_eq!(*value, 999);
        }
    }

    assert_eq!(tree.len(), 1);
    assert_eq!(*tree.get_with_guard(b"newkey", &guard).unwrap(), 999);
}

#[test]
fn test_entry_vacant_try_insert() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    match tree.entry_with_guard(b"newkey", &guard) {
        Entry::Occupied(_) => panic!("Expected Vacant"),
        Entry::Vacant(v) => {
            let result = v.insert(777);
            assert_eq!(*result, 777);
        }
    }
}

#[test]
fn test_entry_insert_entry_from_vacant() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Start vacant, insert_entry returns OccupiedEntry
    let occupied = tree.entry_with_guard(b"key", &guard).insert_entry(42);

    assert_eq!(**occupied.get(), 42);
    assert_eq!(occupied.key(), b"key");

    // Verify in tree
    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 42);
}

#[test]
fn test_entry_insert_entry_from_occupied() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    tree.insert_with_guard(b"key", 10, &guard);

    // Start occupied, insert_entry replaces
    let occupied = tree.entry_with_guard(b"key", &guard).insert_entry(42);

    assert_eq!(**occupied.get(), 42);
    assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 42);
}

#[test]
fn test_entry_try_insert_entry() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    let result = tree.entry_with_guard(b"key", &guard).insert_entry(42);
    assert_eq!(**result.get(), 42);
}

#[test]
fn test_entry_get_on_entry() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Vacant entry - get returns None
    let entry = tree.entry_with_guard(b"key", &guard);
    assert!(entry.get().is_none());

    // Insert and check again
    tree.insert_with_guard(b"key", 42, &guard);
    let entry = tree.entry_with_guard(b"key", &guard);
    assert_eq!(**entry.get().unwrap(), 42);
}

#[test]
fn test_entry_key_method() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Vacant
    let entry = tree.entry_with_guard(b"vacant_key", &guard);
    assert_eq!(entry.key(), b"vacant_key");

    // Occupied
    tree.insert_with_guard(b"occupied_key", 1, &guard);
    let entry = tree.entry_with_guard(b"occupied_key", &guard);
    assert_eq!(entry.key(), b"occupied_key");
}

#[test]
fn test_entry_inline_variant() {
    use crate::MassTree15Inline;

    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    // Test or_insert (inline: value is u64 directly)
    let value = tree.entry_with_guard(b"key", &guard).or_insert(100);
    assert_eq!(value, 100);

    // Test and_modify (inline: v is &u64, dereference once)
    let value = tree
        .entry_with_guard(b"key", &guard)
        .and_modify(|v: &u64| *v + 50)
        .or_insert(0);
    assert_eq!(value, 150);
}

#[test]
fn test_entry_multiple_operations() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Simulate a counter pattern (ValuePtr<u64>: dereference twice)
    for _ in 0..10 {
        tree.entry_with_guard(b"counter", &guard)
            .and_modify(|v| **v + 1)
            .or_insert(1);
    }

    // After 10 iterations: 1 (initial) + 9 increments = 10
    assert_eq!(*tree.get_with_guard(b"counter", &guard).unwrap(), 10);
}

#[test]
fn test_entry_different_keys() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Insert multiple keys via entry API
    tree.entry_with_guard(b"a", &guard).or_insert(1);
    tree.entry_with_guard(b"b", &guard).or_insert(2);
    tree.entry_with_guard(b"c", &guard).or_insert(3);

    assert_eq!(tree.len(), 3);
    assert_eq!(*tree.get_with_guard(b"a", &guard).unwrap(), 1);
    assert_eq!(*tree.get_with_guard(b"b", &guard).unwrap(), 2);
    assert_eq!(*tree.get_with_guard(b"c", &guard).unwrap(), 3);
}

#[test]
fn test_entry_long_key() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Key longer than 8 bytes (triggers multi-layer path)
    let long_key: &'static [u8; 48] = b"this_is_a_very_long_key_that_exceeds_eight_bytes";

    let value: ValuePtr<u64> = tree.entry_with_guard(long_key, &guard).or_insert(42);
    assert_eq!(*value, 42);

    // Verify retrieval
    assert_eq!(*tree.get_with_guard(long_key, &guard).unwrap(), 42);

    // Modify (ValuePtr<u64>: dereference twice)
    let value: ValuePtr<u64> = tree
        .entry_with_guard(long_key, &guard)
        .and_modify(|v: &ValuePtr<u64>| **v * 2)
        .or_insert(0);
    assert_eq!(*value, 84);
}

#[test]
fn test_entry_temporary_key_with_reused_guard() {
    // Verify that the key can be a temporary slice while reusing guard
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // This pattern should work: reuse guard with temporary keys
    for i in 0u64..5 {
        let key: String = format!("key_{i}");
        tree.entry_with_guard(key.as_bytes(), &guard).or_insert(i);
    }

    assert_eq!(tree.len(), 5);
    assert_eq!(*tree.get_with_guard(b"key_0", &guard).unwrap(), 0);
    assert_eq!(*tree.get_with_guard(b"key_4", &guard).unwrap(), 4);
}

#[test]
fn test_entry_or_default() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // or_default on vacant - uses Default::default() = 0
    let value: ValuePtr<u64> = tree.entry_with_guard(b"counter", &guard).or_default();
    assert_eq!(*value, 0);

    // Modify and or_default on occupied - returns cached value
    tree.insert_with_guard(b"other", 42, &guard);
    let value: ValuePtr<u64> = tree.entry_with_guard(b"other", &guard).or_default();
    assert_eq!(*value, 42);
}

#[test]
fn test_entry_or_default_with_and_modify() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Counter pattern using or_default (ValuePtr<u64>: dereference twice)
    for _ in 0..5 {
        tree.entry_with_guard(b"counter", &guard)
            .and_modify(|v: &ValuePtr<u64>| **v + 1)
            .or_default();
    }

    // First iteration: or_default inserts 0
    // Iterations 2-5: and_modify increments (0 -> 1 -> 2 -> 3 -> 4)
    assert_eq!(*tree.get_with_guard(b"counter", &guard).unwrap(), 4);
}

#[test]
fn test_entry_or_try_insert_with_key() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Insert based on key length
    let result: ValuePtr<u64> = tree
        .entry_with_guard(b"hello", &guard)
        .or_insert_with_key(|k| k.len() as u64);

    assert_eq!(*result, 5); // "hello".len() == 5

    // On occupied - returns cached value, closure not called
    let result: ValuePtr<u64> = tree
        .entry_with_guard(b"hello", &guard)
        .or_insert_with_key(|_| 999);

    assert_eq!(*result, 5); // Original value, not 999
}

#[test]
fn test_entry_debug_impl() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard: LocalGuard<'_> = tree.guard();

    // Vacant entry debug
    let entry = tree.entry_with_guard(b"key", &guard);
    let debug_str: String = format!("{entry:?}");
    assert!(debug_str.contains("Vacant"));

    // Occupied entry debug
    tree.insert_with_guard(b"key", 42, &guard);
    let entry = tree.entry_with_guard(b"key", &guard);
    let debug_str: String = format!("{entry:?}");
    assert!(debug_str.contains("Occupied"));
    assert!(debug_str.contains("42"));
}
