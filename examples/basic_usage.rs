//! Basic usage examples for [`MassTree15`] and [`MassTree15Inline`].
//!
//! This example demonstrates:
//! - Creating a tree
//! - Inserting key-value pairs
//! - Point lookups
//! - Range scans
//! - Prefix scans
//! - Iteration
//! - Removal
//!
//! Run with: `cargo run --example basic_usage`

use masstree::{MassTree15, MassTree15Inline, RangeBound};

#[expect(
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]
fn main() {
    println!("=== MassTree Basic Usage Examples ===\n");

    // =========================================================================
    // Example 1: MassTree15 with Box-based storage
    // =========================================================================
    println!("--- Example 1: MassTree15 (Box-based storage) ---\n");

    // Create a new tree that stores u64 values managed by BoxPolicy
    let tree: MassTree15<u64> = MassTree15::new();

    // Get a guard for operations (ties to an epoch for memory safety)
    let guard = tree.guard();

    // Insert some key-value pairs
    // Keys are byte slices, values are your type
    // insert_with_guard returns Option<ValuePtr<V>> - the old value if key existed
    let old = tree.insert_with_guard(b"users/alice", 100, &guard);
    println!("First insert of users/alice: old={old:?}"); // None - no previous value

    let old = tree.insert_with_guard(b"users/bob", 200, &guard);
    assert!(old.is_none()); // New key, no old value

    // Insert is infallible - always succeeds
    tree.insert_with_guard(b"users/charlie", 300, &guard);
    tree.insert_with_guard(b"posts/1", 1001, &guard);
    tree.insert_with_guard(b"posts/2", 1002, &guard);

    println!("Inserted 5 entries");
    println!("Tree size: {}\n", tree.len());

    // Point lookup - returns Option<&V> (for Box-based storage only)
    if let Some(value) = tree.get_ref(b"users/alice", &guard) {
        println!("get_ref(users/alice) = {value}");
    }

    // Point lookup - returns Option<ValuePtr<V>>
    if let Some(value) = tree.get_with_guard(b"users/bob", &guard) {
        println!("get_with_guard(users/bob) = {value}");
    }

    println!();

    // =========================================================================
    // Example 2: Range Scans
    // =========================================================================
    println!("--- Example 2: Range Scans ---\n");

    // Scan all entries with keys starting with "users/"
    println!("Prefix scan for 'users/':");
    tree.scan_prefix(
        b"users/",
        |key, value| {
            println!("  {:?} -> {}", String::from_utf8_lossy(key), value);
            true // continue scanning
        },
        &guard,
    );

    println!();

    // Scan a range using RangeBound (lexicographic ordering)
    println!("Range scan from 'posts/' to 'users/':");
    tree.scan_ref(
        RangeBound::Included(b"posts/".as_slice()),
        RangeBound::Excluded(b"users/".as_slice()),
        |key, value| {
            println!("  {:?} -> {}", String::from_utf8_lossy(key), value);
            true
        },
        &guard,
    );

    println!();

    // =========================================================================
    // Example 3: Iterator-based access
    // =========================================================================
    println!("--- Example 3: Iterator-based Access ---\n");

    // Full iteration
    println!("All entries via iter():");
    for entry in tree.iter(&guard) {
        println!(
            "  {:?} -> {:?}",
            String::from_utf8_lossy(entry.key()),
            entry.value()
        );
    }

    println!();

    // Range iteration with bounds
    println!("Range iteration (posts/ to users/ exclusive):");
    for entry in tree.range(
        RangeBound::Included(b"posts/".as_slice()),
        RangeBound::Excluded(b"users/".as_slice()),
        &guard,
    ) {
        println!(
            "  {:?} -> {:?}",
            String::from_utf8_lossy(entry.key()),
            entry.value()
        );
    }

    println!();

    // =========================================================================
    // Example 4: Updates and Removals
    // =========================================================================
    println!("--- Example 4: Updates and Removals ---\n");

    // Update existing key - insert_with_guard returns the OLD value when key exists
    match tree.insert_with_guard(b"users/alice", 150, &guard) {
        Some(old_value) => println!("Updated users/alice: old={old_value}, new=150"),
        None => println!("Inserted new key users/alice"),
    }

    if let Some(new_val) = tree.get_ref(b"users/alice", &guard) {
        println!("Verified new value: {new_val}");
    }

    // Remove a key - returns Result<Option<ValuePtr<V>>, RemoveError>
    // The Result handles concurrent modification; Option indicates if key existed
    match tree.remove_with_guard(b"users/bob", &guard) {
        Ok(Some(removed)) => println!("Removed users/bob: value was {removed}"),
        Ok(None) => println!("users/bob not found (nothing to remove)"),
        Err(e) => println!("Remove encountered concurrent modification: {e:?}"),
    }
    println!("Tree size after removal: {}", tree.len());

    // Verify removal - should return None
    let not_found = tree.get_ref(b"users/bob", &guard);
    assert!(not_found.is_none());
    println!("users/bob after removal: None (as expected)");

    println!();

    // =========================================================================
    // Example 5: MassTree15Inline for Copy types (no Box overhead)
    // =========================================================================
    println!("--- Example 5: MassTree15Inline (Copy types, no Box) ---\n");

    // For Copy types, use Inline variants to avoid Box allocation
    let inline_tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = inline_tree.guard();

    // Insert operations work the same way (returns old value or None for new keys)
    inline_tree.insert_with_guard(b"counter/a", 1, &guard);
    inline_tree.insert_with_guard(b"counter/b", 2, &guard);
    inline_tree.insert_with_guard(b"counter/c", 3, &guard);

    // get_with_guard returns the value directly (no Box) for Inline variants
    if let Some(value) = inline_tree.get_with_guard(b"counter/a", &guard) {
        println!("Inline get: counter/a = {value}");
    }

    // Iteration works the same way
    println!("All inline entries:");
    for entry in inline_tree.iter(&guard) {
        println!(
            "  {:?} -> {:?}",
            String::from_utf8_lossy(entry.key()),
            entry.value()
        );
    }

    println!();

    // =========================================================================
    // Example 6: Auto-guard API (simpler but slightly more overhead)
    // =========================================================================
    println!("--- Example 6: Auto-guard API ---\n");

    let simple_tree: MassTree15<String> = MassTree15::new();

    // These methods create guards internally - simpler but more overhead per call
    simple_tree.insert(b"greeting", "Hello, World!".to_string());
    simple_tree.insert(b"farewell", "Goodbye!".to_string());

    // get() returns Option<ValuePtr<V>>
    if let Some(greeting) = simple_tree.get(b"greeting") {
        println!("Auto-guard get: {greeting}");
    }

    println!("Simple tree size: {}", simple_tree.len());
    println!("Is empty: {}", simple_tree.is_empty());

    // Remove with auto-guard - returns Result<Option<ValuePtr<V>>, RemoveError>
    if let Ok(Some(_)) = simple_tree.remove(b"farewell") {
        println!("Removed 'farewell'");
    }
    println!("After removal: {}", simple_tree.len());

    println!();

    // =========================================================================
    // Example 7: Collecting results
    // =========================================================================
    println!("--- Example 7: Collecting Scan Results ---\n");

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    for i in 0..10u64 {
        let key = format!("item/{i:03}");
        tree.insert_with_guard(key.as_bytes(), i * 10, &guard);
    }

    // Collect all entries into a Vec
    let entries = tree.collect_entries(&guard);
    println!("Collected {} entries:", entries.len());
    for entry in &entries {
        println!(
            "  {:?} -> {:?}",
            String::from_utf8_lossy(&entry.key),
            entry.value
        );
    }

    // Collect just keys
    let keys = tree.collect_keys(&guard);
    println!("\nCollected {} keys", keys.len());

    println!();

    // =========================================================================
    // Example 8: Working with complex keys
    // =========================================================================
    println!("--- Example 8: Complex Keys ---\n");

    let tree: MassTree15Inline<u32> = MassTree15Inline::new();
    let guard = tree.guard();

    // Keys can be any byte slice - great for hierarchical data
    // URLs, file paths, UUIDs all work well

    // URL-like keys
    tree.insert_with_guard(b"/api/v1/users/123", 1, &guard);
    tree.insert_with_guard(b"/api/v1/users/456", 2, &guard);
    tree.insert_with_guard(b"/api/v1/posts/789", 3, &guard);
    tree.insert_with_guard(b"/api/v2/users/123", 4, &guard);

    println!("All v1 users:");
    tree.scan_prefix(
        b"/api/v1/users/",
        |key, value| {
            println!("  {} -> {}", String::from_utf8_lossy(key), value);
            true
        },
        &guard,
    );

    // Integer keys (use big-endian for correct ordering)
    let int_tree: MassTree15Inline<u32> = MassTree15Inline::new();
    let guard = int_tree.guard();

    for i in [100u64, 50, 200, 75, 150] {
        let key = i.to_be_bytes();
        int_tree.insert_with_guard(&key, i as u32, &guard);
    }

    println!("\nInteger keys in sorted order:");
    for entry in int_tree.iter(&guard) {
        let key_bytes: [u8; 8] = entry.key().try_into().unwrap();
        let key_int = u64::from_be_bytes(key_bytes);
        println!("  {key_int} -> {:?}", entry.value());
    }

    println!("\n=== Examples Complete ===");
}
