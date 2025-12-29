//! Simple MassTree usage example.

use masstree::MassTree;

fn main() {
    // Create a new MassTree with String values
    let tree: MassTree<String> = MassTree::new();

    // Insert "key1" -> "Hello World!"
    let _ = tree.insert(b"key1", "Hello World!".to_string());

    // Get the value back
    if let Some(value) = tree.get(b"key1") {
        println!("key1 = {value}");
    }

    // Try getting a non-existent key
    if tree.get(b"key2").is_none() {
        println!("key2 not found");
    }
}
