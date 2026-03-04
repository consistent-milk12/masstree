//! Tree visualization example.
//!
//! Demonstrates the `print_tree()` debug visualization for inspecting
//! tree structure: leaf nodes, internodes, permutations, layers, and
//! sibling chains.
//!
//! Run with: `cargo run --example print_tree --features debug-print`

use masstree::{MassTree, MassTree15};

fn main() {
    // =========================================================================
    //  Empty tree
    // =========================================================================
    let tree: MassTree<u64> = MassTree::new();
    println!("=== Empty Tree ===\n");
    println!("{tree}");

    // =========================================================================
    //  Single leaf (a few short keys)
    // =========================================================================
    tree.insert(b"apple", 1);
    tree.insert(b"banana", 2);
    tree.insert(b"cherry", 3);
    tree.insert(b"date", 4);
    tree.insert(b"elder", 5);

    println!("Single Leaf (5 keys)\n");
    println!("{tree}");

    // =========================================================================
    //  Trigger layers with long keys sharing a prefix
    // =========================================================================
    let tree2: MassTree<u64> = MassTree::new();
    tree2.insert(b"hello_world_aaa", 100);
    tree2.insert(b"hello_world_bbb", 200);
    tree2.insert(b"hello_world_ccc", 300);
    tree2.insert(b"short", 999);

    println!("Layers (long shared-prefix keys)\n");
    println!("{tree2}");

    // =========================================================================
    //  Force internode splits with many keys
    // =========================================================================
    let tree3: MassTree<u64> = MassTree::new();

    for i in 0u64..40 {
        let key: String = format!("key_{i:04}");
        tree3.insert(key.as_bytes(), i);
    }

    println!("Internodes (40 sequential keys)\n");
    println!("{tree3}");

    // =========================================================================
    //  Two levels of internodes with MassTree15<String> (BoxPolicy)
    // =========================================================================
    let tree4: MassTree15<String> = MassTree15::new();

    for i in 0u64..300 {
        let key: String = format!("k{i:05}");
        let val: String = format!("val_{i}");
        tree4.insert(key.as_bytes(), val);
    }

    println!("Two internode levels, MassTree15<String> (300 keys)\n");
    println!("{tree4}");
}
