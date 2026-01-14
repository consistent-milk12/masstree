//! Print sizes of main Masstree data structures.

use std::mem::size_of;

fn main() {
    println!("=== Masstree Structure Sizes ===\n");

    // Leaf nodes
    println!("Leaf Nodes:");
    println!(
        "  LeafNode15TrueInline<u64>: {} bytes",
        size_of::<masstree::inline::leaf15_true::LeafNode15TrueInline<u64>>()
    );
    println!(
        "  LeafNode15<LeafValue<u64>>: {} bytes",
        size_of::<masstree::leaf15::LeafNode15<masstree::LeafValue<u64>>>()
    );
    println!(
        "  LeafNode24<LeafValue<u64>>: {} bytes",
        size_of::<masstree::leaf24::LeafNode24<masstree::LeafValue<u64>>>()
    );

    // Internode
    println!("\nInternode:");
    println!(
        "  InternodeNode: {} bytes",
        size_of::<masstree::internode::InternodeNode>()
    );

    // Suffix storage
    println!("\nSuffix Storage:");
    println!(
        "  SuffixBag<15>: {} bytes",
        size_of::<masstree::suffix::SuffixBag<15>>()
    );
    println!(
        "  SuffixBag<24>: {} bytes",
        size_of::<masstree::suffix::SuffixBag<24>>()
    );

    // Tree wrapper
    println!("\nTree Wrapper:");
    println!(
        "  MassTree15Inline<u64>: {} bytes",
        size_of::<masstree::MassTree15Inline<u64>>()
    );
    println!(
        "  MassTree15<u64>: {} bytes",
        size_of::<masstree::MassTree15<u64>>()
    );

    // Memory estimate for benchmark
    println!("\n=== Memory Estimates ===\n");

    let leaf_size = size_of::<masstree::inline::leaf15_true::LeafNode15TrueInline<u64>>();
    let internode_size = size_of::<masstree::internode::InternodeNode>();

    println!("For 200M keys (12 threads × ~17M keys each):");
    let keys = 200_000_000u64;
    let keys_per_leaf = 12u64; // typical fill factor
    let leaves = keys / keys_per_leaf;
    let leaf_mem = leaves * leaf_size as u64;

    // Rough estimate: 1 internode per 15 leaves at bottom level, then geometric series
    let internodes = leaves / 15 + leaves / 225 + leaves / 3375;
    let internode_mem = internodes * internode_size as u64;

    println!("  Estimated leaves: {}", leaves);
    println!("  Leaf memory: {:.2} GB", leaf_mem as f64 / 1e9);
    println!("  Estimated internodes: {}", internodes);
    println!("  Internode memory: {:.2} GB", internode_mem as f64 / 1e9);
    println!(
        "  Total tree memory: {:.2} GB",
        (leaf_mem + internode_mem) as f64 / 1e9
    );
}
