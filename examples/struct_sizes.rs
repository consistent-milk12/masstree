//! Print struct sizes and cache line analysis.
//!
//! Run with: cargo run --example struct_sizes

use std::mem::size_of;
use masstree::{
    MassTree15, MassTree24,
    Permuter15, Permuter24, nodeversion::NodeVersion,
    InlineSuffixBag, internode::InternodeNode, value::LeafValue,
};

fn main() {
    println!("=== MassTree Struct Size Analysis ===\n");

    // Trees (for reference)
    println!("MassTree15<u64>: {} bytes", size_of::<MassTree15<u64>>());
    println!("MassTree24<u64>: {} bytes", size_of::<MassTree24<u64>>());
    println!();

    // NodeVersion
    println!("NodeVersion: {} bytes", size_of::<NodeVersion>());
    println!();

    // InternodeNode
    println!("InternodeNode<LeafValue<u64>>: {} bytes ({} cache lines)",
        size_of::<InternodeNode<LeafValue<u64>>>(),
        size_of::<InternodeNode<LeafValue<u64>>>() / 64);
    println!("InternodeNode alignment: {} bytes", std::mem::align_of::<InternodeNode<LeafValue<u64>>>());
    println!();

    // Permuters
    println!("Permuter15: {} bytes (u64)", size_of::<Permuter15>());
    println!("Permuter24: {} bytes (u128)", size_of::<Permuter24>());
    println!();

    // InlineSuffixBag
    println!("InlineSuffixBag<15, 256>: {} bytes", size_of::<InlineSuffixBag<15, 256>>());
    println!();

    // Use type_name trick to get sizes without needing the slot type
    println!("=== LeafNode15 Field Layout (calculated) ===\n");

    let mut offset: usize = 0;
    let cache_line = 64;

    // Cache line 0
    println!("Cache Line 0 (bytes 0-63): Version + Metadata");
    println!("  [{:3}..{:3}] version: NodeVersion ({} bytes)", offset, offset + 4, size_of::<NodeVersion>());
    offset += 4;
    println!("  [{:3}..{:3}] modstate: AtomicU8 (1 byte)", offset, offset + 1);
    offset += 1;
    println!("  [{:3}..{:3}] _pad0: [u8; 55] (55 bytes)", offset, offset + 55);
    offset += 55;

    // Alignment check
    let cl1_start = ((offset + cache_line - 1) / cache_line) * cache_line;
    if cl1_start > offset {
        println!("  [{:3}..{:3}] implicit padding ({} bytes)", offset, cl1_start, cl1_start - offset);
        offset = cl1_start;
    }
    println!();

    // Cache line 1
    println!("Cache Line 1 (bytes 64-127): Permutation (CAS-heavy)");
    println!("  [{:3}..{:3}] permutation: AtomicU64 (8 bytes)", offset, offset + 8);
    offset += 8;
    println!("  [{:3}..{:3}] _pad1: [u8; 56] (56 bytes)", offset, offset + 56);
    offset += 56;
    println!();

    // Cache lines 2-3: ikeys
    println!("Cache Lines 2-3 (bytes 128-247): Keys");
    println!("  [{:3}..{:3}] ikey0: [AtomicU64; 15] (120 bytes)", offset, offset + 120);
    offset += 120;
    println!();

    // keylenx + alignment
    println!("Cache Line 4 (bytes 248-311): Key metadata + values start");
    println!("  [{:3}..{:3}] keylenx: [AtomicU8; 15] (15 bytes)", offset, offset + 15);
    offset += 15;

    // Alignment for AtomicPtr (8-byte aligned)
    let aligned = (offset + 7) & !7;
    if aligned > offset {
        println!("  [{:3}..{:3}] implicit padding ({} bytes)", offset, aligned, aligned - offset);
        offset = aligned;
    }
    println!();

    // leaf_values
    println!("Cache Lines 4-6 (bytes {}..{}): Values", offset, offset + 120);
    println!("  [{:3}..{:3}] leaf_values: [AtomicPtr<u8>; 15] (120 bytes)", offset, offset + 120);
    offset += 120;
    println!();

    // inline_ksuf: InlineSuffixBag<15, 256>
    // slots: [InlineSlotMeta; 15] where InlineSlotMeta = { offset: u16, len: u16 } = 4 bytes
    // size: u16 = 2 bytes
    // data: [u8; 256] = 256 bytes
    // Total: 15 * 4 + 2 + 256 = 318 bytes (but may have alignment)
    let inline_slots = 15 * 4; // 60 bytes
    let inline_size = 2;
    let inline_data = 256;
    let inline_total = inline_slots + inline_size + inline_data;
    println!("Cache Lines 7-11: Inline suffix storage");
    println!("  [{:3}..{:3}] inline_ksuf: InlineSuffixBag<15, 256>", offset, offset + inline_total);
    println!("             - slots: [InlineSlotMeta; 15] ({} bytes)", inline_slots);
    println!("             - size: u16 ({} bytes)", inline_size);
    println!("             - data: [u8; 256] ({} bytes)", inline_data);
    println!("             - total: {} bytes", inline_total);
    offset += inline_total;
    println!();

    // Alignment for AtomicPtr
    let aligned = (offset + 7) & !7;
    if aligned > offset {
        println!("  [{:3}..{:3}] implicit padding ({} bytes)", offset, aligned, aligned - offset);
        offset = aligned;
    }

    // Navigation pointers
    println!("Cache Lines 11-12: Navigation pointers");
    println!("  [{:3}..{:3}] external_ksuf: AtomicPtr (8 bytes)", offset, offset + 8);
    offset += 8;
    println!("  [{:3}..{:3}] next: AtomicPtr (8 bytes)", offset, offset + 8);
    offset += 8;
    println!("  [{:3}..{:3}] prev: AtomicPtr (8 bytes)", offset, offset + 8);
    offset += 8;
    println!("  [{:3}..{:3}] parent: AtomicPtr (8 bytes)", offset, offset + 8);
    offset += 8;
    println!();

    // Final alignment to 64 bytes
    let final_aligned = ((offset + cache_line - 1) / cache_line) * cache_line;
    if final_aligned > offset {
        println!("  [{:3}..{:3}] tail padding ({} bytes)", offset, final_aligned, final_aligned - offset);
        offset = final_aligned;
    }

    println!("=== Summary ===\n");
    println!("Calculated LeafNode15 size: {} bytes", offset);
    println!("Cache lines used: {:.1}", offset as f64 / 64.0);
    println!();

    println!("=== Hot Path Cache Line Access ===\n");
    println!("get() touches minimum 4 cache lines:");
    println!("  CL 0: version (OCC validation)");
    println!("  CL 1: permutation (slot ordering)");
    println!("  CL 2-3: ikey0[0..15] (key comparison)");
    println!("  CL 4-6: leaf_values (on match)");
    println!();

    println!("=== Congee Comparison ===\n");
    println!("Congee Node4:   64 bytes (1 CL) - minimal, fits entirely in L1");
    println!("Congee Node16: ~192 bytes (3 CL)");
    println!("Congee Node48: ~656 bytes (10 CL)");
    println!();
    println!("LeafNode15:    ~{} bytes ({} CL) - trie-of-B+trees design", offset, offset / 64);
    println!();
    println!("Note: Masstree stores full 8-byte keys + values per slot,");
    println!("while ART stores single-byte keys with value at leaves.");
}
