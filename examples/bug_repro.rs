//! Minimal reproduction of the multi-layer key split bug (BugHunt2.md)
//!
//! Run with: cargo run --example bug_repro --features mimalloc
//! Debug:    cargo run --example bug_repro --features "mimalloc debug-routing"

use masstree::tree::MassTree24;

fn main() {
    println!("=== BugHunt2: Multi-Layer Key Split Bug ===\n");

    // All 25 keys from the failing proptest case
    let keys: Vec<Vec<u8>> = vec![
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0], // 0: len=9
        vec![
            13, 72, 140, 213, 52, 145, 160, 6, 73, 131, 220, 74, 36, 226, 54, 60, 131, 221, 235,
            179, 139, 191, 16, 73, 187, 30, 192, 36, 249, 225, 159, 38, 85, 3, 81,
        ], // 1: len=35
        vec![
            251, 182, 157, 215, 115, 184, 125, 60, 204, 204, 2, 82, 34, 70, 37, 169, 104, 239, 253,
            60, 27, 230, 212, 136, 253, 138, 42, 0, 182, 32, 103, 228, 123, 150, 156, 160, 180, 84,
            24, 47, 201, 129, 111, 142, 178, 171, 76, 85, 44, 40, 62, 167,
        ], // 2: len=52
        vec![
            222, 190, 189, 112, 147, 252, 19, 62, 129, 51, 203, 202, 250, 172, 17, 157, 72, 128,
            181, 208, 148, 15, 115, 131, 243, 100, 144, 53, 223, 222, 66, 232, 210, 230, 120, 163,
            136, 163, 136, 96, 128, 138, 234, 8, 196, 17,
        ], // 3: len=46 LOST
        vec![0, 0, 0, 0, 0, 0, 0, 2, 0], // 4: len=9
        vec![
            228, 175, 56, 201, 158, 9, 67, 59, 220, 220, 63, 245, 254, 225, 118, 236, 100, 13, 86,
            90, 135, 247, 235, 229, 159, 111, 41, 122, 30, 82, 146, 202, 191, 244, 106, 236, 202,
            155, 105, 231, 25, 145, 32, 1, 120, 230, 236, 90, 222, 2, 48, 41, 68, 229, 232, 89, 90,
            102, 140, 229, 79, 4, 20, 155,
        ], // 5: len=64
        vec![
            143, 155, 41, 168, 106, 205, 144, 198, 111, 49, 23, 185, 80, 185,
        ], // 6: len=14 LOST
        vec![
            131, 195, 253, 245, 214, 194, 105, 164, 173, 226, 44, 171, 11, 43, 151, 255, 249, 123,
            128, 69, 163,
        ], // 7: len=21
        vec![
            45, 56, 54, 33, 59, 148, 225, 48, 30, 118, 26, 109, 1, 246, 133, 96, 126, 126, 250,
            163, 163, 73, 230, 45, 196, 126, 240, 248, 89, 170, 61, 233, 248, 197, 210, 207, 180,
            85, 60, 40, 171, 243,
        ], // 8: len=42
        vec![232, 250, 21, 41, 2, 24, 175, 139, 44, 205], // 9: len=10
        vec![
            208, 34, 207, 181, 168, 133, 55, 172, 47, 178, 8, 113, 118, 86, 13, 210, 32, 118, 96,
            72, 135, 239, 16, 235, 34, 245,
        ], // 10: len=26 LOST
        vec![
            58, 62, 17, 138, 90, 216, 184, 194, 173, 156, 25, 141, 252, 189, 224, 176, 164, 174,
            51, 173, 105, 119, 54, 49, 140, 127,
        ], // 11: len=26
        vec![
            145, 5, 26, 71, 76, 66, 181, 217, 8, 9, 210, 225, 57, 198, 60, 147, 192, 149, 46, 105,
            152, 49, 106, 193, 157, 166, 144, 168, 238, 236, 144, 53, 226, 192, 201, 254, 160, 173,
            98, 98, 122, 117, 64, 21, 207, 103, 137, 93, 40, 235, 77, 161, 45, 4, 90, 190, 219, 48,
            239,
        ], // 12: len=59 LOST
        vec![0, 0, 0, 6, 133, 214, 188, 152, 130], // 13: len=9
        vec![
            88, 132, 145, 56, 5, 205, 236, 82, 70, 114, 144, 196, 99, 85, 209, 206, 190, 29, 100,
            203, 13, 121, 227, 41, 223, 73, 30, 187, 41, 219, 77, 42, 81, 14, 116, 209, 247,
        ], // 14: len=37
        vec![
            217, 143, 4, 30, 128, 99, 114, 153, 142, 229, 150, 182, 3, 10, 251, 51, 85, 67, 135,
            129, 125, 206, 179, 132, 71,
        ], // 15: len=25 LOST
        vec![0, 0, 0, 0, 0, 0, 0, 1, 0], // 16: len=9
        vec![
            81, 69, 95, 180, 76, 45, 203, 249, 193, 189, 162, 179, 184, 148, 158,
        ], // 17: len=15
        vec![
            195, 83, 16, 103, 243, 16, 192, 175, 148, 7, 199, 192, 70, 253, 240, 151, 17, 155, 19,
            176, 127,
        ], // 18: len=21 LOST
        vec![
            238, 141, 197, 154, 179, 82, 124, 207, 208, 197, 166, 235, 95, 43, 218, 15, 108, 204,
            240, 22, 70, 157, 12, 47, 138, 165, 232, 233, 6, 3, 129, 240, 176, 89, 38, 175, 74,
            150, 131, 5, 219, 81, 213, 198, 39, 212, 37, 110, 73, 170, 79, 121, 82, 96, 211, 212,
        ], // 19: len=56
        vec![
            58, 123, 246, 146, 50, 70, 157, 173, 224, 139, 72, 147, 37, 164, 214, 139, 31, 14, 30,
            69, 30, 67, 222, 74, 19,
        ], // 20: len=25
        vec![
            225, 104, 213, 153, 153, 160, 89, 86, 52, 204, 200, 232, 118, 31, 248, 229, 88, 44, 88,
            179, 103, 180, 222, 87, 79, 161, 46, 7, 208, 247, 78, 239, 47, 3, 50, 73, 20, 102, 74,
            72, 161, 244, 27, 182, 158,
        ], // 21: len=45 LOST
        vec![
            211, 159, 169, 193, 218, 14, 27, 18, 44, 35, 27, 137, 79, 56, 78, 226, 82, 42, 157,
            165, 218, 13, 226, 62, 43, 60, 200, 97, 62, 183, 86, 63, 1, 97, 150, 114, 160, 129,
            145, 188, 255, 87, 247, 14, 75, 228, 96, 215, 97, 207, 97, 242, 128, 32, 240, 239, 136,
        ], // 22: len=57 LOST
        vec![
            103, 89, 78, 255, 155, 168, 99, 177, 201, 34, 118, 244, 121, 96, 196, 190, 223, 171,
            78, 33, 160, 182, 220, 208, 112,
        ], // 23: len=25
        vec![
            161, 22, 157, 11, 164, 234, 95, 191, 6, 89, 228, 180, 161, 156, 94, 199, 12, 210, 42,
            164, 83, 77, 21, 158, 82, 107, 136, 55, 118, 118, 4, 33, 223, 46, 208, 37, 154, 21, 58,
            0, 84, 64, 158, 148, 255, 134, 158, 89, 73, 224, 215, 238, 2, 30, 98, 62, 231, 220,
            133, 90, 224, 61, 118, 16,
        ], // 24: len=64 TRIGGER
    ];

    let lost_indices: [usize; 8] = [3, 6, 10, 12, 15, 18, 21, 22];

    let tree: MassTree24<u64> = MassTree24::new();

    // Insert first 24 keys (no split)
    for (i, key) in keys.iter().take(24).enumerate() {
        tree.insert(key, i as u64).unwrap();
    }

    println!("After inserting 24 keys (before split):");
    for &idx in &lost_indices {
        let key = &keys[idx];
        let found = tree.get(key).is_some();
        println!(
            "  Key {} (len={}): {}",
            idx,
            key.len(),
            if found { "OK" } else { "MISSING" }
        );
    }

    // Insert 25th key (triggers split)
    println!("\nInserting key 24 (len=64) - triggers split...\n");
    tree.insert(&keys[24], 24).unwrap();

    println!("After split:");
    let mut lost_count = 0;
    for &idx in &lost_indices {
        let key = &keys[idx];
        let found = tree.get(key).is_some();
        if !found {
            lost_count += 1;
        }
        println!(
            "  Key {} (len={}): {}",
            idx,
            key.len(),
            if found { "OK" } else { "LOST!" }
        );
    }

    // Verify via iteration
    println!("\nIterating to check physical presence:");
    let guard = tree.guard();
    let mut iter_count = 0;
    let mut wrong_len_count = 0;
    for entry in tree.iter(&guard) {
        iter_count += 1;
        let key = entry.key();
        if key.len() == 8 {
            // Check if this should have been longer
            let ikey = u64::from_be_bytes(key[..8].try_into().unwrap());
            for &idx in &lost_indices {
                let expected_ikey = u64::from_be_bytes(keys[idx][..8].try_into().unwrap());
                if ikey == expected_ikey && keys[idx].len() != 8 {
                    println!(
                        "  CORRUPTED: ikey={:016X} shows len=8, expected len={}",
                        ikey,
                        keys[idx].len()
                    );
                    wrong_len_count += 1;
                }
            }
        }
    }

    println!("\n=== SUMMARY ===");
    println!("Keys lost via get(): {}", lost_count);
    println!("Keys with wrong len in iteration: {}", wrong_len_count);
    println!(
        "Total via iteration: {} (tree.len()={})",
        iter_count,
        tree.len()
    );

    if lost_count > 0 {
        println!("\nBUG CONFIRMED: keylenx corrupted during split!");
        println!("See BugHunt2.md for details.");
    } else {
        println!("\nNo bug: All keys accessible.");
    }
}
