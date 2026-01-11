//! Investigation of tree bug where keys disappear after insertion at scale
//!
//! Findings from mttest parity work:
//! - ~15 out of 1,000,000 keys go missing (~0.0015%)
//! - All missing keys are 8-byte decimal strings
//! - Keys verify immediately after insertion but are missing later
//! - Suggests a bug in split handling or suffix management

use masstree::MassTree15;

fn main() {
    println!("=== Tree Bug Investigation (MassTree15 - non-inline) ===\n");

    // Test with debug routing enabled
    test_routing_trace();
}

/// Test with routing trace enabled (requires --features debug-routing)
fn test_routing_trace() {
    println!("--- Routing Trace Analysis ---");
    println!(
        "(Run with: cargo run --release --features debug-routing --example tree_bug_investigation 2>&1 | tee trace.log)"
    );
    println!();

    let target = b"28631012";
    let target_ikey = u64::from_be_bytes(*target);

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    let mut seed: u32 = 31949;

    println!("Target key: {:?}", std::str::from_utf8(target).unwrap());
    println!("Target ikey: {target_ikey:016x}");
    println!();

    // Insert 132399 keys (one before the triggering insert)
    println!("Inserting 132399 keys...");
    for _ in 0..132_399 {
        let x = kv_rand(&mut seed);
        let key = to_decimal_bytes(x);
        let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
    }

    // Insert the key just before the trigger
    let pre_trigger_x = kv_rand(&mut seed);
    let pre_trigger_key = to_decimal_bytes(pre_trigger_x);
    let _ = tree.insert_with_guard(&pre_trigger_key, u64::from(pre_trigger_x) + 1, &guard);

    println!("After 132400 inserts:");
    println!(
        "  Pre-trigger key: {} -> {:?}",
        pre_trigger_x,
        std::str::from_utf8(&pre_trigger_key).unwrap()
    );

    // Check target before corrupting insert
    println!();
    println!("=== LOOKUP BEFORE CORRUPTING INSERT ===");
    let before_result = tree.get_with_guard(target, &guard);
    println!("Target found: {}", before_result.is_some());

    // Get the corrupting key
    let trigger_x = kv_rand(&mut seed);
    let trigger_key = to_decimal_bytes(trigger_x);
    let trigger_ikey = if trigger_key.len() >= 8 {
        u64::from_be_bytes(trigger_key[..8].try_into().unwrap())
    } else {
        let mut buf = [0u8; 8];
        buf[..trigger_key.len()].copy_from_slice(&trigger_key);
        u64::from_be_bytes(buf)
    };

    println!();
    println!("=== INSERTING CORRUPTING KEY ===");
    println!(
        "Key: {} -> {:?} (len={})",
        trigger_x,
        std::str::from_utf8(&trigger_key).unwrap(),
        trigger_key.len()
    );
    println!("Trigger ikey: {trigger_ikey:016x}");

    // Insert the corrupting key - this will produce split trace output
    let _ = tree.insert_with_guard(&trigger_key, u64::from(trigger_x) + 1, &guard);

    // Check target after corrupting insert
    println!();
    println!("=== LOOKUP AFTER CORRUPTING INSERT ===");
    let after_result = tree.get_with_guard(target, &guard);
    println!("Target found: {}", after_result.is_some());

    // Verify via iteration
    let target_in_iter = tree.iter(&guard).any(|e| e.key == target);
    println!("Target in iteration: {target_in_iter}");

    // Summary
    println!();
    println!("=== SUMMARY ===");
    if before_result.is_some() && after_result.is_none() && target_in_iter {
        println!("BUG CONFIRMED: Key found before insert, missing after, but still in iteration");
        println!("The routing path above shows where the lookup diverges.");
    } else if after_result.is_some() {
        println!("No bug: Key still accessible after insert");
    } else {
        println!(
            "Unexpected state: before={:?} after={:?} iter={}",
            before_result.is_some(),
            after_result.is_some(),
            target_in_iter
        );
    }
}

fn test_range_scan_check() {
    println!("--- Testing if key is physically present but unreachable ---");

    let target = b"28631012";
    let target_ikey = u64::from_be_bytes(*target);

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    let mut seed: u32 = 31949;

    // Insert up to the corruption point
    for _ in 0..132_400 {
        let x = kv_rand(&mut seed);
        let key = to_decimal_bytes(x);
        let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
    }

    println!("  Before corrupting insert:");
    println!(
        "    Direct get: {}",
        tree.get_with_guard(target, &guard).is_some()
    );

    // Count keys via full iteration
    let count_before: usize = tree.iter(&guard).count();
    let target_in_iter_before = tree.iter(&guard).any(|e| e.key == target);
    println!("    Total keys via iter: {count_before}");
    println!("    Target in iter: {}", target_in_iter_before);

    // Now insert the corrupting key
    let x = kv_rand(&mut seed);
    let key = to_decimal_bytes(x);
    let trigger_ikey = if key.len() >= 8 {
        u64::from_be_bytes(key[..8].try_into().unwrap())
    } else {
        let mut buf = [0u8; 8];
        buf[..key.len()].copy_from_slice(&key);
        u64::from_be_bytes(buf)
    };
    println!(
        "\n  Inserting corrupting key: {} -> {:?}",
        x,
        std::str::from_utf8(&key).unwrap()
    );
    println!("    Target ikey:  {:016x}", target_ikey);
    println!("    Trigger ikey: {:016x}", trigger_ikey);
    println!("    Target < Trigger: {}", target_ikey < trigger_ikey);
    let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);

    println!("\n  After corrupting insert:");
    println!(
        "    Direct get: {}",
        tree.get_with_guard(target, &guard).is_some()
    );

    // Count keys via full iteration again
    let count_after: usize = tree.iter(&guard).count();
    let target_in_iter_after = tree.iter(&guard).any(|e| e.key == target);
    println!(
        "    Total keys via iter: {} (expected {}, delta={})",
        count_after,
        count_before + 1,
        count_after as i64 - (count_before + 1) as i64
    );
    println!("    Target in iter: {}", target_in_iter_after);

    if !target_in_iter_after {
        println!("\n  *** KEY IS COMPLETELY UNREACHABLE (not even in full iteration) ***");
        println!("      This confirms the leaf containing the key is orphaned.");
    } else {
        println!("\n  *** KEY IS IN ITERATION BUT NOT IN DIRECT GET ***");
        println!("      Internode routing is incorrect. The separator key from a");
        println!("      split is causing the lookup to go to the wrong subtree.");

        // Find keys near the target in iteration order
        println!("\n  Keys near target in iteration order:");
        let mut found_target = false;
        let mut prev_keys: Vec<String> = Vec::new();
        for entry in tree.iter(&guard) {
            let k = String::from_utf8_lossy(&entry.key).to_string();
            if entry.key == target {
                found_target = true;
                println!("    Previous 3:");
                for pk in prev_keys.iter().rev().take(3).rev() {
                    println!("      {}", pk);
                }
                println!("    >>> {} <<< TARGET", k);
            } else if found_target {
                println!("      {}", k);
                // Print 3 keys after
                static mut AFTER_COUNT: usize = 0;
                unsafe {
                    AFTER_COUNT += 1;
                    if AFTER_COUNT >= 3 {
                        break;
                    }
                }
            }
            prev_keys.push(k);
            if prev_keys.len() > 5 {
                prev_keys.remove(0);
            }
        }
    }
}

fn test_exact_repro() {
    println!("--- Exact reproduction with debugging ---");

    let target = b"28631012";

    // Replay to exact point before corruption
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    let mut seed: u32 = 31949;

    // Insert up to 132395 keys first (5 before trigger)
    for _ in 0..132395 {
        let x = kv_rand(&mut seed);
        let key = to_decimal_bytes(x);
        let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
    }

    println!(
        "  After 132395 inserts, target present: {}",
        tree.get_with_guard(target, &guard).is_some()
    );

    // Now insert one at a time and check after each
    for i in 132395..132402 {
        let x = kv_rand(&mut seed);
        let key = to_decimal_bytes(x);
        let target_before = tree.get_with_guard(target, &guard).is_some();
        let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
        let target_after = tree.get_with_guard(target, &guard).is_some();

        let key_str = std::str::from_utf8(&key).unwrap();
        if target_before == target_after {
            println!(
                "  Insert #{}: {} (len={}) - target OK",
                i,
                key_str,
                key.len()
            );
        } else {
            println!("  *** INSERT #{i} CORRUPTED TARGET ***");
            println!(
                "      Key inserted: {} -> {:?} (len={})",
                x,
                key_str,
                key.len()
            );
            println!(
                "      Target before: {}, after: {}",
                target_before, target_after
            );

            // Print ikey values for analysis
            let target_ikey = u64::from_be_bytes(*target);
            let inserted_ikey = if key.len() >= 8 {
                u64::from_be_bytes(key[..8].try_into().unwrap())
            } else {
                let mut buf = [0u8; 8];
                buf[..key.len()].copy_from_slice(&key);
                u64::from_be_bytes(buf)
            };
            println!("      Target ikey:   {:016x}", target_ikey);
            println!("      Inserted ikey: {:016x}", inserted_ikey);
        }
    }

    // Reset seed to get the triggering key info
    let mut check_seed: u32 = 31949;
    for _ in 0..132400 {
        kv_rand(&mut check_seed);
    }
    let trigger_x = kv_rand(&mut check_seed);
    let trigger_key = to_decimal_bytes(trigger_x);
    println!(
        "\n  Original triggering key would be: {} -> {:?}",
        trigger_x,
        std::str::from_utf8(&trigger_key).unwrap()
    );

    // Check some nearby keys before the trigger
    let nearby_8byte: Vec<u32> = {
        let mut s: u32 = 31949;
        (0..132400)
            .map(|_| kv_rand(&mut s))
            .filter(|k| to_decimal_bytes(*k).len() == 8)
            .collect()
    };
    println!("  Total 8-byte keys before trigger: {}", nearby_8byte.len());

    // Count how many 8-byte keys exist before trigger
    let existing_8byte: usize = nearby_8byte
        .iter()
        .filter(|k| {
            tree.get_with_guard(&to_decimal_bytes(**k), &guard)
                .is_some()
        })
        .count();
    println!(
        "  Existing 8-byte keys: {}/{}",
        existing_8byte,
        nearby_8byte.len()
    );

    // Now insert the triggering key
    println!("\n  Inserting triggering key...");
    let _ = tree.insert_with_guard(&trigger_key, u64::from(trigger_x) + 1, &guard);

    // Check target again
    println!("  After trigger:");
    println!(
        "    Target {:?} present: {:?}",
        std::str::from_utf8(target).unwrap(),
        tree.get_with_guard(target, &guard).is_some()
    );

    // Count how many 8-byte keys now exist
    let existing_8byte_after: usize = nearby_8byte
        .iter()
        .filter(|k| {
            tree.get_with_guard(&to_decimal_bytes(**k), &guard)
                .is_some()
        })
        .count();
    println!(
        "  Existing 8-byte keys: {}/{}",
        existing_8byte_after,
        nearby_8byte.len()
    );

    // Find which keys disappeared
    let disappeared: Vec<u32> = nearby_8byte
        .iter()
        .filter(|k| {
            tree.get_with_guard(&to_decimal_bytes(**k), &guard)
                .is_none()
        })
        .copied()
        .collect();

    if !disappeared.is_empty() {
        println!("\n  Disappeared keys:");
        for k in &disappeared {
            let kb = to_decimal_bytes(*k);
            println!("    {} ({:?})", k, std::str::from_utf8(&kb).unwrap());
        }

        // Check if they share anything with the trigger key
        let trigger_bytes = &trigger_key;
        for k in &disappeared {
            let kb = to_decimal_bytes(*k);
            let common = kb
                .iter()
                .zip(trigger_bytes.iter())
                .take_while(|(a, b)| a == b)
                .count();
            println!("    {} shares {} prefix bytes with trigger", k, common);
        }
    }
}

fn test_8byte_keys() {
    println!("--- Test 1: Basic 8-byte key handling ---");
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    // Insert 8-byte keys
    let keys: Vec<&[u8]> = vec![
        b"10000000",
        b"10000001",
        b"10000002",
        b"10000003",
        b"20000000",
        b"20000001",
        b"20000002",
        b"20000003",
        b"30000000",
        b"30000001",
        b"30000002",
        b"30000003",
    ];

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_with_guard(*key, i as u64, &guard);
    }

    let mut found = 0;
    let mut missing = 0;
    for (i, key) in keys.iter().enumerate() {
        match tree.get_with_guard(*key, &guard) {
            Some(v) if *v == i as u64 => found += 1,
            Some(v) => println!(
                "  Wrong value for {:?}: got {}, expected {}",
                std::str::from_utf8(key).unwrap(),
                *v,
                i
            ),
            None => {
                println!("  Missing: {:?}", std::str::from_utf8(key).unwrap());
                missing += 1;
            }
        }
    }
    println!("  Result: {} found, {} missing\n", found, missing);
}

fn test_minimal_repro() {
    println!("--- Test 2: Finding minimal reproduction scale ---");

    for scale in [100, 1000, 10_000, 50_000, 100_000, 500_000, 1_000_000] {
        let tree: MassTree15<u64> = MassTree15::new();
        let guard = tree.guard();

        // Use the same RNG as mttest
        let mut seed: u32 = 31949;
        let mut keys = Vec::with_capacity(scale);

        // Insert keys
        for _ in 0..scale {
            let x = kv_rand(&mut seed);
            let key = to_decimal_bytes(x);
            let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
            keys.push(x);
        }

        // Count missing
        let mut missing = 0;
        for x in &keys {
            let key = to_decimal_bytes(*x);
            if tree.get_with_guard(&key, &guard).is_none() {
                missing += 1;
            }
        }

        println!(
            "  Scale {}: {} missing ({:.4}%)",
            scale,
            missing,
            (missing as f64 / scale as f64) * 100.0
        );

        if missing > 0 && scale <= 100_000 {
            // Found minimal scale, investigate further
            break;
        }
    }
    println!();
}

fn test_key_disappearance() {
    println!("--- Test 3: Track when keys disappear ---");

    // First, find a key that will go missing using deterministic RNG
    let target_key: u32 = 28631012; // Key that disappears - inserted at 101417, last seen ~130000
    println!(
        "  Tracking key {} to find exact disappearance point",
        target_key
    );

    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let mut seed: u32 = 31949;
    let target_key_bytes = to_decimal_bytes(target_key);

    // Insert keys and track when target disappears
    let mut target_inserted = false;
    let mut last_seen_at: Option<usize> = None;
    let mut disappeared_at: Option<usize> = None;

    for i in 0..600_000 {
        let x = kv_rand(&mut seed);
        let key = to_decimal_bytes(x);
        let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);

        if x == target_key {
            target_inserted = true;
            println!("    Target key inserted at index {}", i);
        }

        // After target is inserted, check periodically if it's still there
        if target_inserted && i % 1000 == 0 {
            if tree.get_with_guard(&target_key_bytes, &guard).is_some() {
                last_seen_at = Some(i);
            } else if disappeared_at.is_none() {
                disappeared_at = Some(i);
                println!(
                    "    Target key DISAPPEARED between {} and {}",
                    last_seen_at.unwrap_or(0),
                    i
                );

                // Find what was inserted in the last 1000 that might have caused this
                println!("    Last 10 inserts before disappearance:");
                let mut check_seed: u32 = 31949;
                for j in 0..i {
                    let check_x = kv_rand(&mut check_seed);
                    if j >= i - 10 {
                        let check_key = to_decimal_bytes(check_x);
                        println!(
                            "      {} -> {:?} (len={})",
                            j,
                            std::str::from_utf8(&check_key).unwrap(),
                            check_key.len()
                        );
                    }
                }
                break;
            }
        }
    }

    if disappeared_at.is_none() && target_inserted {
        println!("    Target key survived all {} insertions!", 600_000);
    }
}

fn test_precise_disappearance() {
    println!("\n--- Test 4: Binary search for exact disappearance ---");

    let target_key: u32 = 28631012;
    let target_key_bytes = to_decimal_bytes(target_key);

    // From test 3: disappeared between ~130000 and ~140000
    // Let's binary search to find the exact insert

    let mut low = 130_000usize;
    let mut high = 140_000usize;

    while low < high {
        let mid = (low + high) / 2;

        // Rebuild tree up to mid
        let tree: MassTree15<u64> = MassTree15::new();
        let guard = tree.guard();
        let mut seed: u32 = 31949;

        for _ in 0..mid {
            let x = kv_rand(&mut seed);
            let key = to_decimal_bytes(x);
            let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
        }

        if tree.get_with_guard(&target_key_bytes, &guard).is_some() {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    println!("  Key {} disappears after insert #{}", target_key, low - 1);

    // Now find what that insert was
    let mut seed: u32 = 31949;
    for i in 0..low {
        let x = kv_rand(&mut seed);
        if i == low - 1 {
            let key = to_decimal_bytes(x);
            println!(
                "  Corrupting insert: {} -> {:?} (len={})",
                x,
                std::str::from_utf8(&key).unwrap(),
                key.len()
            );

            // Check if they share a prefix
            let target_str = std::str::from_utf8(&target_key_bytes).unwrap();
            let corrupt_str = std::str::from_utf8(&key).unwrap();
            let common_prefix = target_str
                .chars()
                .zip(corrupt_str.chars())
                .take_while(|(a, b)| a == b)
                .count();
            println!("  Common prefix length: {} chars", common_prefix);
            println!("  Target:    {}", target_str);
            println!("  Corrupting: {}", corrupt_str);
        }
    }
}

fn test_minimal_case() {
    println!("\n--- Test 5: Minimal reproduction case ---");

    // Try to reproduce with just the two keys that conflict
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let target = b"28631012"; // 8 bytes - the key that disappears
    let corrupt = b"286327548"; // 9 bytes - the key that causes corruption

    println!("  Testing direct conflict between:");
    println!(
        "    Target:  {:?} (len={})",
        std::str::from_utf8(target).unwrap(),
        target.len()
    );
    println!(
        "    Corrupt: {:?} (len={})",
        std::str::from_utf8(corrupt).unwrap(),
        corrupt.len()
    );

    // Insert target first
    let _ = tree.insert_with_guard(target, 1, &guard);
    println!(
        "  After inserting target: {:?}",
        tree.get_with_guard(target, &guard)
    );

    // Insert corrupting key
    let _ = tree.insert_with_guard(corrupt, 2, &guard);
    println!(
        "  After inserting corrupt: target={:?}, corrupt={:?}",
        tree.get_with_guard(target, &guard),
        tree.get_with_guard(corrupt, &guard)
    );

    // If this doesn't reproduce, we need more context - more keys to trigger splits
    if tree.get_with_guard(target, &guard).is_some() {
        println!("  Direct case doesn't reproduce - narrowing down the trigger");

        // Binary search to find minimum number of inserts needed
        let mut low = 101417usize; // After target is inserted
        let mut high = 132401usize;

        while high - low > 1 {
            let mid = (low + high) / 2;

            let tree: MassTree15<u64> = MassTree15::new();
            let guard = tree.guard();
            let mut seed: u32 = 31949;

            for _ in 0..mid {
                let x = kv_rand(&mut seed);
                let key = to_decimal_bytes(x);
                let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
            }

            if tree.get_with_guard(target, &guard).is_some() {
                low = mid;
            } else {
                high = mid;
            }
        }

        println!("  Bug triggers at insert #{}", high);

        // Now replay to that exact point and check neighbors
        let tree: MassTree15<u64> = MassTree15::new();
        let guard = tree.guard();
        let mut seed: u32 = 31949;
        let mut all_keys: Vec<u32> = Vec::new();

        for i in 0..high {
            let x = kv_rand(&mut seed);
            let key = to_decimal_bytes(x);
            all_keys.push(x);

            if i == high - 1 {
                // This is the triggering insert
                println!(
                    "  Triggering insert: {} -> {:?} (len={})",
                    x,
                    std::str::from_utf8(&key).unwrap(),
                    key.len()
                );

                // Check target before this insert
                println!("  Target before: {:?}", tree.get_with_guard(target, &guard));

                let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);

                // Check target after
                println!("  Target after: {:?}", tree.get_with_guard(target, &guard));

                // Check nearby keys to see what else might be affected
                let nearby: Vec<&u32> = all_keys
                    .iter()
                    .filter(|k| {
                        let kb = to_decimal_bytes(**k);
                        kb.len() == 8 && kb.starts_with(b"2863")
                    })
                    .take(20)
                    .collect();

                println!("  Checking nearby 8-byte keys with prefix '2863':");
                for k in nearby {
                    let kb = to_decimal_bytes(*k);
                    let v = tree.get_with_guard(&kb, &guard);
                    let expected = u64::from(*k) + 1;
                    let status = if v.as_ref().map(|arc| **arc) == Some(expected) {
                        "OK"
                    } else {
                        "MISSING"
                    };
                    println!("    {} -> {:?} {}", k, v.map(|arc| *arc), status);
                }
            } else {
                let _ = tree.insert_with_guard(&key, u64::from(x) + 1, &guard);
            }
        }

        // Check how many total keys are now missing
        let mut missing_count = 0;
        let mut missing_8byte = 0;
        for x in &all_keys {
            let key = to_decimal_bytes(*x);
            if tree.get_with_guard(&key, &guard).is_none() {
                missing_count += 1;
                if key.len() == 8 {
                    missing_8byte += 1;
                }
            }
        }
        println!("  Total missing: {missing_count} ({missing_8byte} are 8-byte)");
    }
}

// KvRandom LCG matching C++ kvrandom_lcg_nr
const fn kv_rand(seed: &mut u32) -> u32 {
    const A: u32 = 1_664_525;
    const C: u32 = 1_013_904_223;

    // lcg_step x4, return combination of steps 2 and 4
    *seed = seed.wrapping_mul(A).wrapping_add(C);
    *seed = seed.wrapping_mul(A).wrapping_add(C);
    let x0 = *seed;
    *seed = seed.wrapping_mul(A).wrapping_add(C);
    *seed = seed.wrapping_mul(A).wrapping_add(C);
    let x1 = *seed;

    (x0 >> 15) | ((x1 & 0x7FFE) << 16)
}

// Convert number to decimal string bytes (like QuickIstr)
fn to_decimal_bytes(n: u32) -> Vec<u8> {
    n.to_string().into_bytes()
}
