//! Targeted leak tests for Miri (`cargo miri test --test leak_tests`).
//!
//! Each test exercises a specific memory lifecycle pattern around sublayer
//! creation and `gc_layer` cleanup. Keep tests small — Miri is slow.

#![expect(clippy::indexing_slicing, reason = "fail fast in tests")]

use masstree::tree::{MassTree15, MassTree15Inline};

/// Helper: build a key from an 8-byte layer prefix and a suffix.
fn make_key(layer0: u64, suffix: &[u8]) -> Vec<u8> {
    let mut key = layer0.to_be_bytes().to_vec();
    key.extend_from_slice(suffix);
    key
}

/// Helper: build a key spanning 3 layers (24+ bytes).
fn make_deep_key(layer0: u64, layer1: u64, layer2: u64, suffix: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(24 + suffix.len());
    key.extend_from_slice(&layer0.to_be_bytes());
    key.extend_from_slice(&layer1.to_be_bytes());
    key.extend_from_slice(&layer2.to_be_bytes());
    key.extend_from_slice(suffix);
    key
}

// ============================================================================
//  Macro: generate each test for both MassTree15<u64> and MassTree15Inline<u64>
// ============================================================================

macro_rules! leak_tests {
    ($mod_name:ident, $tree_ty:ty) => {
        mod $mod_name {
            use super::*;

            // ----------------------------------------------------------------
            //  1. Single sublayer: insert one suffix key, remove it.
            //     Exercises basic gc_layer (1-level context).
            // ----------------------------------------------------------------
            #[test]
            fn single_sublayer_insert_remove() {
                let tree = <$tree_ty>::new();
                // 10-byte key → layer0 ikey + 2-byte suffix
                let key = make_key(1, &[0xAB, 0xCD]);
                tree.insert(&key, 42);
                assert!(tree.get(&key).is_some());

                tree.remove(&key).unwrap();
                assert!(tree.get(&key).is_none());

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  2. Multiple keys in one sublayer, remove all.
            //     All share the same 8-byte prefix → same sublayer.
            // ----------------------------------------------------------------
            #[test]
            fn sublayer_remove_all_keys() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..5).map(|i| make_key(1, &[i, 0xFF])).collect();

                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                assert_eq!(tree.len(), 5);

                for key in &keys {
                    tree.remove(key).unwrap();
                }
                assert_eq!(tree.len(), 0);

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  3. Twig chain (2 layers deep): 18-byte keys share 16-byte prefix.
            //     Creates layer0 → layer1 → leaf with suffix.
            //     Remove all → gc_layer must cascade 2 levels.
            // ----------------------------------------------------------------
            #[test]
            fn twig_chain_two_levels() {
                let tree = <$tree_ty>::new();
                // 18-byte keys: 8 (layer0) + 8 (layer1) + 2 (suffix)
                let prefix0: u64 = 0x0000_0000_0000_0001;
                let prefix1: u64 = 0x0000_0000_0000_0002;
                let keys: Vec<Vec<u8>> = (0u8..4)
                    .map(|i| {
                        let mut k = Vec::with_capacity(18);
                        k.extend_from_slice(&prefix0.to_be_bytes());
                        k.extend_from_slice(&prefix1.to_be_bytes());
                        k.extend_from_slice(&[i, 0xAA]);
                        k
                    })
                    .collect();

                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                assert_eq!(tree.len(), 4);

                // Verify all reachable
                for key in &keys {
                    assert!(tree.get(key).is_some());
                }

                // Remove all → should trigger gc_layer cascade
                for key in &keys {
                    tree.remove(key).unwrap();
                }
                assert_eq!(tree.len(), 0);

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  4. Twig chain (3 layers deep): 26-byte keys share 24-byte prefix.
            //     layer0 → layer1 → layer2 → leaf. Deepest cascade.
            // ----------------------------------------------------------------
            #[test]
            fn twig_chain_three_levels() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..3)
                    .map(|i| make_deep_key(1, 2, 3, &[i, 0xBB]))
                    .collect();

                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }

                for key in &keys {
                    tree.remove(key).unwrap();
                }
                assert_eq!(tree.len(), 0);

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  5. Partial removal: remove some keys from sublayer, not all.
            //     gc_layer should NOT fire (sublayer still occupied).
            // ----------------------------------------------------------------
            #[test]
            fn partial_sublayer_removal() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..6).map(|i| make_key(1, &[i])).collect();

                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }

                // Remove first half
                for key in &keys[..3] {
                    tree.remove(key).unwrap();
                }
                assert_eq!(tree.len(), 3);

                // Remaining keys still reachable
                for key in &keys[3..] {
                    assert!(tree.get(key).is_some());
                }

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  6. Remove all then reinsert into same sublayer path.
            //     Verifies the layer slot is properly cleared and a new
            //     sublayer can be created at the same position.
            // ----------------------------------------------------------------
            #[test]
            fn remove_all_then_reinsert() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..3).map(|i| make_key(1, &[i, 0xCC])).collect();

                // Insert round 1
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }

                // Remove all
                for key in &keys {
                    tree.remove(key).unwrap();
                }

                let guard = tree.guard();
                tree.process_coalesce(&guard);
                drop(guard);

                // Reinsert
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, (i + 100) as u64);
                }
                assert_eq!(tree.len(), 3);

                for key in &keys {
                    assert!(tree.get(key).is_some(), "reinserted key missing");
                }

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  7. Mixed layer depths: some keys are 8 bytes (inline),
            //     some are 10 bytes (suffix), some are 18 bytes (2 layers).
            //     Remove all → heterogeneous cleanup.
            // ----------------------------------------------------------------
            #[test]
            fn mixed_depth_remove_all() {
                let tree = <$tree_ty>::new();
                let ikey: u64 = 0x0000_0000_0000_0001;

                // 8-byte key (inline, no sublayer)
                let k_inline = ikey.to_be_bytes().to_vec();
                // 10-byte key (suffix in sublayer)
                let k_suffix = make_key(ikey, &[0x01, 0x02]);
                // 18-byte key (2 layers deep)
                let mut k_deep = Vec::with_capacity(18);
                k_deep.extend_from_slice(&ikey.to_be_bytes());
                k_deep.extend_from_slice(&0x0000_0000_0000_0005u64.to_be_bytes());
                k_deep.extend_from_slice(&[0xDD, 0xEE]);

                tree.insert(&k_inline, 1);
                tree.insert(&k_suffix, 2);
                tree.insert(&k_deep, 3);
                assert_eq!(tree.len(), 3);

                tree.remove(&k_inline).unwrap();
                tree.remove(&k_suffix).unwrap();
                tree.remove(&k_deep).unwrap();
                assert_eq!(tree.len(), 0);

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  8. Multiple independent sublayers under different ikeys.
            //     Remove all from each → each triggers its own gc_layer.
            // ----------------------------------------------------------------
            #[test]
            fn multiple_independent_sublayers() {
                let tree = <$tree_ty>::new();

                // 3 different ikey prefixes, each with 3 suffix keys
                let mut all_keys = Vec::new();

                for prefix in 1u64..=3 {
                    for suffix in 0u8..3 {
                        let key = make_key(prefix, &[suffix, 0xAA]);
                        tree.insert(&key, (prefix * 10 + u64::from(suffix)));
                        all_keys.push(key);
                    }
                }

                assert_eq!(tree.len(), 9);

                for key in &all_keys {
                    tree.remove(key).unwrap();
                }

                assert_eq!(tree.len(), 0);

                let guard = tree.guard();
                tree.process_coalesce(&guard);
            }

            // ----------------------------------------------------------------
            //  9. Interleaved insert-remove cycles on twig chains.
            //     Insert batch → remove all → repeat. Stresses alloc/free cycle.
            // ----------------------------------------------------------------
            #[test]
            fn insert_remove_cycle_twig() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..4).map(|i| make_deep_key(1, 2, 3, &[i])).collect();

                for _cycle in 0..3 {
                    for (i, key) in keys.iter().enumerate() {
                        tree.insert(key, i as u64);
                    }
                    for key in &keys {
                        tree.remove(key).unwrap();
                    }
                    let guard = tree.guard();
                    tree.process_coalesce(&guard);
                }

                assert_eq!(tree.len(), 0);
            }

            // ----------------------------------------------------------------
            // 10. Drop without process_coalesce.
            //     Tree is dropped with pending coalesce entries — teardown
            //     must still free everything (no leaks).
            // ----------------------------------------------------------------
            #[test]
            fn drop_with_pending_coalesce() {
                let tree = <$tree_ty>::new();
                let keys: Vec<Vec<u8>> = (0u8..5).map(|i| make_key(1, &[i, 0xFF])).collect();

                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }

                for key in &keys {
                    tree.remove(key).unwrap();
                }

                // Intentionally do NOT call process_coalesce — tree drop must handle it.
                drop(tree);
            }
        }
    };
}

leak_tests!(arc_values, MassTree15<u64>);
leak_tests!(inline_values, MassTree15Inline<u64>);
