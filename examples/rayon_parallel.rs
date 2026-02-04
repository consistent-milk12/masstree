//! Rayon parallel processing example for [`MassTree15`] with [`TreeIndex`] comparison.
//!
//! [`MassTree15`] is thread-safe and excels under concurrent access. This example
//! demonstrates how to use Rayon's parallel iterators for high-throughput
//! bulk operations, comparing performance against [`scc::TreeIndex`].
//!
//! Key patterns:
//! - **Chunked parallel processing** (amortizes guard overhead - RECOMMENDED)
//! - Parallel bulk insert and lookup
//! - Map-reduce patterns
//! - Performance comparison with [`TreeIndex`]
//!
//! **IMPORTANT**: Always use `par_chunks` to amortize guard creation overhead.
//! Creating a guard per-iteration is an anti-pattern that hurts performance.
//!
//! Run with: `cargo run --example rayon_parallel --release`

use masstree::{MassTree15, MassTree15Inline};
use rayon::prelude::*;
use scc::TreeIndex;
use std::sync::Arc;
use std::time::Instant;

/// Fixed-size key for [`TreeIndex`] comparison ([`TreeIndex`] needs Ord keys)
const KEY_SIZE: usize = 16;

fn make_key(i: u64) -> [u8; KEY_SIZE] {
    let mut key = [0u8; KEY_SIZE];
    key[..8].copy_from_slice(b"key/0000");
    key[8..].copy_from_slice(&i.to_be_bytes());
    key
}

#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]
fn main() {
    println!("=== MassTree + Rayon Parallel Processing ===\n");

    // =========================================================================
    // Performance Comparison: MassTree15Inline vs TreeIndex
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          PERFORMANCE COMPARISON: MassTree15 vs TreeIndex         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let n = 1_000_000u64;
    let chunk_size = 1000usize; // Amortize guard overhead across 1000 operations

    // -------------------------------------------------------------------------
    // Parallel Bulk Insert Comparison (CHUNKED - recommended pattern)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Parallel Bulk Insert ({n} entries, chunked)                │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // MassTree15Inline - using par_chunks to amortize guard overhead
    let masstree: Arc<MassTree15Inline<u64>> = Arc::new(MassTree15Inline::new());
    let keys: Vec<u64> = (0..n).collect();
    let start = Instant::now();
    keys.par_chunks(chunk_size).for_each(|chunk| {
        let guard = masstree.guard(); // One guard per chunk!
        for &i in chunk {
            let key = make_key(i);
            let _ = masstree.insert_with_guard(&key, i, &guard);
        }
    });
    let masstree_insert = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_insert,
        n as f64 / masstree_insert.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex
    let treeindex: Arc<TreeIndex<[u8; KEY_SIZE], u64>> = Arc::new(TreeIndex::new());
    let start = Instant::now();
    keys.par_chunks(chunk_size).for_each(|chunk| {
        for &i in chunk {
            let key = make_key(i);
            let _ = treeindex.insert_sync(key, i);
        }
    });
    let treeindex_insert = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_insert,
        n as f64 / treeindex_insert.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_insert.as_secs_f64() / masstree_insert.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Parallel Bulk Lookup Comparison (CHUNKED)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Parallel Bulk Lookup ({n} entries, chunked)                │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // MassTree15Inline - chunked lookups
    let start = Instant::now();
    let masstree_found: u64 = keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i);
                    u64::from(masstree.get_with_guard(&key, &guard).is_some())
                })
                .sum::<u64>()
        })
        .sum();
    let masstree_lookup = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)  found: {}",
        masstree_lookup,
        n as f64 / masstree_lookup.as_secs_f64() / 1_000_000.0,
        masstree_found
    );

    // TreeIndex - chunked for fair comparison
    let start = Instant::now();
    let treeindex_found: u64 = keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i);
                    u64::from(treeindex.peek_with(&key, |_, _| ()).is_some())
                })
                .sum::<u64>()
        })
        .sum();
    let treeindex_lookup = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)  found: {}",
        treeindex_lookup,
        n as f64 / treeindex_lookup.as_secs_f64() / 1_000_000.0,
        treeindex_found
    );

    let ratio = treeindex_lookup.as_secs_f64() / masstree_lookup.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Mixed Read/Write Workload (90% read, 10% write) - CHUNKED
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Mixed Workload: 90% Read, 10% Write (500k ops, chunked)         │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let ops = 500_000u64;
    let op_indices: Vec<u64> = (0..ops).collect();

    // MassTree15Inline - chunked mixed workload
    let start = Instant::now();
    let _: u64 = op_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            for &i in chunk {
                let key = make_key(i % n);
                if i % 10 == 0 {
                    let _ = masstree.insert_with_guard(&key, i + 1_000_000, &guard);
                } else {
                    let _ = masstree.get_with_guard(&key, &guard);
                }
            }
            chunk.len() as u64
        })
        .sum();
    let masstree_mixed = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_mixed,
        ops as f64 / masstree_mixed.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex - chunked for fair comparison
    let start = Instant::now();
    let _: u64 = op_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            for &i in chunk {
                let key = make_key(i % n);
                if i % 10 == 0 {
                    // TreeIndex insert fails if key exists, so remove first
                    treeindex.remove_sync(&key);
                    let _ = treeindex.insert_sync(key, i + 1_000_000);
                } else {
                    let _ = treeindex.peek_with(&key, |_, _| ());
                }
            }
            chunk.len() as u64
        })
        .sum();
    let treeindex_mixed = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_mixed,
        ops as f64 / treeindex_mixed.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_mixed.as_secs_f64() / masstree_mixed.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Parallel Map-Reduce (sum all values) - CHUNKED
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Parallel Map-Reduce: Sum All Values (chunked)                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // MassTree15Inline - chunked reduce
    let start = Instant::now();
    let masstree_sum: u64 = keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i);
                    masstree.get_with_guard(&key, &guard).unwrap_or(0)
                })
                .sum::<u64>()
        })
        .sum();
    let masstree_reduce = start.elapsed();
    println!("  MassTree15Inline: {masstree_reduce:>8.2?}  sum = {masstree_sum}");

    // TreeIndex - chunked for fair comparison
    let start = Instant::now();
    let treeindex_sum: u64 = keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i);
                    treeindex.peek_with(&key, |_, v| *v).unwrap_or(0)
                })
                .sum::<u64>()
        })
        .sum();
    let treeindex_reduce = start.elapsed();
    println!("  TreeIndex:        {treeindex_reduce:>8.2?}  sum = {treeindex_sum}");

    let ratio = treeindex_reduce.as_secs_f64() / masstree_reduce.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // =========================================================================
    // HIGH CONTENTION SCENARIOS (chunked for fair comparison)
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              HIGH CONTENTION SCENARIOS (MassTree shines)         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // -------------------------------------------------------------------------
    // Hot Spot: All threads hitting 100 keys (90% read, 10% write)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Hot Spot: 100 Keys, 500k ops (90% read, 10% write)              │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let hot_keys = 100u64;
    let hot_ops = 500_000u64;
    let hot_indices: Vec<u64> = (0..hot_ops).collect();

    // Pre-populate hot keys in both structures
    {
        let guard = masstree.guard();
        for i in 0..hot_keys {
            let key = make_key(i);
            let _ = masstree.insert_with_guard(&key, i, &guard);
            let _ = treeindex.insert_sync(key, i);
        }
    }

    // MassTree - chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            for &i in chunk {
                let key = make_key(i % hot_keys);
                if i % 10 == 0 {
                    let _ = masstree.insert_with_guard(&key, i, &guard);
                } else {
                    let _ = masstree.get_with_guard(&key, &guard);
                }
            }
            chunk.len() as u64
        })
        .sum();
    let masstree_hot = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_hot,
        hot_ops as f64 / masstree_hot.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex - chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            for &i in chunk {
                let key = make_key(i % hot_keys);
                if i % 10 == 0 {
                    treeindex.remove_sync(&key);
                    let _ = treeindex.insert_sync(key, i);
                } else {
                    let _ = treeindex.peek_with(&key, |_, _| ());
                }
            }
            chunk.len() as u64
        })
        .sum();
    let treeindex_hot = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_hot,
        hot_ops as f64 / treeindex_hot.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_hot.as_secs_f64() / masstree_hot.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Single Hot Key: Extreme contention on ONE key
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Single Hot Key: 1 Key, 500k ops (95% read, 5% write)            │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let single_key = make_key(42);

    // MassTree - chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            for &i in chunk {
                if i % 20 == 0 {
                    let _ = masstree.insert_with_guard(&single_key, i, &guard);
                } else {
                    let _ = masstree.get_with_guard(&single_key, &guard);
                }
            }
            chunk.len() as u64
        })
        .sum();
    let masstree_single = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_single,
        hot_ops as f64 / masstree_single.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex - chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            for &i in chunk {
                if i % 20 == 0 {
                    treeindex.remove_sync(&single_key);
                    let _ = treeindex.insert_sync(single_key, i);
                } else {
                    let _ = treeindex.peek_with(&single_key, |_, _| ());
                }
            }
            chunk.len() as u64
        })
        .sum();
    let treeindex_single = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_single,
        hot_ops as f64 / treeindex_single.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_single.as_secs_f64() / masstree_single.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Zipfian-like: Power law distribution (80% of ops hit 20% of keys)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Zipfian-like: 80% ops on 20% of keys (500k ops)                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let total_keys = 10_000u64;
    let hot_portion = total_keys / 5; // 20% of keys

    // Pre-populate
    {
        let guard = masstree.guard();
        for i in 0..total_keys {
            let key = make_key(i);
            let _ = masstree.insert_with_guard(&key, i, &guard);
            let _ = treeindex.insert_sync(key, i);
        }
    }

    // MassTree - Zipfian access, chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            for &i in chunk {
                // 80% chance to hit hot keys (0..hot_portion), 20% cold
                let key_idx = if i % 5 == 0 {
                    // 20% cold access
                    hot_portion + (i % (total_keys - hot_portion))
                } else {
                    // 80% hot access
                    i % hot_portion
                };
                let key = make_key(key_idx);
                if i % 10 == 0 {
                    let _ = masstree.insert_with_guard(&key, i, &guard);
                } else {
                    let _ = masstree.get_with_guard(&key, &guard);
                }
            }
            chunk.len() as u64
        })
        .sum();
    let masstree_zipf = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_zipf,
        hot_ops as f64 / masstree_zipf.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex - Zipfian access, chunked
    let start = Instant::now();
    let _: u64 = hot_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            for &i in chunk {
                let key_idx = if i % 5 == 0 {
                    hot_portion + (i % (total_keys - hot_portion))
                } else {
                    i % hot_portion
                };
                let key = make_key(key_idx);
                if i % 10 == 0 {
                    treeindex.remove_sync(&key);
                    let _ = treeindex.insert_sync(key, i);
                } else {
                    let _ = treeindex.peek_with(&key, |_, _| ());
                }
            }
            chunk.len() as u64
        })
        .sum();
    let treeindex_zipf = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_zipf,
        hot_ops as f64 / treeindex_zipf.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_zipf.as_secs_f64() / masstree_zipf.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // High Write Contention: 50% read, 50% write on small key set
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Write Storm: 50% Read, 50% Write on 1000 keys (200k ops)        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let write_keys = 1000u64;
    let write_ops = 200_000u64;
    let write_indices: Vec<u64> = (0..write_ops).collect();

    // MassTree - chunked
    let start = Instant::now();
    let _: u64 = write_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            for &i in chunk {
                let key = make_key(i % write_keys);
                if i % 2 == 0 {
                    let _ = masstree.insert_with_guard(&key, i, &guard);
                } else {
                    let _ = masstree.get_with_guard(&key, &guard);
                }
            }
            chunk.len() as u64
        })
        .sum();
    let masstree_write = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)",
        masstree_write,
        write_ops as f64 / masstree_write.as_secs_f64() / 1_000_000.0
    );

    // TreeIndex - chunked
    let start = Instant::now();
    let _: u64 = write_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            for &i in chunk {
                let key = make_key(i % write_keys);
                if i % 2 == 0 {
                    treeindex.remove_sync(&key);
                    let _ = treeindex.insert_sync(key, i);
                } else {
                    let _ = treeindex.peek_with(&key, |_, _| ());
                }
            }
            chunk.len() as u64
        })
        .sum();
    let treeindex_write = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)",
        treeindex_write,
        write_ops as f64 / treeindex_write.as_secs_f64() / 1_000_000.0
    );

    let ratio = treeindex_write.as_secs_f64() / masstree_write.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // -------------------------------------------------------------------------
    // Read-only Hot Spot: Pure reads on small key set
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Read-Only Hot Spot: 100% reads on 100 keys (1M ops)             │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let read_only_ops = 1_000_000u64;
    let read_indices: Vec<u64> = (0..read_only_ops).collect();

    // MassTree - chunked
    let start = Instant::now();
    let found: u64 = read_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let guard = masstree.guard();
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i % hot_keys);
                    u64::from(masstree.get_with_guard(&key, &guard).is_some())
                })
                .sum::<u64>()
        })
        .sum();
    let masstree_readonly = start.elapsed();
    println!(
        "  MassTree15Inline: {:>8.2?}  ({:>6.2} M ops/sec)  found: {}",
        masstree_readonly,
        read_only_ops as f64 / masstree_readonly.as_secs_f64() / 1_000_000.0,
        found
    );

    // TreeIndex - chunked
    let start = Instant::now();
    let found: u64 = read_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&i| {
                    let key = make_key(i % hot_keys);
                    u64::from(treeindex.peek_with(&key, |_, _| ()).is_some())
                })
                .sum::<u64>()
        })
        .sum();
    let treeindex_readonly = start.elapsed();
    println!(
        "  TreeIndex:        {:>8.2?}  ({:>6.2} M ops/sec)  found: {}",
        treeindex_readonly,
        read_only_ops as f64 / treeindex_readonly.as_secs_f64() / 1_000_000.0,
        found
    );

    let ratio = treeindex_readonly.as_secs_f64() / masstree_readonly.as_secs_f64();
    println!(
        "  → MassTree is {:.1}x {}\n",
        if ratio > 1.0 { ratio } else { 1.0 / ratio },
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // =========================================================================
    // Additional MassTree-specific Examples
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    MASSTREE-SPECIFIC FEATURES                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // -------------------------------------------------------------------------
    // Chunked Processing (amortizes guard overhead)
    // -------------------------------------------------------------------------
    println!("--- Chunked Parallel Processing ---\n");

    let tree: Arc<MassTree15Inline<u64>> = Arc::new(MassTree15Inline::new());
    let data: Vec<(String, u64)> = (0..500_000).map(|i| (format!("chunk/{i:07}"), i)).collect();

    let start = Instant::now();
    data.par_chunks(1000).for_each(|chunk| {
        let guard = tree.guard();
        for (key, value) in chunk {
            let _ = tree.insert_with_guard(key.as_bytes(), *value, &guard);
        }
    });
    let elapsed = start.elapsed();
    println!(
        "Inserted {} entries using chunks in {elapsed:?}",
        data.len()
    );
    println!(
        "Rate: {:.2} M ops/sec",
        data.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!("Tree size: {}\n", tree.len());

    // -------------------------------------------------------------------------
    // Parallel Filter and Collect (chunked)
    // -------------------------------------------------------------------------
    println!("--- Parallel Filter and Collect ---\n");

    let filter_indices: Vec<u64> = (0..500_000).collect();
    let start = Instant::now();
    let filtered: Vec<(String, u64)> = filter_indices
        .par_chunks(1000)
        .flat_map(|chunk| {
            let guard = tree.guard();
            chunk
                .iter()
                .filter_map(|&i| {
                    let key = format!("chunk/{i:07}");
                    tree.get_with_guard(key.as_bytes(), &guard)
                        .and_then(|v| if v > 400_000 { Some((key, v)) } else { None })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let elapsed = start.elapsed();
    println!(
        "Filtered {} entries (value > 400,000) in {elapsed:?}",
        filtered.len()
    );
    println!("Sample: {:?}\n", &filtered[..3.min(filtered.len())]);

    // -------------------------------------------------------------------------
    // Scoped Thread Pool Work
    // -------------------------------------------------------------------------
    println!("--- Scoped Thread Pool Work ---\n");

    let tree: MassTree15<String> = MassTree15::new();
    let entries_per_thread = 10_000;

    rayon::scope(|s| {
        for thread_id in 0..4 {
            let tree_ref = &tree;
            s.spawn(move |_| {
                let guard = tree_ref.guard();
                for i in 0..entries_per_thread {
                    let key = format!("thread{thread_id}/item{i}");
                    let value = format!("value_from_thread_{thread_id}_{i}");
                    let _ = tree_ref.insert_with_guard(key.as_bytes(), value, &guard);
                }
            });
        }
    });

    println!("4 threads each inserted {entries_per_thread} entries");
    println!("Total tree size: {}", tree.len());

    let guard = tree.guard();
    for thread_id in 0..4 {
        let prefix = format!("thread{thread_id}/");
        let mut count = 0;
        tree.scan_prefix(
            prefix.as_bytes(),
            |_, _| {
                count += 1;
                true
            },
            &guard,
        );
        println!("  Thread {thread_id}: {count} entries");
    }
    println!();

    // -------------------------------------------------------------------------
    // Parallel Prefix Scan (MassTree advantage: prefix queries)
    // -------------------------------------------------------------------------
    println!("--- Parallel Prefix Scan (MassTree-specific) ---\n");

    let tree: Arc<MassTree15Inline<u64>> = Arc::new(MassTree15Inline::new());

    // Insert with partitioned prefixes
    (0..100_000u64).into_par_iter().for_each(|i| {
        let partition = i / 10_000;
        let key = format!("p{partition:02}/item{i:06}");
        let guard = tree.guard();
        let _ = tree.insert_with_guard(key.as_bytes(), i, &guard);
    });
    println!("Inserted 100,000 entries across 10 partitions");

    let start = Instant::now();
    let partition_sums: Vec<u64> = (0..10)
        .into_par_iter()
        .map(|partition| {
            let prefix = format!("p{partition:02}/");
            let guard = tree.guard();
            let mut sum = 0u64;
            tree.scan_prefix(
                prefix.as_bytes(),
                |_, value| {
                    sum += value;
                    true
                },
                &guard,
            );
            sum
        })
        .collect();
    let elapsed = start.elapsed();

    let total: u64 = partition_sums.iter().sum();
    println!("Partition sums computed in {elapsed:?}");
    println!("Sums: {partition_sums:?}");
    println!("Total: {total}");

    println!("\n=== Rayon Examples Complete ===");
}
