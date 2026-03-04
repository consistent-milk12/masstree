//! DHAT allocation-site profiling for Masstree.
//!
//! Produces per-allocation-site breakdowns showing where memory goes
//! (leaf nodes, internodes, suffix bags, reclamation queues).
//!
//! Incompatible with `mimalloc`/`tcmalloc` features (two global allocators conflict).
//!
//! ```bash
//! cargo bench --bench memory_dhat
//! # Open dhat-heap.json in https://nnethercote.github.io/dh_view/dh_view.html
//! ```

// When mimalloc or tcmalloc is active, the custom allocator would conflict.
// Print a message and exit cleanly instead of failing to compile.
#[cfg(feature = "mimalloc")]
fn main() {
    eprintln!(
        "memory_dhat: skipped (incompatible with mimalloc/tcmalloc feature). \
         Run with: cargo bench --bench memory_dhat"
    );
}

#[cfg(not(feature = "mimalloc"))]
mod bench_utils;

#[cfg(not(feature = "mimalloc"))]
fn main() {
    inner::run();
}

#[cfg(not(feature = "mimalloc"))]
mod inner {
    #![expect(clippy::unwrap_used)]
    #![expect(clippy::pedantic)]
    #![expect(clippy::indexing_slicing)]
    #![allow(dead_code)]

    use super::bench_utils::keys;
    use masstree::MassTree15Inline;

    #[global_allocator]
    static ALLOC: dhat::Alloc = dhat::Alloc;

    // =========================================================================
    // Scenarios
    // =========================================================================

    fn scenario_baseline_insert() {
        println!("=== Scenario 1: Baseline Insert (100K entries, 32B keys) ===");
        println!();

        let profiler = dhat::Profiler::builder().testing().build();

        let n = 100_000;
        let data: Vec<[u8; 32]> = keys::<32>(n);

        let tree = MassTree15Inline::<u64>::new();
        {
            let guard = tree.guard();
            for (i, key) in data.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }

        let stats = dhat::HeapStats::get();
        print_stats("Baseline Insert", &stats);
        println!();

        drop(tree);
        drop(profiler);
    }

    fn scenario_long_keys() {
        println!("=== Scenario 2: Long Keys (100K entries, 128B keys) ===");
        println!();

        let profiler = dhat::Profiler::builder().testing().build();

        let n = 100_000;
        let data: Vec<[u8; 128]> = keys::<128>(n);

        let tree = MassTree15Inline::<u64>::new();
        {
            let guard = tree.guard();
            for (i, key) in data.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }

        let stats = dhat::HeapStats::get();
        print_stats("Long Keys", &stats);
        println!();

        drop(tree);
        drop(profiler);
    }

    fn scenario_mixed_churn() {
        println!("=== Scenario 3: Mixed Churn (insert 100K, delete 50K, insert 25K) ===");
        println!();

        // This scenario saves dhat-heap.json for interactive viewer analysis.
        let profiler = dhat::Profiler::builder()
            .file_name("dhat-heap.json")
            .build();

        let n = 100_000;
        let data: Vec<[u8; 32]> = keys::<32>(n);

        let tree = MassTree15Inline::<u64>::new();

        // Phase 1: Insert 100K.
        {
            let guard = tree.guard();
            for (i, key) in data.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }

        // Phase 2: Delete first 50K.
        {
            let guard = tree.guard();
            for key in data.iter().take(50_000) {
                let _ = tree.remove_with_guard(key, &guard);
            }
        }

        // Advance epoch.
        {
            let _guard = tree.guard();
        }

        // Phase 3: Re-insert 25K new keys (offset to avoid duplicates).
        let extra: Vec<[u8; 32]> = {
            let mut all = keys::<32>(n + 25_000);
            all.split_off(n)
        };

        {
            let guard = tree.guard();
            for (i, key) in extra.iter().enumerate() {
                let _ = tree.insert_with_guard(key, (n + i) as u64, &guard);
            }
        }

        drop(tree);
        drop(profiler);

        println!("  Saved dhat-heap.json for interactive analysis.");
        println!("  Open with: https://nnethercote.github.io/dh_view/dh_view.html");
        println!();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    fn print_stats(label: &str, stats: &dhat::HeapStats) {
        println!("  [{label}]");
        println!("  Total blocks:     {}", fmt_count(stats.total_blocks));
        println!("  Total bytes:      {}", fmt_bytes(stats.total_bytes));
        println!("  Peak blocks:      {}", fmt_count(stats.max_blocks as u64));
        println!("  Peak bytes:       {}", fmt_bytes(stats.max_bytes as u64));
        println!(
            "  Currently live:   {} blocks, {}",
            fmt_count(stats.curr_blocks as u64),
            fmt_bytes(stats.curr_bytes as u64),
        );
    }

    fn fmt_bytes(bytes: u64) -> String {
        let b = bytes as f64;
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", b / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.2} MB", b / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.2} KB", b / 1_000.0)
        } else {
            format!("{} B", bytes)
        }
    }

    fn fmt_count(n: u64) -> String {
        let f = n as f64;
        if n >= 1_000_000 {
            format!("{:.2}M", f / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", f / 1_000.0)
        } else {
            format!("{}", n)
        }
    }

    // =========================================================================
    // Entry point
    // =========================================================================

    pub fn run() {
        println!();
        println!("=== Masstree DHAT Allocation Profile ===");
        println!();

        scenario_baseline_insert();
        scenario_long_keys();
        scenario_mixed_churn();
    }
}
