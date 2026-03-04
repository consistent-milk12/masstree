//! Total memory profiling for Masstree.
//!
//! Measures bytes-per-entry, overhead ratio, and reclamation effectiveness
//! at various scales. Uses a lightweight tracking allocator wrapping System.
//!
//! Incompatible with `mimalloc`/`tcmalloc` features (two global allocators conflict).
//!
//! ```bash
//! cargo bench --bench memory_total
//! ```

// When mimalloc or tcmalloc is active, the custom allocator would conflict.
// Print a message and exit cleanly instead of failing to compile.
#[cfg(feature = "mimalloc")]
fn main() {
    eprintln!(
        "memory_total: skipped (incompatible with mimalloc/tcmalloc feature). \
         Run with: cargo bench --bench memory_total"
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
    #![expect(clippy::cast_precision_loss)]
    #![allow(dead_code)]

    use super::bench_utils::{keys, keys_shared_prefix};
    use masstree::{MassTree15, MassTree15Inline};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    // =========================================================================
    // Tracking Allocator
    // =========================================================================

    struct TrackingAllocator {
        allocated: AtomicUsize,
        deallocated: AtomicUsize,
    }

    impl TrackingAllocator {
        const fn new() -> Self {
            Self {
                allocated: AtomicUsize::new(0),
                deallocated: AtomicUsize::new(0),
            }
        }

        fn snapshot(&self) -> MemSnapshot {
            MemSnapshot {
                allocated: self.allocated.load(Relaxed),
                deallocated: self.deallocated.load(Relaxed),
            }
        }
    }

    // SAFETY: Delegates to System allocator, adding atomic counter updates.
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                self.allocated.fetch_add(layout.size(), Relaxed);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            self.deallocated.fetch_add(layout.size(), Relaxed);
            unsafe { System.dealloc(ptr, layout) };
        }
    }

    #[global_allocator]
    static ALLOC: TrackingAllocator = TrackingAllocator::new();

    #[derive(Clone, Copy)]
    struct MemSnapshot {
        allocated: usize,
        deallocated: usize,
    }

    impl MemSnapshot {
        fn net(&self) -> usize {
            self.allocated.saturating_sub(self.deallocated)
        }

        fn delta_net(&self, before: &MemSnapshot) -> usize {
            self.net().saturating_sub(before.net())
        }

        fn delta_allocated(&self, before: &MemSnapshot) -> usize {
            self.allocated.saturating_sub(before.allocated)
        }

        fn delta_deallocated(&self, before: &MemSnapshot) -> usize {
            self.deallocated.saturating_sub(before.deallocated)
        }
    }

    // =========================================================================
    // Formatting helpers
    // =========================================================================

    fn fmt_bytes(bytes: usize) -> String {
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.2} MB", bytes as f64 / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.2} KB", bytes as f64 / 1_000.0)
        } else {
            format!("{} B", bytes)
        }
    }

    fn fmt_count(n: usize) -> String {
        if n >= 1_000_000 {
            format!("{}M", n / 1_000_000)
        } else if n >= 1_000 {
            format!("{}K", n / 1_000)
        } else {
            format!("{}", n)
        }
    }

    fn theoretical_min(key_size: usize) -> usize {
        key_size + 8
    }

    // =========================================================================
    // Scenarios
    // =========================================================================

    fn scale_sweep_inline() {
        println!("--- Scale Sweep (MassTree15Inline<u64>, 32B keys) ---");
        println!(
            "  {:>10}  {:>12}  {:>11}  {:>9}",
            "Entries", "Net Alloc", "Bytes/Entry", "Overhead"
        );

        let counts = [1_000, 10_000, 100_000, 1_000_000];
        let tmin = theoretical_min(32);

        for &n in &counts {
            let data: Vec<[u8; 32]> = keys::<32>(n);
            let before = ALLOC.snapshot();

            let tree = MassTree15Inline::<u64>::new();
            {
                let guard = tree.guard();
                for (i, key) in data.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }

            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            let bpe = net / n;
            let overhead = bpe as f64 / tmin as f64;

            println!(
                "  {:>10}  {:>12}  {:>9} B  {:>8.3}x",
                fmt_count(n),
                fmt_bytes(net),
                bpe,
                overhead,
            );

            drop(tree);
        }
        println!();
    }

    fn scale_sweep_box() {
        println!("--- Scale Sweep (MassTree15<u64> box policy, 32B keys) ---");
        println!(
            "  {:>10}  {:>12}  {:>11}  {:>9}",
            "Entries", "Net Alloc", "Bytes/Entry", "Overhead"
        );

        let counts = [1_000, 10_000, 100_000, 1_000_000];
        let tmin = theoretical_min(32);

        for &n in &counts {
            let data: Vec<[u8; 32]> = keys::<32>(n);
            let before = ALLOC.snapshot();

            let tree = MassTree15::<u64>::new();
            {
                let guard = tree.guard();
                for (i, key) in data.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }

            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            let bpe = net / n;
            let overhead = bpe as f64 / tmin as f64;

            println!(
                "  {:>10}  {:>12}  {:>9} B  {:>8.3}x",
                fmt_count(n),
                fmt_bytes(net),
                bpe,
                overhead,
            );

            drop(tree);
        }
        println!();
    }

    fn key_size_sweep() {
        println!("--- Key Size Sweep (MassTree15Inline<u64>, 100K entries) ---");
        println!(
            "  {:>8}  {:>12}  {:>11}  {:>9}",
            "Key Size", "Net Alloc", "Bytes/Entry", "Overhead"
        );

        let n = 100_000;

        macro_rules! measure_key_size {
            ($k:expr) => {{
                let data: Vec<[u8; $k]> = keys::<$k>(n);
                let before = ALLOC.snapshot();

                let tree = MassTree15Inline::<u64>::new();
                {
                    let guard = tree.guard();
                    for (i, key) in data.iter().enumerate() {
                        let _ = tree.insert_with_guard(key, i as u64, &guard);
                    }
                }

                let after = ALLOC.snapshot();
                let net = after.delta_net(&before);
                let bpe = net / n;
                let tmin = theoretical_min($k);
                let overhead = bpe as f64 / tmin as f64;

                println!(
                    "  {:>6} B  {:>12}  {:>9} B  {:>8.3}x",
                    $k,
                    fmt_bytes(net),
                    bpe,
                    overhead,
                );

                drop(tree);
            }};
        }

        measure_key_size!(8);
        measure_key_size!(16);
        measure_key_size!(32);
        measure_key_size!(64);
        measure_key_size!(128);

        println!();
    }

    fn churn_test() {
        println!("--- Churn Test (MassTree15Inline<u64>, 100K insert, 50K delete) ---");

        let n = 100_000;
        let delete_count = n / 2;
        let data: Vec<[u8; 32]> = keys::<32>(n);

        let before_insert = ALLOC.snapshot();

        let tree = MassTree15Inline::<u64>::new();
        {
            let guard = tree.guard();
            for (i, key) in data.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }

        let after_insert = ALLOC.snapshot();
        let insert_net = after_insert.delta_net(&before_insert);
        let insert_bpe = insert_net / n;

        // Delete half the keys, then run maintenance to reclaim memory.
        {
            let mut guard = tree.guard();
            for key in data.iter().take(delete_count) {
                let _ = tree.remove_with_guard(key, &guard);
            }
            tree.maintenance(&mut guard);
        }

        let after_delete = ALLOC.snapshot();
        let delete_net = after_delete.delta_net(&before_insert);
        let remaining = n - delete_count;
        let delete_bpe = if remaining > 0 {
            delete_net / remaining
        } else {
            0
        };

        let total_dealloc = after_delete.delta_deallocated(&after_insert);
        let total_alloc_at_insert = after_insert.delta_allocated(&before_insert);
        let reclaimed_pct = if total_alloc_at_insert > 0 {
            (total_dealloc as f64 / total_alloc_at_insert as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "  After insert:  {} ({} B/entry)",
            fmt_bytes(insert_net),
            insert_bpe
        );
        println!(
            "  After delete:  {} ({} B/entry for {} remaining)",
            fmt_bytes(delete_net),
            delete_bpe,
            fmt_count(remaining)
        );
        println!("  Reclaimed:     {:.1}%", reclaimed_pct);
        println!();

        drop(tree);
    }

    fn shared_prefix_test() {
        println!("--- Shared Prefix Keys (MassTree15Inline<u64>, 100K entries, 64B keys) ---");
        println!(
            "  {:>12}  {:>12}  {:>11}",
            "Prefixes", "Net Alloc", "Bytes/Entry"
        );

        let n = 100_000;
        let prefix_counts: [u64; 4] = [10, 100, 1_000, 10_000];

        for &buckets in &prefix_counts {
            let data: Vec<[u8; 64]> = keys_shared_prefix::<64>(n, buckets);
            let before = ALLOC.snapshot();

            let tree = MassTree15Inline::<u64>::new();
            {
                let guard = tree.guard();
                for (i, key) in data.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }

            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            let bpe = net / n;

            println!(
                "  {:>12}  {:>12}  {:>9} B",
                fmt_count(buckets as usize),
                fmt_bytes(net),
                bpe,
            );

            drop(tree);
        }
        println!();
    }

    fn competitor_comparison() {
        println!("--- Competitor Comparison (100K entries, 32B keys) ---");
        println!(
            "  {:>25}  {:>12}  {:>11}",
            "Structure", "Net Alloc", "Bytes/Entry"
        );

        let n = 100_000;
        let data: Vec<[u8; 32]> = keys::<32>(n);

        // MassTree15Inline
        {
            let before = ALLOC.snapshot();
            let tree = MassTree15Inline::<u64>::new();
            {
                let guard = tree.guard();
                for (i, key) in data.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            println!(
                "  {:>25}  {:>12}  {:>9} B",
                "MassTree15Inline",
                fmt_bytes(net),
                net / n,
            );
            drop(tree);
        }

        // MassTree15 (box)
        {
            let before = ALLOC.snapshot();
            let tree = MassTree15::<u64>::new();
            {
                let guard = tree.guard();
                for (i, key) in data.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            println!(
                "  {:>25}  {:>12}  {:>9} B",
                "MassTree15 (box)",
                fmt_bytes(net),
                net / n,
            );
            drop(tree);
        }

        // scc::TreeIndex
        {
            use scc::TreeIndex;
            let before = ALLOC.snapshot();
            let tree: TreeIndex<[u8; 32], u64> = TreeIndex::new();
            for (i, key) in data.iter().enumerate() {
                let _ = tree.insert_sync(*key, i as u64);
            }
            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            println!(
                "  {:>25}  {:>12}  {:>9} B",
                "scc::TreeIndex",
                fmt_bytes(net),
                net / n,
            );
            drop(tree);
        }

        // crossbeam_skiplist::SkipMap
        {
            use crossbeam_skiplist::SkipMap;
            let before = ALLOC.snapshot();
            let map: SkipMap<[u8; 32], u64> = SkipMap::new();
            for (i, key) in data.iter().enumerate() {
                map.insert(*key, i as u64);
            }
            let after = ALLOC.snapshot();
            let net = after.delta_net(&before);
            println!(
                "  {:>25}  {:>12}  {:>9} B",
                "crossbeam::SkipMap",
                fmt_bytes(net),
                net / n,
            );
            drop(map);
        }

        println!();
    }

    // =========================================================================
    // Entry point
    // =========================================================================

    pub fn run() {
        println!();
        println!("=== Masstree Memory Profile ===");
        println!();

        scale_sweep_inline();
        scale_sweep_box();
        key_size_sweep();
        churn_test();
        shared_prefix_test();
        competitor_comparison();
    }
}
