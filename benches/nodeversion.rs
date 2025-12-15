//! Benchmarks for `NodeVersion` using Divan.
//!
//! Run with: `cargo bench --bench nodeversion`

use divan::{Bencher, black_box};
use masstree::nodeversion::NodeVersion;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
mod construction {
    use super::{Bencher, NodeVersion, black_box};

    #[divan::bench]
    const fn new_leaf() -> NodeVersion {
        NodeVersion::new(black_box(true))
    }

    #[divan::bench]
    const fn new_internode() -> NodeVersion {
        NodeVersion::new(black_box(false))
    }

    #[divan::bench]
    const fn from_value() -> NodeVersion {
        NodeVersion::from_value(black_box(0x8000_0000))
    }

    #[divan::bench]
    fn clone(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).clone());
    }
}

// =============================================================================
// Flag Accessors (hot path for readers)
// =============================================================================

#[divan::bench_group]
mod accessors {
    use super::{Bencher, NodeVersion, black_box};

    #[divan::bench]
    fn is_leaf(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_leaf());
    }

    #[divan::bench]
    fn is_root(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_root());
    }

    #[divan::bench]
    fn is_deleted(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_deleted());
    }

    #[divan::bench]
    fn is_locked(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_locked());
    }

    #[divan::bench]
    fn is_inserting(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_inserting());
    }

    #[divan::bench]
    fn is_splitting(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_splitting());
    }

    #[divan::bench]
    fn is_dirty(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).is_dirty());
    }

    #[divan::bench]
    fn value(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).value());
    }
}

// =============================================================================
// Optimistic Read Operations
// =============================================================================

#[divan::bench_group]
mod optimistic_read {
    use super::{Bencher, NodeVersion, black_box};

    #[divan::bench]
    fn stable(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| black_box(&v).stable());
    }

    #[divan::bench]
    fn has_changed_false(bencher: Bencher) {
        let v = NodeVersion::new(true);
        let stable = v.stable();
        bencher.bench_local(|| black_box(&v).has_changed(black_box(stable)));
    }

    #[divan::bench]
    fn has_changed_true(bencher: Bencher) {
        let v = NodeVersion::new(true);
        let old_stable = v.stable();
        // Simulate a version change
        {
            let mut guard = v.lock();
            guard.mark_insert();
        }
        bencher.bench_local(|| black_box(&v).has_changed(black_box(old_stable)));
    }

    #[divan::bench]
    fn has_split_false(bencher: Bencher) {
        let v = NodeVersion::new(true);
        let stable = v.stable();
        bencher.bench_local(|| black_box(&v).has_split(black_box(stable)));
    }

    #[divan::bench]
    fn has_split_true(bencher: Bencher) {
        let v = NodeVersion::new(true);
        let old_stable = v.stable();
        // Simulate a split
        {
            let mut guard = v.lock();
            guard.mark_split();
        }
        bencher.bench_local(|| black_box(&v).has_split(black_box(old_stable)));
    }
}

// =============================================================================
// Lock Operations
// =============================================================================

#[divan::bench_group]
mod lock_ops {
    use super::{Bencher, NodeVersion};

    #[divan::bench]
    fn lock_unlock(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let guard = v.lock();
                drop(guard);
            });
    }

    #[divan::bench]
    fn try_lock_success(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let guard = v.try_lock();
                drop(guard);
            });
    }

    #[divan::bench]
    fn try_lock_fail(bencher: Bencher) {
        let v = NodeVersion::new(true);
        let _guard = v.lock(); // Hold lock
        bencher.bench_local(|| v.try_lock());
    }

    #[divan::bench]
    fn locked_value(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let guard = v.lock();
                guard.locked_value()
            });
    }
}

// =============================================================================
// Mark Operations (require lock)
// =============================================================================

#[divan::bench_group]
mod mark_ops {
    use super::{Bencher, NodeVersion};

    #[divan::bench]
    fn mark_insert(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_insert();
            });
    }

    #[divan::bench]
    fn mark_split(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_split();
            });
    }

    #[divan::bench]
    fn mark_deleted(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_deleted();
            });
    }

    #[divan::bench]
    fn mark_nonroot(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let v = NodeVersion::new(true);
                v.mark_root();
                v
            })
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_nonroot();
            });
    }

    #[divan::bench]
    fn mark_root(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                v.mark_root();
            });
    }
}

// =============================================================================
// Realistic Workloads
// =============================================================================

#[divan::bench_group]
mod workload {
    use super::{Bencher, NodeVersion, black_box};

    /// Optimistic read pattern: get stable, read data, validate
    #[divan::bench]
    fn optimistic_read_success(bencher: Bencher) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| {
            let stable = black_box(&v).stable();
            // Simulate reading some data
            let _ = black_box(42u64);
            black_box(&v).has_changed(stable)
        });
    }

    /// Write pattern: lock, mark insert, unlock
    #[divan::bench]
    fn write_with_insert(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_insert();
                // Guard drops, releasing lock and incrementing version
            });
    }

    /// Write pattern: lock, mark split, unlock
    #[divan::bench]
    fn write_with_split(bencher: Bencher) {
        bencher
            .with_inputs(NodeVersion::default)
            .bench_local_values(|v| {
                let mut guard = v.lock();
                guard.mark_split();
                // Guard drops, releasing lock and incrementing version
            });
    }

    /// Multiple flag checks (common in traversal)
    #[divan::bench]
    fn check_all_flags(bencher: Bencher) {
        let v = NodeVersion::new(true);
        v.mark_root();
        bencher.bench_local(|| {
            let vref = black_box(&v);
            (
                vref.is_leaf(),
                vref.is_root(),
                vref.is_deleted(),
                vref.is_locked(),
                vref.is_dirty(),
            )
        });
    }

    /// Repeated optimistic reads (simulating concurrent readers)
    #[divan::bench(args = [1, 4, 8, 16])]
    fn repeated_optimistic_reads(bencher: Bencher, n: usize) {
        let v = NodeVersion::new(true);
        bencher.bench_local(|| {
            for _ in 0..n {
                let stable = black_box(&v).stable();
                let _ = black_box(stable);
                let _ = black_box(&v).has_changed(stable);
            }
        });
    }
}
