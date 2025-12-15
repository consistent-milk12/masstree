//! Fast benchmarks for `Permuter` using Divan.
//!
//! Run with: `cargo bench --bench permuter`

use divan::{Bencher, black_box};
use masstree::permuter::Permuter;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
mod construction {
    use super::{Permuter, black_box};

    #[divan::bench]
    const fn empty_15() -> Permuter<15> {
        Permuter::empty()
    }

    #[divan::bench]
    const fn empty_7() -> Permuter<7> {
        Permuter::empty()
    }

    #[divan::bench]
    const fn empty_3() -> Permuter<3> {
        Permuter::empty()
    }

    #[divan::bench(args = [0, 1, 5, 10, 15])]
    fn make_sorted_15(n: usize) -> Permuter<15> {
        Permuter::make_sorted(black_box(n))
    }

    #[divan::bench(args = [0, 1, 3, 7])]
    fn make_sorted_7(n: usize) -> Permuter<7> {
        Permuter::make_sorted(black_box(n))
    }
}

// =============================================================================
// Accessors
// =============================================================================

#[divan::bench_group]
mod accessors {
    use super::{Bencher, Permuter, black_box};

    #[divan::bench]
    fn size(bencher: Bencher) {
        let p: Permuter<15> = Permuter::make_sorted(10);
        bencher.bench_local(|| black_box(&p).size());
    }

    #[divan::bench(args = [0, 7, 14])]
    fn get(bencher: Bencher, pos: usize) {
        let p: Permuter<15> = Permuter::make_sorted(15);
        bencher.bench_local(|| black_box(&p).get(black_box(pos)));
    }

    #[divan::bench]
    fn back(bencher: Bencher) {
        let p: Permuter<15> = Permuter::make_sorted(7);
        bencher.bench_local(|| black_box(&p).back());
    }

    #[divan::bench]
    fn value(bencher: Bencher) {
        let p: Permuter<15> = Permuter::make_sorted(15);
        bencher.bench_local(|| black_box(&p).value());
    }

    #[divan::bench]
    fn scan_all_15(bencher: Bencher) {
        let p: Permuter<15> = Permuter::make_sorted(15);
        bencher.bench_local(|| {
            let p = black_box(&p);
            let mut sum: usize = 0;
            for i in 0..15 {
                sum = sum.wrapping_add(p.get(i));
            }
            sum
        });
    }
}

// =============================================================================
// Insertion
// =============================================================================

#[divan::bench_group]
mod insert {
    use super::{Bencher, Permuter};

    #[divan::bench]
    fn insert_at_0_empty(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                let slot = p.insert_from_back(0);
                (p, slot)
            });
    }

    #[divan::bench]
    fn insert_at_0_half(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(7))
            .bench_local_values(|mut p| {
                let slot = p.insert_from_back(0);
                (p, slot)
            });
    }

    #[divan::bench]
    fn insert_at_end(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(7))
            .bench_local_values(|mut p| {
                let slot = p.insert_from_back(7);
                (p, slot)
            });
    }

    #[divan::bench]
    fn insert_at_middle(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                let slot = p.insert_from_back(5);
                (p, slot)
            });
    }

    #[divan::bench]
    fn fill_15_at_end(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                for i in 0..15 {
                    let _ = p.insert_from_back(i);
                }
                p
            });
    }

    #[divan::bench]
    fn fill_15_at_beginning(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                for _ in 0..15 {
                    let _ = p.insert_from_back(0);
                }
                p
            });
    }
}

// =============================================================================
// Removal
// =============================================================================

#[divan::bench_group]
mod remove {
    use super::{Bencher, Permuter};

    #[divan::bench]
    fn remove_first(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.remove(0);
                p
            });
    }

    #[divan::bench]
    fn remove_last(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.remove(9);
                p
            });
    }

    #[divan::bench]
    fn remove_middle(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.remove(5);
                p
            });
    }

    #[divan::bench]
    fn remove_to_back_first(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.remove_to_back(0);
                p
            });
    }

    #[divan::bench]
    fn remove_to_back_middle(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.remove_to_back(5);
                p
            });
    }

    #[divan::bench]
    fn drain_15_from_end(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                for i in (0..15).rev() {
                    p.remove(i);
                }
                p
            });
    }

    #[divan::bench]
    fn drain_15_from_beginning(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                for _ in 0..15 {
                    p.remove(0);
                }
                p
            });
    }
}

// =============================================================================
// Exchange and Rotate
// =============================================================================

#[divan::bench_group]
mod exchange_rotate {
    use super::{Bencher, Permuter};

    #[divan::bench]
    fn exchange_adjacent(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                p.exchange(5, 6);
                p
            });
    }

    #[divan::bench]
    fn exchange_distant(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                p.exchange(0, 14);
                p
            });
    }

    #[divan::bench]
    fn exchange_same(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                p.exchange(7, 7);
                p
            });
    }

    #[divan::bench]
    fn rotate_small(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                p.rotate(5, 7);
                p
            });
    }

    #[divan::bench]
    fn rotate_large(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(15))
            .bench_local_values(|mut p| {
                p.rotate(2, 12);
                p
            });
    }
}

// =============================================================================
// Realistic Workloads
// =============================================================================

#[divan::bench_group]
mod workload {
    use super::{Bencher, Permuter, black_box};

    #[divan::bench]
    fn build_sorted_10(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                for i in 0..10 {
                    let _ = p.insert_from_back(i);
                }
                p
            });
    }

    #[divan::bench]
    fn build_interleaved_10(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                let positions = [0, 0, 1, 0, 2, 1, 3, 2, 4, 3];
                for &pos in &positions {
                    let _ = p.insert_from_back(pos);
                }
                p
            });
    }

    #[divan::bench]
    fn binary_search_pattern(bencher: Bencher) {
        let p: Permuter<15> = Permuter::make_sorted(15);
        bencher.bench_local(|| {
            let p = black_box(&p);
            (p.get(7), p.get(11), p.get(9), p.get(10))
        });
    }

    #[divan::bench]
    fn insert_remove_mix(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                for i in 0..8 {
                    let _ = p.insert_from_back(i);
                }
                p.remove(3);
                p.remove(4);
                let _ = p.insert_from_back(2);
                let _ = p.insert_from_back(4);
                p
            });
    }

    #[divan::bench]
    fn set_size(bencher: Bencher) {
        bencher
            .with_inputs(|| Permuter::<15>::make_sorted(10))
            .bench_local_values(|mut p| {
                p.set_size(5);
                p
            });
    }
}

// =============================================================================
// Width Comparison
// =============================================================================

#[divan::bench_group]
mod width_cmp {
    use super::{Bencher, Permuter};

    #[divan::bench]
    fn fill_width_15(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<15>::empty)
            .bench_local_values(|mut p| {
                for i in 0..15 {
                    let _ = p.insert_from_back(i);
                }
                p
            });
    }

    #[divan::bench]
    fn fill_width_7(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<7>::empty)
            .bench_local_values(|mut p| {
                for i in 0..7 {
                    let _ = p.insert_from_back(i);
                }
                p
            });
    }

    #[divan::bench]
    fn fill_width_3(bencher: Bencher) {
        bencher
            .with_inputs(Permuter::<3>::empty)
            .bench_local_values(|mut p| {
                for i in 0..3 {
                    let _ = p.insert_from_back(i);
                }
                p
            });
    }
}
