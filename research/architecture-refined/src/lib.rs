//! Executable evidence for ARCHITECTURERefined.md, not a replacement tree.

#[cfg(test)]
mod tests {
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicU64, Ordering, fence};
    use loom::thread;

    const LOCK: u64 = 1;
    const DIRTY: u64 = 2;

    // Two independently atomic fields represent one logical metadata pair.
    // This deliberately does not model pointers, EBR, or a complete leaf.
    fn check_snapshot(writer_fence: bool, reader_fence: bool) {
        loom::model(move || {
            let state = Arc::new((AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)));
            let writer_state = state.clone();
            let writer = thread::spawn(move || {
                let (version, x, y) = &*writer_state;
                version
                    .compare_exchange(0, LOCK, Ordering::Acquire, Ordering::Relaxed)
                    .unwrap();
                version.store(LOCK | DIRTY, Ordering::Relaxed);
                if writer_fence {
                    fence(Ordering::Release);
                }
                x.store(1, Ordering::Relaxed);
                y.store(1, Ordering::Relaxed);
                version.store(4, Ordering::Release);
            });
            let (version, x, y) = &*state;
            let before = version.load(Ordering::Acquire);
            if before & DIRTY == 0 {
                let sx = x.load(Ordering::Relaxed);
                let sy = y.load(Ordering::Relaxed);
                if reader_fence {
                    fence(Ordering::Acquire);
                }
                let after = version.load(Ordering::Relaxed);
                if (before & !LOCK) == (after & !LOCK) && after & DIRTY == 0 {
                    assert_eq!(sx, sy, "accepted a torn metadata snapshot");
                }
            }
            writer.join().unwrap();
        });
    }

    #[test]
    fn proposed_snapshot_protocol() {
        check_snapshot(true, true);
    }

    #[test]
    #[should_panic(expected = "accepted a torn metadata snapshot")]
    fn negative_control_without_fences() {
        check_snapshot(false, false);
    }

    #[test]
    #[should_panic(expected = "accepted a torn metadata snapshot")]
    fn negative_control_without_writer_fence() {
        check_snapshot(false, true);
    }

    #[test]
    #[should_panic(expected = "accepted a torn metadata snapshot")]
    fn negative_control_without_reader_fence() {
        check_snapshot(true, false);
    }
}
