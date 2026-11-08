use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

use masstree::{MassTree, MassTree15, ValueRef};

#[test]
fn guarded_insert_returns_stable_write_through_snapshots() {
    let tree = MassTree15::<u64>::new();
    let guard = tree.guard();
    assert!(tree.insert_with_guard(b"key", 1, &guard).is_none());
    let old = tree.insert_with_guard(b"key", 2, &guard).unwrap();
    assert!(matches!(old, ValueRef::Owned(1)));

    let barrier = Barrier::new(2);
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let writer_guard = tree.guard();
            barrier.wait();
            for value in 3..=10 {
                assert_eq!(
                    *tree
                        .insert_with_guard(b"key", value, &writer_guard)
                        .unwrap(),
                    value - 1,
                );
            }
            done_tx.send(()).unwrap();
        });
        barrier.wait();
        assert_eq!(*old, 1);
        done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        assert_eq!(*old, 1);
    });
    assert_eq!(tree.get(b"key"), Some(10));
}

#[test]
fn guarded_insert_borrows_nonclone_old_values_until_guard_ends() {
    struct Tracked {
        id: usize,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    {
        let tree = MassTree15::<Tracked>::new();
        let guard = tree.guard();
        let tracked = |id| Tracked {
            id,
            drops: Arc::clone(&drops),
        };
        assert!(tree.insert_with_guard(b"key", tracked(1), &guard).is_none());
        let first = tree.insert_with_guard(b"key", tracked(2), &guard).unwrap();
        let second = tree.insert_with_guard(b"key", tracked(3), &guard).unwrap();
        assert!(matches!(first, ValueRef::Borrowed(_)));
        assert_eq!(first.id, 1);
        assert_eq!(second.id, 2);
        assert_eq!(tree.get_with_guard(b"key", &guard).unwrap().id, 3);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
    }
    assert_eq!(drops.load(Ordering::Relaxed), 3);
}

#[test]
fn guarded_insert_returns_owned_inline_values() {
    let tree = MassTree::<u64>::new();
    let guard = tree.guard();
    assert!(tree.insert_with_guard(b"key", 41, &guard).is_none());
    let old = tree.insert_with_guard(b"key", 42, &guard).unwrap();
    assert!(matches!(old, ValueRef::Owned(41)));
    assert_eq!(tree.insert_value_with_guard(b"key", 43, &guard), Some(42));
    assert_eq!(tree.insert(b"key", 44), Some(43));
}
