use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::{BoxPolicy, MassTreeGeneric, SeizeAllocator, TreeAllocator};
use seize::LocalGuard;

#[derive(Default)]
struct Control {
    watched: AtomicPtr<NodeVersion>,
    locked_allocations: AtomicUsize,
    retired: AtomicUsize,
    pause_once: AtomicBool,
    started: Mutex<Option<mpsc::Sender<()>>>,
    resume: Mutex<Option<mpsc::Receiver<()>>>,
    timed_out: AtomicBool,
}

struct ObservedAllocator<P: LeafPolicy> {
    inner: SeizeAllocator<P>,
    control: Arc<Control>,
}

// SAFETY: Every allocation/reclamation operation delegates unchanged to
// SeizeAllocator. The observer never panics and only probes a version whose
// leaf the tests keep protected by a guard for the observation's lifetime.
unsafe impl<P: LeafPolicy> TreeAllocator<P> for ObservedAllocator<P> {
    fn alloc_leaf_direct(&self, root: bool, layer: bool) -> *mut LeafNode15<P> {
        let watched = self.control.watched.load(Ordering::Acquire);
        if layer && !watched.is_null() {
            // SAFETY: The test keeps this node protected until it disarms us.
            let version = unsafe { &*watched };
            if version.try_lock().is_none() {
                self.control
                    .locked_allocations
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if layer && self.control.pause_once.swap(false, Ordering::Relaxed) {
            let started = self
                .control
                .started
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(started) = started {
                let _ = started.send(());
            }
            let resume = self
                .control
                .resume
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(resume) = resume
                && resume.recv_timeout(Duration::from_secs(30)).is_err()
            {
                self.control.timed_out.store(true, Ordering::Relaxed);
            }
        }
        self.inner.alloc_leaf_direct(root, layer)
    }

    unsafe fn retire_leaf(&self, leaf: *mut LeafNode15<P>, guard: &LocalGuard<'_>) {
        self.control.retired.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The caller transfers an unreachable node, as required by the trait.
        unsafe { self.inner.retire_leaf(leaf, guard) };
    }

    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        self.inner.alloc_internode_direct_root(height)
    }

    fn alloc_internode_direct_for_split(&self, version: &NodeVersion, height: u32) -> *mut u8 {
        self.inner.alloc_internode_direct_for_split(version, height)
    }

    unsafe fn retire_internode_erased(&self, node: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: The caller supplies an unreachable internode from this allocator.
        unsafe { self.inner.retire_internode_erased(node, guard) };
    }

    unsafe fn teardown_tree(&self, root: *mut u8) {
        // SAFETY: The caller has exclusive ownership of the complete subtree.
        unsafe { self.inner.teardown_tree(root) };
    }
}

#[test]
fn twig_allocations_release_the_parent_lock() {
    let control = Arc::new(Control::default());
    let tree = MassTreeGeneric::<BoxPolicy<u64>, _>::with_allocator(ObservedAllocator {
        inner: SeizeAllocator::new(),
        control: Arc::clone(&control),
    });
    let guard = tree.guard();
    let first = [b'x'; 256];
    let mut second = first;
    second[255] += 1;
    tree.insert_with_guard(&first, 1, &guard);
    // SAFETY: A one-key tree has a leaf root, protected by guard until disarmed.
    let parent = unsafe {
        &*tree
            .load_root_ptr_generic(&guard)
            .cast::<LeafNode15<BoxPolicy<u64>>>()
    };
    control.watched.store(
        ptr::from_ref(parent.version()).cast_mut(),
        Ordering::Release,
    );
    tree.insert_with_guard(&second, 2, &guard);
    control.watched.store(ptr::null_mut(), Ordering::Release);
    assert_eq!(tree.get(&first), Some(1));
    assert_eq!(tree.get(&second), Some(2));
    assert_eq!(control.locked_allocations.load(Ordering::Relaxed), 0);
}

struct Tracked {
    value: u64,
    drops: Arc<AtomicUsize>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
enum Mutation {
    Update,
    Convert,
    Remove,
}

#[expect(
    clippy::too_many_lines,
    reason = "One bounded interleaving with ownership assertions"
)]
fn revalidate_after(mutation: Mutation) {
    let control = Arc::new(Control::default());
    let drops = Arc::new(AtomicUsize::new(0));
    let tracked = |value| Tracked {
        value,
        drops: Arc::clone(&drops),
    };
    let tree = MassTreeGeneric::<BoxPolicy<Tracked>, _>::with_allocator(ObservedAllocator {
        inner: SeizeAllocator::new(),
        control: Arc::clone(&control),
    });
    let guard = tree.guard();
    let first = [b'x'; 256];
    let mut second = first;
    second[255] += 1;
    let mut third = first;
    third[255] += 2;
    tree.insert_with_guard(&first, tracked(0), &guard);

    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    *control.started.lock().unwrap() = Some(started_tx);
    *control.resume.lock().unwrap() = Some(resume_rx);
    control.pause_once.store(true, Ordering::Relaxed);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let guard = tree.guard();
            tree.insert_with_guard(&second, tracked(1), &guard);
        });
        let started = started_rx.recv_timeout(Duration::from_secs(30));
        let (done_tx, done_rx) = mpsc::channel();
        let tree = &tree;
        scope.spawn(move || {
            let guard = tree.guard();
            match mutation {
                Mutation::Update => {
                    tree.insert_with_guard(&first, tracked(99), &guard);
                }
                Mutation::Convert => {
                    tree.insert_with_guard(&third, tracked(2), &guard);
                }
                Mutation::Remove => {
                    tree.remove_with_guard(&first, &guard).unwrap();
                }
            }
            let _ = done_tx.send(());
        });
        let progressed = done_rx.recv_timeout(Duration::from_secs(30));
        // Always release the allocator before asserting, including on failure.
        let _ = resume_tx.send(());
        assert!(
            started.is_ok(),
            "twig builder did not reach the allocation hook"
        );
        assert!(
            progressed.is_ok(),
            "another writer could not acquire the parent"
        );
    });

    assert!(!control.timed_out.load(Ordering::Relaxed));
    assert!(
        control.retired.load(Ordering::Relaxed) >= 31,
        "the abandoned maximum-depth chain must be retired"
    );
    assert_eq!(tree.get_with_guard(&second, &guard).unwrap().value, 1);
    match mutation {
        Mutation::Update => {
            assert_eq!(tree.get_with_guard(&first, &guard).unwrap().value, 99);
            assert_eq!(tree.len(), 2);
        }
        Mutation::Convert => {
            assert_eq!(tree.get_with_guard(&first, &guard).unwrap().value, 0);
            assert_eq!(tree.get_with_guard(&third, &guard).unwrap().value, 2);
            assert_eq!(tree.len(), 3);
        }
        Mutation::Remove => {
            assert!(tree.get_with_guard(&first, &guard).is_none());
            assert_eq!(tree.len(), 1);
        }
    }
    drop(guard);
    drop(tree);
    let expected = if matches!(mutation, Mutation::Remove) {
        2
    } else {
        3
    };
    assert_eq!(drops.load(Ordering::Relaxed), expected);
}

#[test]
fn twig_preparation_retries_after_value_update() {
    revalidate_after(Mutation::Update);
}

#[test]
fn twig_preparation_retries_after_layer_conversion() {
    revalidate_after(Mutation::Convert);
}

#[test]
fn twig_preparation_retries_after_source_removal() {
    revalidate_after(Mutation::Remove);
}
