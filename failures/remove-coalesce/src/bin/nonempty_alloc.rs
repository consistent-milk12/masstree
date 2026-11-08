//! Count real allocations while removing keys from nonempty layered leaves.
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use masstree::{LeafPolicy, MassTree, MassTree15, MassTreeGeneric, SeizeAllocator};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Requests are forwarded unchanged to System. The counter neither
// accesses allocation contents nor changes ownership or layout.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: GlobalAlloc's caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The caller returns the same pointer/layout to its allocator.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn exercise<P: LeafPolicy<Value = u64>>(
    tree: MassTreeGeneric<P, SeizeAllocator<P>>,
    key_len: usize,
) -> usize {
    let keys: Vec<_> = (0..10u64)
        .map(|i| {
            let mut key = vec![b'p'; key_len];
            key[key_len - 8..].copy_from_slice(&i.to_be_bytes());
            key
        })
        .collect();
    let mut model = BTreeMap::new();
    let guard = tree.guard();
    for (i, key) in keys.iter().enumerate() {
        tree.insert_with_guard(key, i as u64, &guard);
        model.insert(key.clone(), i as u64);
    }
    // Prime EBR retirement bookkeeping; the measurement isolates the route.
    tree.remove_with_guard(&keys[0], &guard).unwrap();
    tree.insert_with_guard(&keys[0], 0, &guard);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for (i, key) in keys.iter().take(8).enumerate() {
        assert_eq!(
            *tree.remove_with_guard(key, &guard).unwrap().unwrap(),
            i as u64
        );
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;

    for key in keys.iter().take(8) {
        model.remove(key);
        assert!(tree.get(key).is_none());
    }
    assert_eq!(tree.len(), model.len());
    let ordered: Vec<_> = tree
        .iter(&guard)
        .map(|entry| entry.key().to_vec())
        .collect();
    assert_eq!(ordered, model.keys().cloned().collect::<Vec<_>>());
    for (key, value) in &model {
        assert_eq!(tree.get(key), Some(*value));
        tree.remove_with_guard(key, &guard).unwrap();
    }
    tree.process_coalesce(&guard);
    assert!(tree.is_empty());
    assert_eq!(tree.iter(&guard).count(), 0);
    allocations
}

fn main() {
    let allocations = [
        exercise(MassTree15::new(), 32),
        exercise(MassTree::new(), 32),
        exercise(MassTree15::new(), 256),
        exercise(MassTree::new(), 256),
    ];
    println!("eight nonempty removals, boxed/inline at 32/256 bytes: {allocations:?}");
    assert_eq!(allocations, [0; 4]);
}
