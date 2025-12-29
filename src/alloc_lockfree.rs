//! Lock-free node allocation for MassTree.
//!
//! Uses atomic operations instead of mutex for tracking, eliminating
//! contention on the allocation hot path.

#![allow(missing_docs)]

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use seize::LocalGuard;

use crate::internode::InternodeNode;
use crate::leaf_trait::TreeLeafNode;
use crate::slot::ValueSlot;

/// Width constant for internodes.
const INTERNODE_WIDTH: usize = 15;

/// A node in the lock-free tracking list.
///
/// We use an intrusive linked list where each tracked pointer
/// is wrapped in a Node that points to the next.
struct TrackNode<T> {
    ptr: *mut T,
    next: AtomicPtr<TrackNode<T>>,
}

impl<T> TrackNode<T> {
    fn new(ptr: *mut T) -> Box<Self> {
        Box::new(Self {
            ptr,
            next: AtomicPtr::new(ptr::null_mut()),
        })
    }
}

/// Lock-free stack (Treiber stack) for tracking allocations.
struct LockFreeStack<T> {
    head: AtomicPtr<TrackNode<T>>,
    len: AtomicUsize,
}

impl<T> LockFreeStack<T> {
    const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            len: AtomicUsize::new(0),
        }
    }

    /// Push a pointer onto the stack (lock-free).
    fn push(&self, ptr: *mut T) {
        let node = Box::into_raw(TrackNode::new(ptr));

        loop {
            let head = self.head.load(Ordering::Acquire);

            // SAFETY: node is valid, we just allocated it
            unsafe {
                (*node).next.store(head, Ordering::Relaxed);
            }

            if self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.len.fetch_add(1, Ordering::Relaxed);
                return;
            }
            // CAS failed, retry
            std::hint::spin_loop();
        }
    }

    /// Take all items from the stack (for cleanup).
    fn take_all(&self) -> Vec<*mut T> {
        // Atomically swap head with null
        let head = self.head.swap(ptr::null_mut(), Ordering::AcqRel);

        if head.is_null() {
            return Vec::new();
        }

        // Traverse the list and collect all pointers
        let mut result = Vec::new();
        let mut current = head;

        while !current.is_null() {
            // SAFETY: current is a valid TrackNode we allocated
            let node = unsafe { Box::from_raw(current) };
            result.push(node.ptr);
            current = node.next.load(Ordering::Relaxed);
            // node is dropped here, freeing the TrackNode (but not the payload)
        }

        self.len.store(0, Ordering::Relaxed);
        result
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        // Just free the TrackNodes, not the payloads
        let head = *self.head.get_mut();
        let mut current = head;

        while !current.is_null() {
            let mut node = unsafe { Box::from_raw(current) };
            current = *node.next.get_mut();
            // Don't free node.ptr - that's handled by take_all or teardown
        }
    }
}

// SAFETY: The stack only contains raw pointers which are Send
unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

/// Lock-free allocator for tree nodes.
///
/// Uses atomic CAS operations instead of mutex for tracking allocations.
/// This eliminates contention on the allocation hot path.
pub struct LockFreeAllocator<L: TreeLeafNode<S>, S: ValueSlot> {
    leaf_stack: LockFreeStack<L>,
    internode_stack: LockFreeStack<InternodeNode<S, INTERNODE_WIDTH>>,
    _marker: std::marker::PhantomData<S>,
}

impl<L: TreeLeafNode<S>, S: ValueSlot> std::fmt::Debug for LockFreeAllocator<L, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockFreeAllocator")
            .field("leaf_count", &self.leaf_stack.len())
            .field("internode_count", &self.internode_stack.len())
            .finish()
    }
}

impl<L: TreeLeafNode<S>, S: ValueSlot> LockFreeAllocator<L, S> {
    pub const fn new() -> Self {
        Self {
            leaf_stack: LockFreeStack::new(),
            internode_stack: LockFreeStack::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_stack.len()
    }

    pub fn internode_count(&self) -> usize {
        self.internode_stack.len()
    }
}

impl<L: TreeLeafNode<S>, S: ValueSlot> Default for LockFreeAllocator<L, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L, S> crate::alloc_trait::NodeAllocatorGeneric<S, L> for LockFreeAllocator<L, S>
where
    L: TreeLeafNode<S> + Send + Sync + 'static,
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<L>) -> *mut L {
        let ptr = Box::into_raw(node);
        self.leaf_stack.push(ptr);
        ptr
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut L) {
        self.leaf_stack.push(ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, _ptr: *mut L, _guard: &LocalGuard<'_>) {
        // For now, don't remove from tracking - let teardown handle it
        // A more sophisticated implementation would use hazard pointers or
        // epoch-based removal
    }

    #[inline(always)]
    #[expect(clippy::cast_ptr_alignment)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        let node: Box<InternodeNode<S, INTERNODE_WIDTH>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S, INTERNODE_WIDTH>>()) };
        let ptr = Box::into_raw(node);
        self.internode_stack.push(ptr);
        ptr.cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        self.internode_stack.push(ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, _ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // For now, don't remove from tracking - let teardown handle it
    }

    #[inline(always)]
    fn teardown_tree(&self, _root_ptr: *mut u8) {
        // Free all tracked leaves
        for ptr in self.leaf_stack.take_all() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        // Free all tracked internodes
        for ptr in self.internode_stack.take_all() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, _root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        // Not implemented
    }
}

impl<L: TreeLeafNode<S>, S: ValueSlot> Drop for LockFreeAllocator<L, S> {
    fn drop(&mut self) {
        // Free all tracked nodes
        for ptr in self.leaf_stack.take_all() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        for ptr in self.internode_stack.take_all() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf24::LeafNode24;
    use crate::value::LeafValue;

    #[test]
    fn test_lockfree_allocator_new() {
        let alloc: LockFreeAllocator<LeafNode24<LeafValue<u64>>, LeafValue<u64>> =
            LockFreeAllocator::new();
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_lockfree_allocator_alloc_leaf() {
        use crate::alloc_trait::NodeAllocatorGeneric;

        let alloc: LockFreeAllocator<LeafNode24<LeafValue<u64>>, LeafValue<u64>> =
            LockFreeAllocator::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let ptr = alloc.alloc_leaf(leaf);
        assert!(!ptr.is_null());
        assert_eq!(alloc.leaf_count(), 1);

        unsafe {
            assert!((*ptr).is_empty());
        }
    }

    #[test]
    fn test_lockfree_stack_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let stack: Arc<LockFreeStack<u64>> = Arc::new(LockFreeStack::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let stack = Arc::clone(&stack);
                thread::spawn(move || {
                    for i in 0..1000 {
                        let ptr = Box::into_raw(Box::new(t * 1000 + i));
                        stack.push(ptr);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(stack.len(), 8000);

        // Cleanup
        for ptr in stack.take_all() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}
