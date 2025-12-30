//! Lock-free pool allocator for [`MassTree`] nodes.
//!
//! This module provides O(1) node allocation from lock-free free lists,
//! eliminating global allocator overhead on the hot path.
//!
//! # Design
//!
//! - **Lock-free pools**: Atomic CAS operations for thread-safe allocation
//! - **Allocator-owned chunks**: Memory lifetime tied to allocator, not threads
//! - **Intrusive linked lists**: Free blocks store next pointer in first 8 bytes
//! - **2MB chunks**: Large allocations amortize refill cost
//! - **seize integration**: Deferred return to pool after epoch grace period
//!
//! # Reference
//!
//! Based on C++ Masstree's `kvthread.hh:226-259` pool allocator.
//!
//! See `CODE_003.md`, `CODE_004.md`, `CODE_005.md` for design documents.

use std::alloc::{Layout, alloc, dealloc};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, Ordering};

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::internode::InternodeNode;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;

// =============================================================================
// Constants
// =============================================================================

/// Size of allocation chunks (2MB, matching C++ `refill_pool`).
const CHUNK_SIZE: usize = 2 << 20; // 2MB

/// Cache line size for alignment.
const CACHE_LINE: usize = 64;

/// Width constant for internodes (limited by 4-bit permutation slots).
const INTERNODE_WIDTH: usize = 15;

/// Block size for leaf nodes (rounded up to cache line).
const LEAF_BLOCK_SIZE: usize = 2048; // 32 cache lines

/// Block size for internodes (rounded up to cache line).
const INTERNODE_BLOCK_SIZE: usize = 320; // 5 cache lines

// =============================================================================
// FreeBlock - Intrusive Linked List Node
// =============================================================================

/// A node in the intrusive free list.
///
/// When a block is free, its first 8 bytes store the next pointer.
/// This struct is only used to interpret freed memory - the actual
/// node data is overwritten when the block is in the free list.
#[repr(C)]
struct FreeBlock {
    next: *mut Self,
}

// =============================================================================
// PoolChunk - 2MB Memory Chunks
// =============================================================================

/// A 2MB chunk of memory for pool allocation.
///
/// Chunks are allocated via the global allocator and carved into
/// fixed-size blocks linked via intrusive pointers.
struct PoolChunk {
    /// Pointer to the chunk's memory.
    ptr: NonNull<u8>,
    /// Layout used for deallocation.
    layout: Layout,
}

// SAFETY: PoolChunk contains only a pointer and layout.
// The pointer is only accessed through synchronized methods.
unsafe impl Send for PoolChunk {}

impl PoolChunk {
    /// Allocate a new 2MB chunk aligned to cache line boundaries.
    fn new() -> Option<Self> {
        let layout = Layout::from_size_align(CHUNK_SIZE, CACHE_LINE).ok()?;
        // SAFETY: layout is valid (non-zero size, power-of-two alignment)
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr)?;
        Some(Self { ptr, layout })
    }

    /// Carve this chunk into a linked list of fixed-size blocks.
    ///
    /// Returns the head of the linked list.
    ///
    /// # Safety
    ///
    /// - `block_size` must be >= 8 bytes
    /// - `block_size` must be a multiple of [`CACHE_LINE`]
    unsafe fn carve_into_list(&self, block_size: usize) -> *mut FreeBlock {
        debug_assert!(block_size >= mem::size_of::<*mut ()>());
        debug_assert!(block_size.is_multiple_of(CACHE_LINE));

        let base = self.ptr.as_ptr();
        let num_blocks = CHUNK_SIZE / block_size;

        if num_blocks == 0 {
            return ptr::null_mut();
        }

        // Link blocks in reverse order so first block is at lowest address
        let mut head: *mut FreeBlock = ptr::null_mut();
        for i in (0..num_blocks).rev() {
            // SAFETY: i * block_size is always < CHUNK_SIZE
            let block_ptr = unsafe { base.add(i * block_size).cast::<FreeBlock>() };
            // SAFETY: block_ptr is valid and aligned
            unsafe {
                (*block_ptr).next = head;
            }
            head = block_ptr;
        }
        head
    }
}

impl Drop for PoolChunk {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout in new()
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// =============================================================================
// LockFreePool - Atomic Free List
// =============================================================================

/// A lock-free pool for a single block size.
///
/// Uses a Treiber stack (lock-free LIFO) for O(1) allocation and deallocation.
struct LockFreePool {
    /// Head of the free list (atomic for lock-free access).
    head: AtomicPtr<FreeBlock>,
    /// Size of each block.
    block_size: usize,
    /// Owned chunks (protected by mutex for refill only).
    chunks: Mutex<Vec<PoolChunk>>,
}

impl LockFreePool {
    /// Create a new lock-free pool for blocks of the given size.
    const fn new(block_size: usize) -> Self {
        // Round up to cache line multiple
        let aligned_size = block_size.div_ceil(CACHE_LINE) * CACHE_LINE;
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            block_size: aligned_size,
            chunks: Mutex::new(Vec::new()),
        }
    }

    /// Allocate a block from the pool (lock-free).
    ///
    /// Returns null if allocation fails (OOM).
    #[inline]
    fn allocate(&self) -> *mut u8 {
        loop {
            let head = self.head.load(Ordering::Acquire);

            if head.is_null() {
                // Need to refill - this path is rare and uses a lock
                if !self.refill() {
                    return ptr::null_mut(); // OOM
                }
                continue;
            }

            // SAFETY: head is valid (from our chunk) and properly aligned
            let next = unsafe { (*head).next };

            // Try to pop the head
            if self
                .head
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return head.cast::<u8>();
            }
            // CAS failed, retry
        }
    }

    /// Return a block to the pool (lock-free).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated from this pool
    /// - `ptr` must not be used after this call
    #[inline]
    unsafe fn deallocate(&self, ptr: *mut u8) {
        debug_assert!(!ptr.is_null());
        let block: *mut FreeBlock = ptr.cast();

        loop {
            let head = self.head.load(Ordering::Acquire);

            // SAFETY: block is valid and aligned (caller guarantees)
            unsafe {
                (*block).next = head;
            }

            // Try to push as new head (CAS: expect head, set to block)
            if self
                .head
                .compare_exchange_weak(head, block, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
            // CAS failed, retry
        }
    }

    /// Refill the pool by allocating a new chunk.
    ///
    /// Returns true if refill succeeded, false on OOM.
    #[cold]
    fn refill(&self) -> bool {
        let mut chunks = self.chunks.lock();

        // Double-check if another thread already refilled
        if !self.head.load(Ordering::Acquire).is_null() {
            return true;
        }

        // Allocate a new chunk
        let chunk = match PoolChunk::new() {
            Some(c) => c,
            None => return false,
        };

        // Carve into blocks and prepend to free list
        // SAFETY: block_size is valid (>= 8, cache-line aligned)
        let new_head = unsafe { chunk.carve_into_list(self.block_size) };

        // Store the chunk (keeps memory alive)
        chunks.push(chunk);

        // Find the tail of the new list to link with existing head
        if !new_head.is_null() {
            let mut tail = new_head;
            // SAFETY: tail is valid and from our chunk
            while unsafe { !(*tail).next.is_null() } {
                tail = unsafe { (*tail).next };
            }

            // Link tail to current head and set new head
            loop {
                let old_head = self.head.load(Ordering::Acquire);
                // SAFETY: tail is valid
                unsafe {
                    (*tail).next = old_head;
                }
                if self
                    .head
                    .compare_exchange_weak(old_head, new_head, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }

        true
    }

    /// Returns the number of allocated chunks.
    #[cfg(test)]
    fn chunk_count(&self) -> usize {
        self.chunks.lock().len()
    }
}

#[expect(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for LockFreePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockFreePool")
            .field("block_size", &self.block_size)
            .field("chunks", &self.chunks.lock().len())
            .finish()
    }
}

// =============================================================================
// Global Static Pools
// =============================================================================

// Global pools allow deferred retirement callbacks to access the pool
// without capturing state (seize requires function pointers, not closures).

static GLOBAL_LEAF_POOL: OnceLock<LockFreePool> = OnceLock::new();
static GLOBAL_INTERNODE_POOL: OnceLock<LockFreePool> = OnceLock::new();

/// Get or initialize the global leaf pool.
fn global_leaf_pool() -> &'static LockFreePool {
    GLOBAL_LEAF_POOL.get_or_init(|| LockFreePool::new(LEAF_BLOCK_SIZE))
}

/// Get or initialize the global internode pool.
fn global_internode_pool() -> &'static LockFreePool {
    GLOBAL_INTERNODE_POOL.get_or_init(|| LockFreePool::new(INTERNODE_BLOCK_SIZE))
}

/// Reclaim a leaf node (callback for seize `defer_retire`).
///
/// This is a function pointer (not closure) so it can be used with seize.
///
/// # Safety
///
/// - `ptr` must be a valid initialized [`LeafNode24<S>`]
/// - Called by seize when epoch is safe (no concurrent readers)
unsafe fn reclaim_leaf<S: ValueSlot>(ptr: *mut LeafNode24<S>, _collector: &seize::Collector) {
    // SAFETY: ptr is valid, called when epoch is safe
    unsafe {
        ptr::drop_in_place(ptr);
        global_leaf_pool().deallocate(ptr.cast::<u8>());
    }
}

/// Reclaim an internode (callback for seize `defer_retire`).
///
/// # Safety
///
/// - `ptr` must be a valid initialized [`InternodeNode<S, 15>`]
/// - Called by seize when epoch is safe
unsafe fn reclaim_internode<S: ValueSlot>(
    ptr: *mut InternodeNode<S, INTERNODE_WIDTH>,
    _collector: &seize::Collector,
) {
    // SAFETY: ptr is valid, called when epoch is safe
    unsafe {
        ptr::drop_in_place(ptr);
        global_internode_pool().deallocate(ptr.cast::<u8>());
    }
}

// =============================================================================
// PoolAllocator24 - Main Allocator
// =============================================================================

/// Lock-free pool allocator for [`MassTree`] nodes.
///
/// Provides O(1) allocation from global lock-free free lists while
/// integrating with seize for epoch-based memory reclamation.
///
/// # Design
///
/// - **Global pools**: Shared across all allocators for callback compatibility
/// - **Allocation**: Atomic CAS pop from lock-free free list (O(1), no tracking)
/// - **Deallocation**: Atomic CAS push to lock-free free list (O(1), no tracking)
/// - **Retirement**: Defer return to pool via seize guard
/// - **Teardown**: Tree traversal to free all nodes (like C++ Masstree)
///
/// # Memory Model
///
/// Memory is allocated in 2MB chunks from the global allocator. Blocks are
/// recycled via the pool free list but chunks are never freed (they persist
/// until process exit). This matches typical allocator behavior for long-running
/// servers.
///
/// Unlike `SeizeAllocator24` which tracks every allocation in a [`HashSet`],
/// this allocator has zero per-allocation overhead. Teardown uses tree
/// traversal to find and free all nodes.
///
/// # Example
///
/// ```rust,ignore
/// use masstree::{MassTreePooled, PoolAllocator24};
///
/// let tree: MassTreePooled<u64> = MassTreePooled::with_allocator(
///     PoolAllocator24::new()
/// );
/// tree.insert(b"key", 42u64);
/// ```
pub struct PoolAllocator24<S: ValueSlot> {
    /// Marker for the slot type.
    _marker: PhantomData<S>,
}

// SAFETY: The allocator uses lock-free pools. All operations are thread-safe.
unsafe impl<S: ValueSlot + Send + Sync> Send for PoolAllocator24<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for PoolAllocator24<S> {}

impl<S: ValueSlot> std::fmt::Debug for PoolAllocator24<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolAllocator24")
            .field("leaf_pool_chunks", &global_leaf_pool().chunks.lock().len())
            .field(
                "internode_pool_chunks",
                &global_internode_pool().chunks.lock().len(),
            )
            .finish()
    }
}

impl<S: ValueSlot> PoolAllocator24<S> {
    /// Create a new pool allocator.
    ///
    /// All allocators share global pools for memory efficiency and
    /// compatibility with seize's function pointer callbacks.
    ///
    /// # Note
    ///
    /// This allocator does NOT track individual allocations. Cleanup is done
    /// via tree traversal in `teardown_tree`.
    #[must_use]
    pub fn new() -> Self {
        // Ensure global pools are initialized
        let _ = global_leaf_pool();
        let _ = global_internode_pool();

        Self {
            _marker: PhantomData,
        }
    }
}

impl<S: ValueSlot> Default for PoolAllocator24<S> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// NodeAllocatorGeneric Implementation
// =============================================================================

impl<S> NodeAllocatorGeneric<S, LeafNode24<S>> for PoolAllocator24<S>
where
    S: ValueSlot + Send + Sync + 'static,
{
    /// Allocate a leaf node from the global lock-free pool.
    ///
    /// The incoming Box is consumed for its initialized data, which is
    /// moved to pool-allocated memory. No per-allocation tracking - O(1).
    #[inline]
    fn alloc_leaf(&self, node: Box<LeafNode24<S>>) -> *mut LeafNode24<S> {
        // Allocate from global lock-free pool
        let ptr = global_leaf_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box if pool allocation fails (OOM)
            return Box::into_raw(node);
        }

        // Move the Box contents into pool memory
        let ptr = ptr.cast::<LeafNode24<S>>();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        unsafe {
            ptr::write(ptr, *node);
        }
        // Box wrapper is dropped (deallocates its backing memory, not contents)
        ptr
    }

    /// Allocate a leaf directly from the pool without Box overhead.
    ///
    /// This writes directly to pool memory, avoiding the Box allocation
    /// and copy that `alloc_leaf` requires. No per-allocation tracking - O(1).
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode24<S> {
        // Allocate from global lock-free pool
        let ptr = global_leaf_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box path on OOM
            let node = if is_layer_root {
                LeafNode24::new_layer_root()
            } else if is_root {
                LeafNode24::new_root()
            } else {
                LeafNode24::new()
            };
            return self.alloc_leaf(node);
        }

        let ptr: *mut LeafNode24<S> = ptr.cast();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        // Initialize directly in pool memory - no stack allocation!
        unsafe {
            LeafNode24::init_at(ptr, is_root || is_layer_root);
        }
        ptr
    }

    /// Track a leaf pointer for cleanup.
    ///
    /// No-op for pool allocator - we don't track individual allocations.
    #[inline]
    fn track_leaf(&self, _ptr: *mut LeafNode24<S>) {
        // No tracking - teardown uses tree traversal
    }

    /// Retire a leaf node for deferred reclamation.
    ///
    /// The node will be returned to the global pool once all concurrent readers
    /// have exited their critical sections.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator
    /// - `ptr` must be unreachable from the tree
    #[inline]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode24<S>, guard: &LocalGuard<'_>) {
        // Defer cleanup via seize - function pointer accesses global pool
        // SAFETY: ptr is valid and from our allocator (caller guarantees)
        unsafe {
            guard.defer_retire(ptr, reclaim_leaf::<S>);
        }
    }

    /// Allocate an internode from the global lock-free pool (type-erased).
    ///
    /// No per-allocation tracking - O(1).
    #[inline]
    #[expect(clippy::cast_ptr_alignment)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // node_ptr is Box::into_raw().cast() - reconstruct the Box
        // SAFETY: Caller guarantees node_ptr is a valid Box<InternodeNode<S, 15>>
        let node: Box<InternodeNode<S, INTERNODE_WIDTH>> =
            unsafe { Box::from_raw(node_ptr.cast()) };

        // Allocate from global lock-free pool
        let ptr = global_internode_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box if pool allocation fails
            return Box::into_raw(node).cast::<u8>();
        }

        // Move Box contents into pool memory
        let ptr = ptr.cast::<InternodeNode<S, INTERNODE_WIDTH>>();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        unsafe {
            ptr::write(ptr, *node);
        }

        ptr.cast::<u8>()
    }

    /// Allocate an internode directly from the pool without Box overhead.
    ///
    /// Writes directly to pool memory, avoiding the Box allocation.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        let ptr = global_internode_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box path on OOM
            let node: Box<InternodeNode<S, INTERNODE_WIDTH>> = InternodeNode::new(height);
            return self.alloc_internode_erased(Box::into_raw(node).cast());
        }

        let ptr: *mut InternodeNode<S, INTERNODE_WIDTH> = ptr.cast();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        unsafe {
            InternodeNode::init_at(ptr, height);
        }
        ptr.cast::<u8>()
    }

    /// Allocate an internode as root directly from the pool.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        let ptr = global_internode_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box path on OOM
            let node: Box<InternodeNode<S, INTERNODE_WIDTH>> = InternodeNode::new_root(height);
            return self.alloc_internode_erased(Box::into_raw(node).cast());
        }

        let ptr: *mut InternodeNode<S, INTERNODE_WIDTH> = ptr.cast();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        unsafe {
            InternodeNode::init_at_root(ptr, height);
        }
        ptr.cast::<u8>()
    }

    /// Allocate an internode for split directly from the pool.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> *mut u8 {
        let ptr = global_internode_pool().allocate();
        if ptr.is_null() {
            // Fallback to Box path on OOM
            let node: Box<InternodeNode<S, INTERNODE_WIDTH>> =
                InternodeNode::new_for_split(parent_version, height);
            return self.alloc_internode_erased(Box::into_raw(node).cast());
        }

        let ptr: *mut InternodeNode<S, INTERNODE_WIDTH> = ptr.cast();
        // SAFETY: ptr is valid, aligned (from pool), we have exclusive access
        unsafe {
            InternodeNode::init_at_for_split(ptr, parent_version, height);
        }
        ptr.cast::<u8>()
    }

    /// Track an internode pointer for cleanup.
    ///
    /// No-op for pool allocator - we don't track individual allocations.
    #[inline]
    fn track_internode_erased(&self, _ptr: *mut u8) {
        // No tracking - teardown uses tree traversal
    }

    /// Retire an internode for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid `InternodeNode<S, 15>`
    /// - `ptr` must be unreachable from the tree
    #[inline]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Defer cleanup via seize - function pointer accesses global pool
        // SAFETY: ptr is valid and from our allocator (caller guarantees)
        unsafe {
            guard.defer_retire(
                ptr.cast::<InternodeNode<S, INTERNODE_WIDTH>>(),
                reclaim_internode::<S>,
            );
        }
    }

    /// Teardown all nodes at tree drop via tree traversal.
    ///
    /// Walks the tree structure starting from root, freeing all nodes.
    /// This matches the C++ Masstree approach of traversing rather than tracking.
    fn teardown_tree(&self, root_ptr: *mut u8) {
        if root_ptr.is_null() {
            return;
        }
        // SAFETY: root_ptr is valid tree root, no concurrent access at teardown
        unsafe {
            self.teardown_node(root_ptr);
        }
    }

    /// Retire a subtree root via traversal.
    ///
    /// # Safety
    ///
    /// - The subtree must be fully unlinked from the main tree
    /// - `root_ptr` must point to a valid leaf or internode
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, _guard: &LocalGuard<'_>) {
        if root_ptr.is_null() {
            return;
        }
        // SAFETY: caller guarantees subtree is unlinked
        unsafe {
            self.teardown_node(root_ptr);
        }
    }
}

// =============================================================================
// Tree Traversal for Teardown
// =============================================================================

impl<S: ValueSlot + Send + Sync + 'static> PoolAllocator24<S> {
    /// Recursively teardown a node and all its descendants.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must be a valid pointer to either a leaf or internode
    /// - No concurrent access to these nodes (we're at tree drop)
    unsafe fn teardown_node(&self, node_ptr: *mut u8) {
        use crate::nodeversion::NodeVersion;

        // Read version to determine node type
        // SAFETY: node_ptr is valid, first field is NodeVersion
        let version_ptr = node_ptr as *const NodeVersion;

        let is_leaf = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf confirmed this is a LeafNode24
            let leaf: *mut LeafNode24<S> = node_ptr.cast();
            unsafe {
                self.teardown_leaf(leaf);
            }
        } else {
            // SAFETY: !is_leaf confirmed this is an InternodeNode
            let internode: *mut InternodeNode<S> = node_ptr.cast();
            unsafe {
                self.teardown_internode(internode);
            }
        }
    }

    /// Teardown a leaf node and any layer pointers it contains.
    ///
    /// # Safety
    ///
    /// - `leaf` must be a valid [`LeafNode24`] pointer
    /// - No concurrent access
    unsafe fn teardown_leaf(&self, leaf: *mut LeafNode24<S>) {
        use crate::leaf24::LAYER_KEYLENX;

        let leaf_ref = unsafe { &*leaf };

        // Check each slot for layer pointers (nested trees)
        let perm = leaf_ref.permutation();
        for i in 0..perm.size() {
            let slot = perm.get(i);
            let keylenx = leaf_ref.keylenx(slot);

            if keylenx >= LAYER_KEYLENX {
                // This is a layer pointer - recursively teardown the nested tree
                let layer_ptr = leaf_ref.leaf_value_ptr(slot);
                if !layer_ptr.is_null() {
                    unsafe {
                        self.teardown_node(layer_ptr);
                    }
                }
            }
        }

        // Drop the leaf contents and return to pool
        // SAFETY: leaf is valid, we have exclusive access
        unsafe {
            ptr::drop_in_place(leaf);
            global_leaf_pool().deallocate(leaf.cast::<u8>());
        }
    }

    /// Teardown an internode and all its children.
    ///
    /// # Safety
    ///
    /// - `internode` must be a valid [`InternodeNode`] pointer
    /// - No concurrent access
    unsafe fn teardown_internode(&self, internode: *mut InternodeNode<S, INTERNODE_WIDTH>) {
        let internode_ref = unsafe { &*internode };
        let nkeys = internode_ref.nkeys();

        // Recursively teardown all children (0..=nkeys)
        for i in 0..=nkeys {
            let child = internode_ref.child(i);
            if !child.is_null() {
                unsafe {
                    self.teardown_node(child);
                }
            }
        }

        // Drop the internode and return to pool
        // SAFETY: internode is valid, we have exclusive access
        unsafe {
            ptr::drop_in_place(internode);
            global_internode_pool().deallocate(internode.cast::<u8>());
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    #[test]
    fn test_lockfree_pool_alloc_dealloc() {
        let pool = LockFreePool::new(2048);

        let ptr1 = pool.allocate();
        assert!(!ptr1.is_null());

        let ptr2 = pool.allocate();
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);

        unsafe {
            pool.deallocate(ptr1);
            pool.deallocate(ptr2);
        }

        // Reallocation should return pointers (order depends on CAS timing)
        let ptr3 = pool.allocate();
        assert!(!ptr3.is_null());
    }

    #[test]
    fn test_lockfree_pool_chunk_carving() {
        let pool = LockFreePool::new(2048);

        // First allocation triggers chunk creation
        let ptr1 = pool.allocate();
        assert!(!ptr1.is_null());
        assert_eq!(pool.chunk_count(), 1);

        // 2MB / 2048 = 1024 blocks per chunk
        // Allocate all blocks in first chunk
        for _ in 1..1024 {
            let ptr = pool.allocate();
            assert!(!ptr.is_null());
        }

        // Next allocation should trigger new chunk
        let ptr = pool.allocate();
        assert!(!ptr.is_null());
        assert_eq!(pool.chunk_count(), 2);
    }

    #[test]
    fn test_pool_allocator_basic() {
        use crate::alloc_trait::NodeAllocatorGeneric;

        let alloc: PoolAllocator24<LeafValue<u64>> = PoolAllocator24::new();
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let ptr = alloc.alloc_leaf(leaf);
        assert!(!ptr.is_null());

        // Verify accessible
        unsafe {
            assert!((*ptr).is_empty());
        }

        // Clean up via teardown
        alloc.teardown_tree(ptr.cast::<u8>());
    }

    #[test]
    fn test_pool_allocator_alloc_leaf_direct() {
        use crate::alloc_trait::NodeAllocatorGeneric;

        let alloc: PoolAllocator24<LeafValue<u64>> = PoolAllocator24::new();

        let ptr = alloc.alloc_leaf_direct(false, false);
        assert!(!ptr.is_null());

        // Verify accessible and properly initialized
        unsafe {
            assert!((*ptr).is_empty());
        }

        // Clean up via teardown
        alloc.teardown_tree(ptr.cast::<u8>());
    }

    #[test]
    fn test_concurrent_allocation() {
        use crate::alloc_trait::NodeAllocatorGeneric;
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(PoolAllocator24::<LeafValue<u64>>::new());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let alloc = Arc::clone(&alloc);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
                        let ptr = alloc.alloc_leaf(leaf);
                        assert!(!ptr.is_null());
                        // Return to pool immediately (no tracking to verify)
                        unsafe {
                            ptr::drop_in_place(ptr);
                            global_leaf_pool().deallocate(ptr.cast::<u8>());
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        // Success = no crashes from concurrent allocation/deallocation
    }

    #[test]
    fn test_internode_pool_allocation() {
        let pool = LockFreePool::new(320);

        // First allocation
        let ptr1 = pool.allocate();
        assert!(!ptr1.is_null());
        assert_eq!(pool.chunk_count(), 1);

        // Block size should be rounded up to cache line
        assert_eq!(pool.block_size, 320);

        // 2MB / 320 = 6553 blocks per chunk
        // Allocate many blocks
        for _ in 1..6553 {
            let ptr = pool.allocate();
            assert!(!ptr.is_null());
        }

        // Next should trigger new chunk
        let ptr = pool.allocate();
        assert!(!ptr.is_null());
        assert_eq!(pool.chunk_count(), 2);
    }

    #[test]
    fn test_concurrent_alloc_dealloc() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(LockFreePool::new(2048));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    let mut ptrs = Vec::with_capacity(100);
                    for _ in 0..100 {
                        let ptr = pool.allocate();
                        assert!(!ptr.is_null());
                        ptrs.push(ptr);
                    }
                    // Deallocate all
                    for ptr in ptrs {
                        unsafe {
                            pool.deallocate(ptr);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}
