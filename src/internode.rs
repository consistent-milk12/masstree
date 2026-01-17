//! Filepath: src/internode.rs
//!
//! Internode (internal node) for `MassTree`.
//!
//! Internodes route traversals through the tree. They contain only
//! keys and child pointers, no values. Keys are always in sorted order
//! (no permutation array needed).
//!
//! # Memory Layout (WIDTH=15)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Cache Line 0 (64 bytes)                                         │
//! │   version: NodeVersion (4 bytes)                                │
//! │   nkeys: AtomicU8 (1 byte)                                      │
//! │   height: u8 (1 byte)                                           │
//! │   _pad: [u8; 2] (2 bytes alignment)                             │
//! │   parent: AtomicPtr<u8> (8 bytes)                               │
//! │   ikey0[0..6]: [AtomicU64; 6] (48 bytes)                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Cache Lines 1-2 (128 bytes)                                     │
//! │   ikey0[6..15]: [AtomicU64; 9] (72 bytes)                       │
//! │   child[0..7]: [AtomicPtr<u8>; 7] (56 bytes)                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Cache Lines 3-4 (128 bytes)                                     │
//! │   child[7..16]: [AtomicPtr<u8>; 9] (72 bytes)                   │
//! └─────────────────────────────────────────────────────────────────┘
//! Total: ~280 bytes (5 cache lines)
//! ```
//!
//! # B+Tree Routing Model
//!
//! ```text
//!         [K0 | K1 | K2]           <- Internode (3 keys, 4 children)
//!        /    |    |    \
//!    C0     C1    C2     C3        <- Children
//!
//!    C0: keys < K0
//!    C1: keys >= K0 and < K1
//!    C2: keys >= K1 and < K2
//!    C3: keys >= K2
//! ```
//!
//! # Thread Safety
//!
//! `InternodeNode` is `Send + Sync` when `S: Send + Sync`. Thread safety
//! is provided by the tree's concurrency protocol:
//!
//! - **Readers:** Use optimistic concurrency control. Read version before
//!   accessing data, read version after, and retry if version changed.
//! - **Writers:** Acquire the [`NodeVersion`] lock before modifications.
//!   The lock uses CAS-based spinlock semantics.
//! - **Memory Ordering:** Atomic fields use `Acquire`/`Release` ordering
//!   to ensure proper visibility of modifications across threads.

use static_assertions::const_assert_eq;
use std::array as StdArray;
use std::cmp::Ordering;
use std::fmt as StdFmt;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64};

use seize::Guard;

use crate::leaf_trait::TreeInternode;
use crate::nodeversion::NodeVersion;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};

mod accessors;
mod layout;

// ============================================================================
//  Constants
// ============================================================================

/// Number of keys in an internode.
/// Fixed at 15 to match leaf WIDTH and enable unified child array.
pub const WIDTH: usize = 15;

/// Number of children in an internode (WIDTH + 1).
const NUM_CHILDREN: usize = WIDTH + 1;

// ============================================================================
//  InternodeNode
// ============================================================================

/// An internal routing node in the `MassTree`.
///
/// Stores up to 15 keys and 16 child pointers. Keys are always
/// in sorted physical order (no permutation needed).
///
/// # Design Note
///
/// Unlike leaf nodes, internodes don't store values - only `u64` keys and
/// `*mut u8` child pointers. This struct is therefore non-generic, reducing
/// code monomorphization (one `InternodeNode` implementation regardless of
/// the tree's slot type).
///
/// # Invariants
/// - `nkeys <= WIDTH` (max 15 keys)
/// - For `nkeys` keys, there are `nkeys + 1` valid children (child[0..=nkeys])
/// - Keys are in ascending order: `ikey[i] < ikey[i+1]` for all `i < nkeys-1`
/// - `child[i]` contains keys `< ikey[i]`
/// - `child[i+1]` contains keys `>= ikey[i]`
///
/// # Memory Layout
/// Uses `#[repr(C, align(64))]` for cache-line alignment.
/// Total size is ~264 bytes (5 cache lines).
#[repr(C, align(64))]
pub struct InternodeNode {
    // ========================================================================
    // Cache Line 0 (64 bytes)
    // ========================================================================
    /// Version for optimistic concurrency control.
    version: NodeVersion, // 4 bytes

    /// Number of keys (0 to WIDTH).
    nkeys: AtomicU8, // 1 byte

    /// Tree height (0 = children are leaves, 1+ = children are internodes).
    /// Max practical height is ~15 (supports billions of keys).
    height: u8, // 1 byte

    /// Padding for 8-byte alignment of parent pointer.
    _pad: [u8; 2], // 2 bytes

    /// Parent internode pointer (null for root).
    parent: AtomicPtr<u8>, // 8 bytes

    // ========================================================================
    // Cache Lines 0-2 (keys - contiguous for prefetcher)
    // ========================================================================
    /// Routing keys in sorted order.
    ikey0: [AtomicU64; WIDTH], // 120 bytes

    // ========================================================================
    // Cache Lines 2-4 (children)
    // ========================================================================
    /// Child pointers (16 children for 15 keys).
    /// - child[i] contains keys < ikey0[i]
    /// - child[nkeys] is the rightmost child (keys >= ikey0[nkeys-1])
    child: [AtomicPtr<u8>; NUM_CHILDREN], // 128 bytes
}

impl StdFmt::Debug for InternodeNode {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        // SAFETY: Debug impl - parent pointer is stable during formatting.
        f.debug_struct("InternodeNode")
            .field("nkeys", &self.nkeys())
            .field("height", &self.height)
            .field(
                "has_parent",
                &(!unsafe { self.parent_unguarded() }.is_null()),
            )
            .finish_non_exhaustive()
    }
}

// Compile-time layout verification.
// InternodeNode must be cache-line aligned (64 bytes) for optimal performance.
const_assert_eq!(std::mem::align_of::<InternodeNode>(), 64);

impl InternodeNode {
    // ========================================================================
    //  In-Place Initialization (for pool allocators)
    // ========================================================================

    /// Initialize an internode in-place at the given pointer.
    ///
    /// This is used by pool allocators to initialize directly in pool memory,
    /// avoiding the intermediate Box allocation.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for writes of `size_of::<Self>()` bytes
    /// - `ptr` must be properly aligned for `Self`
    /// - The caller must have exclusive access to the memory
    /// - The memory does not need to be initialized (will be overwritten)
    #[inline]
    pub unsafe fn init_at(ptr: *mut Self, height: u32) {
        debug_assert!(height <= 15, "init_at: height {height} exceeds maximum 15");

        // SAFETY: All operations here are safe because:
        // - ptr is valid for writes and properly aligned (caller guarantees)
        // - We have exclusive access to the memory (caller guarantees)
        // - write_bytes initializes all bytes to zero, making the memory valid
        // - After zeroing, we can safely create a mutable reference because:
        //   - All atomic types (AtomicU8, AtomicU64, AtomicPtr) are valid when zeroed
        //   - PhantomData is zero-sized
        //   - NodeVersion contains AtomicU32 which is valid when zeroed
        // - ptr::write is used for NodeVersion to properly initialize it
        unsafe {
            // Zero the entire struct first (most fields are zero-initialized)
            StdPtr::write_bytes(ptr, 0, 1);

            // Now write the non-zero fields
            let node = &mut *ptr;

            // Version: internode (not leaf), not root
            StdPtr::write(&raw mut node.version, NodeVersion::new(false));

            // Height (truncate to u8 - max practical height is ~15)
            #[expect(clippy::cast_possible_truncation, reason = "height <= 15 in practice")]
            {
                node.height = height as u8;
            }

            // nkeys, ikey0, child, parent are all zero/null
            // which is correct for a fresh internode
        }
    }

    /// Initialize an internode in-place as a root node.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::init_at`].
    #[inline]
    pub unsafe fn init_at_root(ptr: *mut Self, height: u32) {
        // SAFETY: Caller guarantees ptr validity
        unsafe {
            Self::init_at(ptr, height);
            (*ptr).version.mark_root();
        }
    }

    /// Initialize an internode in-place for a split operation.
    ///
    /// Creates a split-locked version copied from the parent's locked version.
    /// This prevents other threads from locking the sibling until installed.
    ///
    /// # Safety
    ///
    /// - Same requirements as [`Self::init_at`]
    /// - `parent_version` must be from a locked node
    #[inline]
    pub unsafe fn init_at_for_split(ptr: *mut Self, parent_version: &NodeVersion, height: u32) {
        // SAFETY: Caller guarantees ptr validity
        unsafe {
            // Zero the entire struct first
            StdPtr::write_bytes(ptr, 0, 1);

            let node = &mut *ptr;

            // Create split-locked version from parent's locked version
            let split_version: NodeVersion = NodeVersion::new_for_split(parent_version);
            StdPtr::write(&raw mut node.version, split_version);

            // Height (truncate to u8)
            #[expect(clippy::cast_possible_truncation, reason = "height <= 15 in practice")]
            {
                node.height = height as u8;
            }
        }
    }

    // ========================================================================
    //  Boxed Constructors (test utilities)
    // ========================================================================
    //
    // NOTE: Production code uses pool allocators with `init_at()`.
    // These boxed constructors are primarily used in unit tests.

    /// Create a new internode at the given height.
    ///
    /// # Arguments
    /// * `height` - Tree height (0 = children are leaves)
    ///
    /// # Returns
    /// A boxed internode with zero keys and null children.
    #[must_use]
    #[inline]
    pub fn new(height: u32) -> Box<Self> {
        #[expect(clippy::cast_possible_truncation, reason = "height <= 15 in practice")]
        Box::new(Self {
            version: NodeVersion::new(false), // false = not a leaf
            nkeys: AtomicU8::new(0),
            height: height as u8,
            _pad: [0; 2],
            parent: AtomicPtr::new(StdPtr::null_mut()),
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            child: StdArray::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
        })
    }

    /// Create a new internode as root of a tree/layer.
    ///
    /// Same as `new()` but marks the node as root.
    #[must_use]
    #[inline(always)]
    pub fn new_root(height: u32) -> Box<Self> {
        let node: Box<Self> = Self::new(height);
        node.version.mark_root();
        node
    }

    /// Create a new internode sibling for a split operation.
    ///
    /// The new internode is created with a **split-locked** version copied from the
    /// locked parent. This prevents other threads from locking the sibling until
    /// it is installed into the tree and its parent pointer is set.
    ///
    /// # Help-Along Protocol
    ///
    /// This is the internode equivalent of leaf `NodeVersion::new_for_split()`.
    /// The caller MUST call `version().unlock_for_split()` exactly once after:
    /// 1. The sibling is inserted into its parent (grandparent or new root)
    /// 2. The sibling's parent pointer is set
    ///
    /// # C++ Reference
    ///
    /// Matches `next_child->assign_version(*p)` in `masstree_split.hh:234`:
    /// ```cpp
    /// next_child = internode_type::make(height + 1, ti);
    /// next_child->assign_version(*p);
    /// next_child->mark_nonroot();
    /// ```
    ///
    /// # Safety
    ///
    /// The `parent_version` must be from a locked node (the parent being split).
    #[must_use]
    #[inline]
    pub fn new_for_split(parent_version: &NodeVersion, height: u32) -> Box<Self> {
        // Create split-locked version from parent's locked version.
        // This ensures the sibling cannot be locked by other threads until
        // we call unlock_for_split() after installation.
        let split_version: NodeVersion = NodeVersion::new_for_split(parent_version);

        #[expect(clippy::cast_possible_truncation, reason = "height <= 15 in practice")]
        Box::new(Self {
            version: split_version,
            nkeys: AtomicU8::new(0),
            height: height as u8,
            _pad: [0; 2],
            parent: AtomicPtr::new(StdPtr::null_mut()),
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            child: StdArray::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
        })
    }

    // ========================================================================
    //  Insertion Operations
    // ========================================================================

    /// Insert a key and child at position `p`, shifting existing entries right.
    ///
    /// After insertion:
    /// - `ikey[p] = new_ikey`
    /// - `child[p + 1] = new_child`
    /// - Keys/children at positions >= p are shifted right by 1
    ///
    /// Used when propagating a split up the tree: the `new_ikey` is the popup key
    /// from the child split, and `new_child` is the new right sibling.
    ///
    /// # Arguments
    /// * `p` - Position to insert at (0 <= p <= nkeys)
    /// * `new_ikey` - The popup key from the child split
    /// * `new_child` - The new right child (right sibling of the split)
    ///
    /// # Memory Ordering
    ///
    /// Keys use Relaxed stores (validated via version checking by readers).
    /// Child pointers use Release stores (readers use Acquire loads during traversal).
    /// The final `nkeys` store with Release ordering publishes the key count.
    ///
    /// # Panics
    /// Panics in debug mode if node is full or position out of bounds.
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn insert_key_and_child(&self, p: usize, new_ikey: u64, new_child: *mut u8) {
        let n: usize = self.nkeys.load(RELAXED) as usize;

        debug_assert!(n < WIDTH, "insert_key_and_child: node is full");
        debug_assert!(
            p <= n,
            "insert_key_and_child: position {p} out of bounds (n={n})"
        );

        // Shift keys and children to the right (interleaved for proper ordering)
        // Keys: ikey[p..n] -> ikey[p+1..n+1]
        // Children: child[p+1..n+1] -> child[p+2..n+2]
        //
        // Keys use Relaxed stores (validated via version checking by readers).
        // Child pointers use Release stores (readers use Acquire loads during traversal).
        // The interleaved pattern ensures each Release store on child[i+2] orders
        // the prior Relaxed store on ikey[i+1].
        for i in (p..n).rev() {
            let key: u64 = self.ikey0[i].load(RELAXED);
            self.ikey0[i + 1].store(key, RELAXED);

            let child: *mut u8 = self.child[i + 1].load(RELAXED);
            self.child[i + 2].store(child, WRITE_ORD);
        }

        // Insert new key (Relaxed) and child (Release)
        self.ikey0[p].store(new_ikey, RELAXED);
        self.child[p + 1].store(new_child, WRITE_ORD);

        // Publish via nkeys - this is the synchronization point for key count.
        #[expect(clippy::cast_possible_truncation)]
        self.nkeys.store((n + 1) as u8, WRITE_ORD);
    }

    /// Shift entries from another internode.
    ///
    /// Copies `count` entries starting at `src_pos` from `src` to `dst_pos` in self.
    /// Used during internode splits.
    ///
    /// # Arguments
    /// * `dst_pos` - Starting position in self
    /// * `src` - Source internode
    /// * `src_pos` - Starting position in source
    /// * `count` - Number of entries to copy
    ///
    /// # Memory Ordering
    ///
    /// Keys use Relaxed stores (validated via version checking by readers).
    /// Child pointers use Release stores (readers use Acquire loads during traversal).
    /// The caller publishes changes via `nkeys.store(WRITE_ORD)` after this returns.
    ///
    /// # Safety
    ///
    /// Caller must ensure exclusive access (hold lock on both nodes).
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked by caller via split logic"
    )]
    pub unsafe fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize) {
        if count == 0 {
            return;
        }

        // SAFETY: Caller guarantees exclusive access - no concurrent retirement.
        // Copy keys and children (interleaved for proper ordering)
        //
        // Keys use Relaxed stores (validated via version checking by readers).
        // Child pointers use Release stores (readers use Acquire loads during traversal).
        // The interleaved pattern ensures each Release store on child orders the prior
        // Relaxed store on the corresponding key.
        for i in 0..count {
            let key: u64 = src.ikey0[src_pos + i].load(RELAXED);
            self.ikey0[dst_pos + i].store(key, RELAXED);

            let child: *mut u8 = src.child[src_pos + 1 + i].load(RELAXED);
            self.child[dst_pos + 1 + i].store(child, WRITE_ORD);
        }
    }

    // ========================================================================
    //  Split Operation (with simultaneous insertion)
    // ========================================================================

    /// Split this internode into `self + new_right`, simultaneously inserting a new key/child.
    ///
    /// This matches the C++ `internode::split_into()` semantics from `reference/masstree_split.hh`.
    ///
    /// # Operation
    ///
    /// 1. Splits keys and children between `self` and `new_right` at midpoint
    /// 2. Inserts `(insert_ikey, insert_child)` at position `insert_pos`
    /// 3. Updates all children's parent pointers in `new_right` (for internode children)
    ///
    /// After split:
    /// - `self` contains keys `[0, mid)`
    /// - `new_right` contains keys `[mid+1, WIDTH+1)`
    /// - The key at post-insert position `mid` becomes the popup key
    ///
    /// # Arguments
    ///
    /// * `new_right` - The new right sibling (pre-allocated by caller)
    /// * `new_right_ptr` - Raw pointer to `new_right` for setting parent pointers
    /// * `insert_pos` - Position to insert the new key/child (0..=WIDTH)
    /// * `insert_ikey` - The key to insert (popup key from child split)
    /// * `insert_child` - The child to insert (new right sibling from child split)
    ///
    /// # Returns
    ///
    /// `(popup_key, insert_went_left)` where:
    /// - `popup_key` is the separator key to propagate to the parent
    /// - `insert_went_left` is true if insert went into `self`, false if into `new_right`
    ///
    /// # Caller Responsibilities
    ///
    /// **CRITICAL: When `height == 0` (leaf children), the caller MUST update the parent
    /// pointers of all leaf children that moved to `new_right`.** This function only
    /// updates internode children's parent pointers (when `height > 0`).
    ///
    /// # Safety
    ///
    /// * `new_right_ptr` must point to `new_right`
    /// * The caller must hold the lock on `self`
    ///
    /// # Reference
    ///
    /// `reference/masstree_split.hh:123-175`
    #[must_use = "popup_key must be inserted into parent node to complete the split"]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "WIDTH <= 15, so mid and WIDTH-mid fit in u8"
    )]
    pub fn split_into(
        &self,
        new_right: &mut Self,
        new_right_ptr: *mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool) {
        debug_assert!(
            self.nkeys.load(RELAXED) as usize == WIDTH,
            "split_into: node must be full"
        );

        let mid: usize = WIDTH.div_ceil(2); // ceil(WIDTH / 2)

        // Determine where the insertion goes and compute popup key
        // SAFETY: split_into is called under exclusive lock - no concurrent retirement.
        let (popup_key, insert_went_left) = match insert_pos.cmp(&mid) {
            Ordering::Less => {
                // Case 1: Insert goes into left (self)
                new_right.set_child(0, unsafe { self.child_unguarded(mid) });
                unsafe { new_right.shift_from(0, self, mid, WIDTH - mid) };
                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                let popup: u64 = self.ikey_relaxed(mid - 1);

                // Now insert into left side
                self.nkeys.store((mid - 1) as u8, WRITE_ORD);
                self.insert_key_and_child(insert_pos, insert_ikey, insert_child);

                (popup, true)
            }

            Ordering::Equal => {
                // Case 2: Insert becomes the popup key
                new_right.set_child(0, insert_child);
                unsafe { new_right.shift_from(0, self, mid, WIDTH - mid) };
                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                self.nkeys.store(mid as u8, WRITE_ORD);

                (insert_ikey, false)
            }

            Ordering::Greater => {
                // Case 3: Insert goes into right (new_right)
                let right_insert_pos: usize = insert_pos - (mid + 1);

                new_right.set_child(0, unsafe { self.child_unguarded(mid + 1) });
                unsafe { new_right.shift_from(0, self, mid + 1, right_insert_pos) };

                new_right.set_ikey_relaxed(right_insert_pos, insert_ikey);
                new_right.set_child(right_insert_pos + 1, insert_child);

                let count_after: usize = WIDTH - insert_pos;
                unsafe {
                    new_right.shift_from(right_insert_pos + 1, self, insert_pos, count_after);
                };

                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                let popup: u64 = self.ikey_relaxed(mid);
                self.nkeys.store(mid as u8, WRITE_ORD);

                (popup, false)
            }
        };

        // Set new_right's height to match self
        new_right.height = self.height;

        // Update children's parent pointers (internode children only)
        // SAFETY: We have exclusive access during split.
        if self.height > 0 {
            let nr_nkeys: usize = new_right.nkeys.load(RELAXED) as usize;
            let new_right_ptr_u8: *mut u8 = new_right_ptr.cast::<u8>();

            for i in 0..=nr_nkeys {
                let child: *mut u8 = unsafe { new_right.child_unguarded(i) };
                if !child.is_null() {
                    // SAFETY: height > 0 means children are InternodeNode
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "height > 0 means children are InternodeNode"
                    )]
                    unsafe {
                        (*child.cast::<Self>()).set_parent(new_right_ptr_u8);
                    }
                }
            }
        }

        (popup_key, insert_went_left)
    }

    // ========================================================================
    //  Parent Accessors
    // ========================================================================

    /// Get the parent pointer (as `*mut u8`) with guard protection.
    ///
    /// Cast to `*mut InternodeNode` at usage sites.
    #[must_use]
    #[inline(always)]
    pub fn parent(&self, guard: &impl Guard) -> *mut u8 {
        guard.protect(&self.parent, READ_ORD)
    }

    /// Get the parent pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the parent pointer won't be retired during use.
    /// Valid when:
    /// - Called during `Drop` (no concurrent access)
    /// - Called while holding exclusive lock that prevents retirement
    #[must_use]
    #[inline(always)]
    pub unsafe fn parent_unguarded(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    /// Set the parent pointer.
    ///
    /// Accepts `*mut u8` for uniformity with `LeafNode`.
    #[inline(always)]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    /// Check if this is a root node (no parent or version says root).
    #[must_use]
    #[inline(always)]
    pub fn is_root(&self) -> bool {
        self.version.is_root()
    }

    // ========================================================================
    //  Comparison (for binary search)
    // ========================================================================

    /// Compare a search key against the key at position `p`.
    ///
    /// Returns:
    /// - `Ordering::Less` if `search_ikey < ikey[p]`
    /// - `Ordering::Equal` if `search_ikey == ikey[p]`
    /// - `Ordering::Greater` if `search_ikey > ikey[p]`
    #[must_use]
    #[inline(always)]
    pub fn compare_key(&self, search_ikey: u64, p: usize) -> Ordering {
        search_ikey.cmp(&self.ikey(p))
    }

    /// Find the position where a key should be inserted.
    ///
    /// Returns the index where `insert_ikey` should go, such that
    /// `ikey(i-1) < insert_ikey <= ikey(i)` (or at the end if greater than all).
    ///
    /// Uses linear search with loop unrolling and prefetching. Linear search is
    /// faster than binary for small nodes (WIDTH ≤ 16) due to predictable branches
    /// and sequential memory access.
    ///
    /// # Memory Layout
    ///
    /// ```text
    /// Offset 0-63 (CL0):   header (16B) + ikey0[0..5] (48B)
    /// Offset 64-127 (CL1): ikey0[6..13] (64B)
    /// Offset 128+ (CL2):   ikey0[14] (8B) + children
    /// ```
    ///
    /// We prefetch the next cache line of keys while processing the current batch.
    #[inline]
    pub fn find_insert_position(&self, insert_ikey: u64) -> usize {
        let n: usize = self.nkeys();
        let mut i: usize = 0;

        // Prefetch cache line 1 (ikey0[6..13]) before we need it
        // CL0 is already in cache from loading nkeys
        if n > 6 {
            crate::prefetch::prefetch_read(&raw const self.ikey0[6]);
        }

        // Unrolled loop: process 4 keys per iteration
        while i + 4 <= n {
            if self.ikey(i) >= insert_ikey {
                return i;
            }
            if self.ikey(i + 1) >= insert_ikey {
                return i + 1;
            }
            if self.ikey(i + 2) >= insert_ikey {
                return i + 2;
            }
            if self.ikey(i + 3) >= insert_ikey {
                return i + 3;
            }
            i += 4;
        }

        // Handle remainder (0-3 keys)
        while i < n {
            if self.ikey(i) >= insert_ikey {
                return i;
            }
            i += 1;
        }

        n
    }

    // ========================================================================
    //  Invariant Checker
    // ========================================================================

    /// Verify internode invariants (debug builds only).
    ///
    /// Checks:
    /// - nkeys <= WIDTH
    /// - Keys are in ascending order
    /// - Children for valid indices are potentially non-null (soft check)
    ///
    /// # Panics
    /// If any invariant is violated.
    #[cfg(debug_assertions)]
    pub fn debug_assert_invariants(&self) {
        // Check nkeys bound
        assert!(
            self.nkeys() <= WIDTH,
            "nkeys {} exceeds WIDTH {}",
            self.nkeys(),
            WIDTH
        );

        let size: usize = self.size();

        // Check key ordering
        if size > 1 {
            for i in 1..size {
                assert!(
                    self.ikey_relaxed(i - 1) < self.ikey_relaxed(i),
                    "keys not in ascending order: ikey[{}] ({:#x}) >= ikey[{}] ({:#x})",
                    i - 1,
                    self.ikey_relaxed(i - 1),
                    i,
                    self.ikey_relaxed(i)
                );
            }
        }
    }

    /// No-op in release builds.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub fn debug_assert_invariants(&self) {}
}

impl Default for InternodeNode {
    fn default() -> Self {
        Self {
            version: NodeVersion::new(false),
            nkeys: AtomicU8::new(0),
            height: 0,
            _pad: [0; 2],
            parent: AtomicPtr::new(StdPtr::null_mut()),
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            child: StdArray::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
        }
    }
}

// ============================================================================
//  Send + Sync
// ============================================================================

// SAFETY: InternodeNode is safe to send/share between threads.
//
// Thread safety is provided by:
// 1. Atomic fields (nkeys, ikey0, child, parent) use appropriate memory
//    orderings for concurrent access
// 2. The NodeVersion field provides locking and optimistic concurrency control
// 3. Raw pointers (child, parent) are protected by the tree's concurrency
//    protocol:
//    - Readers use version validation to detect concurrent modifications
//    - Writers hold the node lock before modifying children
//
// Unlike leaf nodes, internodes don't store values (only u64 keys and *mut u8
// child pointers), so there's no generic parameter to propagate Send/Sync from.
unsafe impl Send for InternodeNode {}
unsafe impl Sync for InternodeNode {}

// ============================================================================
//  Compile-Time Layout Assertions
// ============================================================================
//
// See `layout` submodule for comprehensive compile-time assertions that verify:
// - Exact size (320 bytes, 5 cache lines)
// - Cache-line alignment (64 bytes)
// - Field offsets for hot-path optimization
// - Cache line boundary constraints (ikey0[6] at CL1, child[7] at CL3)
//
// The layout module ensures any refactoring that changes field positions
// will fail at compile time with clear error messages.

// ============================================================================
//  TreeInternode Implementation
// ============================================================================

impl TreeInternode for InternodeNode {
    const WIDTH: usize = WIDTH;

    #[inline(always)]
    fn new_boxed(height: u32) -> Box<Self> {
        Self::new(height)
    }

    #[inline(always)]
    fn new_root_boxed(height: u32) -> Box<Self> {
        Self::new_root(height)
    }

    #[inline(always)]
    fn new_boxed_for_split(
        parent_version: &crate::nodeversion::NodeVersion,
        height: u32,
    ) -> Box<Self> {
        Self::new_for_split(parent_version, height)
    }

    #[inline(always)]
    fn version(&self) -> &crate::nodeversion::NodeVersion {
        Self::version(self)
    }

    #[inline(always)]
    fn height(&self) -> u32 {
        Self::height(self)
    }

    #[inline(always)]
    fn children_are_leaves(&self) -> bool {
        Self::children_are_leaves(self)
    }

    #[inline(always)]
    fn nkeys(&self) -> usize {
        Self::nkeys(self)
    }

    #[inline(always)]
    fn set_nkeys(&self, n: u8) {
        Self::set_nkeys(self, n);
    }

    #[inline(always)]
    fn inc_nkeys(&self) {
        Self::inc_nkeys(self);
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        Self::is_full(self)
    }

    #[inline(always)]
    fn ikey(&self, idx: usize) -> u64 {
        Self::ikey(self, idx)
    }

    // #[inline(always)]
    // fn ikey_relaxed(&self, idx: usize) -> u64 {
    //     Self::ikey_relaxed(self, idx)
    // }

    /// Get the key at the given index using Relaxed ordering.
    ///
    /// Used in internal operations where ordering is handled by caller.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    fn ikey_relaxed(&self, i: usize) -> u64 {
        debug_assert!(i < WIDTH, "ikey_relaxed: index {i} out of bounds");
        self.ikey0[i].load(RELAXED)
    }

    /// Get raw pointer to the ikey array for SIMD operations.
    ///
    /// Returns pointer to `self.ikey0[0]`, valid for `WIDTH` (15) contiguous u64 reads.
    ///
    /// # Layout
    ///
    /// `ikey0` starts at offset 16 in [`InternodeNode`]:
    /// - Offset 0-15: version (4B) + nkeys (1B) + height (1B) + pad (2B) + parent (8B)
    /// - Offset 16-135: ikey0[0..15] (120 bytes, 15 x 8B)
    #[inline(always)]
    fn ikey_ptr(&self) -> *const u64 {
        // SAFETY: AtomicU64 has identical layout to u64.
        // Pointer arithmetic is valid within the ikey0 array bounds.
        self.ikey0.as_ptr().cast::<u64>()
    }

    #[inline(always)]
    fn set_ikey(&self, idx: usize, key: u64) {
        Self::set_ikey(self, idx, key);
    }

    #[inline(always)]
    fn compare_key(&self, search_ikey: u64, p: usize) -> Ordering {
        Self::compare_key(self, search_ikey, p)
    }

    #[inline(always)]
    fn find_insert_position(&self, insert_ikey: u64) -> usize {
        Self::find_insert_position(self, insert_ikey)
    }

    /// # Safety
    ///
    /// This trait method is intended for use under exclusive lock.
    /// Uses unguarded loads - caller must ensure no concurrent retirement.
    #[inline(always)]
    fn child(&self, idx: usize) -> *mut u8 {
        // SAFETY: BTreeNodeCommon trait methods are called during locked operations.
        unsafe { Self::child_unguarded(self, idx) }
    }

    /// # Safety
    ///
    /// This trait method is intended for use under exclusive lock.
    /// Uses unguarded loads - caller must ensure no concurrent retirement.
    #[inline(always)]
    fn child_with_prefetch(&self, idx: usize, nkeys: usize) -> *mut u8 {
        // SAFETY: BTreeNodeCommon trait methods are called during locked operations.
        // Note: We lose the prefetch behavior here, but trait usage is rare.
        let _ = nkeys;
        unsafe { Self::child_unguarded(self, idx) }
    }

    /// # Safety
    ///
    /// This trait method is intended for use under exclusive lock.
    /// Uses unguarded loads - caller must ensure no concurrent retirement.
    #[inline(always)]
    fn child_with_depth_prefetch(&self, idx: usize) -> *mut u8 {
        // SAFETY: Trait methods called during locked operations where
        // the caller holds a guard at a higher scope level.
        let ptr: *mut u8 = unsafe { Self::child_unguarded(self, idx) };
        Self::prefetch_child_internal(ptr);
        ptr
    }

    /// # Safety
    ///
    /// This trait method is intended for use under exclusive lock.
    /// Uses unguarded loads - caller must ensure no concurrent retirement.
    #[inline(always)]
    fn child_with_full_prefetch(&self, idx: usize) -> *mut u8 {
        // SAFETY: Trait methods called during locked operations where
        // the caller holds a guard at a higher scope level.
        let ptr: *mut u8 = unsafe { Self::child_unguarded(self, idx) };
        Self::prefetch_child_full(ptr);
        ptr
    }

    #[inline(always)]
    fn set_child(&self, idx: usize, child: *mut u8) {
        Self::set_child(self, idx, child);
    }

    #[inline(always)]
    fn assign(&self, p: usize, ikey: u64, right_child: *mut u8) {
        Self::assign(self, p, ikey, right_child);
    }

    #[inline(always)]
    fn insert_key_and_child(&self, p: usize, new_ikey: u64, new_child: *mut u8) {
        Self::insert_key_and_child(self, p, new_ikey, new_child);
    }

    /// # Safety
    ///
    /// This trait method is intended for use under exclusive lock.
    /// Uses unguarded loads - caller must ensure no concurrent retirement.
    #[inline(always)]
    fn parent(&self) -> *mut u8 {
        // SAFETY: BTreeNodeCommon trait methods are called during locked operations.
        unsafe { Self::parent_unguarded(self) }
    }

    #[inline(always)]
    fn set_parent(&self, parent: *mut u8) {
        Self::set_parent(self, parent);
    }

    #[inline(always)]
    fn is_root(&self) -> bool {
        Self::is_root(self)
    }

    /// # Safety
    ///
    /// Called under exclusive lock - uses unguarded child loads.
    #[inline(always)]
    fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize) {
        // SAFETY: Trait method is called during locked operations.
        unsafe { Self::shift_from(self, dst_pos, src, src_pos, count) };
    }

    #[inline(always)]
    fn split_into(
        &self,
        new_right: &mut Self,
        new_right_ptr: *mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool) {
        Self::split_into(
            self,
            new_right,
            new_right_ptr,
            insert_pos,
            insert_ikey,
            insert_child,
        )
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;

// ============================================================================
//  Loom Tests
// ============================================================================

/// Loom tests for concurrent internode operations.
///
/// These tests verify that concurrent reads and writes to internodes
/// are properly synchronized through the version protocol.
///
/// Run with: `RUSTFLAGS="--cfg loom" cargo test --lib internode::loom_tests`
#[cfg(loom)]
mod loom_tests;
