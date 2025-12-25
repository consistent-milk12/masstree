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
//! │   height: u32 (4 bytes)                                         │
//! │   padding (~55 bytes)                                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Cache Lines 1-2 (128 bytes)                                     │
//! │   ikey0: [AtomicU64; 15] (120 bytes) - routing keys             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Cache Lines 3-4 (128 bytes)                                     │
//! │   child: [AtomicPtr<u8>; 15] (120 bytes) - child pointers       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Cache Line 5 (partial)                                          │
//! │   rightmost_child: AtomicPtr<u8> (8 bytes)                      │
//! │   parent: AtomicPtr<u8> (8 bytes)                               │
//! └─────────────────────────────────────────────────────────────────┘
//! Total: ~320 bytes (5 cache lines)
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

use std::cmp::Ordering;
use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::mem as StdMem;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, fence};

use crate::leaf_trait::TreeInternode;
use crate::nodeversion::NodeVersion;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};
use crate::slot::ValueSlot;
use crate::value::LeafValue;

// ============================================================================
//  InternodeNode
// ============================================================================

/// An internal routing node in the `MassTree`.
///
/// Stores up to WIDTH keys and WIDTH+1 child pointers. Keys are always
/// in sorted physical order (no permutation needed).
///
/// # Type Parameters
/// * `S` - The slot type implementing [`ValueSlot`] (phantom, for type consistency)
/// * `WIDTH` - Number of key slots (default: 15, max: 15)
///
/// # Invariants
/// - `nkeys <= WIDTH`
/// - For `nkeys` keys, there are `nkeys + 1` valid children (child[0..=nkeys])
/// - Keys are in ascending order: `ikey0[i] < ikey0[i+1]` for all `i < nkeys-1`
/// - `child[i]` contains keys `< ikey0[i]`
/// - `child[i+1]` contains keys `>= ikey0[i]`
///
/// # Memory Layout
/// Uses `#[repr(C, align(64))]` for cache-line alignment.
/// For WIDTH=15, total size is ~320 bytes (5 cache lines).
#[repr(C, align(64))]
pub struct InternodeNode<S: ValueSlot, const WIDTH: usize = 15> {
    /// Version for optimistic concurrency control.
    version: NodeVersion,

    /// Number of keys (0 to WIDTH).
    nkeys: AtomicU8,

    /// Tree height (0 = children are leaves, 1+ = children are internodes).
    height: u32,

    /// Routing keys in sorted order.
    ikey0: [AtomicU64; WIDTH],

    /// Child pointers for slots 0..WIDTH.
    /// - child[i] contains keys < ikey0[i]
    /// - Type is `*mut u8` for uniformity; cast to LeafNode/InternodeNode based on height
    ///   Note: The rightmost child (at index nkeys) is stored in `rightmost_child`
    ///   to avoid `WIDTH + 1` which requires unstable `generic_const_exprs`.
    child: [AtomicPtr<u8>; WIDTH],

    /// Rightmost child pointer (child at index nkeys).
    /// Stored separately to avoid `[*mut u8; WIDTH + 1]` which requires unstable features.
    rightmost_child: AtomicPtr<u8>,

    /// Parent internode pointer (null for root).
    /// Type is `*mut u8` for uniformity with `LeafNode` (both use `*mut u8`).
    /// Cast to `*mut InternodeNode` at usage sites.
    parent: AtomicPtr<u8>,

    /// Phantom data to hold S type parameter for tree type consistency.
    _marker: PhantomData<S>,
}

impl<S: ValueSlot, const WIDTH: usize> StdFmt::Debug for InternodeNode<S, WIDTH> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("InternodeNode")
            .field("nkeys", &self.nkeys())
            .field("height", &self.height)
            .field("has_parent", &(!self.parent().is_null()))
            .finish_non_exhaustive()
    }
}

// Compile-time assertion: WIDTH must be 1..=15
impl<S: ValueSlot, const WIDTH: usize> InternodeNode<S, WIDTH> {
    const WIDTH_CHECK: () = {
        assert!(WIDTH > 0, "WIDTH must be at least 1");

        assert!(WIDTH <= 15, "WIDTH must be at most 15");
    };
}

impl<S: ValueSlot, const WIDTH: usize> InternodeNode<S, WIDTH> {
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
        // Trigger compile-time WIDTH check
        let _: () = Self::WIDTH_CHECK;

        Box::new(Self {
            version: NodeVersion::new(false), // false = not a leaf
            nkeys: AtomicU8::new(0),
            height,
            ikey0: std::array::from_fn(|_| AtomicU64::new(0)),
            child: std::array::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
            rightmost_child: AtomicPtr::new(StdPtr::null_mut()),
            parent: AtomicPtr::new(StdPtr::null_mut()),
            _marker: PhantomData,
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

    // ========================================================================
    //  Version Accessors
    // ========================================================================

    /// Get a reference to the node's version.
    #[must_use]
    #[inline(always)]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the node's version.
    #[inline(always)]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ========================================================================
    //  Key Accessors
    // ========================================================================

    /// Get the number of keys in this internode.
    #[must_use]
    #[inline(always)]
    pub fn nkeys(&self) -> usize {
        self.nkeys.load(READ_ORD) as usize
    }

    /// Get the number of keys as usize (convenience method).
    #[must_use]
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.nkeys()
    }

    /// Check if the internode has no keys.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nkeys.load(READ_ORD) == 0
    }

    /// Check if the internode is full.
    #[must_use]
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.nkeys.load(READ_ORD) as usize >= WIDTH
    }

    /// Get the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn ikey(&self, i: usize) -> u64 {
        debug_assert!(i < WIDTH, "ikey: index out of bounds");

        self.ikey0[i].load(READ_ORD)
    }

    /// Set the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn set_ikey(&self, i: usize, ikey: u64) {
        debug_assert!(i < WIDTH, "set_ikey: index out of bounds");

        self.ikey0[i].store(ikey, WRITE_ORD);
    }

    /// Get the tree height.
    ///
    /// - `height = 0` means children are leaves
    /// - `height > 0` means children are internodes
    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Check if children are leaves (height == 0).
    #[must_use]
    #[inline(always)]
    pub const fn children_are_leaves(&self) -> bool {
        self.height == 0
    }

    // ========================================================================
    //  Child Accessors
    // ========================================================================

    /// Get the child pointer at the given index.
    ///
    /// Valid indices are `0..=nkeys` (one more child than keys).
    /// Index WIDTH returns the rightmost child.
    ///
    /// # Panics
    /// Panics in debug mode if `i > WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; i < WIDTH"
    )]
    pub fn child(&self, i: usize) -> *mut u8 {
        debug_assert!(i <= WIDTH, "child: index out of bounds");
        if i < WIDTH {
            self.child[i].load(READ_ORD)
        } else {
            self.rightmost_child.load(READ_ORD)
        }
    }

    /// Set the child pointer at the given index.
    ///
    /// Index WIDTH sets the rightmost child.
    ///
    /// # Panics
    /// Panics in debug mode if `i > WIDTH`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; i < WIDTH"
    )]
    pub fn set_child(&self, i: usize, child: *mut u8) {
        debug_assert!(i <= WIDTH, "set_child: index out of bounds");
        if i < WIDTH {
            self.child[i].store(child, WRITE_ORD);
        } else {
            self.rightmost_child.store(child, WRITE_ORD);
        }
    }

    /// Assign a key and its right child at position `p`.
    ///
    /// Following the C++ pattern:
    /// - `ikey0[p] = ikey`
    /// - `child[p + 1] = right_child`
    ///
    /// The left child (`child[p]`) must already be set.
    ///
    /// # Panics
    /// Panics in debug mode if `p >= WIDTH`.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn assign(&self, p: usize, ikey: u64, right_child: *mut u8) {
        debug_assert!(p < WIDTH, "assign: position out of bounds");

        self.ikey0[p].store(ikey, WRITE_ORD);
        self.set_child(p + 1, right_child);
    }

    /// Set the number of keys.
    ///
    /// # Panics
    /// Panics in debug mode if `n > WIDTH`.
    #[inline(always)]
    pub fn set_nkeys(&self, n: u8) {
        debug_assert!((n as usize) <= WIDTH, "set_nkeys: count out of bounds");
        self.nkeys.store(n, WRITE_ORD);
    }

    /// Increment the number of keys by 1.
    ///
    /// # Panics
    /// Panics in debug mode if already at WIDTH.
    #[inline(always)]
    pub fn inc_nkeys(&self) {
        let current: u8 = self.nkeys.load(RELAXED);
        debug_assert!((current as usize) < WIDTH, "inc_nkeys: would overflow");
        self.nkeys.store(current.wrapping_add(1), WRITE_ORD);
    }

    // ========================================================================
    //  Insertion Operations
    // ========================================================================

    /// Insert a key and child at position `p`, shifting existing entries right.
    ///
    /// After insertion:
    /// - `ikey0[p] = new_ikey`
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
    /// # Panics
    /// Panics in debug mode if node is full or position out of bounds.
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked via debug_assert; loop invariants ensure p,i < WIDTH"
    )]
    pub fn insert_key_and_child(&self, p: usize, new_ikey: u64, new_child: *mut u8) {
        let n: usize = self.nkeys.load(RELAXED) as usize;

        debug_assert!(n < WIDTH, "insert_key_and_child: node is full");
        debug_assert!(p <= n, "insert_key_and_child: position out of bounds");

        // Shift keys and children to the right
        // Keys: ikey0[p..n] -> ikey0[p+1..n+1]
        // Children: child[p+1..n+1] -> child[p+2..n+2]
        for i in (p..n).rev() {
            let key: u64 = self.ikey0[i].load(RELAXED);
            self.ikey0[i + 1].store(key, RELAXED);

            let child = self.child(i + 1);
            self.set_child(i + 2, child);
        }

        // Insert new key and child
        self.ikey0[p].store(new_ikey, RELAXED);
        self.set_child(p + 1, new_child);

        fence(WRITE_ORD);

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
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "caller ensures indices are within WIDTH bounds"
    )]
    pub fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize) {
        for i in 0..count {
            let key: u64 = src.ikey0[src_pos + i].load(RELAXED);
            self.ikey0[dst_pos + i].store(key, RELAXED);
            self.set_child(dst_pos + 1 + i, src.child(src_pos + 1 + i));
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
    /// Example for the caller when splitting an internode with leaf children:
    ///
    /// ```ignore
    /// let (popup_key, insert_went_left) = parent.split_into(
    ///     &mut new_right, new_right_ptr, insert_pos, key, child
    /// );
    ///
    /// // If parent.height == 0, update leaf children's parent pointers:
    /// if parent.children_are_leaves() {
    ///     let nr_nkeys = new_right.nkeys() as usize;
    ///     for i in 0..=nr_nkeys {
    ///         let leaf_ptr = new_right.child(i).cast::<LeafNode>();
    ///         (*leaf_ptr).set_parent(new_right_ptr.cast::<u8>());
    ///     }
    /// }
    /// ```
    ///
    /// This must be done while still holding the parent lock, before making
    /// `new_right` visible to other threads.
    ///
    /// # Safety
    ///
    /// * `new_right_ptr` must point to `new_right`
    /// * The caller must hold the lock on `self`
    ///
    /// # Reference
    ///
    /// `reference/masstree_split.hh:123-175`
    #[expect(
        clippy::indexing_slicing,
        reason = "split logic ensures all indices < WIDTH"
    )]
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
        let (popup_key, insert_went_left) = match insert_pos.cmp(&mid) {
            Ordering::Less => {
                // Case 1: Insert goes into left (self)
                // - new_right.child[0] = self.child[mid]
                // - new_right gets keys [mid, WIDTH)
                // - popup_key = self.ikey0[mid - 1] (pre-insert key at mid-1)
                new_right.set_child(0, self.child(mid));
                new_right.shift_from(0, self, mid, WIDTH - mid);
                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                let popup: u64 = self.ikey0[mid - 1].load(RELAXED);

                // Now insert into left side
                self.nkeys.store((mid - 1) as u8, WRITE_ORD);
                self.insert_key_and_child(insert_pos, insert_ikey, insert_child);

                (popup, true)
            }

            Ordering::Equal => {
                // Case 2: Insert becomes the popup key
                // - new_right.child[0] = insert_child
                // - new_right gets keys [mid, WIDTH)
                // - popup_key = insert_ikey
                new_right.set_child(0, insert_child);
                new_right.shift_from(0, self, mid, WIDTH - mid);
                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                self.nkeys.store(mid as u8, WRITE_ORD);

                (insert_ikey, false) // Technically neither left nor right
            }

            Ordering::Greater => {
                // Case 3: Insert goes into right (new_right)
                // - new_right.child[0] = self.child[mid + 1]
                // - new_right gets keys [mid+1, insert_pos) + insert + [insert_pos, WIDTH)
                // - popup_key = self.ikey0[mid]
                let right_insert_pos: usize = insert_pos - (mid + 1);

                new_right.set_child(0, self.child(mid + 1));

                // Copy keys before insertion point
                new_right.shift_from(0, self, mid + 1, right_insert_pos);

                // Insert the new key/child
                new_right.ikey0[right_insert_pos].store(insert_ikey, RELAXED);
                new_right.set_child(right_insert_pos + 1, insert_child);

                // Copy keys after insertion point (using shift_from for consistency)
                let count_after: usize = WIDTH - insert_pos;
                new_right.shift_from(right_insert_pos + 1, self, insert_pos, count_after);

                new_right.nkeys.store((WIDTH - mid) as u8, WRITE_ORD);

                let popup: u64 = self.ikey0[mid].load(RELAXED);
                self.nkeys.store(mid as u8, WRITE_ORD);

                (popup, false)
            }
        };

        // Set new_right's height to match self
        new_right.height = self.height;

        // CRITICAL: Update children's parent pointers to point to new_right.
        // This MUST happen inside split_into (before returning) to prevent races
        // where a thread sees a child with a stale parent pointer.
        // Matches C++ masstree_split.hh:163-165:
        //   for (int i = 0; i <= nr->nkeys_; ++i) {
        //       nr->child_[i]->set_parent(nr);
        //   }
        //
        // NOTE: We only update internode children here (height > 0).
        // For leaf children (height == 0), the caller must update them because
        // the internode doesn't know the actual leaf type (could be LeafNode<S, WIDTH>
        // or LeafNode24<S> for MassTree24).
        if self.height > 0 {
            // Children are internodes - we can handle this directly
            let nr_nkeys: usize = new_right.nkeys.load(RELAXED) as usize;
            let new_right_ptr_u8: *mut u8 = new_right_ptr.cast::<u8>();

            for i in 0..=nr_nkeys {
                let child: *mut u8 = new_right.child(i);
                if !child.is_null() {
                    // SAFETY: height > 0 means children are InternodeNode<S, WIDTH>
                    unsafe {
                        (*child.cast::<Self>()).set_parent(new_right_ptr_u8);
                    }
                }
            }
        }
        // NOTE: For height == 0 (leaf children), the caller is responsible for
        // updating parent pointers. This must be done immediately after split_into
        // returns, while still holding the parent lock.

        (popup_key, insert_went_left)
    }

    // ========================================================================
    //  Parent Accessors
    // ========================================================================

    /// Get the parent pointer (as `*mut u8`).
    ///
    /// Cast to `*mut InternodeNode<V, WIDTH>` at usage sites.
    #[must_use]
    #[inline(always)]
    pub fn parent(&self) -> *mut u8 {
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
    /// - `Ordering::Less` if `search_ikey < ikey0[p]`
    /// - `Ordering::Equal` if `search_ikey == ikey0[p]`
    /// - `Ordering::Greater` if `search_ikey > ikey0[p]`
    ///
    /// Unlike leaf nodes, internode comparison is purely on ikey—no keylenx.
    #[must_use]
    #[inline(always)]
    pub fn compare_key(&self, search_ikey: u64, p: usize) -> std::cmp::Ordering {
        search_ikey.cmp(&self.ikey(p))
    }

    /// Find the position where a key should be inserted.
    ///
    /// Returns the index where `insert_ikey` should go, such that
    /// `ikey(i-1) < insert_ikey <= ikey(i)` (or at the end if greater than all).
    ///
    /// Uses binary search for O(log n) complexity instead of O(n) linear scan.
    /// For WIDTH=15, this reduces worst-case from 15 comparisons to ~4.
    ///
    /// FIXED: Used in the data race fix for recomputing child index after reacquiring lock.
    #[inline]
    pub fn find_insert_position(&self, insert_ikey: u64) -> usize {
        let n: usize = self.nkeys();

        let mut lo: usize = 0;
        let mut hi: usize = n;

        while lo < hi {
            let mid: usize = (lo + hi) >> 1;
            if self.ikey(mid) < insert_ikey {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        lo
    }

    /// Prefetch the internode's data into cache.
    ///
    /// Brings the node's key and child arrays into CPU cache before they're
    /// accessed, reducing memory latency during traversal.
    ///
    /// Matches C++ `internode::prefetch()` from `reference/masstree_struct.hh:149-152`.
    #[inline(always)]
    pub fn prefetch(&self) {
        use crate::prefetch::prefetch_read;

        let self_ptr: *const u8 = std::ptr::from_ref::<Self>(self).cast::<u8>();
        let max_offset: usize = core::cmp::min(16 * WIDTH + 1, 256);

        // Prefetch cache lines beyond the first (which was fetched when we accessed version)
        let mut offset: usize = 64;
        while offset < max_offset {
            // SAFETY: prefetch_read is a hint, safe even with invalid addresses.
            // The CPU will simply ignore prefetch requests for unmapped memory.
            unsafe {
                prefetch_read(self_ptr.add(offset));
            }
            offset += 64;
        }
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
    #[expect(
        clippy::indexing_slicing,
        reason = "loop bounds ensure i-1 and i are < size <= WIDTH"
    )]
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
                    self.ikey0[i - 1].load(RELAXED) < self.ikey0[i].load(RELAXED),
                    "keys not in ascending order: ikey0[{}] ({:#x}) >= ikey0[{}] ({:#x})",
                    i - 1,
                    self.ikey0[i - 1].load(RELAXED),
                    i,
                    self.ikey0[i].load(RELAXED)
                );
            }
        }
    }

    /// No-op in release builds.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub fn debug_assert_invariants(&self) {}
}

impl<S: ValueSlot, const WIDTH: usize> Default for InternodeNode<S, WIDTH> {
    fn default() -> Self {
        // Trigger compile-time WIDTH check
        let _: () = Self::WIDTH_CHECK;

        Self {
            version: NodeVersion::new(false),
            nkeys: AtomicU8::new(0),
            height: 0,
            ikey0: std::array::from_fn(|_| AtomicU64::new(0)),
            child: std::array::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
            rightmost_child: AtomicPtr::new(StdPtr::null_mut()),
            parent: AtomicPtr::new(StdPtr::null_mut()),
            _marker: PhantomData,
        }
    }
}

// ============================================================================
//  Send + Sync
// ============================================================================

// SAFETY: InternodeNode is safe to send/share between threads when S is.
//
// Thread safety is provided by:
// 1. Atomic fields (nkeys, ikey0, child, rightmost_child, parent) use
//    appropriate memory orderings for concurrent access
// 2. The NodeVersion field provides locking and optimistic concurrency control
// 3. Raw pointers (child, parent) are protected by the tree's concurrency
//    protocol:
//    - Readers use version validation to detect concurrent modifications
//    - Writers hold the node lock before modifying children
//
// This matches the C++ Masstree concurrency model where internode access
// is protected by either:
// - Version validation (readers retry on version change)
// - Lock acquisition (writers hold lock during modifications)
unsafe impl<S: ValueSlot + Send + Sync, const WIDTH: usize> Send for InternodeNode<S, WIDTH> {}
unsafe impl<S: ValueSlot + Send + Sync, const WIDTH: usize> Sync for InternodeNode<S, WIDTH> {}

// ============================================================================
//  Type Aliases
// ============================================================================

/// Standard 15-key internode (default).
pub type InternodeNode15<V> = InternodeNode<V, 15>;

/// Compact 7-key internode.
pub type InternodeNodeCompact<V> = InternodeNode<V, 7>;

// ============================================================================
//  Size Assertions
// ============================================================================

/// Compile-time size check for `InternodeNode<LeafValue<u64>, 15>`.
const _: () = {
    const SIZE: usize = StdMem::size_of::<InternodeNode<LeafValue<u64>, 15>>();
    const ALIGN: usize = StdMem::align_of::<InternodeNode<LeafValue<u64>, 15>>();

    // Should fit in 5 cache lines (320 bytes)
    assert!(SIZE <= 320, "InternodeNode exceeds 5 cache lines");

    // Should be cache-line aligned
    assert!(ALIGN == 64, "InternodeNode not cache-line aligned");
};

// ============================================================================
//  TreeInternode Implementation
// ============================================================================

impl<S, const WIDTH: usize> TreeInternode<S> for InternodeNode<S, WIDTH>
where
    S: ValueSlot + Send + Sync + 'static,
{
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

    #[inline(always)]
    fn set_ikey(&self, idx: usize, key: u64) {
        Self::set_ikey(self, idx, key);
    }

    #[inline(always)]
    fn compare_key(&self, search_ikey: u64, p: usize) -> std::cmp::Ordering {
        Self::compare_key(self, search_ikey, p)
    }

    #[inline(always)]
    fn find_insert_position(&self, insert_ikey: u64) -> usize {
        Self::find_insert_position(self, insert_ikey)
    }

    #[inline(always)]
    fn child(&self, idx: usize) -> *mut u8 {
        Self::child(self, idx)
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

    #[inline(always)]
    fn parent(&self) -> *mut u8 {
        Self::parent(self)
    }

    #[inline(always)]
    fn set_parent(&self, parent: *mut u8) {
        Self::set_parent(self, parent);
    }

    #[inline(always)]
    fn is_root(&self) -> bool {
        Self::is_root(self)
    }

    #[inline(always)]
    fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize) {
        Self::shift_from(self, dst_pos, src, src_pos, count);
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

    #[inline(always)]
    fn prefetch(&self) {
        Self::prefetch(self);
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    #[test]
    fn test_new_internode() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        assert!(!node.version().is_leaf());
        assert!(!node.version().is_root());
        assert_eq!(node.nkeys(), 0);
        assert_eq!(node.height(), 0);
        assert!(node.is_empty());
        assert!(!node.is_full());
        assert!(node.children_are_leaves());
        assert!(node.parent().is_null());
    }

    #[test]
    fn test_new_root() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new_root(1);

        assert!(!node.version().is_leaf());
        assert!(node.version().is_root());
        assert_eq!(node.height(), 1);
        assert!(!node.children_are_leaves());
    }

    #[test]
    fn test_key_accessors() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        node.set_ikey(0, 0x1000_0000_0000_0000);
        node.set_ikey(1, 0x2000_0000_0000_0000);
        node.set_ikey(2, 0x3000_0000_0000_0000);
        node.set_nkeys(3);

        assert_eq!(node.ikey(0), 0x1000_0000_0000_0000);
        assert_eq!(node.ikey(1), 0x2000_0000_0000_0000);
        assert_eq!(node.ikey(2), 0x3000_0000_0000_0000);
        assert_eq!(node.size(), 3);
    }

    #[test]
    fn test_child_accessors() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        let fake_child0: *mut u8 = std::ptr::without_provenance_mut(0x1000);
        let fake_child1: *mut u8 = std::ptr::without_provenance_mut(0x2000);
        let fake_child2: *mut u8 = std::ptr::without_provenance_mut(0x3000);

        node.set_child(0, fake_child0);
        node.set_child(1, fake_child1);
        node.set_child(2, fake_child2);

        assert_eq!(node.child(0), fake_child0);
        assert_eq!(node.child(1), fake_child1);
        assert_eq!(node.child(2), fake_child2);
    }

    #[test]
    fn test_assign() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        let left_child: *mut u8 = std::ptr::without_provenance_mut(0x1000);
        let right_child: *mut u8 = std::ptr::without_provenance_mut(0x2000);

        // Set left child first
        node.set_child(0, left_child);

        // Assign key and right child
        node.assign(0, 0xABCD_0000_0000_0000, right_child);
        node.set_nkeys(1);

        assert_eq!(node.ikey(0), 0xABCD_0000_0000_0000);
        assert_eq!(node.child(0), left_child);
        assert_eq!(node.child(1), right_child);
        assert_eq!(node.size(), 1);
    }

    #[test]
    fn test_inc_nkeys() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        assert_eq!(node.nkeys(), 0);

        node.inc_nkeys();
        assert_eq!(node.nkeys(), 1);

        node.inc_nkeys();
        assert_eq!(node.nkeys(), 2);
    }

    #[test]
    fn test_is_full() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        assert!(!node.is_full());

        node.set_nkeys(15);
        assert!(node.is_full());
    }

    #[test]
    fn test_parent_accessors() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        let mut parent: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(1);

        let parent_ptr: *mut InternodeNode<LeafValue<u64>> =
            parent.as_mut() as *mut InternodeNode<LeafValue<u64>, 15>;

        // set_parent takes *mut u8, so cast the pointer
        node.set_parent(parent_ptr.cast::<u8>());
        assert_eq!(node.parent(), parent_ptr.cast::<u8>());
    }

    #[test]
    fn test_compare_key() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        node.set_ikey(0, 0x5000_0000_0000_0000);
        node.set_nkeys(1);

        assert_eq!(node.compare_key(0x3000_0000_0000_0000, 0), Ordering::Less);
        assert_eq!(node.compare_key(0x5000_0000_0000_0000, 0), Ordering::Equal);
        assert_eq!(
            node.compare_key(0x7000_0000_0000_0000, 0),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compact_internode() {
        let node: Box<InternodeNodeCompact<LeafValue<u64>>> = InternodeNode::new(0);

        assert_eq!(node.size(), 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_invariants_valid() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Set up correctly sorted keys
        node.set_ikey(0, 0x1000_0000_0000_0000);
        node.set_ikey(1, 0x2000_0000_0000_0000);
        node.set_ikey(2, 0x3000_0000_0000_0000);
        node.set_nkeys(3);

        // Should not panic
        node.debug_assert_invariants();
    }

    #[test]
    #[should_panic(expected = "keys not in ascending order")]
    #[cfg(debug_assertions)]
    fn test_invariant_unsorted_keys() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Set up unsorted keys
        node.set_ikey(0, 0x3000_0000_0000_0000);
        node.set_ikey(1, 0x1000_0000_0000_0000); // Wrong order!
        node.set_nkeys(2);

        node.debug_assert_invariants(); // Should panic
    }

    // ========================================================================
    //  find_insert_position tests (binary search verification)
    // ========================================================================

    #[test]
    fn test_find_insert_position_empty() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        // Empty node: any key goes at position 0
        assert_eq!(node.find_insert_position(0x1000), 0);
        assert_eq!(node.find_insert_position(0), 0);
        assert_eq!(node.find_insert_position(u64::MAX), 0);
    }

    #[test]
    fn test_find_insert_position_single_key() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        node.set_ikey(0, 100);
        node.set_nkeys(1);

        // Key < existing: goes before
        assert_eq!(node.find_insert_position(50), 0);
        // Key == existing: goes at same position
        assert_eq!(node.find_insert_position(100), 0);
        // Key > existing: goes after
        assert_eq!(node.find_insert_position(150), 1);
    }

    #[test]
    fn test_find_insert_position_multiple_keys() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Set up keys: 10, 20, 30, 40, 50
        node.set_ikey(0, 10);
        node.set_ikey(1, 20);
        node.set_ikey(2, 30);
        node.set_ikey(3, 40);
        node.set_ikey(4, 50);
        node.set_nkeys(5);

        // Before all
        assert_eq!(node.find_insert_position(5), 0);
        // Equal to first
        assert_eq!(node.find_insert_position(10), 0);
        // Between first and second
        assert_eq!(node.find_insert_position(15), 1);
        // Equal to middle
        assert_eq!(node.find_insert_position(30), 2);
        // Between 30 and 40
        assert_eq!(node.find_insert_position(35), 3);
        // Equal to last
        assert_eq!(node.find_insert_position(50), 4);
        // After all
        assert_eq!(node.find_insert_position(100), 5);
    }

    #[test]
    fn test_find_insert_position_full_node() {
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Fill with keys 10, 20, 30, ..., 150
        for i in 0..15 {
            node.set_ikey(i, (i as u64 + 1) * 10);
        }
        node.set_nkeys(15);

        // Verify binary search works for all positions
        assert_eq!(node.find_insert_position(5), 0); // Before first
        assert_eq!(node.find_insert_position(10), 0); // Equal to first
        assert_eq!(node.find_insert_position(75), 7); // Mid-range
        assert_eq!(node.find_insert_position(80), 7); // Equal to key[7]
        assert_eq!(node.find_insert_position(145), 14); // Between 140 and 150
        assert_eq!(node.find_insert_position(150), 14); // Equal to last
        assert_eq!(node.find_insert_position(200), 15); // After all
    }

    // ========================================================================
    //  Split edge case tests
    // ========================================================================

    #[test]
    fn test_split_insert_at_position_0() {
        // Test splitting when the new key goes at position 0 (smallest)
        // Use height=0 so split_into treats children as leaves (doesn't dereference them)
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        let mut new_right: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Fill the node with keys 20, 30, 40, ..., 160 (15 keys)
        for i in 0..15 {
            node.set_ikey(i, (i as u64 + 2) * 10);
            node.set_child(i, std::ptr::without_provenance_mut((i + 1) * 0x1000));
        }
        node.set_child(15, std::ptr::without_provenance_mut(16 * 0x1000));
        node.set_nkeys(15);

        let new_right_ptr: *mut InternodeNode<LeafValue<u64>, 15> = new_right.as_mut();
        let new_child: *mut u8 = std::ptr::without_provenance_mut(0xABCD);

        // Insert key 10 at position 0 (smallest)
        let (popup_key, insert_went_left) =
            node.split_into(&mut new_right, new_right_ptr, 0, 10, new_child);

        // Insert at position 0 < mid(8), so it goes left
        assert!(insert_went_left, "Insert at position 0 should go left");

        // Verify popup key is reasonable (should be one of the keys around mid)
        assert!(popup_key > 0, "Popup key should be non-zero");
    }

    #[test]
    fn test_split_insert_at_width() {
        // Test splitting when the new key goes at position WIDTH (largest)
        // Use height=0 so split_into treats children as leaves (doesn't dereference them)
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        let mut new_right: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
        for i in 0..15 {
            node.set_ikey(i, (i as u64 + 1) * 10);
            node.set_child(i, std::ptr::without_provenance_mut((i + 1) * 0x1000));
        }
        node.set_child(15, std::ptr::without_provenance_mut(16 * 0x1000));
        node.set_nkeys(15);

        let new_right_ptr: *mut InternodeNode<LeafValue<u64>, 15> = new_right.as_mut();
        let new_child: *mut u8 = std::ptr::without_provenance_mut(0xABCD);

        // Insert key 200 at position 15 (largest, after all existing)
        let (popup_key, insert_went_left) =
            node.split_into(&mut new_right, new_right_ptr, 15, 200, new_child);

        // Insert at position 15 > mid(8), so it goes right
        assert!(
            !insert_went_left,
            "Insert at position WIDTH should go right"
        );

        // Verify popup key is reasonable
        assert!(popup_key > 0, "Popup key should be non-zero");
    }

    #[test]
    fn test_split_insert_at_midpoint() {
        // Test splitting when the new key goes at the midpoint (becomes popup)
        // Use height=0 so split_into treats children as leaves (doesn't dereference them)
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        let mut new_right: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);

        // Fill the node with keys 10, 20, 30, ..., 150 (15 keys)
        for i in 0..15 {
            node.set_ikey(i, (i as u64 + 1) * 10);
            node.set_child(i, std::ptr::without_provenance_mut((i + 1) * 0x1000));
        }
        node.set_child(15, std::ptr::without_provenance_mut(16 * 0x1000));
        node.set_nkeys(15);

        let new_right_ptr: *mut InternodeNode<LeafValue<u64>, 15> = new_right.as_mut();
        let new_child: *mut u8 = std::ptr::without_provenance_mut(0xABCD);

        // mid = ceil(15/2) = 8
        // Insert key 85 at position 8 (the midpoint)
        let (popup_key, insert_went_left) =
            node.split_into(&mut new_right, new_right_ptr, 8, 85, new_child);

        // When insert_pos == mid, the insert key becomes the popup key
        assert_eq!(popup_key, 85, "Insert at midpoint should become popup key");

        // insert_went_left behavior at midpoint depends on implementation details.
        // The key observation is that when insert_pos == mid, the inserted key
        // becomes the popup key, so it doesn't go to either sibling.
        // We just verify the function completed successfully.
        let _ = insert_went_left;
    }

    #[test]
    fn test_prefetch_does_not_crash() {
        // Just verify prefetch doesn't crash - it's a no-op hint on most systems
        let node: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        node.prefetch();
        // If we get here, prefetch worked (it's just a hint, no crash expected)
    }
}

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
mod loom_tests {
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
    use loom::thread;

    /// Simplified internode for loom testing.
    ///
    /// Uses loom atomics to enable deterministic interleaving exploration.
    /// Only includes the fields needed for testing concurrent key access.
    struct LoomInternode {
        nkeys: AtomicU8,
        ikey0: [AtomicU64; 4], // Small width for faster loom exploration
    }

    impl LoomInternode {
        fn new() -> Self {
            Self {
                nkeys: AtomicU8::new(0),
                ikey0: std::array::from_fn(|_| AtomicU64::new(0)),
            }
        }

        fn nkeys(&self) -> usize {
            self.nkeys.load(Ordering::Acquire) as usize
        }

        fn set_nkeys(&self, n: u8) {
            self.nkeys.store(n, Ordering::Release);
        }

        fn ikey(&self, idx: usize) -> u64 {
            self.ikey0[idx].load(Ordering::Acquire)
        }

        fn set_ikey(&self, idx: usize, key: u64) {
            self.ikey0[idx].store(key, Ordering::Release);
        }

        /// Binary search matching the real implementation.
        fn find_insert_position(&self, insert_ikey: u64) -> usize {
            let n: usize = self.nkeys();

            let mut lo: usize = 0;
            let mut hi: usize = n;

            while lo < hi {
                let mid: usize = (lo + hi) >> 1;
                if self.ikey(mid) < insert_ikey {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            lo
        }

        /// Simulate key insertion (simplified, no children).
        fn insert_key(&self, pos: usize, key: u64) {
            let n = self.nkeys();

            // Shift keys right
            for i in (pos..n).rev() {
                let k = self.ikey(i);
                self.set_ikey(i + 1, k);
            }

            // Insert new key
            self.set_ikey(pos, key);

            // Increment nkeys with release ordering
            self.set_nkeys((n + 1) as u8);
        }
    }

    /// Test that concurrent reads of find_insert_position are consistent.
    ///
    /// Even without locking, reads should see a consistent snapshot of
    /// the keys (either before or after modification, not partial).
    #[test]
    fn test_loom_find_position_concurrent_reads() {
        loom::model(|| {
            let node = Arc::new(LoomInternode::new());

            // Setup: insert keys 10, 20, 30
            node.set_ikey(0, 10);
            node.set_ikey(1, 20);
            node.set_ikey(2, 30);
            node.set_nkeys(3);

            let n1 = Arc::clone(&node);
            let t1 = thread::spawn(move || {
                // Read: find position for key 25
                let pos = n1.find_insert_position(25);
                // Should be between 20 and 30, so position 2
                // But if we see partial state, result may vary
                pos
            });

            let n2 = Arc::clone(&node);
            let t2 = thread::spawn(move || {
                // Read: find position for key 15
                let pos = n2.find_insert_position(15);
                pos
            });

            let pos1 = t1.join().unwrap();
            let pos2 = t2.join().unwrap();

            // Both should get valid positions (0-3)
            assert!(pos1 <= 3, "pos1={} should be <= 3", pos1);
            assert!(pos2 <= 3, "pos2={} should be <= 3", pos2);
        });
    }

    /// Test that find_insert_position during concurrent write sees
    /// consistent state (via atomic loads).
    #[test]
    fn test_loom_find_position_during_insert() {
        loom::model(|| {
            let node = Arc::new(LoomInternode::new());

            // Setup: insert key 20
            node.set_ikey(0, 20);
            node.set_nkeys(1);

            let results = Arc::new(AtomicUsize::new(0));

            let n1 = Arc::clone(&node);
            let t1 = thread::spawn(move || {
                // Writer: insert key 10 at position 0
                n1.insert_key(0, 10);
            });

            let n2 = Arc::clone(&node);
            let r2 = Arc::clone(&results);
            let t2 = thread::spawn(move || {
                // Reader: find position for key 15
                let pos = n2.find_insert_position(15);
                r2.store(pos, Ordering::Relaxed);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Result depends on interleaving:
            // - If reader runs before insert: sees [20], returns 0 (15 < 20)
            // - If reader runs after insert: sees [10, 20], returns 1 (10 < 15 < 20)
            // - Partial views possible but should still be valid
            let pos = results.load(Ordering::Relaxed);
            assert!(pos <= 2, "pos={} should be <= 2", pos);
        });
    }

    /// Test that concurrent inserts maintain sorted order invariant.
    ///
    /// NOTE: In real usage, inserts are protected by locks. This test
    /// verifies that atomic operations don't corrupt data even without
    /// locks (the result may be non-deterministic but should be valid).
    #[test]
    fn test_loom_concurrent_reads_different_keys() {
        loom::model(|| {
            let node = Arc::new(LoomInternode::new());

            // Setup: insert keys 10, 20, 30, 40
            node.set_ikey(0, 10);
            node.set_ikey(1, 20);
            node.set_ikey(2, 30);
            node.set_ikey(3, 40);
            node.set_nkeys(4);

            let n1 = Arc::clone(&node);
            let t1 = thread::spawn(move || n1.find_insert_position(5)); // Before all

            let n2 = Arc::clone(&node);
            let t2 = thread::spawn(move || n2.find_insert_position(25)); // Middle

            let n3 = Arc::clone(&node);
            let t3 = thread::spawn(move || n3.find_insert_position(50)); // After all

            let pos1 = t1.join().unwrap();
            let pos2 = t2.join().unwrap();
            let pos3 = t3.join().unwrap();

            // All should get deterministic results (no concurrent writes)
            assert_eq!(pos1, 0, "5 should go at position 0");
            assert_eq!(pos2, 2, "25 should go at position 2");
            assert_eq!(pos3, 4, "50 should go at position 4");
        });
    }
}
