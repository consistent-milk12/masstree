//! Filepath: src/internode.rs
//!
//! Internode (internal node) for `MassTree`.
//!
//! Internodes route traversals through the tree. They contain only
//! keys and child pointers, no values. Keys are always in sorted order
//! (no permutation array needed).

use std::cmp::Ordering;
use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::mem as StdMem;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, fence};

use crate::leaf::LeafValue;
use crate::nodeversion::NodeVersion;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};
use crate::slot::ValueSlot;

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
    pub fn new_root(height: u32) -> Box<Self> {
        let node: Box<Self> = Self::new(height);
        node.version.mark_root();
        node
    }

    // ========================================================================
    //  Version Accessors
    // ========================================================================

    /// Get a reference to the node's version.
    #[inline]
    #[must_use]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the node's version.
    #[inline]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ========================================================================
    //  Key Accessors
    // ========================================================================

    /// Get the number of keys in this internode.
    #[inline]
    #[must_use]
    pub fn nkeys(&self) -> usize {
        self.nkeys.load(READ_ORD) as usize
    }

    /// Get the number of keys as usize (convenience method).
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.nkeys()
    }

    /// Check if the internode has no keys.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nkeys.load(READ_ORD) == 0
    }

    /// Check if the internode is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.nkeys.load(READ_ORD) as usize >= WIDTH
    }

    /// Get the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[inline]
    #[must_use]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn ikey(&self, i: usize) -> u64 {
        debug_assert!(i < WIDTH, "ikey: index out of bounds");

        self.ikey0[i].load(READ_ORD)
    }

    /// Set the key at the given index.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[inline]
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn set_ikey(&self, i: usize, ikey: u64) {
        debug_assert!(i < WIDTH, "set_ikey: index out of bounds");

        self.ikey0[i].store(ikey, WRITE_ORD);
    }

    /// Get the tree height.
    ///
    /// - `height = 0` means children are leaves
    /// - `height > 0` means children are internodes
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Check if children are leaves (height == 0).
    #[inline]
    #[must_use]
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
    #[inline]
    #[must_use]
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
    #[inline]
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
    #[inline]
    pub fn set_nkeys(&self, n: u8) {
        debug_assert!((n as usize) <= WIDTH, "set_nkeys: count out of bounds");
        self.nkeys.store(n, WRITE_ORD);
    }

    /// Increment the number of keys by 1.
    ///
    /// # Panics
    /// Panics in debug mode if already at WIDTH.
    #[inline]
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
    /// This matches the C++ `internode::split_into()` semantics from `reference/masstree_split.hh`:
    /// - Splits the full internode at midpoint
    /// - Simultaneously inserts `(insert_ikey, insert_child)` at position `insert_pos`
    /// - Returns the popup key for the parent
    ///
    /// After split:
    /// - `self` contains keys [0, mid)
    /// - `new_right` contains keys [mid+1, WIDTH+1)
    /// - The key at post-insert position `mid` becomes the popup key
    ///
    /// # Arguments
    /// * `new_right` - The new right sibling (caller allocates)
    /// * `insert_pos` - Position to insert the new key/child `(0 <= insert_pos <= WIDTH)`
    /// * `insert_ikey` - The key to insert (popup key from child split)
    /// * `insert_child` - The child to insert (new right sibling from child split)
    ///
    /// # Returns
    /// * `popup_key` - The key to propagate to the parent
    /// * `insert_went_left` - True if the insertion went into self (left), false if into `new_right`
    ///
    /// # Reference
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

        (popup_key, insert_went_left)
    }

    // ========================================================================
    //  Parent Accessors
    // ========================================================================

    /// Get the parent pointer (as `*mut u8`).
    ///
    /// Cast to `*mut InternodeNode<V, WIDTH>` at usage sites.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    /// Set the parent pointer.
    ///
    /// Accepts `*mut u8` for uniformity with `LeafNode`.
    #[inline]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    /// Check if this is a root node (no parent or version says root).
    #[inline]
    #[must_use]
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
    #[inline]
    #[must_use]
    pub fn compare_key(&self, search_ikey: u64, p: usize) -> std::cmp::Ordering {
        search_ikey.cmp(&self.ikey(p))
    }

    /// Find the position where a key should be inserted.
    ///
    /// Returns the index where `insert_ikey` should go, such that
    /// `ikey(i-1) < insert_ikey <= ikey(i)` (or at the end if greater than all).
    ///
    /// FIXED: Used in the data race fix for recomputing child index after reacquiring lock.
    pub fn find_insert_position(&self, insert_ikey: u64) -> usize {
        let n = self.nkeys();
        for i in 0..n {
            if insert_ikey <= self.ikey(i) {
                return i;
            }
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

// NOTE: Send/Sync impls are intentionally omitted for now.
// Tree operations are single-threaded (Phase 3.2-3.3 pending). NodeVersion
// has CAS-based locking but tree ops don't use it yet. When optimistic get
// and locked insert are implemented, appropriate Send/Sync impls can be
// added with correct SAFETY documentation.
//
// Until then, raw pointers make InternodeNode !Send + !Sync by default,
// which is the correct conservative choice.

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
}
