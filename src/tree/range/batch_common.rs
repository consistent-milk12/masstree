//! Shared helpers for forward and reverse batch processing.
//!
//! Extracted to eliminate key-building duplication across `find.rs` and `find_rev.rs`.
//!
//! # `SlotVisitor` Trait
//!
//! Abstracts the ref vs copy value-loading difference. Two implementations:
//! - [`RefSlotVisitor`]: loads `&P::Value` via pointer dereference (zero-copy)
//! - [`CopySlotVisitor`]: loads `P::Output` via `load_value()` (universal)
//!
//! After monomorphization, the trait dispatch is fully inlined — zero overhead.

use crate::key::IKEY_SIZE;
use crate::leaf15::LeafNode15;
use crate::leaf15::KSUF_KEYLENX;
use crate::policy::LeafPolicy;

use super::cursor_key::CursorKey;

/// Build the full key in `cursor_key` from slot data.
///
/// Stores the ikey unconditionally, then handles suffix or inline length.
/// Callers should call `cursor_key.mark_key_complete()` after this returns.
#[inline(always)]
pub fn build_slot_key<P: LeafPolicy>(
    cursor_key: &mut CursorKey,
    leaf: &LeafNode15<P>,
    slot: usize,
    slot_ikey: u64,
    slot_keylenx: u8,
) {
    cursor_key.assign_store_ikey(slot_ikey);

    if slot_keylenx == KSUF_KEYLENX {
        if let Some(suffix) = leaf.ksuf(slot) {
            let suffix_len = suffix.len();
            let _ = cursor_key.assign_store_suffix(suffix);
            cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
        } else {
            cursor_key.assign_store_length(IKEY_SIZE);
        }
    } else {
        let len = slot_keylenx as usize;
        cursor_key.assign_store_length(len);
    }
}

/// Visitor for batch slot processing — abstracts ref vs copy value loading.
///
/// Each implementation defines how to load and deliver a value from a leaf slot
/// to the user's callback. The batch loop calls [`visit`](SlotVisitor::visit)
/// after building the key, letting the visitor handle value access.
pub trait SlotVisitor<P: LeafPolicy> {
    /// Visit a slot after key has been built in `cursor_key`.
    ///
    /// # Returns
    ///
    /// - `None`: Value was concurrently removed (TOCTOU race) — skip this slot
    /// - `Some(true)`: Continue scanning
    /// - `Some(false)`: Visitor requested stop
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool>;
}

/// Ref visitor: loads `&P::Value` via pointer dereference (zero-copy).
///
/// For use with `RefLeafPolicy` types where values are pointer-backed (Arc, Box).
pub struct RefSlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for RefSlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], &P::Value) -> bool,
{
    #[inline(always)]
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool> {
        // SAFETY: Guard protects value, slot is valid (in permutation).
        // Null-check handles TOCTOU race — value may have been concurrently
        // removed between caller's is_value_empty check and this dereference.
        let ptr = unsafe { leaf.load_value_ptr(slot) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: Non-null pointer to a valid P::Value, protected by OCC guard.
        let value_ref: &P::Value = unsafe { &*ptr };
        Some((self.0)(key, value_ref))
    }
}

/// Copy visitor: loads `P::Output` via `load_value()` (universal).
///
/// Works for all `LeafPolicy` types including true-inline storage.
pub struct CopySlotVisitor<F>(pub F);

impl<P, F> SlotVisitor<P> for CopySlotVisitor<F>
where
    P: LeafPolicy,
    F: FnMut(&[u8], P::Output) -> bool,
{
    #[inline(always)]
    fn visit(&mut self, leaf: &LeafNode15<P>, slot: usize, key: &[u8]) -> Option<bool> {
        // Use load_value to handle TOCTOU race — value may have been
        // concurrently removed between is_value_empty check and here.
        let output = leaf.load_value(slot)?;
        Some((self.0)(key, output))
    }
}
