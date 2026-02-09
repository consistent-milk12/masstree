//! Key search algorithms for `MassTree`.
//!
//! Provides scalar ikey matching for internode routing.
//!
//! # Reference
//! Based on `ksearch.hh` from the C++ Masstree implementation.

use crate::ksearch::scalar::Scalar;
use crate::leaf_trait::TreeInternode;

pub mod scalar;

// ============================================================================
//  Internode Search
// ============================================================================

/// Upper bound search for internode routing.
///
/// Returns child index (0 to nkeys) to follow for the given ikey.
#[inline]
pub fn upper_bound_internode_generic<I: TreeInternode>(search_ikey: u64, node: &I) -> usize {
    Scalar::upper_bound_internode_scalar(search_ikey, node)
}
