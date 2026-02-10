//! Marker trait for leaf policies that support reference returns.
//!
//! Re-exports [`RefPolicy`] as `RefLeafPolicy` for backward compatibility
//! with existing import sites.

pub use crate::policy::RefPolicy as RefLeafPolicy;
