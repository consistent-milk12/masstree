use super::{Backoff, LockGuard, NodeVersion, StdThread};

const MAX_SPINS: u32 = 16;

impl NodeVersion {
    /// Acquire the lock with bounded spinning before yielding.
    ///
    /// This method spins for up to [`MAX_SPINS`] iterations using exponential
    /// backoff before yielding the CPU. Optimized for workloads where:
    ///
    /// - Contention is transient (locks held briefly)
    /// - Context switch overhead exceeds spin time
    /// - Insert-heavy patterns target the same leaf repeatedly
    ///
    /// # Algorithm
    ///
    /// 1. Try to acquire the lock with `try_lock()`
    /// 2. If failed, spin with exponential backoff (up to 16 iterations)
    /// 3. After 16 spins, yield CPU and restart spin counter
    /// 4. Repeat until lock acquired
    ///
    /// # When to Use
    ///
    /// Use `lock_bounded()` for:
    /// - Leaf-level insert ops (short critical sections)
    /// - Workloads with predictable, brief lock holds
    ///
    /// Use `lock_with_yield` for:
    /// - Split propagation (longer critical sections)
    /// - Unknown contentions patterns
    ///
    /// # Memory Ordering
    ///
    /// Uses [`Acquire`](std::sync::atomic::Ordering::Acquire) ordering on successful lock
    /// acquisition.
    #[must_use = "releasing a lock without using the guard is a logic error"]
    pub fn lock_bounded(&self) -> LockGuard<'_> {
        let mut backoff = Backoff::new();
        let mut spin_count: u32 = 0;

        loop {
            // Fast Path: try to acquire immediately
            if let Some(guard) = self.try_lock() {
                return guard;
            }

            spin_count += 1;

            if spin_count < MAX_SPINS {
                // Spin with exponential backoff
                // This reduces cache line contention compared to tight polling
                backoff.spin();
            } else {
                // Fallback: yield after bounded spinning.
                // This prevents starvation under sustained contention
                StdThread::yield_now();
                spin_count = 0;
                backoff = Backoff::new();
            }
        }
    }
}
