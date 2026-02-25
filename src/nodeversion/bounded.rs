use super::{Backoff, LOCK_BIT, LockGuard, NodeVersion, StdThread};
use core::marker::PhantomData;
use core::ptr as StdPtr;
use core::sync::atomic::Ordering;

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
    /// 1. Try to acquire the lock with a CAS
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
        // Fast path: try immediate acquire.
        let value: u32 = self.value.load(Ordering::Relaxed);

        if (value & LOCK_BIT) == 0 {
            let locked: u32 = value | LOCK_BIT;

            if self
                .value
                .compare_exchange_weak(value, locked, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return LockGuard {
                    version: StdPtr::from_ref(self),
                    locked_value: locked,
                    _lifetime: PhantomData,
                    _marker: PhantomData,
                };
            }
        }

        self.lock_bounded_contended()
    }

    #[cold]
    #[inline(never)]
    fn lock_bounded_contended(&self) -> LockGuard<'_> {
        let mut backoff = Backoff::new();
        let mut spin_count: u32 = 0;
        let mut value: u32 = self.value.load(Ordering::Relaxed);

        loop {
            if (value & LOCK_BIT) == 0 {
                let locked: u32 = value | LOCK_BIT;

                match self.value.compare_exchange_weak(
                    value,
                    locked,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return LockGuard {
                            version: StdPtr::from_ref(self),
                            locked_value: locked,
                            _lifetime: PhantomData,
                            _marker: PhantomData,
                        };
                    }
                    Err(v) => {
                        value = v;
                        continue; // Retry immediately with fresh value
                    }
                }
            }

            spin_count += 1;

            if spin_count < MAX_SPINS {
                backoff.spin();
            } else {
                // Fallback: yield after bounded spinning.
                // This prevents starvation under sustained contention
                StdThread::yield_now();
                spin_count = 0;
                backoff = Backoff::new();
            }

            value = self.value.load(Ordering::Relaxed);
        }
    }
}
