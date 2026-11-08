# Suffix allocation regression probe

`src/main.rs` counts global allocations for two workloads and asserts that
suffix storage remains amortized. It also checks every surviving value and len.

1. **PERF-01** — 13 keys with 40-byte suffixes, then 260 value updates:
   zero allocations (already fixed before this series).
2. **PERF-02** — 15 keys with 200-byte suffixes, then 20 remove/reinsert cycles:
   baseline 42 allocations / 71,584 bytes; after sizing rebuilds from the complete
   live set and skipping throwaway preallocation, 6 allocations / 20,560 bytes.
   The bound allows EBR bookkeeping while rejecting per-operation rebuilding.

```bash
CARGO_TARGET_DIR=/tmp/masstree-target cargo run --quiet --manifest-path failures/suffix/Cargo.toml --bin probe
```

The probe failed at the allocation bound before the fix. Allocation counts are
from the system allocator, default suffix capacity, and the inline-value policy;
they are not throughput measurements. `compact_empty` remains a separate SUF-02
regression. Run each failure-corpus binary in its own process.
