# Suffix allocation/reproduction probe

`src/main.rs` counts global allocations for two workloads:

1. **PERF-SUF-01** — 13 keys with 40-byte suffixes in one leaf, then 260 value
   updates. Expected: ~1 allocation and ~600 bytes per update even though only
   the value changes.
2. **PERF-SUF-02** — 15 keys with 200-byte suffixes in a full leaf, then 20
   remove+reinsert cycles. Expected: ~2 allocations and ~3.5 KB per cycle from
   drain-and-rebuild.

```bash
cd failures/suffix
cargo run --quiet
```

The suffix functional failures in SUF-01…SUF-05 (plus SAFE-06 for the external-bag race) are small public-API snippets
documented in `SystemAudit.md`; the allocation measurements here are the
runtime evidence.
