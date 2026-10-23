# Sort-order probes

These are standalone Rust probes used to characterize `sort_unstable_by_key`
reordering of equal keys. They do not depend on masstree.

```bash
rustc probes/sortcheck.rs -o /tmp/sortcheck && /tmp/sortcheck
rustc probes/sortcheck2.rs -o /tmp/sortcheck2 && /tmp/sortcheck2
```

They show that pdqsort reorders equal ikeys for realistic batch sizes, which is
the mechanism behind BATCH-04.
