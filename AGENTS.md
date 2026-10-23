# AGENTS.md

Instructions for AI coding agents and human contributors working on this
repository. Keep this file concise and update it when the project's workflow or
invariants change.

## 1. Project overview

`masstree` is a Rust implementation of the Masstree ordered-map data structure:
a trie of B+trees with concurrent OCC readers, leaf/internode locks, B-link
siblings, EBR reclamation, inline or boxed values, and optional suffix storage.

- Crate: `masstree` 0.9.5, Rust edition 2024.
- Public aliases: `MassTree` = `MassTree15Inline` (inline values);
  `MassTree15` = boxed small-value `BoxPolicy`/sidecar-suffix tree.
- Default feature: `debug-print`.
- Other features: `mimalloc`, `tracing`, `debug-routing`, `no-prefetch`,
  `small-suffix-capacity`, `sidecar-suffix`.
- Architecture reference: `ARCHITECTURE.md`.
- Audit findings and fix order: `SystemAudit.md`.
- Reproducible failure corpus: `failures/` (intentionally failing programs).

Do not assume comments or `ARCHITECTURE.md` are correct. The audit found
several architecture claims contradicted by the code. Verify against source and
tests.

## 2. First steps for any task

1. Read `ARCHITECTURE.md` and the relevant section of `SystemAudit.md`.
2. If the task is a bug fix, search `failures/` for an existing reproducer.
   Prefer extending an existing reproducer over writing a new stress test.
3. Read the actual call paths in `src/`; do not trust names such as
   `is_value_empty`, `is_layer`, `assign_init`, or `has_changed` without checking
   their implementation.
4. Make a focused regression test that fails for the expected reason before
   changing behavior.
5. Keep changes minimal and subsystem-local unless the fix requires a protocol
   change.
6. Run targeted tests, then the relevant feature/architecture checks.

## 3. Build and test commands

Start with the standard library tests:

```bash
cargo test --lib
cargo test --test <name>
cargo test --test batch_insert_tests --test true_inline
```

Targeted/feature builds:

```bash
cargo check
cargo check --no-default-features
cargo check --all-features
cargo check --features sidecar-suffix
cargo check --features small-suffix-capacity
cargo check --features no-prefetch
cargo check --tests
```

Known baseline failures (do not treat as a reason to ignore the whole command):

```bash
cargo check --all-targets --no-default-features   # fails on examples/print_tree.rs
cargo clippy --all-targets --all-features -- -D warnings  # currently 4 errors
```

These are tracked in `SystemAudit.md` as `TOOL-01` and `TOOL-04`. Fixing them is
welcome, but do not mix unrelated lint cleanup into a correctness fix.

Loom:

```bash
RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
```

Miri:

```bash
cargo +nightly miri test --lib <test-filter>
cargo +nightly miri test --test leak_tests
cargo +nightly miri run --bin <failure-bin>
```

Use `MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check"`.
Miri does not model all weak-memory reorderings; see `OCC-03` in the audit.

Never run the entire `failures/` corpus in one process. Several binaries
intentionally abort, hang, or trigger UB. Use `timeout` and a disposable process.

## 4. Repository map

| Path | Responsibility |
|---|---|
| `src/tree/generic.rs` | top-level dispatch, reach/advance, insert/remove entry points, node accounting |
| `src/tree/generic/{search,insert,insert_strategy}.rs` | optimistic search, lock/retry, insert/update paths |
| `src/tree/generic/{split,layer,root_creation}.rs` | split propagation, layer creation, root promotion |
| `src/tree/generic/{batch,entry}.rs` | batch and entry APIs |
| `src/tree/generic/optimistic_reads*.rs` | OCC point-read paths |
| `src/tree/remove.rs` | optimistic remove, parent updates, cascades |
| `src/tree/coalesce.rs` | lazy leaf/sublayer GC, raw/deferred queues |
| `src/tree/range/` | forward/reverse/prefix/values scans, cursors |
| `src/leaf15.rs`, `src/leaf15/layout.rs` | 15-slot leaf, keys/values/suffixes, split/link |
| `src/internode.rs`, `src/internode/{accessors,layout}.rs` | 15-key/16-child internodes |
| `src/policy.rs`, `src/policy/*.rs` | `LeafPolicy`, `BoxPolicy`, `InlinePolicy`, value ownership, suffix ops |
| `src/suffix.rs`, `src/suffix/*.rs` | inline/external suffix bags, compaction |
| `src/nodeversion.rs` | OCC version/lock bits, split/delete/root state |
| `src/permuter.rs` | 15-slot permutation encoding and operations |
| `src/alloc15.rs`, `src/node_pool.rs` | allocator, EBR reclamation, TLS node pool, teardown |
| `src/key.rs`, `src/ksearch/` | key chunks/ordering and internode routing |
| `src/shard_counter.rs` | approximate concurrent `len()` counter |
| `tests/` | integration, stress, proptests, leak tests, C++-shaped parity |
| `benches/` | divan/pbench/custom benchmark binaries |
| `failures/` | standalone reproducers for audited bugs; not part of the crate |

## 5. Critical invariants

These are not suggestions; breaking them causes data loss, UAF, or hangs.

### Leaf/internode layout

- `WIDTH = 15`; a full leaf has 15 active slots; an internode has at most 15
  separator keys and 16 children.
- Maximum key length is 256 bytes; a suffix is at most 248 bytes.
- The first 8 bytes are the `ikey`; keys shorter than 8 bytes are zero-padded.
  Two distinct keys can share an ikey. Never use ikey equality alone to identify
  a key.
- `keylenx`:
  - terminal inline key: key length (`0..=8`);
  - `KSUF_KEYLENX` (64): suffix is stored in the suffix store;
  - `>= LAYER_KEYLENX` (128): slot is a layer pointer, not a value.
- A layer conversion is a structural mutation: it clears the old value/suffix
  and changes `keylenx` before publishing the layer pointer. OCC readers must not
  accept a partial conversion.

### Layer roots

- On promotion, the parent slot continues to point at the original **left leaf**;
  the true layer root is an internode reached by climbing `parent` pointers.
  This invariant is the source of several bugs (`RNG-04`, `GC-02`).
- Any code that assumes a layer-slot pointer is a leaf must either climb to the
  root or explicitly reject promoted layers.

### B-link chain and retirement

- A split publishes the right sibling and the B-link mark; readers/inserters
  must follow `next` when the target is to the right.
- A coalesced leaf is marked deleted, unlinked, then retired. Do not retire a
  node while a published child/parent pointer still reaches it.
- Raw pointers stored across maintenance calls are unsafe. Prefer `ikey_bound`
  re-traversal or a generation token.
- Ordinary chain coalesce takes lock order leaf → parent. Sublayer GC takes
  parent leaf → sublayer leaf/root. Do not introduce a reverse cycle.

### OCC

- `NodeVersion::stable()`/`has_changed()` are the reader validation primitives.
  Current `has_changed` deliberately ignores `LOCK_BIT | INSERTING_BIT`
  (`OCC-02`); changing it requires auditing every read path.
- New-slot inserts currently skip `mark_insert` in the normal state
  (`OCC-01`); do not rely on the version word changing after an insert until
  this is fixed.
- Writers must publish slot/value/keylenx/permutation before the Release unlock;
  readers must validate after loading the data. See `OCC-03` for the weak-memory
  fence problem.
- A negative multi-layer read (`None`, empty layer pointer, cleared value) must
  retry if the version is dirty. Do not turn a transient missing value into a
  linearizable `None`.

### Value ownership

- `P::Output` for `BoxPolicy` is currently a non-owning `ValuePtr<V>` with no
  guard lifetime (`SAFE-01`). It must not be inserted into another slot, held
  past the guard, or dereferenced concurrently with a write-through update.
- `BatchEntry` currently exposes `output` and `key`; do not use
  `BatchEntry::from_output(get_with_guard(...))` or aliasing will double-free.
- `move_value_to`/`move_value_to_relaxed` copy the pointer without clearing the
  source. The caller must clear/retire correctly; the safe public method is
  currently a soundness hole (`SAFE-04`).
- Write-through values are mutated atomically in place; do not add a plain
  `&V` accessor without using `atomic_read_value`.

### Suffix storage

- Inline suffix metadata is atomic; external `SuffixBag` metadata and
  `Vec::len` are currently plain and race with lock-free readers
  (`SAFE-06`). Do not add more non-atomic reader-visible fields.
- Published bags must never reallocate. Capacity checks before append are
  required; growing is only for unpublished/exclusively owned leaves.
- `assign_init` requires an unpublished/exclusively owned leaf, not merely a
  locked one (`SUF-04`).

### Allocation/reclamation

- Do not use `POOL.with` on a path that can run during TLS destruction; use
  `try_with` with a direct allocator fallback (`ALLOC-01`).
- A node may be retired only after it is unreachable from the root and all
  published child pointers are updated (`GC-03`).
- `Drop` calls `reclaim_all()` and then teardown; a stale linked node can be
  freed twice.

## 6. Coding rules

- Run `cargo fmt --all` before finishing.
- Prefer explicit `Ordering` on every atomic. Do not add `SeqCst` without a
  memory-model reason; do not rely on x86 TSO for correctness.
- Every `unsafe` block needs a `// SAFETY:` comment naming the invariant,
  lifetime, and ownership argument.
- Do not introduce safe public APIs that return raw pointers, raw nodes, or
  unguarded values. If a raw API is required for internals, keep it
  `pub(crate)` and document the precondition.
- Keep the leaf/internode layout stable unless the change includes compile-time
  size/offset assertions and updates to `src/leaf15/layout.rs` or
  `src/internode/layout.rs`.
- Avoid adding `#[allow(dead_code)]`. Delete dead code or wire it into a test.
- Do not use `debug_assert!` for a precondition whose violation is memory
  unsafety or public-API silent corruption. Use `assert!` or `unsafe fn`.
- Prefer one shared helper over a second copy of search/bound/insert/remove
  logic. The codebase already suffers from duplicated state machines.
- Comments that claim “this cannot happen” need a proof or an assertion.
- Public API changes are semver-relevant. Note them in the PR/commit and update
  docs/tests.

## 7. Testing guidance

### Regression tests

- Use a `BTreeMap` model for sequential behavior; compare keys, values, len,
  and ordered iteration.
- For a concurrency bug, use a barrier and a bounded watchdog; do not use
  sleeps as the primary synchronization.
- For a hang, run the test with `timeout` and assert a sentinel/exit code.
- For ownership bugs, use a `Drop` counter and/or Miri; a plain value assertion
  does not detect double-free.
- For allocation bugs, use a counting allocator and assert deltas; a single
  `Vec::capacity` check is not enough.
- For range bugs, test all of `range`, `.rev()`, `scan`, `scan_ref`,
  `scan_batch_ref`, `scan_intra_leaf_batch`, `scan_values`, `scan_prefix`, and
  the mixed `next`/`next_back` contract. A keyed path can be correct while the
  values path is not.
- For layer bugs, force promotion with >15 keys under one 8-byte prefix; test
  start positions 0, 15, 39 and a full-key prefix.

### Determinism

- Do not use `HashSet` iteration order to generate keys. Use `BTreeSet` or sort
  before insertion.
- Do not rely on unstable sort preserving input order for equal ikeys.
- Saved proptest failures must replay the same sequence; check the strategy.
- Stress tests should report the round/attempt at which they fail.

### Miri/Loom

- Miri is required for safe-API pointer/ownership claims. It is not a
  substitute for weak-memory litmus tests.
- The current Loom tests model `NodeVersion` and a simplified suffix word, not
  the real leaf or `SuffixBag`. Do not claim Loom coverage of a protocol unless
  the model actually includes it.
- Many concurrency tests are `#[cfg_attr(miri, ignore)]`. If you add one, say
  why and ensure the non-Miri test is still meaningful.

## 8. Known audit landmines

Before touching these areas, read the corresponding `SystemAudit.md` finding:

| Area | Finding |
|---|---|
| `ValuePtr`, `ScanEntry`, `get_with_guard`, Entry outputs | `SAFE-01`, `SAFE-02` |
| `BatchEntry`/`insert_batch_entries` | `SAFE-03`, `BATCH-01`, `BATCH-03` |
| `TreeAllocator`, `debug_root_ptr`, exported leaf/internode | `SAFE-04` |
| `CAN_WRITE_THROUGH` gate | `SAFE-05` |
| external `SuffixBag` | `SAFE-06` |
| `assign_slot_generic`, `insert_new_value`, batch insert | `OCC-01` |
| `NodeVersion::has_changed`, layer conversion | `OCC-02` |
| `try_cascade_internodes`, `reach` | `GC-03`, `OCC-04` |
| `coalesce.rs` raw chain entries | `GC-01` |
| `try_remove_sublayer` | `GC-02` |
| `remove.rs` post-lock verification | `REM-01` |
| `remove.rs` B-link/advance | `REM-02` |
| `calculate_split_point` | `INS-01`, `INS-02` |
| `advance_to_key_by_bound_generic` rightmost delete | `INS-03` |
| `range/helper.rs`, `forward_ctx`, `reverse_ctx`, `traversal.rs` | `RNG-01` … `RNG-09` |
| `node_pool.rs` TLS deallocation | `ALLOC-01` |
| reclaimer value `Drop` | `ALLOC-02` |
| `ShardedCounter::load` | `PRIM-01` |
| `Permuter<WIDTH<15>` | `PRIM-02` |
| `print_tree` feature gate | `TOOL-01` |
| test tracing/logging | `TOOL-06` |

Do not “fix” a failing test in `failures/` by weakening its assertion. It is
supposed to fail until the underlying bug is fixed. If the bug is fixed, flip
the reproducer to a passing regression and update `SystemAudit.md`.

## 9. Definition of done

A change is ready only when:

- [ ] The failure is reproduced before the fix (or a deterministic argument is
      documented for a `Likely` finding).
- [ ] A focused regression test exists and fails without the fix.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] Relevant `cargo test --lib` / integration tests pass.
- [ ] Feature builds relevant to the touched code pass.
- [ ] Miri/Loom is run for ownership/concurrency changes.
- [ ] No new safe-API unsoundness, `unsafe` block without a proof, or
      debug-only precondition is introduced.
- [ ] Public API/docs/semver impact is noted.
- [ ] `SystemAudit.md` and/or `failures/README.md` are updated if the audit
      status changes.

## 10. Useful commands

```bash
# Find a symbol or an unsafe site
rg -n "has_changed|ValuePtr|BatchEntry|calculate_split_point" src
rg -n "unsafe" src

# Run one deterministic failure
cd failures/range && cargo run --quiet --bin nul
cd failures/insert-split && cargo test --test prepend -- --nocapture
cd failures/remove-coalesce && cargo test --release --test wrong_key -- --nocapture

# Miri soundness checks
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_escape
cargo +nightly miri run --bin padded_write_through

# Tooling defects
cd failures/tooling && ./run-all.sh
```

Use a target directory outside the repository when running the failure corpus:

```bash
export CARGO_TARGET_DIR=/tmp/masstree-target
```
