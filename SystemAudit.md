# System Audit — `masstree` 0.9.5

**Repository:** `masstree`  
**Commit audited:** `eb8bee2e546e` (`REFACTOR: Deduplicate insert and batch paths via strategy traits`)  
**Crate:** `masstree` 0.9.5, Rust edition 2024  
**Audit date:** 2026-10-23  
**Host:** Linux x86_64, AMD Ryzen 5 3600 (6C/12T), `rustc`/`cargo` 1.99.1, nightly Miri available  
**Deliverable:** this merged audit plus the runnable corpus in `failures/`

**Follow-up sweep:** 2026-11-08, source at `93e65eb` with pre-existing dependency
updates in the worktree. Added `OCC-06`, `INS-04`, `GC-04`, `GC-05`, `RNG-10`,
and `TOOL-14`; extended SAFE-05 and corrected SAFE-04. Follow-up locations
refer to that source checkout. See Appendix E for coverage and fresh evidence.

**Contents**

1. [Scope, method, and evidence](#1-scope-method-and-evidence)
2. [Executive summary](#2-executive-summary)
3. [Safety and public API soundness](#3-safety-and-public-api-soundness)
4. [OCC and concurrency protocol](#4-occ-and-concurrency-protocol)
5. [Insert, layer creation, and split](#5-insert-layer-creation-and-split)
6. [Remove, coalesce, and reclamation](#6-remove-coalesce-and-reclamation)
7. [Allocation, internode, and primitives](#7-allocation-internode-and-primitives)
8. [Range scans](#8-range-scans)
9. [Suffix storage](#9-suffix-storage)
10. [Batch and public API](#10-batch-and-public-api)
11. [Tests, tooling, and benchmarks](#11-tests-tooling-and-benchmarks)
12. [Simplification opportunities](#12-simplification-opportunities)
13. [Performance opportunities](#13-performance-opportunities)
14. [Recommended fix order](#14-recommended-fix-order)
15. [Test-gap summary](#15-test-gap-summary)
- [Appendix A — Canonical finding index](#appendix-a--canonical-finding-index)
- [Appendix B — Reproduction quick reference](#appendix-b--reproduction-quick-reference)
- [Appendix C — Environment and limitations](#appendix-c--environment-limitations-and-assumptions)
- [Appendix D — Files reviewed](#appendix-d--files-reviewed)
- [Appendix E — Follow-up source sweep](#appendix-e--follow-up-source-sweep)

---

## 1. Scope, method, and evidence

This audit covers every subsystem in the crate:

1. primitives (`Key`, `NodeVersion`, `Permuter`, `Linker`, prefetch, sharded counter);
2. leaf layout, value policies, split/link helpers;
3. suffix storage and migration;
4. internodes, allocation, reclamation, teardown;
5. insert, layer creation, split propagation;
6. remove, coalescing, GC;
7. range scans, prefix scans, values-only scans;
8. batch operations, entry API, public surface;
9. tests, benchmarks, and tooling;
10. a concurrency/`unsafe` cross-cut.

The audit used ten independent subsystem passes. Each pass read the architecture
baseline, walked the assigned source and tests, and wrote a separate report. The
reports were then reconciled against an independent differential harness and
Miri. Duplicate findings were merged under the canonical IDs used here; the
original report IDs are retained in each finding and in Appendix A.

### 1.1 Evidence labels

- **Confirmed** — reproduced by a deterministic test, a public-API stress test,
  a version/allocation counter, or Miri. The repro is in `failures/`.
- **Likely** — the full interleaving is mechanically established, but the
  schedule or weak-memory hardware needed to observe it was not available.
- **Speculative** — a contract-level or defensive risk that requires a caller
  outside the current tree, or a future change, to become a live bug.

A finding labelled Confirmed may still be *rare*; a finding labelled Likely may
still be a serious release blocker. The label describes evidence, not severity.

### 1.2 What was run

```text
cargo test --lib                                     610 passed, 0 failed
cargo test --test leak_tests                          20 passed
cargo test --test deleted_leaf_handling ...              passed in repeated runs
RUSTFLAGS="--cfg loom" cargo test --lib loom_tests    12 passed
cargo test --doc                                      14 passed, 20 ignored
cargo test --test batch_insert_tests --test true_inline  28 + 6 passed
cargo check [default, no-default, all-features, feature matrix]  pass
cargo check --all-targets --no-default-features        FAILS (TOOL-01)
cargo clippy --all-targets --all-features -- -D warnings FAILS (TOOL-04)
```

Miri was used for the focused tests listed in each soundness finding. The full
suite was not run under Miri; the repository's existing Miri exclusions are
themselves a finding (TOOL-08).

### 1.3 Reproducer corpus

The runnable reproducers are in:

```text
failures/
  range/                 forward/reverse/prefix/values bounds
  insert-split/          prepend split, SplitFailed panic, insert livelock
  remove-coalesce/       wrong-key remove, lost remove, duplicate chain entries, GC leak
  batch-api/             BatchEntry aliasing, stale key cache, leaks, mimalloc
  suffix/                suffix allocation probes
  primitives/            version-word, layer race, sharded counter, Permuter
  internode-alloc/       TLS teardown abort, panic-in-reclaimer, cascade probes
  concurrency/           OCC probes, remove race, Miri lifetime/race tests
  leaf-values/           public Miri tests + patch-ready private unit tests
  independent/           the independent differential/Miri harness
  tooling/               scripts that assert the tooling defects
```

Start with `failures/README.md`. Every confirmed finding below has a command.
The intentionally failing tests are the reproduction; success is a test that
**fails**, hangs, aborts, or produces a Miri `Undefined Behavior` report.

---

## 2. Executive summary

### 2.1 Verdict

This implementation is **not production-safe for concurrent use**, and it is
not safe for a downstream user of the `MassTree15`/`BoxPolicy` API even without
concurrency. The sequential single-threaded core is substantially healthier:
ordinary insert/get/remove/`len` bookkeeping, split arithmetic, internode
routing, batch counters, and teardown of reachable promoted layers passed the
independent checks. The failures cluster in six systemic areas:

1. **The OCC publication protocol is broken.** New-key inserts can leave the
   version word unchanged, and `has_changed` deliberately ignores `INSERTING`.
   A reader can validate during a layer conversion and return a missing value
   or a layer pointer dressed as a value. This is both a linearizability bug
   and, with the public pointer outputs, a type-confusion/soundness bug.
2. **Guard-scoped outputs are not lifetime-bound.** `ValuePtr`, `ScanEntry`,
   `Entry`, and batch old-values can outlive the guard and tree. Miri confirms
   use-after-free and a write-through data race in safe code.
3. **Remove does not revalidate key identity or follow the B-link.** A
   concurrent same-ikey insert can shift the logical position and make
   `remove(k)` delete a different key; a concurrent split can make
   `remove(k)` return `None` while `k` is still present.
4. **The coalesce/GC ownership model is unsound.** Raw leaf pointers survive
   empty-leaf reuse and duplicate chain entries, causing maintenance hangs and
   cross-call UAF. Promoted multi-leaf sublayers are never reclaimed.
5. **Range bounds are not full-key bounds.** Forward and reverse scans use
   8-byte ikeys only, return keys outside the requested range, and skip the
   start key in promoted layers. Values-only scans ignore bounds altogether.
6. **The public low-level API leaks unsafe ownership.** Safe `TreeAllocator`,
   `InternodeNode` mutators, and `LeafNode15` methods can cause OOB, double
   free, or type confusion. The `mimalloc` feature also commandeers the
   downstream process's global allocator.

### 2.2 Release blockers (fix before any production use)

| Priority | Finding | Why |
|---|---|---|
| P0 | SAFE-01, SAFE-02, SAFE-03 | Safe code can UAF, data-race, double-free, or corrupt another key's value |
| P0 | SAFE-04, SAFE-05 | Safe public API permits OOB/double-free/uninitialized atomic reads |
| P0 | OCC-01, OCC-02 | Silent false misses and type-confused reads; weak-memory publication |
| P0 | REM-01, REM-02 | Removes the wrong key or loses a present key |
| P0 | GC-01 | Maintenance hangs or UAFs after normal reuse/coalesce |
| P0 | ALLOC-01 | Normal thread-local teardown aborts the process |
| P0 | INS-01 | A split makes a live key unreachable; count drifts |
| P0 | RNG-01 … RNG-05 | Scans return out-of-range keys, omit valid keys, or duplicate values |
| P1 | INS-02, INS-03, GC-02, GC-03, ALLOC-02, RNG-07, BATCH-01/02/04 | DoS, leaks, weak-memory/UAF exposure |
| P1 | OCC-06, INS-04, GC-04, GC-05, RNG-10 | Protected-load ordering gap, deterministic post-cleanup/post-panic hangs, lost cleanup retries, invalid callback batches |
| P1 | TOOL-01 … TOOL-05, TOOL-09 | The advertised build/test/benchmark gates cannot be trusted |

### 2.3 What passed

The following are positive results, not assumptions:

- Sequential insert/get/remove/`len` and linearizable single-thread behavior
  passed a 3,000-op randomized `BTreeMap` oracle over short keys, NUL keys,
  suffix keys, and 8/16/24/64/255/256-byte keys, with duplicates and updates.
- Batch bookkeeping (`inserted + updated == total`, `len`, old values) was
  correct in all tested non-panicking batches, including same-ikey layers.
- Internode `split_into` child/key arithmetic, equality-routes-right routing,
  root-bit capture, reparenting, pool size classes, and iterative teardown of
  reachable promoted layers passed source review and targeted tests.
- Normal tree Drop frees reachable promoted layers; the leak is the
  *unreachable after maintenance* case, not ordinary teardown.
- `cargo fmt --check` passes; the library and tests compile across the tested
  feature matrix; no split/coalesce lock-order cycle was found.
- No deterministic single-threaded failure was found in the basic
  `MassTree`, `MassTree15`, or `MassTree15Inline` configuration outside the
  specific bugs below.

### 2.4 How to read a finding

Each finding uses this shape:

- **Severity / confidence**
- **Location** — paths and line numbers at the audited commit
- **Mechanism** — the exact state or interleaving
- **Evidence / repro** — command and observed result
- **Impact**
- **Fix**
- **Test gap**
- **Legacy IDs** — the original subsystem-report IDs

The line numbers are intentionally specific. They refer to `eb8bee2` and may
move after a refactor.

---

### 2.5 Remediation status (2026-11-08)

Every P0/P1 finding in this audit has been fixed in the worktree. The table
below records the fix commit and the verification that was run. The original
findings are kept unchanged below as the historical audit record; where a
location or mechanism has changed, the fix commit and `ARCHITECTURE.md` are
authoritative.

Verification used for the series:

```text
cargo test --lib                                      613 passed
cargo test --test tree_proptests                       51 passed
cargo test --test range_scan_tests                     25 passed, 1 ignored
cargo test --test batch_insert_tests                   28 passed
cargo test --test cpp_tests                            all C++-shaped tests
cargo clippy --all-targets --all-features -- -D warnings  clean
cargo check --all-targets --no-default-features         clean
cargo check --all-targets --all-features                clean
cargo check --features sidecar-suffix                   clean
cargo check --features small-suffix-capacity            clean
Miri (valueptr_escape/valueptr_race, leak_tests)        clean
range/property/batch reproducers                        pass
```

| Finding | Resolution | Commit |
| --- | --- | --- |
| SAFE-01/02/03 | `ValuePtr` opaque; public reads return guard-scoped `ValueRef`; batch entries own `P::Value` | `deabbcd` |
| SAFE-04 | Ownership-bearing node/allocation/leaf mutators are `unsafe`, with documented contracts; layer-gated value reads | `e94308d`, `bcff95c`, `a1bc057` |
| SAFE-05 | Write-through restricted to a closed primitive-type list | `0af684c` |
| SAFE-06 | `SuffixBag` metadata/used length atomic; `as_writer`, no `&mut` aliasing | `d6e0a07` |
| OCC-01 | Every new-slot insert marks the version dirty and publishes with Release | `989246e` |
| OCC-02 | `has_changed` no longer ignores the insert bit; counters are authoritative | `24cfe33` |
| OCC-03 | Store-store ordering before dirty publication; Release store-store fence; Acquire validation | `f11f583`, `cfa1188` |
| OCC-04 | Descent restarts when a child internode is deleted | `81ea0fc` |
| OCC-05 | Ancestor separator redirects go through `mark_insert` | `7302514` |
| OCC-06 | Guard-protected root/parent/child loads; SeqCst value/suffix pointer loads | `947681b` |
| INS-01/02 | Split only on strict ikey boundaries; post-insert boundary and equal-ikey side choice; split path never fails | `de19063`, `9ffacd5`, `8c4021c` |
| INS-03 | Insert restarts when it reaches a deleted rightmost leaf | `14a4da4` |
| INS-04 | User value encoding happens before every structural insert mutation (split, empty-leaf reuse, layer creation) | `c686940`, `9147069` |
| REM-01/02/03 | Full-key revalidation under the lock; B-link follow; retry budget charged only on contention; promoted-root climb | `8ea351d`, `7339d70` |
| GC-01 | Chain entries carry only `ikey_bound`; duplicate entries are consumed safely | `0265567` |
| GC-02 | Promoted/multi-leaf sublayers reclaimed as a validated whole subtree | `e351a36` |
| GC-03/04 | Unlink before retire; preserve live unary levels; retry parent acquisition | `1d6ef92` |
| GC-05 | One deferred attempt per maintenance call | `d5b081d` |
| ALLOC-01 | TLS-destruction fallback to the global allocator | `f2d8175` |
| ALLOC-02 | User `Drop` panics contained during reclamation | `29e84fd` |
| ALLOC-03 | `TreeAllocator` unsafe contract requires abort, never unwind | `b70ebe1` |
| PRIM-01 | `ShardedCounter::load` no longer asserts on a valid snapshot | `40c70af` |
| PRIM-02 | `Permuter` results masked to the encoded width | `7a439fe` |
| RNG-01..10 | Full-key bounds, promoted-layer root climb, stable B-link scan, reverse/values/batch bounds, fused DoubleEnded; values-only interior fast path | `ee3364a`, `e3eed9f` |
| SUF-01 | Replaced slot's inline suffix cleared after a drain | `abb3f5d` |
| SUF-02 | Zero-length suffixes skipped during compaction | `cfaf419` |
| SUF-03 | Public suffix length/bounds checks unconditional | `3047f97` |
| SUF-04 | `assign_init` requires exclusive ownership; layer init uses `assign_ksuf_init` | `f5ce6b3` |
| SUF-05 | Saturating headroom subtraction | `f99f4a6` |
| BATCH-01..06 | Private owning entries, deterministic sort, guard-scoped old values, no global allocator | `5a5a8e2`, `54df313` |
| TOOL-01 | `print_tree` example feature-gated; all targets build with `--no-default-features` | `b70ebe1` |
| TOOL-02/03/05/07/11/12/13/14 | Shuttle removed; Miri leak target fixed; benchmark paths/redirection fixed; tracing honest; release-visible tests; CI; promoted-layer printing | `c82e9dc` |
| TOOL-04 | Strict Clippy gate passes | `9c6e8d1` |
| TOOL-06/10 | Test logs out of the repo; deterministic proptest order and complete scan assertions | `a13ed47` |
| TOOL-08 | Miri exclusions enumerated in `ARCHITECTURE.md` §11.4 | this docs commit |
| TOOL-09 | Stress/parity tests assert values and final state | `a3be3e2`, `4694467` |

A follow-up review pass then closed the remaining correctness and performance
gaps in the first remediation:

- the split point is now chosen against the **post-insert** key sequence, and
  an equal-ikey boundary insert is routed left so the separator is strictly
  greater than every left key (`9ffacd5`, `8c4021c`; the strengthened
  `failures/insert-split/tests/equal_run2.rs` checks every key, and the
  200-seed `insert-split` oracle passes);
- every structural insert path (`assign_slot_generic`, empty-leaf reuse,
  split, layer creation) stores a pre-encoded `PreparedValue`, so a panicking
  `InlineBits::to_bits` cannot leave a partially published leaf or leak twigs
  (`9147069`);
- dirty publication uses a Release store-store fence instead of `SeqCst`
  (`cfa1188`), which removes a serializing barrier from every write;
- values-only scans materialize a full key only at the bound's 8-byte ikey,
  keeping the interior/unbounded path allocation-free (`e3eed9f`);
- remove normalizes promoted layer roots, re-checks too-far-right after every
  B-link advance, and charges its retry budget only to actual contention
  (`7339d70`); scan traversal re-climbs the true layer root on every restart.

A second review pass (recorded in full in `Findings.md`) closed the residual
issues that survived the first wave:

- the split boundary and side choice are computed against the **post-insert**
  sequence (`9ffacd5`, `8c4021c`), and every structural insert path pre-encodes
  its value (`9147069`);
- the scan cursor carries an explicit unshift sentinel instead of overloading
  length 9, excluded bounds descend into equal layer slots, and batch scans
  validate before committing cursor/stack state (`e7e42f4`, `0d45a5f`);
- register/print and remove fast paths were tightened (`f3e957a`, `e6c8bd9`,
  `d34a5fd`), ancestor redirects retry instead of abandoning a null child, and
  both coalesce deferred queues are drained once per call (`68c4ea9`);
- `SuffixBag` exclusive mutation is `&mut self` and the concurrent append path
  writes through raw pointers without aliasing reader slices (`0c1a71e`);
- write-through is used by the Entry and batch-entry APIs, full-leaf batches
  still apply in-place updates, suffix preallocation is deferred until a split
  is known, and maintenance flushes after coalescing (`4960f8d`, `da74a75`).

The simplification opportunities (SIM-01..16) are **not correctness defects**.
The branch addresses the ones that carry the wrong ownership/contract signal:
`InsertError` was collapsed to `SplitFailed` (SIM-01), unused dev-dependencies
were removed (SIM-15), tracing is real (SIM-16), and the sentinel tests are
wired (SIM-12). The remaining SIM entries are larger structural refactors that
do not change observable behavior; they are specified in
`ArchitectureProposals.md` rather than release blockers.

## 3. Safety and public API soundness

### SAFE-01 — Guard-scoped outputs can outlive the guard and tree (safe-code UAF)

**Severity:** Critical · **Confidence:** Confirmed (Miri)  
**Legacy IDs:** `BUG-API-02a`, `BUG-LEAF-05` (UAF), `BUG-CON-05`, `BUG-RNG-01`

**Location:** `src/policy.rs:48-99` (`ValuePtr`), `src/tree/range/iterator/scan_entry.rs:19-29`,
`src/tree/generic/optimistic_reads/get_guarded.rs:28`, `src/tree/range/iterator.rs:758-763`,
`src/tree/generic/entry.rs:126/226/253/259`, `src/tree/generic/batch.rs:705-728`,
`src/tree/range/api.rs:446-457`.

`ValuePtr<V>` is a `Copy + Send + Sync` `NonNull<V>` with no lifetime parameter
and a safe `Deref` (`src/policy.rs:83-91`). `get_with_guard` returns
`Option<P::Output>`; for `MassTree15` that is `Option<ValuePtr<V>>`. The same
pattern is used by `ScanEntry<P::Output>`, range iterators, entry outputs,
`remove_with_guard`, and batch old-values. The docs say the value is valid
only while the guard is held, but the type system enforces nothing.

**Repro:**

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_escape
```

Observed Miri error:

```text
Undefined Behavior: constructing invalid value of type &String:
encountered a dangling reference (use-after-free)
  --> src/policy.rs:89
```

The same escape works through a range iterator:

```rust
let e = {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"key", "a-long-string-value-to-allocate".to_string());
    let guard = tree.guard();
    tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard)
        .next().unwrap()
}; // guard and tree dropped
println!("{:?}", &*e.value); // safe deref of a freed value
```

`failures/range/src/bin/uaf.rs` contains the scan variant; Miri reports a
segfault/use-after-free.

**Impact:** Any downstream user of `MassTree15` can retain a `ValuePtr` or
`ScanEntry` past the guard, send it to another thread, and dereference it in
safe code. The documented lifetime precondition is not enforceable by callers
who never use `unsafe`. This is the most direct soundness hole in the crate.

**Fix:** Make all guard-scoped outputs lifetime-bearing:

- `ValuePtr<'g, V>` with `PhantomData<&'g V>` (and a `!Send` marker if the
  probe may not cross threads while protected), or
- return owned `P::Value`/`ValueRef<'g, V>` from these APIs; `get_ref` already
  uses the sound `ValueRef` model (`src/policy.rs:151-175`).

`ScanEntry` must be `ScanEntry<'g, O>`. Iterator items cannot borrow `self`
directly, so either clone into `P::Value` for owning iterators (as
`first()`/`last()` do) or use a callback/lending iterator for BoxPolicy.
`get()` can retain the owned-clone behavior. This is an API break and should
be treated as a major-version change.

**Test gap:** No compile-fail test for an escaped `ValuePtr`; no Miri test for
`get_with_guard`, `remove_with_guard`, `ScanEntry`, `collect_entries`, or
`BatchInsertResult` old-values. The `failures/leaf-values/public` tests are
the minimal candidates for CI.

---

### SAFE-02 — `ValuePtr::Deref` races write-through atomic updates

**Severity:** Critical · **Confidence:** Confirmed (Miri)  
**Legacy IDs:** `BUG-API-02b`, `BUG-LEAF-05` (race), `BUG-CON-06`

**Location:** `src/policy.rs:83-99` (`Deref`, `Eq`, `Ord`, `Hash`),
`src/policy/box_values.rs:25-57,132-152` (atomic load/store), `src/tree/generic/insert.rs:161`
(write-through update), `src/tree/generic/optimistic_reads/get_guarded.rs:28`.

For write-through values (`BoxPolicy<V>`, `CAN_WRITE_THROUGH`), an update
mutates the Box in place through `AtomicU64`/`AtomicU32` loads and stores.
`ValuePtr::Deref` returns a plain `&V`, and `PartialEq`, `Ord`, `Hash`, and
`Debug` also read it non-atomically. A caller holding the guard can therefore
dereference the pointer while another thread performs a valid write-through
update. This is UB even if the machine-sized read does not tear.

**Repro:**

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_race
```

Observed Miri error:

```text
Data race detected between (1) non-atomic read ... and (2) atomic store
  --> src/policy/box_values.rs:52
```

The public repro keeps the guard alive and uses a scoped writer; no lifetime
escape is required.

**Impact:** Safe-code data race on the recommended `MassTree15<u64>`
configuration. The architecture's claim that a concurrent write-through update
cannot produce a torn read (`ARCHITECTURE.md:664`) is false for `ValuePtr`;
`get_ref`/`output_as_ref_sound` use `atomic_read_value`, but the output APIs do
not.

**Fix:** Do not expose a plain `&V` for `CAN_WRITE_THROUGH`. Return
`ValueRef::Owned` (atomic copy), an `AtomicValue<V>` with `load()`, or an owned
`V` from `get_with_guard`/entry/scan APIs. If the API cannot change, make
`Deref`/comparison/hash `unsafe` for the write-through case and document the
exact synchronization contract. `Debug`/`Display` for `BoxPolicy` values must
also use the atomic path (`print_tree` inherits this race).

**Test gap:** No write-through test in `src/policy/tests.rs`; no Miri test
covering `get_with_guard` plus `insert`; `failures/leaf-values/public` has the
minimal regression.

---

### SAFE-03 — `BatchEntry` can alias a tree-owned value: double-free, UAF, and cross-key corruption

**Severity:** Critical · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-API-01`

**Location:** `src/tree/generic/batch.rs:61-90` (`pub key`, `pub output`,
`from_output`), `src/tree/generic/batch.rs:773-787`
(`insert_batch_entries`), `src/policy.rs:776-780` (`BoxPolicy::into_output`),
`src/tree/remove.rs:233-237` (retire on remove).

`BatchEntry` exposes `output` and `key` publicly, while its `ikey` is a private
cache. `from_output` accepts any `P::Output`. For `BoxPolicy`, `P::Output` is a
non-owning `ValuePtr<V>`. Wrapping a pointer returned by `get_with_guard` in a
new `BatchEntry` and inserting it under another key stores the same Box in two
slots. Updating one key mutates the other; removing both retires the same
allocation twice.

**Repro:**

```bash
cd failures/batch-api
cargo run --bin alias       # cross-key corruption and double-free at exit
cargo run --bin doublefree  # glibc double-free abort
```

Observed in the alias sequence:

```text
get(a)=2, get(b)=2          # one update changed both keys
free(): double free detected in tcache 2
```

With a `Drop`-counting value, the doublefree binary prints two drops for one
address and aborts.

**Impact:** Safe code can cause use-after-free, double-free, and silent
cross-key value corruption through a public API. This is independent of the
write-through race: even a non-write-through `BoxPolicy` tree is affected,
because `ValuePtr` has no ownership and `remove` retires whatever the slot
holds.

**Fix:** `BatchEntry` must own a real user value:

- make `key`, `output`, and `ikey` private; add `key()`/`into_parts()`;
- remove `from_output` (or make it `unsafe` with an explicit unique-ownership
  contract, which cannot currently be checked);
- store `Option<P::Value>` (as `BatchValueEntry` already does), convert with
  `P::into_output` only at the insertion point, and implement `Drop` normally.

Also stop exporting tree-owned `P::Output` from any API that can feed
`insert_batch_entries`.

**Test gap:** No test constructs a `BatchEntry` from `get_with_guard`, no
Drop-counter/Miri test for double retire, and no compile-fail test if
`from_output` remains. Add the alias/doublefree tests to CI under Miri.

---

### SAFE-04 — Safe public allocator/leaf/internode APIs permit double-free and type confusion

**Severity:** High · **Confidence:** Confirmed (compile/Miri/source)  
**Legacy IDs:** `BUG-API-07`, `BUG-ALLOC-06`, `BUG-CON-13`, `BUG-LEAF-08`

**Location:**

- `src/alloc_trait.rs:24-71` — public safe `TreeAllocator`; methods return raw
  nodes and `teardown_tree` is safe.
- `src/tree/generic.rs:216-218` — safe public `debug_root_ptr`.
- `src/alloc15.rs:160-162` — safe `SeizeAllocator::teardown_tree`.
- `src/leaf15.rs:470-477` — safe `move_value_to`.
- `src/leaf15.rs:257-297` — safe `load_value`/`take_value` without keylenx
  gating.
- `src/internode.rs:224-246,287-380` — safe internode mutators/split with
  debug-only bounds.
- `src/internode/accessors.rs:77-103,140-143,237-240` — safe `set_ikey`/
  `set_child` indexed by a caller-supplied slot.
- `src/lib.rs:201-208` — exports `InternodeNode`, `LeafNode15`, `SuffixBag`.

**Repro A — safe allocator teardown UAF/double-free:**

```rust
let tree: MassTree15<u64> = MassTree15::new();
tree.insert(b"k", 1);
let g = tree.guard();
let root = tree.debug_root_ptr(&g);
SeizeAllocator::<BoxPolicy<u64>>::new().teardown_tree(root);
let _ = tree.get(b"k");          // UAF
// tree Drop calls teardown_tree again -> double free
```

The exact program compiles in safe Rust; `failures/batch-api/src/bin/teardown_safe.rs`
is the runnable form (it aborts/corrupts memory). `TreeAllocator` is also safe
to implement: a safe implementation can return null/dangling leaves, and the
tree dereferences them immediately (`generic.rs:330`).

**Repro B — safe `LeafNode15::move_value_to` double-free:**

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin move_value_same_leaf
```

Miri: `constructing invalid value of type Box<u64>: encountered a dangling box
(use-after-free)` during `Drop`. `move_slot` copies the pointer but does not
clear the source or retire an existing destination value.

**Correction from the follow-up — indexed internode access panics in release:**

```rust
let node = InternodeNode::new(0);
node.set_ikey(15, 0);            // bounds-check panic, including in release
node.set_child(16, std::ptr::null_mut()); // bounds-check panic
```

The original audit incorrectly inferred unchecked writes from `debug_assert!`.
These methods use ordinary Rust array indexing, which retains bounds checks.
The release control in `failures/leaf-values/public/tests/internode_bounds.rs`
catches all three panics (`set_ikey`, `set_child`, and insertion into a full
node). Those calls are not evidence of an out-of-bounds memory write.

The internode soundness concern remains: safe `split_into` reparents children
through raw-pointer dereferences when `height > 0` (`src/internode.rs:360-375`).
Safe `set_child` accepts arbitrary pointers, and `split_into` does not enforce
their validity or ownership. Invalid positions also cause arithmetic/bounds
panics; the subsequent checked indexing must not be described as unchecked UB.

**Impact:** Safe downstream code can cause invalid pointer dereferences, double-free, UAF, and
type confusion. The external allocator trait is the cleanest route because
`with_allocator` trusts any safe implementation. This is an API soundness
problem, not a call-site bug.

**Fix:**

- Make `TreeAllocator` an `unsafe trait` (or seal it), add `# Safety` contracts
  to allocation/retirement/teardown, and make `teardown_tree` `unsafe`.
- Make `debug_root_ptr` `unsafe`, `#[doc(hidden)]`, or remove it.
- Make pointer-dereferencing internode operations `pub(crate)`/`unsafe` and
  enforce child validity/ownership; validate `insert_pos <= WIDTH` in `split_into`.
  Keep the existing array bounds checks.
- Make `move_value_to` a move/swap that nulls the source and retires the
  destination, or make it `unsafe` and stop exporting `LeafNode15`.
- Gate `load_value`/`take_value` on `keylenx < LAYER_KEYLENX`, or require a
  typed slot token.

**Test gap:** No Miri/compile-fail test for a malicious allocator, no test for
`move_value_to` ownership, no test for layer-slot `load_value`/`take_value`,
The follow-up adds a passing release bounds-check control; a safe-code Miri
test for `split_into` with invalid children is still needed.

---

### SAFE-05 — `CAN_WRITE_THROUGH` admits padding, ownership, and pointer representations that integer atomics cannot preserve

**Severity:** High · **Confidence:** Confirmed (Miri)  
**Legacy IDs:** `BUG-LEAF-06`

**Location:** `src/policy.rs:729-732` (`CAN_WRITE_THROUGH` gate),
`src/policy/box_values.rs:25-57` (`atomic_load_dispatch`/`atomic_store_dispatch`),
`src/policy/box_values.rs:132-149` (write-through update).

The gate is:

```rust
!needs_drop::<V>()
    && size_of::<V>() <= 8
    && size_of::<V>().is_power_of_two()
    && align_of::<V>() >= size_of::<V>()
```

That does not imply every byte of `V` is initialized or that `V: Copy`. A
`#[repr(align(8))] struct Padded { a: u32 }` has size 8, align 8, no drop, and
four padding bytes; `Box::new(Padded { a: 7 })` leaves the padding
uninitialized. `clone_value_from_output` then loads the whole 8-byte word as
`AtomicU64`.

**Repro:**

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin padded_write_through
```

Miri:

```text
Undefined Behavior: reading memory ... memory is uninitialized at [0x4..0x8]
  --> src/policy/box_values.rs:34
```

The `&mut` case is also admitted: `!needs_drop::<&'static mut T>()` is true and
the bitwise duplication in `atomic_read_value` can create aliasing mutable
references in safe code. No separate Miri run was done for that variant, but it
follows from the same transmute.

**Follow-up repro — even a `Copy`, padding-free shared reference fails:**

```bash
cd failures/leaf-values/public
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly miri run --bin reference_write_through
```

`MassTree15<&'static u64>` admits write-through. After inserting a reference
to a static, `get_ref` loads its representation through `AtomicU64` and
transmutes the integer into a reference. Miri reports at
`src/policy/box_values.rs:34`: `encountered a dangling reference (... has no
provenance)`. This is a single-threaded read with the guard alive. Requiring
only `Copy` and initialized bytes would therefore still be insufficient;
pointer-bearing representations must be excluded or handled with a
provenance-preserving typed atomic representation.

**Impact:** UB when the integer-atomic read encounters padding or reconstructs
an invalid reference representation; ordinary native runs can hide it.
Common types such as `[u8; 8]`, `(u32, u8, u8,
u8)`, and `#[repr(C)] { a: u32, b: u32 }` are rejected, so the gate both admits
unsafe types and misses useful safe ones.

**Fix:** Replace the arithmetic gate with a sealed, explicit marker:

- require `V: Copy` and restrict to primitive numeric/bool/char types whose
  representation has no padding, or
- require a user `unsafe impl AtomicWord` with no-padding, no-ownership, and
  integer-representation/provenance contracts. Zeroing an allocation alone
  does not make arbitrary typed values suitable for integer atomics.

**Test gap:** No write-through tests at all in `src/policy/tests.rs`; add the
padded struct and the `&mut` case under Miri.

---

### SAFE-06 — External `SuffixBag` has non-atomic metadata/Vec races and `&mut` aliasing

**Severity:** High · **Confidence:** Confirmed (language-level UB; practical
visible corruption is unlikely on current paths)  
**Legacy IDs:** `BUG-SUF-01`, `BUG-CON-12`

**Location:** `src/suffix.rs:32-58` (`SlotMeta`),
`src/suffix.rs:273-288` (`get`), `src/suffix.rs:380-406` (append),
`src/suffix.rs:459-467` (clear), `src/policy/suffix_ops.rs:118-121`
(`as_mut` while readers hold `as_ref`), `src/suffix.rs:543-560`
(`SuffixBagCell::as_ref`/`as_mut`), `src/nodeversion.rs:443-445`
(`has_changed` can validate inside the writer's lock).

`SlotMeta` (`offset: u32`, `len: u16`) and `Vec::len` are plain fields. A
lock-free reader can load `slots[s]` and `data.len()` while a writer under the
leaf lock executes `data.extend_from_slice(...)` and then stores
`slots[s] = ...` with a plain store. `clear(s)` similarly writes
`SlotMeta::EMPTY`. `SuffixBagCell::as_mut` creates `&mut SuffixBag` while a
reader may hold `&SuffixBag`, which is a Stacked/Tree Borrows aliasing
violation independent of the torn value.

The easiest interleaving is remove: reader takes a stable version and the old
permutation, writer locks and clears the value/suffix, reader loads the plain
metadata, then `has_changed` computes `(old ^ current) == 3` and returns false
because it requires `> LOCK_BIT | INSERTING_BIT` (see OCC-02). The bounds check
usually prevents an out-of-bounds slice, but the read is still a data race and
can select the wrong in-buffer region.

**Impact:** Rust data-race UB in normal long-key workloads when the external
bag overflows inline capacity. Current tree paths make a wrong logical suffix
match unlikely because the value is already removed, but the code undermines
the crate's “OCC makes lock-free reads sound” claim and will be flagged by
Miri/TSan. `BUG-SUF-02` describes a latent path that could turn it into a
visible wrong-key result.

**Fix:** Make reader-visible metadata atomic:

- pack `offset`/`len` into one `AtomicU64` (Acquire load, Release store), or
  use `AtomicU32 offset` + `AtomicU16 len`;
- keep an `AtomicUsize used` updated after bytes are copied, and build the
  slice with `from_raw_parts` rather than indexing `Vec` (which reads
  `Vec::len` again);
- never reallocate a published bag; keep the capacity precheck;
- replace `&mut` creation with field writes through raw pointers or an
  `UnsafeCell` writer API.

**Test gap:** The existing Loom model (`src/inline/loom_tests.rs`) models only
an `AtomicU32` array, not the real `SuffixBag`, byte buffer, `Vec::len`, or
`SuffixBagCell`. Add a real-bag Miri/TSan test with a drain/clear concurrent
with `get`.

---

## 4. OCC and concurrency protocol

### OCC-01 — New-key inserts do not mark the version dirty or bump `vinsert`

**Severity:** Critical · **Confidence:** Confirmed (version probe); weak-memory
consequence Likely  
**Legacy IDs:** `BUG-INS-05`, `BUG-LEAF-01`, `BUG-CON-02`, `BUG-X-01`

**Location:** `src/tree/generic.rs:844-847` (`assign_slot_generic`),
`src/tree/generic/insert.rs:196-199` (new permutation, Relaxed), `:267-268`
(empty-leaf reuse), `src/tree/generic/batch.rs:1029-1035` (batch empty-leaf),
`src/nodeversion.rs:117-127` (unlock adds nothing if `INSERTING` was never set).

`assign_slot_generic` only calls `lock.mark_insert()` when
`leaf.modstate_relaxed() != MODSTATE_INSERT`. `MODSTATE_INSERT` is the normal
state, not a dirty marker. A normal/fresh leaf therefore takes the new-slot
path with no `INSERTING` bit; `LockGuard::drop` releases the same version word,
and `vinsert` does not change. The permutation store is `Relaxed`.

**Repro:**

```bash
cd failures/remove-coalesce
cargo test --release --test version_probe -- --nocapture
# or
cd failures/concurrency/vercheck
cargo run --quiet
```

Observed:

```text
v0=0xc0000000 p0=...   (before insert)
v1=0xc0000000 p1=...   (after first new key)
v2=0xc0000000 p2=...   (after second new key)
v3=0xc0000008 p3=...   (after update)
```

The version word is identical across three new-key inserts while the
permutation changes.

**Impact:** A reader that snapshotted the pre-insert version cannot detect the
new permutation. On x86's TSO the permutation store is last, so the practical
window is smaller, but the code is not a valid OCC publication: all slot stores
and the permutation are `Relaxed`, and the unlock has no distinct value for an
acquire to synchronize with. On aarch64 a reader can see the new permutation
before the slot metadata, including a stale `keylenx >= LAYER` or a stale
inline payload. It also removes the only version evidence a pre-lock remove
could use (REM-01).

Git evidence: commit `c7781c9` (“Micro optimizations”) replaced an
unconditional `lock.mark_insert()` with the modstate condition.

**Fix:** Call `lock.mark_insert()` unconditionally before the first slot or
permutation mutation, including `insert_new_value`, `insert_into_empty_leaf`,
and batch empty-leaf insertion. Publish a published permutation with
`set_permutation` (Release) or an explicit Release fence; keep Relaxed only for
an unpublished right sibling. A combined `lock_for_insert()` CAS from clean to
`LOCK|INSERTING` avoids a second atomic store. Add a version-change assertion
test for every insert path.

**Test gap:** No test asserts that the version word changes after a new-key
insert; no Loom model includes permutation publication. The current Loom tests
model `NodeVersion` in isolation and encode the old semantics.

---

### OCC-02 — `has_changed` ignores `INSERTING`; layer conversion can return a missing value or a layer pointer as a value

**Severity:** Critical · **Confidence:** Confirmed (x86 public stress and
crate-internal deterministic test)  
**Legacy IDs:** `BUG-PRI-01`, `BUG-LEAF-04`, `BUG-CON-01`, `BUG-CON-09`

**Location:** `src/nodeversion.rs:443-445` (`has_changed`),
`src/tree/generic/insert.rs:364-392` (`handle_suffix_conflict`),
`src/tree/generic/optimistic_reads/get_guarded.rs:267-284`,
`src/tree/generic/optimistic_reads.rs:749-775`,
`src/tree/remove.rs:682-685` (null layer descent), `src/tree/generic/insert.rs:373-391`.

`has_changed` compares `(old ^ current) > (LOCK_BIT | INSERTING_BIT)`. A
writer that holds the lock with only `INSERTING` set is invisible: the XOR is
`LOCK|INSERTING` (3), which is not `> 3`. Layer creation does exactly that:
it marks insert, builds a sublayer, takes the old value (`take_value_for_layer`
nulls it), clears the suffix, sets `keylenx = LAYER`, and only then stores the
layer pointer. A reader that searched before the clear can validate during that
window and return `None` for a key that exists before and after. A reader that
sees the new `keylenx` with the old value/tag can return the layer pointer as
`P::Output` (type confusion); for `InlinePolicy`, a stale sentinel is
dereferenced as a `NodeVersion`.

**Repro A — public `get_ref` misses:**

```bash
cd failures/concurrency/layer_race
cargo run --release
```

Observed in a 30k-cycle run: `stable_none=73` transient misses for a key that
was present. A scratch copy with only `has_changed` changed to `> LOCK_BIT`
reported `stable_none=0`.

**Repro B — type-confused value:**

```bash
cd failures/concurrency/typeconf
cargo run --release
```

Observed: `bad=28` reads returned `Some` but `v.as_str() != "A"` (printed
`""`) while the tree held a `String`. The returned pointer was a layer node.

**Repro C — deterministic crate-internal test:**

```bash
# apply failures/leaf-values/private-tests/leaf15_unit_tests.add.rs
cargo test --lib audit_value_cleared_under_inserting -- --nocapture
```

It asserts `is_value_empty(slot) == true` and `has_changed(old) == false` after
`lock.mark_insert()` + `take_value_for_layer`. That is exactly the invalid
snapshot.

**Impact:** Non-linearizable point reads, false `contains_key`, false
`Entry::or_insert` decisions (overwrites an existing value), and a
type-confused `&V`/`ValuePtr` on weak memory. Range scans that use the same
validator inherit the miss. Remove's layer descent can additionally call
`is_valid_sublayer(null)` (`remove.rs:682-685`), a null dereference, if it
observes `keylenx = LAYER` before the pointer store.

**Fix (primary):** Match the C++ reference: ignore only `LOCK_BIT`, i.e.
`(old ^ current) > LOCK_BIT`, so an in-flight structural write forces a retry.
Update the Loom model and all docs. Alternatively, add a dedicated structural
dirty bit (bit 28 is unused and cleared by both unlock masks) for operations
that reinterpret an existing slot; ordinary value updates/new-slot inserts can
use the cheaper path after OCC-01 is fixed.

Also add the missing negative-return dirty retry in every multi-layer read
path, and make `is_valid_sublayer` reject null. This is defense in depth, not a
substitute for the protocol fix.

**Test gap:** No test takes a snapshot, marks `INSERTING`, clears a value, and
asserts `has_changed` is true. No Loom test includes payload mutation. Add
`layer_race`/`typeconf` as stress tests (with a generation oracle) and a
deterministic version test.

---

### OCC-03 — Weak-memory publication and validation fences are ordered the wrong way

**Severity:** High on aarch64/weak-memory targets; no x86 effect · **Confidence:** Likely (fence semantics)  
**Legacy IDs:** `BUG-PRI-02`, `BUG-PRI-03`

**Location:** `src/nodeversion.rs:145-192` (`mark_insert`/`mark_split`/
`mark_deleted`), `src/nodeversion.rs:422-445` (`has_changed`),
`src/leaf15.rs:276-286` (`store_value_relaxed`), `:798-820` (`set_ikey_relaxed`),
`:924-932` (`set_keylenx_relaxed`), `src/tree/generic/insert.rs:199,276`
(permutation Relaxed).

The marker functions do `store(dirty_bit, Release)` followed by
`fence(Acquire)`. A Release store orders *preceding* operations before the
store; it does not prevent *subsequent* slot/permutation stores from becoming
visible first. An Acquire fence after the store is not a store-store barrier.
Separately, `has_changed` uses `load(Acquire)` without an acquire fence before
the final version load. An acquire load orders later operations after the load,
but not prior data reads before validation. A seqlock-style reader needs an
Acquire fence immediately before the final version read.

**Interleaving (aarch64):**

1. Reader snapshots clean version `V`.
2. Writer locks and stores `LOCK|INSERTING`, but the store is not yet visible.
3. Writer clears a value, writes keylenx/layer/permutation.
4. Reader loads the new data, then loads `V` (clean) because the dirty store is
   not visible.
5. `has_changed(V)` is false; the reader accepts a mixture.

The reverse direction (reader sees dirty bit but not data) is caught by
`stable()` and retries.

**Impact:** On weak memory, every `get`/scan validation can accept a mixed
snapshot; combined with OCC-02/SAFE-01 this can expose an invalid layer
pointer or an unpublished Box value. x86 TSO hides the reorderings. This is a
portability/soundness issue, not just stale data.

**Fix:** Replace the post-store `fence(Acquire)` with `fence(Release)` (or use
`store(SeqCst)` + `fence(SeqCst)`) in all three markers, and put
`fence(Acquire)` immediately before the final version load in
`has_changed`/`has_split` on non-TSO targets. Publish live permutations with
Release. Add aarch64 litmus/QEMU CI; Loom does not model these reorderings.

**Test gap:** No aarch64 test or litmus; no CI for non-x86. The current Loom
model uses per-thread program order and cannot catch store-store or load-load
reordering.

---

### OCC-04 — Descent through a deleted/empty internode has no escape

**Severity:** High · **Confidence:** Likely (source-level; no fault-injection repro)  
**Legacy IDs:** `BUG-INS-04` (consumer), `BUG-CON-07`, `BUG-X-04`

**Location:** `src/tree/generic.rs:396-401` (`if child.is_null() { continue
'retry; }` with unchanged `start`), `src/tree/generic.rs:341-442` (no
`is_deleted` check on child internodes), `src/tree/remove.rs:433-461`
(cascade producer).

`reach_leaf_concurrent_generic` checks `is_deleted` only for the climb start,
not for a child internode. `try_cascade_internodes` can leave an empty
internode marked `DELETED` with `child[0] = null` linked in its grandparent on
`RetryExhausted` (see GC-03). Every descent into it hits the null child and
`continue 'retry` on the unchanged root. Before reclamation this is a
permanent livelock for that key range; after EBR reclaims the node it is a UAF.

**Impact:** A rare cascade failure can make a whole subtree unreachable and
spin a core forever. The same path is the consumer half of GC-03's UAF/double
free at teardown.

**Fix:** In the descending loop, load the child version and restart from
`root_ptr` if the child is deleted/dirty or if a zero-key internode has a null
child, with a bounded retry. The producer must not abandon the cascade (GC-03).

**Test gap:** No fault injection for `MAX_PARENT_RETRIES`; no test covers a
deleted internode still linked. Add a test-only retry limiter and a
`with_batch_size(1)` teardown test.

---

### OCC-05 — Ancestor separator updates bypass the version protocol

**Severity:** Low · **Confidence:** Likely (routing-safe, protocol blind spot)  
**Legacy IDs:** `BUG-PRI-02` note, `BUG-CON-07` note, `SIM-CON-02`

**Location:** `src/tree/remove.rs:832-911` (`redirect_ikey_bounds_generic`),
especially `:896-898`.

`redirect_ikey_bounds_generic` locks ancestors and calls `parent.set_ikey(...)`
but never calls `mark_insert`; a pure lock/unlock does not change the version.
The change is a monotone increase of a lower-bound separator for an empty
leftmost leaf, so no live key range changes and null-child/B-link recovery
covers readers, but it violates the stated invariant that every mutation is
published by a version change.

**Fix:** Call `mark_insert` on the parent (or document this as a monotone route
hint that deliberately needs no version bump). Add a test that a reader either
sees the old or new separator, never a torn child/key pair.

**Test gap:** No test observes a concurrent separator redirect with an OCC
reader.

---

### OCC-06 — Reader pointer loads bypass seize's protected-load ordering

**Severity:** High · **Confidence:** Likely (dependency contract/source proof;
no weak-memory execution reproduced)

**Location:** `src/tree/generic.rs:211-217,245-261` (root/parent loads),
`src/tree/range/traversal.rs:21-82` (ignores guard; `child_unguarded`),
`src/policy/box_values.rs:101-115,202-220` (value pointer loads),
`src/policy/sidecar_suffix.rs:37-38`, `src/policy/suffix_ops.rs:32-41`
(sidecar/external bag loads). Contrast `src/leaf15.rs:1057-1059` and
`src/internode/accessors.rs:140-143`, which use `guard.protect`.

Holding a `LocalGuard` does not, by itself, prove that every raw atomic load
observes a still-live object. The pinned dependency `seize` 0.5.1 implements
`Guard::protect` by strengthening the pointer load to `SeqCst`
(`seize/src/guard.rs:64-65`, `raw/collector.rs:75-104`). Its source explicitly
requires these loads to participate in the total order between entering a
guard and the retirement barrier. Many tree readers instead use plain
Acquire or Relaxed loads and then dereference the pointer. The safe guard
argument is sometimes unused altogether.

**Mechanism:** A retiring thread removes/replaces a pointer and executes the
collector's retirement barrier while a reader is inactive. A subsequently
entering reader must observe that removal, since the retired object need not
be reserved for it. The collector's protected-load ordering provides that
guarantee; an unprotected Acquire/Relaxed load has no established equivalent
argument here. A stale pointer can then be dereferenced before node-version
validation. This is distinct from OCC-03's dirty-bit/payload fences and from
GC-03's retirement of a still-published node.

**Evidence/limits:** Confirmed the ordering mismatch against the installed
source for the locked `seize = 0.5.1`; no UAF is claimed as reproduced by this
follow-up. Native x86 execution and Miri success do not establish the missing
weak-memory guarantee. Some loads under a writer lock or from already
protected immutable storage may have independent proofs; those must be
identified individually.

**Fix:** Inventory dereferenced pointer loads, use `Guard::protect` (or its
guarded swap/CAS counterparts) where required, and pass the guard through
value/suffix loaders. Retain raw loads only where a concrete lifetime and
ordering proof replaces the collector's contract. Add a reclamation handoff
model covering inactive → enter versus replace → retire, with the real
collector's barrier strategy represented.

---

## 5. Insert, layer creation, and split

### INS-01 — Prepend split can put a key equal to the parent separator on the left, making it unreachable

**Severity:** Critical · **Confidence:** Confirmed (deterministic single-thread)  
**Legacy IDs:** `BUG-INS-01`, `BUG-LEAF-01` interop note

**Location:** `src/leaf15.rs:1531-1538` (`calculate_split_point`, prepend
branch), routing `src/ksearch/scalar.rs:22-24`, consequence checks
`src/tree/generic/optimistic_reads.rs:606-608,727-729`.

The prepend fast path returns `SplitPoint { pos: 1, split_ikey: perm.get(1).ikey }`
whenever `insert_pos == 0`, `prev.is_null()`, `size > 1`, and
`perm[1].ikey != insert_ikey`. It does not require `perm[0].ikey !=
perm[1].ikey`. Keys with the same 8-byte chunk but different `keylenx` are legal
neighbours: an 8-byte terminal key and an 8-byte key plus suffix, or terminal
keys whose lengths differ only by trailing NULs. If the split point is 1, the
left leaf keeps `perm[0]` whose ikey equals the separator. The parent routes
equality to the right child (`upper_bound_internode` returns `i+1` on equality),
and the too-far-right test is strict `<`, so the left copy is behind a
separator it is not less than.

**Repro:**

```bash
cd failures/insert-split
cargo test --test prepend -- --nocapture
cargo test --test prepend_impact -- --nocapture
```

Observed:

```text
len after split = 16
get("AAAAAAAA")  -> None       (want 1)
get("AAAAAAAAA") -> Some(2)
scan contains "AAAAAAAA" = true
reinsert("AAAAAAAA") -> old=None, len 16 -> 17
```

The key is physically present and visible to a scan, but unreachable by
`get`, `insert`, and `remove`. Reinserting the same key treats it as absent,
returns `None`, increments `len()` to 17 for 16 distinct keys, and leaves the
old value permanently orphaned.

**Impact:** Silent data loss and `len()` drift on a normal prepend pattern
involving suffix or NUL-padded keys. A later scan and a point lookup disagree.

**Fix:** Require a strict ikey boundary for the prepend fast path:
`perm[0].ikey != perm[1].ikey`. Otherwise fall through to the midpoint path,
which enforces `perm[p-1].ikey < perm[p].ikey`, or advance `pos` to the next
strict boundary (see INS-02).

**Test gap:** No test fills a leaf with a same-ikey run and then prepends a
smaller key. Add the exact `AAAAAAAA`/`AAAAAAAAA`/`B..N`/`AAA` sequence, then
assert `get`, `remove`, `len`, and scan membership agree.

---

### INS-02 — `calculate_split_point` can return `None` for an equal-ikey run at position 0; public insert panics with `SplitFailed`

**Severity:** High · **Confidence:** Confirmed (two deterministic repros)  
**Legacy IDs:** `BUG-INS-02`

**Location:** `src/leaf15.rs:1546-1565` (run adjustment:
`Less | Greater => split_pos -= 1`), `:1567-1583` (boundary decrement),
`:1585-1587` (`None`), panic wrapper `src/tree/generic.rs:673-681`.

With `WIDTH=15`, `mid=7`. If the midpoint boundary falls inside a run of equal
ikeys, the loop moves the split **left** for both `insert_ikey < run` and
`insert_ikey > run`. Moving left is only correct for `Less`; for `Greater` it
must move right. When the run starts at position 0, `split_pos` decays to 0 and
the function returns `None`. `prepare_split` maps that to
`InsertError::SplitFailed`, and every public insert wrapper panics.

A full leaf can hold up to nine entries with one ikey: terminal lengths 1..8
via trailing NULs plus one suffix key. This is not a pathological key set.

**Repro 1:**

```bash
cd failures/insert-split
cargo test --test equal_run -- --nocapture
```

Eight keys `A`, `A\0`, …, `A`+7 NULs (ikey `A`), seven keys `B`..`B`+6 NULs,
then insert `B`+7 NULs. The insert panics:

```text
Insert failed unexpectedly: SplitFailed. This indicates a bug in the tree implementation.
  at src/tree/generic.rs:677
```

**Repro 2:**

```bash
cargo test --test equal_run2 -- --nocapture
```

Seven `A`-terminal keys + eight `B`-terminal keys, then insert the 9-byte key
`A`+7 NULs+`X` (ikey `A`, suffix). Its sorted position is 7 == `mid`; the
boundary decrement walks to 0 and panics.

**Impact:** A normal key set makes `insert` panic, a denial of service for
that tree. The tree remains usable after the panic (the 15 entries are intact)
but the operation can never complete.

**Fix:** In the run loop, move the split right when `insert_ikey > left_ikey`,
left when `<`, and only adjust away from `insert_pos` for `==` (an existing key
should have been `Found`). After adjustment, scan outward for the nearest
strict boundary `perm[p-1].ikey < perm[p].ikey`, preferring the side that keeps
`insert_pos` with its ikey group. Only return `None` for an empty leaf. Stop
panicking on `SplitFailed`; either make the split point infallible with a
1/`size-1` fallback or return a non-public internal error that callers handle.

**Test gap:** Existing tests use at most 2–3 keys with a shared prefix and never
reach a 15-slot split. Add both repros and a randomized short/NUL-key oracle.

---

### INS-03 — Insert can livelock forever on a concurrently coalesced rightmost leaf

**Severity:** High · **Confidence:** Confirmed (subsystem hang + gdb)  
**Legacy IDs:** `BUG-INS-03`, `BUG-CON-11`

**Location:** `src/tree/generic.rs:561-575` (deleted leaf with null `next`
returns the same leaf), `src/tree/generic/insert.rs:527-551`
(`SplitInProgress|KeyMovedToSibling` re-enters `'forward`), `:328-333`
(`validate_membership` rejects a marked `next`), `src/leaf15.rs:1183-1232`
(`unlink_from_chain` leaves `self.next` marked; only the predecessor is
repaired).

A coalesced leaf is `DELETED`, removed from its parent, and its own `next`
remains marked forever. If it was the rightmost leaf, `unmark_ptr(next)` is
null. An inserter that reached it just before coalesce:

1. reaches the leaf and passes `advance_to_key_by_bound_generic`;
2. coalescer locks it, removes it from the parent, marks deleted, unlinks,
   unlocks/retires;
3. inserter reads the post-delete version (it may spin in `stable()` while the
   lock is held), locks, `validate_post_lock` passes, `deleted_layer()` is
   false because modstate is `MODSTATE_EMPTY`;
4. `validate_membership` sees the marked `next` and `wait_for_split` returns
   immediately because `is_deleted()`, so the result is `SplitInProgress`;
5. `advance_to_key_by_bound_generic` sees a deleted leaf with `next == null`,
   breaks, and returns the same leaf;
6. `'forward` repeats forever at 100% CPU.

**Repro:**

```bash
cd failures/insert-split
timeout 120s cargo run --release --bin livelock2
```

`livelock2` repeatedly inserts a new maximum key (append split creates a fresh
rightmost leaf), removes it (leaf empty + queued), and inserts it again while a
background thread spins on `process_coalesce`. The subsystem run hung at round
290,325 (200% CPU, no progress for 20 s). Under gdb the main thread was in
`insert_concurrent` at `insert.rs:530`, background in `collector.leave` (no
locks held). A live leaf's mark clears and `KeyMovedToSibling` advances to
`next`; only the deleted+null case can re-enter the arm forever, so the
interleaving is forced.

**Impact:** A pure CPU livelock (not a wait-for-deadlock) with no other thread
holding a lock. A single insert can pin a core indefinitely.

**Fix:** After acquiring the lock, treat `leaf.version().is_deleted()` like
`deleted_layer()`: drop the lock, `key.unshift_all()`, reload `root_ptr`, and
`continue 'retry`. In `advance_to_key_by_bound_generic`, a deleted leaf with a
null successor must return “exceeded”/force a root restart instead of returning
the same leaf. Make `'forward` exit on no progress.

**Test gap:** No concurrency test mixes remove + `process_coalesce` with inserts
into the vacated rightmost range. Add a bounded/watchdog stress test.

---

### INS-04 — A safe `InlineBits` encoder panic during split leaves the tree unusable

**Severity:** High · **Confidence:** Confirmed (deterministic public API hang)

**Location:** `src/policy/inline_values.rs:113-117` (`to_bits`),
`src/tree/generic/split.rs:241-253` (`mark_split` before split/insert),
`src/leaf15.rs:1692-1710,1747-1753` (entry moves and value encoding),
`src/nodeversion.rs:119-126` (split unlock clears root bit).

`InlineBits` is a public safe trait; its encoder may panic. A leaf split calls
that encoder after marking the left node split and after potentially moving
existing entries to the unpublished right node. Unwinding drops the left
`LockGuard`, which clears `ROOT_BIT`; there is no unwind owner for the right
node, no rollback, and no completed propagation. With a main-root leaf, the
published root now has neither a root bit nor a parent. The climb in
`reach_leaf_concurrent_generic` loops forever, even for an unrelated existing
key. Moved entries and the unpublished sibling can also be stranded.

**Repro:**

```bash
cd failures/insert-split
cargo build --bin inline_encode_panic
timeout 3s "$CARGO_TARGET_DIR/debug/inline_encode_panic"
```

Use the absolute target directory recommended in `failures/README.md`.
The binary fills a leaf with 15 even-numbered keys, then inserts between them
using an encoder that panics for that new value. It catches the intended panic,
prints `caught encoder panic; checking existing key`, and times out with exit
124 in `get` for key zero. No custom allocator or concurrency is involved.

**Fix:** Run user encoding before entering a structural mutation, or make
split construction transactional with an unwind owner and rollback. A safe
trait's panic cannot be excluded by a documentation precondition. ALLOC-03's
allocator-only remedy does not cover this path. Test caught encoder panics
for new-slot, split, and layer-construction paths and verify old entries,
routing, count, and allocation balance after unwinding.

---

## 6. Remove, coalesce, and reclamation

### REM-01 — `remove` can delete a different key that shares the 8-byte ikey

**Severity:** Critical · **Confidence:** Confirmed (deterministic and stress)  
**Legacy IDs:** `BUG-REM-01`, `BUG-LEAF-02`, `BUG-CON-03`

**Location:** `src/tree/remove.rs:715-783` (`lock_and_verify_for_remove`),
`:224-264` (`finish_remove`), `:645-711` (`search_for_remove`).

The optimistic search returns `Found { ki, kp }`, where `ki` is a logical
position. After taking the leaf lock, `lock_and_verify_for_remove` re-reads
`new_perm.get(ki)`, checks `size`, and compares only `slot_ikey` to the target
ikey (`remove.rs:748-758`). It never compares `keylenx`, suffix bytes, value
identity, or the pre-lock permutation/version. Any concurrent insert or remove
of a different key with the same ikey can shift `ki`; the new slot's ikey is
still equal, so the cursor removes the wrong key and returns its value.

**Deterministic sequence:**

1. Leaf contains `K1 = b"abcdefghZZZ"` (ikey `A`, suffix key, value 100) at
   `ki=0` and `K0 = b"abcdefgh"` (same ikey, 8-byte terminal, value 7) absent.
2. `remove(K1)` searches before locking and records `ki=0`.
3. Another thread inserts `K0`; it sorts before `K1` and takes logical
   position 0. `K1` shifts to 1.
4. `remove(K1)` locks; `new_perm.get(0)` is `K0`; ikey check passes.
5. `finish_remove` removes `K0`, returns 7, decrements the count, and leaves
   `K1` (the requested key) present.

This exact sequence is in
`failures/leaf-values/private-tests/remove_unit_tests.add.rs`.

**Public stress repro:**

```bash
cd failures/remove-coalesce
cargo test --release --test wrong_key -- --nocapture
```

Observed: **624,818 wrong-key removals in 8,380,480 operations**. The
subsystem's earlier run observed 635,706/8,165,634. The first independent
mismatch was `remove(key len 6) returned value 108 (expected 106)`.

**Impact:** Silent data loss of an unrelated key, a wrong returned value, and a
linearizability violation. Count arithmetic follows the slot actually removed,
so `len()` stays internally consistent with the corrupted contents and does not
expose the error. Nine legal short keys all have ikey 0, so this is easy to hit
in workloads using empty/NUL-padded keys.

**Fix:** Under the lock, require full key identity:
`slot_keylenx == expected_keylenx`, and for `KSUF_KEYLENX` compare the slot's
suffix bytes with the remaining key. The simplest robust fix is to re-run
`search_for_remove` under the lock and use the returned `ki/kp` (O(15)).
Also pass the pre-lock version and raw permutation into
`lock_and_verify_for_remove` and require `!has_changed` plus the same
permutation (as insert's `validate_post_lock` does). Reject
`is_value_empty(new_kp)` as defense in depth. Fixing OCC-01 is necessary but
**not sufficient**: a same-ikey insert can occur after remove's pre-lock check.

**Test gap:** No existing test churns multiple same-ikey short keys.
`test_same_prefix_different_lengths` is sequential; `splitremove_concurrent`
ignores remove return values. Add the deterministic private test and the public
stress test to CI.

---

### REM-02 — `remove` loses a present key when a concurrent split moves it right

**Severity:** High · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-REM-02`, `BUG-LEAF-03`, `BUG-CON-08`

**Location:** `src/tree/remove.rs:550-634` (outer loop/reach/search),
`:562-580` (`NotFound -> Ok(None)`), contrast insert
`src/tree/generic/insert.rs:453-463` (calls
`advance_to_key_by_bound_generic`) and get
`src/tree/generic/optimistic_reads.rs:582-669` (deleted-leaf, too-far-right,
`check_blink_chain`).

`remove` loads the root, calls `reach_leaf_concurrent_generic`, and searches
only the returned leaf. It has no `advance_to_key_by_bound_generic`, no
`next`/`ikey_bound` forward check, no “too far right” (`prev != null && target <
ikey_bound`) restart, and no deleted-leaf follow. If a concurrent insert splits
the leaf and moves the target to the new right sibling, the left leaf search
returns `NotFound` and remove returns `Ok(None)`. The post-search version check
does not help once the split is complete; on a version change the code
re-searches the same left leaf.

**Repro:**

```bash
cd failures/remove-coalesce
cargo test --release --test lost_remove -- --nocapture
```

Observed: **23 lost removals in 300 attempts**, e.g. `attempt 9: remove(90)
returned None but key is present`. The fixture is one full leaf
`0,10,…,140`; a concurrent `insert(75)` splits and moves `70..140` right; eight
removers target the moved keys.

**Impact:** `remove` reports “not found” for a key that is present before and
after the call. The count is not decremented, so `len()` remains correct, but
the API result is wrong; callers may insert a duplicate/overwrite. Applies at
every layer, including promoted layers.

**Fix:** After `reach`, call `advance_to_key_by_bound_generic` (make it
`pub(crate)`) and use the advanced leaf; repeat on `Retry`. On `NotFound`,
follow `next` when `target_ikey >= next.ikey_bound()`, re-reach from the root
when `prev != null && target_ikey < ikey_bound()`, and treat
`version.is_deleted()` like get. On `has_split`, restart from the layer root
instead of re-searching the same leaf. The cleanest implementation is a shared
stable-leaf/advance helper used by get, insert, and remove.

**Test gap:** Existing concurrent remove tests either remove only or ignore
return values. Add the exact 15-key split fixture with 8 removers and assert
`Some` for every known-present key.

---

### REM-03 — The 1000-retry cap does not bound the inner remove loop or `reach`

**Severity:** Medium · **Confidence:** Confirmed code; impact Likely  
**Legacy IDs:** `BUG-REM-03`, `BUG-CON-07` note

**Location:** `src/tree/remove.rs:551-554` (`retry_count` only at the top of
`'layer_loop`), `:562-574` (`continue 'forward` on version change), `:603-606`
(`Retry` re-scans, no count), `src/tree/generic.rs:323-443` (unbounded
`'retry` in `reach`).

For a single-layer key, `retry_count` is incremented once; subsequent
`Retry`/version-change iterations stay in `'forward` and never increment it. A
hot leaf can spin indefinitely and the documented `RetryLimitExceeded`
contract is not enforced. `reach`'s internal loop is also unbounded; a
permanently null child (OCC-04/GC-03) or a stuck lock spins forever.

**Fix:** Increment one attempt counter at every back-edge (`'layer_loop`,
`'forward`, `Retry`, `RestartFromRoot`), bound `reach` with a retryable result,
and add a short backoff before returning `RetryLimitExceeded`. Add a test-only
contention hook.

---

### GC-01 — Duplicate raw `ChainEntry` pointers survive empty-leaf reuse: maintenance hang and cross-call UAF

**Severity:** Critical · **Confidence:** Confirmed (deterministic)  
**Legacy IDs:** `BUG-COAL-01`, `BUG-ALLOC-02`, `BUG-CON-04`

**Location:** `src/tree/remove.rs:271-291` (`schedule_chain` stores a raw
`*mut u8`), `src/leaf15.rs:1383-1389` (`clear_empty_state` clears only the
queued bit), `src/leaf15.rs:1402-1418` (`try_mark_queued`), `src/tree/coalesce.rs:47-56,344-400,480-545`,
`src/leaf15.rs:1151-1175,1183-1227` (`lock_next`/`unlink_from_chain`).

`chains` stores raw leaf pointers and is documented as “only dereferenced in
the same maintenance call”. That is false: entries are pushed by remove and can
sit across maintenance calls. The queued bit is only a current dedup token;
insert reuse clears it, so a leaf can be emptied again and a second entry
enqueued while the first is still pending. There is no generation or identity
check.

**Deterministic sequence:**

1. Non-leftmost leaf `L` empties; `E1 = ChainEntry { L, ... }` queued.
2. `insert(K)` reuses `L`; `clear_empty_state` clears the queued bit.
3. `remove(K)` empties `L` again; `E2` queued. `pending_coalesce() == 2`.
4. `process_coalesce` pops E1: locks L, size 0, `prev != null`, removes it
   from the parent, `mark_deleted`, `unlink_from_chain` (marks L.next),
   retires L.
5. It pops E2 in the same guard: L is still alive under EBR, lock succeeds,
   `size()==0`, `deleted_layer()` is false (chain deletion uses the version
   DELETED bit, not modstate), `prev` is still non-null. `finish_chain_coalesce`
   calls `unlink_from_chain` again. `lock_next` sees L.next still marked and
   `wait_for_split` returns immediately because `version.is_deleted()`; the CAS
   can never succeed, so the loop spins forever.
6. If E2 is popped after reclamation (e.g. `process_coalesce_batch(1)` and a
   later call with a new guard), the first dereference is a UAF; if the block
   was reused, it can operate on an unrelated leaf/internode (type confusion).

**Repro:**

```bash
cd failures/remove-coalesce
cargo test --release --test double_unlink_deterministic -- --nocapture
```

Observed: `pending_coalesce()==2`, then `WEDGED: process_coalesce did not
return`, watchdog exit 42. The stress variant `double_unlink` and gdb show the
wedged stack `finish_chain_coalesce -> unlink_from_chain -> lock_next ->
wait_for_split`; leaf version word `0xa0009405` = `ISLEAF | DELETED |
SPLITTING | LOCKED`.

**Impact:** Permanent maintenance hang (application DoS), and latent
UAF/type-confused coalesce when a stale entry crosses a guard/epoch or the
pool reuses the block. The same duplication can occur between a deferred entry
and a newer immediate entry because the queued bit is shared.

**Fix:** Do not store raw node pointers in a durable queue. Use the existing
`ikey_bound`-only deferred shape for all chain cleanup; re-traverse from the
root, validate `is_empty_state`/`is_queued`/`is_deleted` under the leaf lock,
and clear the queued bit only when the entry is consumed. If the pointer fast
path must stay, add a per-leaf generation token checked after re-traversal,
never before; additionally check `version().is_deleted()` under the lock in
both chain paths, and make `lock_next` return an error rather than spin on a
deleted leaf.

**Test gap:** Existing tests cover stale/mismatched deferred entries only.
Add the deterministic two-entry test, the `process_coalesce_batch(1)` +
epoch-advance test, and a pooled-reuse UAF test under Miri.

---

### GC-02 — Promoted and multi-leaf sublayers are never reclaimed

**Severity:** High · **Confidence:** Confirmed (counting allocator + counters)  
**Legacy IDs:** `BUG-GC-01`, `BUG-ALLOC-01`, `BUG-CON-10`, independent `gc` repro

**Location:** `src/tree/coalesce.rs:610-728` (`try_remove_sublayer`),
especially `:654-668` (cast the parent-slot pointer to `LeafNode15`, lock it),
`:671-685` (`deleted_layer`/`size`/`prev`/`next` checks), `:687-698` (clear
parent slot), `:704-713` (retire the pointed-to leaf); promotion
`src/tree/split/root_creation.rs:92-120` (does not rewrite the parent slot);
scheduling `src/tree/remove.rs:278-290` (sublayer leaves always use route
entries, never chain entries).

After layer-root promotion the parent slot still points at the original left
leaf `L0`; the true layer root is a new internode with `L0`, `L1`, … as
children. `try_remove_sublayer` locks the slot pointer, casts it to
`LeafNode15`, and requires this one leaf to have `prev == null && next == null`.
A promoted layer always has at least two linked leaves, and nothing ever
chain-coalesces leaves inside a sublayer, so `L0.next` is non-null. The entry is
requeued until `MAX_REQUEUE_COUNT = 10` and then abandoned; the parent slot is
never cleared and the internode is never retired.

**Repro A:**

```bash
cd failures/remove-coalesce
cargo test --release --test sublayer_leak -- --nocapture
```

40 keys `b"ABCDEFGH" + u32` (one ikey, >15 keys force promotion), then remove
all and call `process_coalesce` 200 times. Observed: **before 4 live leaves +
1 live internode; after 4 live leaves + 1 live internode**, `pending=0`,
`deferred=0`, `abandoned=3`.

**Repro B (independent):**

```bash
cd failures/independent
cargo run --quiet --bin gc
```

20 keys `shared00{i:04}`, remove all, maintenance 30×. `coalesce_abandoned()`
is nonzero and `print_tree()` still shows an empty `SUBLAYER`.

**Impact:** Unbounded retention until tree Drop: roughly 4 leaves + 1 internode
per drained promoted layer (for 1M prefixes, on the order of 1.2 GB with
BoxPolicy). Lookups in the dead range still pay a layer descent. The parent
slot retains a pointer to a dead tree, and the queued bit is also mishandled:
`try_remove_sublayer` clears the queued bit of `L0` rather than the leaf that
was actually emptied, stranding empty sibling leaves with no entry.

Ordinary `Drop` still frees the reachable subtree, so this is not a permanent
leak at process exit, but it is unbounded while the tree lives and contradicts
the architecture's reclamation claims.

**Fix:** Follow parent pointers from the slot target to the true layer root
(`find_layer_root`, currently private in `alloc15.rs:37-59`), verify all leaves
are empty, mark the layer deleted early, clear the parent slot, and retire every
leaf and internode in the subtree. Preserve the current lock order (parent leaf
→ sublayer leaves/root) and re-check insert's `deleted_layer` handling. Store
the actually-queued leaf identity (or a token) rather than assuming `L0`.
Alternatively, change the parent slot to point at the promoted root with a
distinguishable tag and update reach/validation/teardown, but that is a larger
invariant change.

**Test gap:** `test_gc_layer_basic_sublayer_cleanup` uses a 2-key
unpromoted layer; `test_gc_layer_max_requeue_drop` accepts “processed or
dropped” and never checks reclamation; `phase2_regression` promotes but never
removes. Add a counting-allocator test asserting live-node counts return to
baseline after draining a promoted layer.

---

### GC-03 — Internode cascade retires a node before unlinking/reparenting it; `RetryExhausted` strands a stale child

**Severity:** High · **Confidence:** Likely (source proof; no forced schedule)  
**Legacy IDs:** `BUG-CASCADE-01`, `BUG-ALLOC-03`, `BUG-CON-07`

**Location:** `src/tree/remove.rs:411-520` (`try_cascade_internodes`),
especially `:431-445` (`mark_deleted`, clear `child[0]`, **retire** `current`,
then lock the grandparent), `:455-461` (`RetryExhausted` returns),
`:487-488` (grandparent child replaced after retirement); consumer
`src/tree/generic.rs:341-443` (no deleted check); teardown
`src/alloc15.rs:69-134`, `src/tree.rs:196-210`.

`try_cascade_internodes` retires the empty internode `I` before it locks the
grandparent `G` and replaces `G.child[k]`. Between retirement and the update,
`G.child[k]` still points at `I`. If the retire batch flushes (especially with
a small batch size), a reader that enters after the retire barrier can still
see the stale child and dereference reclaimed memory. On `RetryExhausted`
(100 consecutive parent changes), the function returns after retirement without
ever replacing the child; the stale pointer is permanent.

The comment claims traversal detects deleted nodes and restarts, but
`reach_leaf_concurrent_generic` does not check a child's `DELETED_BIT` (OCC-04).
An empty cascaded internode has `nkeys==0` and `child[0]==null`; descent hits
`continue 'retry` forever. After reuse, routing can enter an unrelated subtree.
At Drop, `collector.reclaim_all()` may free the node, then `teardown_tree`
traverses the still-linked pointer and frees it again.

**Impact:** Permanent livelock for the affected key range, then UAF/type
confusion after reclamation, and possible double-free/heap corruption at tree
Drop. This violates the “retire only after unreachable” invariant.

**Fix:** Hold `current`'s lock, acquire the grandparent, locate `kp`, install
`child0`, reparent the surviving child, publish the deletion, then drop the
current lock and retire. On `RetryExhausted`, do not return after retirement;
leave the node allocated/linked (marked deleted if desired) and retry or abort,
never leave a retired node reachable. Add a deleted/dirty check in `reach`'s
descending loop.

**Test gap:** No test drives `MAX_PARENT_RETRIES` or a small retirement batch.
Add a test-only retry limiter, a `with_batch_size(1)` cascade test, and a debug
teardown assertion that no traversed child is deleted.

---

### GC-04 — Collapsing a live unary internode breaks height and later split reparenting

**Severity:** High · **Confidence:** Confirmed (quiescent shape test and
deterministic public API hang)

**Location:** `src/tree/remove.rs:431-439,487-491`
(`try_cascade_internodes` replaces a zero-key internode with its `child[0]`),
`src/internode.rs:360-375` (height-based child cast),
`src/tree/split/propagation.rs:226-230` (leaf reparenting only at height zero).

A zero-key internode can still have a live child. Replacing that non-root
internode with its child reduces the depth of only one branch, while the
grandparent's immutable `height` stays unchanged. A height-1 internode then
contains both leaves and height-0 internodes. On a later split,
`InternodeNode::split_into` treats every moved child as an internode because
`height > 0`. For a real leaf it stores the parent at the internode offset
(8), rather than the leaf's actual parent field. Moved leaves retain stale
parents. Splitting one of those leaves loops forever in parent membership
validation.

This happens with successful, single-threaded cascade and no reclamation race.
Reordering retirement as recommended in GC-03 alone does not repair it.

**Repro:**

```bash
cd failures/remove-coalesce
cargo test --test cascade_height -- --nocapture
cargo build --bin cascade_reinsert
timeout 4s "$CARGO_TARGET_DIR/debug/cascade_reinsert"
```

The shape test inserts 300 keys, removes 150..299, and runs maintenance.
It fails because the height-1 root's second child is a leaf. The public-only
binary spaces keys by 1,000, performs the same cleanup, grows the tree, verifies
all resident values, then times out (124) after `splitting a moved leaf at
450001`. A strict-provenance Miri run also reached that sentinel and timed out;
it did not report UB before the hang.

**Fix:** Preserve non-root unary internodes when their child is live. Remove
truly empty branches without shortening a single live branch. Any deliberate
root contraction needs a separate publication/reparenting protocol. Check
height and real node kind together in structural invariant tests; then force
another split in a moved child after cleanup. Keep GC-03's retirement fix too.

---

### GC-05 — A deferred chain entry consumes its entire retry budget in one call

**Severity:** Medium · **Confidence:** Confirmed (deterministic contention test)

**Location:** `src/tree/coalesce.rs:246-259,288-303,549-560`.

`try_remove_one` prioritizes `deferred_chains`. A failed chain attempt pushes
back into that same queue. `process_all` immediately pops the requeued entry
again, so one briefly locked leaf consumes the initial attempt plus all ten
retries during a single maintenance call. It increments `abandoned` and leaves
the empty leaf's queued bit set. Later maintenance has no entry to retry.
`process_batch` can do the same when its limit is large enough. Sublayer
deferral uses a separate queue and does not have this exact feedback loop.

**Repro:**

```bash
cd failures/remove-coalesce
cargo test --test requeue_budget -- --nocapture
```

The test empties the right leaf of a 20-key tree and holds that leaf's normal
version lock while processing cleanup. Quiescent, guard-protected node
inspection obtains the lock; it does not fabricate or alter node state.
Observed: `processed=11 deferred=0 abandoned=1`, followed by the expected
assertion failure. Once the lock is released, the cleanup opportunity is lost.

**Fix:** Separate the current call's work from entries deferred by this call,
or process only a captured initial work budget. Increment a retry count at
most once per entry per maintenance call. Verify both `process_all` and
`process_batch`, eventual reclamation after contention ends, and that the
queued bit still agrees with a live queue entry.

---

## 7. Allocation, internode, and primitives

### ALLOC-01 — Pool deallocation after TLS destruction aborts the process

**Severity:** Critical · **Confidence:** Confirmed (reproduced)  
**Legacy IDs:** `BUG-ALLOC-04`

**Location:** `src/node_pool.rs:292-354` (all entry points use `POOL.with`),
`:419-424` (`reclaim_leaf15` calls `pool_dealloc_leaf`), `src/tree.rs:201-209`
(`Drop` calls `collector.reclaim_all()` before teardown), `src/node_pool.rs:27-30`
(pool capacity/refill), `src/alloc15.rs:223-230` (teardown).

`POOL.with` panics if the thread-local pool has already been destroyed. A tree
stored in a `thread_local!` can create exactly that order: the tree's TLS
destructor is registered first, then the first insert/remove initializes
`POOL` (registering its destructor later). At thread exit, destructors run LIFO,
so `POOL` is destroyed before the tree. Tree `Drop` → `collector.reclaim_all()`
→ `reclaim_leaf15` → `pool_dealloc_leaf` → `POOL.with` panics. Depending on the
seize version/path, the unwind then hits `LocalBatch::DROP` and aborts, or
turns into a SIGSEGV.

**Repro:**

```bash
cd failures/internode-alloc
cargo test --release --test tls_abort -- --nocapture
```

Observed:

```text
retired, pending=0 abandoned=0
worker exiting
thread '<unnamed>' panicked ... cannot access a Thread Local Storage value
  during or after destruction: AccessError
...
signal: 11 (SIGSEGV: invalid memory reference)
```

The subsystem also observed SIGABRT/misaligned-pointer in the same path with a
different seize unwind state.

**Impact:** Normal teardown of a tree moved into a thread-local can abort the
process. This is a common embedding pattern (caches, worker-local indexes), not
malicious use.

**Fix:** Use `POOL.try_with` in every allocation and deallocation path. On
failure, deallocate with the bucket layout directly (or the original allocation
layout for oversized blocks); on allocation failure, fall back to `alloc` +
`handle_alloc_error`/abort. `ThreadPool::drop` already avoids `POOL.with`; the
fallback closes the teardown hole. Add the TLS repro as a regression test.

**Test gap:** No test covers TLS destructor ordering or moving a tree across
threads. `tests/leak_tests.rs` only drops on the owning thread.

---

### ALLOC-02 — A panicking user value `Drop` corrupts seize's batch state and aborts on tree drop

**Severity:** High · **Confidence:** Confirmed (reproduced)  
**Legacy IDs:** `BUG-ALLOC-05`

**Location:** `src/node_pool.rs:419-424` (`drop_in_place` in the reclaimer),
`src/tree.rs:201-209` (`Drop` calls `reclaim_all` twice, explicitly and via
`Collector::drop`), seize 0.5.1 `raw/collector.rs:436-451,475-481`.

Rust permits a value's `Drop` to panic. `reclaim_leaf15` runs
`drop_in_place` on a leaf; for `BoxPolicy` that calls user `V::drop`. If it
panics, seize's `free_batch` has already set the local batch to
`LocalBatch::DROP`, but the panic skips the reset and `LocalBatch::free`.
`Collector::drop` calls `reclaim_all` again; `reclaim_all` checks only
`batch.is_null()`, not `batch == LocalBatch::DROP`, and calls
`free_batch(0xffff…)`. The next `(*batch).entries` is a misaligned-pointer
read, observed as a second panic and SIGABRT.

**Repro:**

```bash
cd failures/internode-alloc
cargo test --release --test panic_value -- --nocapture
```

The test uses a `Boom` value whose `Drop` panics, overwrites a key once to
retire the old value, then drops the tree. It prints two `boom value drop`
panics followed by a misaligned pointer/abort.

**Impact:** Panicking value destructors are legal Rust; dropping a tree can
therefore abort or execute UB instead of unwinding predictably. This is partly
an upstream seize bug, but the crate can defend its reclaimer and document the
constraint.

**Fix:** Upstream: `reclaim_all` must treat `LocalBatch::DROP` as an empty
batch and `free_batch` must reset the local batch before invoking reclaimers or
be unwind-guarded. In masstree, ensure `pool_dealloc` runs even if
`drop_in_place` unwinds (unwind guard), fix ALLOC-01 so the panic does not
cascade, and document that `V::drop` must not panic during destruction. Add an
upstream issue with the minimal `Boom` repro.

**Test gap:** No test covers panicking `Drop`; add one under `catch_unwind` and
assert the tree remains droppable/allocator state is clean.

---

### ALLOC-03 — A custom allocator panic after `mark_split` can leave a split-locked, linked sibling

**Severity:** Medium · **Confidence:** Likely (safe trait allows panic)  
**Legacy IDs:** `BUG-ALLOC-08`

**Location:** `src/tree/generic/split.rs:241` (`lock.mark_split()`),
`src/leaf15.rs:1807-1824` (link right sibling), `src/tree/split/propagation.rs:209`
(`alloc_internode_direct_for_split`), `src/tree/split/root_creation.rs:33,67,102,135`
(allocations), `src/alloc_trait.rs:24-71` (safe trait).

`handle_leaf_split_and_insert_generic` marks split, links the right sibling,
then enters propagation, which allocates parent siblings/root internodes. The
built-in `SeizeAllocator` aborts on OOM, but `TreeAllocator` is a safe public
trait and an implementation may panic (or a debug assertion may fire). On
unwind, the left lock is released, but the right sibling keeps its
`new_for_split` version (`LOCK|SPLITTING`), is linked in the leaf chain, and
has no parent. Any reader that reaches it via `next` spins in
`NodeVersion::stable` forever; the sibling and allocated twigs leak.

**Fix:** Allocate the parent sibling/root before `mark_split` where feasible
(the split point is known), or install an unwind guard that unlocks the sibling
and truncates the link on panic. Document/enforce “allocator must abort, never
panic”, or make the trait `unsafe`. Add a test with a custom allocator that
panics on the second internode allocation and assert the tree remains usable.

---

### PRIM-01 — `ShardedCounter::load` can debug-assert on a valid concurrent snapshot

**Severity:** Low · **Confidence:** Likely (deterministic arithmetic, no
scheduler repro)  
**Legacy IDs:** `BUG-PRI-04`

**Location:** `src/shard_counter.rs:124-143`; `len()` at
`src/tree/generic.rs:141`; test comment at `src/shard_counter/unit_tests.rs:199-203`.

`load` sums 16 independently loaded `Relaxed` shard values and then
`debug_assert!(total >= 0)`. The sum is not a linearizable snapshot. With two
shards holding `+k` and `-k` and a logical count of 0, a load can read the
negative shard after its decrement and the positive shard before its next
increment, yielding `-1` while the logical count is never negative. The unit
test pre-populates the counter “to avoid negative intermediate states”, which
confirms the transient is known.

**Repro:**

```bash
cd failures/primitives/shard
cargo run --quiet
```

`ShardedCounter` is included from a local copy of the source; the binary hammers
increment/decrement/load and lets the debug assertion fire. Release builds
return 0 for a transient negative sum, so `len()` is approximate (documented),
but the debug panic violates the “approximate” contract.

**Fix:** Remove the debug assertion; clamp unconditionally in both profiles and
document that the sum is not a snapshot. A stronger fix would maintain a single
approximate total or tolerate bounded skew, but that is a design change.

---

### PRIM-02 — `Permuter<WIDTH<15>` leaves bits above `WIDTH_MASK` after `remove_to_back`/`rotate`

**Severity:** Low · **Confidence:** Confirmed (direct probe)  
**Legacy IDs:** `BUG-PRI-05`

**Location:** `src/permuter.rs:82-89` (`WIDTH_MASK`), `:318-330`
(`remove_to_back`), `:408-428` (`rotate`), `:439-457` (`debug_assert_valid`
only checks positions `0..WIDTH`).

`remove_to_back` computes `(x & mask) << shift_to_back` and ORs it into the
result without a final `& WIDTH_MASK`; `rotate` likewise shifts a masked value
left without masking the final result. For `WIDTH<15`, bits above the encoded
width are written into unused nibbles. The logical permutation
(`get(0..WIDTH)`, `size`) remains correct, but the raw `value()` is not
canonical: `Eq` compares raw values and raw CAS APIs can disagree with logical
equality.

**Repro:**

```bash
cd failures/primitives/nodeversion
cargo run --quiet --bin perm_raw
```

Observed:

```text
Permuter<7>::make_sorted(5)          raw=0x56432105
after remove_to_back(0)              raw=0x56432105643214
bits above width                    0x56432100000000
```

`debug_assert_valid` passes because it only inspects `0..WIDTH`.

**Fix:** End both operations with `& WIDTH_MASK`; add
`debug_assert_eq!(value & !WIDTH_MASK, 0)`; extend proptests to WIDTH 3/7 and
compare raw values. Production `Permuter15` is unaffected.

---

## 8. Range scans

The range subsystem is the largest source of deterministic correctness bugs.
The common root causes are:

- `lower_with_position` (`src/tree/range/helper.rs:69-103`) compares only the
  first 8-byte ikey, not the full key (`keylenx`, suffix bytes, layer kind);
- reverse initial positioning reuses that helper and then applies scan-wide
  `emit_equal` flags to individual slots;
- `reach_leaf_for_scan` (`src/tree/range/traversal.rs:21-82`) is a second,
  weaker traversal implementation that does not climb promoted layer roots or
  follow the full B-link/deleted-leaf protocol;
- values-only strategies skip the bound checks that keyed strategies perform.

### RNG-01 — Forward lower bound ignores full-key order inside one ikey group

**Severity:** Critical · **Confidence:** Confirmed (deterministic)  
**Legacy IDs:** `BUG-RNG-02`, independent `repros`

**Location:** `src/tree/range/helper.rs:69-103` (`lower_with_position`),
`src/tree/range/forward_ctx.rs:255-278` (`find_initial`, `final_pos = kx.i + 1`),
`:327-333` (`handle_initial_match` emits inline slots without a full-key check),
`src/tree/range/api.rs:427-438` (`scan_prefix` fallback), forward batch paths
`forward_ctx.rs:986-999,1100-1125,1197-1221`, `batch_common.rs:422-448`.

The lower-bound search returns the first slot of an equal-ikey group. `find_initial`
then starts after it, and `handle_initial_match` emits any inline slot when
`emit_equal` is true without verifying the slot key is `>=` the bound. Equal
ikeys are common: short keys are zero-padded, so `""`, `"\0"`, `"\0\0"` all
have ikey 0; an 8-byte key `K` and `K + suffix` share the first 8 bytes.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin nul
cargo run --quiet --bin nulpfx
```

Keys `""`, `"\0"`, `"\0\0"`, `"a"`:

```text
range(Included("\0"), Unbounded)   actual ["", "\0", "\0\0", "a"]
                                   expected ["\0", "\0\0", "a"]
range(Excluded("\0"), Unbounded)   actual ["\0", "\0\0", "a"]
                                   expected ["\0\0", "a"]
scan_prefix("\0")                  actual ["", "\0", "\0\0"]
                                   expected ["\0", "\0\0"]
```

The `Excluded` case emits the excluded bound itself. `scan_prefix("\0")`
returns a key that does not have the prefix. With `abcdefgh` and
`abcdefghX`, `range(Included("abcdefghX"), Unbounded)` emits `abcdefgh`
first, and an excluded 9-byte bound can emit the 8-byte terminal (strictly below
the start).

**Impact:** All forward iterator, ref, batch, and prefix APIs can return keys
outside the requested range. This is not the documented “same ikey boundary”
approximation: the key is fully available, and the code does not compare it.

**Fix:** Implement one full-key lower-bound routine for forward scans. Within an
equal-ikey group compare kind and bytes: layer pointer (128) sorts after `KSUF`
(64) after terminal lengths; inline slots compare lengths; `KSUF` slots compare
stored suffix bytes. Return the first key `>=` the cursor and remember whether
it is exactly equal (for `Excluded`). Use the same helper in `find_initial`,
`find_retry`, and `advance_leaf` with an explicit “cursor already emitted” flag.

**Test gap:** Existing proptests use mostly unique/random keys and do not test
zero-padding-equivalent keys or a bound inside an 8-byte+suffix group. Add the
four-key NUL case and the `abcdefgh`/`abcdefghX` case.

---

### RNG-02 — Reverse scan descends a layer whose keys are all greater than the end bound

**Severity:** High · **Confidence:** Confirmed (iterator and batch)  
**Legacy IDs:** `BUG-RNG-03`, independent `repros`

**Location:** `src/tree/range/reverse_ctx.rs:266-277`
(`try_initial_position_reverse`), `:346-354`
(`handle_layer_descent_reverse`), `:399-402` (`try_emit_slot_reverse`).

`try_initial_position_reverse` descends whenever `lower_reverse` selects a
layer pointer. `handle_layer_descent_reverse`, when the cursor has no suffix
(`!cursor_key.has_suffix()`, i.e. the end key ends at or before the chunk
boundary), calls `shift_clear_reverse()` and sets `upper_bound`, so the
sublayer is scanned from its maximum. But if the end key is a prefix of the
layer's keys (for example 8-byte `PREFIX00` and keys `PREFIX00 || u64`), every
sublayer key is greater than the end bound and must not be visited.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin layerend
```

For keys `abcdefghX`, `abcdefghY`, `abcdefghZ`:

```text
range(Unbounded, Included("abcdefgh")).rev()  -> 3 keys   expected 0
range(Unbounded, Excluded("abcdefgh")).rev()  -> 3 keys   expected 0
```

For 40 keys `PREFIX00 || u64(0..39)`, both `Included("PREFIX00")` and
`Excluded("PREFIX00")` return all 40; `scan_rev_batch` returns all 40 too.

**Impact:** A bounded reverse scan can return the entire sublayer above the end
bound, violating both the keyed iterator and batch contracts. `Excluded` is
treated like `Included`.

**Fix:** Do not descend on an initial reverse end position when
`cursor_key.current_len() <= IKEY_SIZE` (or, more precisely, when the end key
ends at or before this layer chunk). Position `ki - 1` in the parent and emit
only parent inline keys `<=` the bound. Only descend when the cursor has a real
remaining suffix. Audit the same descent rule in `reposition_back`/`find_prev`
for scan-discovered layers.

**Test gap:** Existing reverse-bound unit tests use 24-byte keys that continue
past the chunk; none ends exactly on a layer chunk. Add the 3-key and 40-key
cases.

---

### RNG-03 — Reverse initial positioning is not a full-key upper bound

**Severity:** High · **Confidence:** Confirmed (iterator and batch)  
**Legacy IDs:** `BUG-RNG-04`, independent `repros`

**Location:** `src/tree/range/helper.rs:389-409` (`lower_reverse` built on the
ikey-only `lower_with_position`), `src/tree/range/reverse_ctx.rs:399-402`
(scan-wide `emit_equal` suppresses earlier inline slots), `:454-463`
(`try_emit_suffix_slot_reverse` compares suffixes even when
`slot_ikey < cursor_ikey`).

Three related defects:

1. `lower_reverse` returns the first slot of the equal-ikey group, not the last
   slot `<=` the bound.
2. `try_emit_slot_reverse` suppresses **all** inline candidates when the end
   bound is `Excluded`, even when the candidate is an earlier, unrelated key
   strictly below the bound.
3. Suffix comparison runs whenever `upper_bound` is false, even when the slot
   ikey is less than the cursor ikey; such a slot is wholly below the bound and
   must be emitted regardless of suffix.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin nul
cargo run --quiet --bin apis
```

With `""`, `"\0"`, `"\0\0"`:

```text
range(Unbounded, Included("\0")).rev()  -> [""]      expected ["\0", ""]
range(Unbounded, Excluded("\0")).rev()  -> []        expected [""]
```

With `abcdefgh`, `abcdefghX`, `abcdefghX0`, `abcdefghY`:

```text
range(Unbounded, Included("abcdefghY")).rev()
  -> ["abcdefgh"] expected ["abcdefghY","abcdefghX0","abcdefghX","abcdefgh"]
range(Included("abcdefghX"), Included("abcdefghY")).rev()
  -> only "abcdefgh" (batch: 0 rows)
```

The last line is a complete loss of a valid non-empty reverse range in
`scan_rev_batch`; the iterator emits an out-of-bound key below the start bound.

**Impact:** Missing keys, extra keys, and empty results for valid bounded
reverse ranges. Keys can be emitted below the start or above the end.

**Fix:** Implement an exact reverse upper-bound search: compare ikey, then full
key (`keylenx`/suffix), only skip a candidate when the full key equals an
`Excluded` bound, and only compare suffixes when `slot_ikey == cursor_ikey`.
Do not use a scan-global `emit_equal` to decide whether to emit an earlier slot.

**Test gap:** No proptest/unit test mixes an 8-byte terminal with longer
suffixes under a bounded reverse scan, and no test asserts an excluded bound is
not emitted.

---

### RNG-04 — Bounded forward scans skip the start key in a promoted (multi-leaf) layer

**Severity:** High · **Confidence:** Confirmed (iterator, batch, prefix)  
**Legacy IDs:** `BUG-RNG-05`, `BUG-RNG-09` (root cause)

**Location:** `src/tree/range/traversal.rs:21-82` (`reach_leaf_for_scan`),
`src/tree/range/forward_ctx.rs:697` (advance uses `lower_with_suffix`),
`src/tree/range/helper.rs:145-150` (`Equal => i + 1` assumes the cursor was
already emitted), `src/tree/range/api.rs:503-566` (`find_layer_child_root`).

`reach_leaf_for_scan` returns the first leaf when the supplied root is a leaf.
After layer-root promotion, that “leaf” is the original left leaf with a parent
pointer to the new internode; scans never climb to it. Point reads do walk up
to the root (`generic.rs:305-340`). `find_initial` therefore searches only the
leftmost leaf and returns `FindNext` with `ki = size`. The first
`advance_leaf_generic` uses `lower_with_suffix`, which returns `i + 1` for an
exact inline key on the assumption that the cursor key was already emitted. It
was not, because `find_initial` never reached its leaf, so the start key is
skipped. For a full 16-byte prefix, the prefix fast path also falls back to
this range and returns nothing.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin promoted
```

40 keys `PREFIX00 || u64(i)`, `i = 0..40`:

```text
range(Included(keys[15]), Unbounded) -> 24 keys starting keys[16]
                                        expected 25 starting keys[15]
range(Included(keys[39]), Unbounded) -> empty
                                        expected keys[39]
scan_prefix(keys[15])                -> 0 rows
                                        expected 1
```

`keys[0..15]`, `Excluded` starts, and all reverse scans are correct.

**Impact:** Bounded range and full-key prefix scans omit the requested start key
when it lands in a non-leftmost leaf of a promoted layer. A scan starting at the
last key returns nothing.

**Fix:** In `reach_leaf_for_scan`, climb parent pointers to the true layer root
before descending, or reuse the stable-leaf walk from
`reach_leaf_concurrent_generic`/`advance_to_key_by_bound`. Track
`cursor_emitted` separately from “cursor value”: on the first leaf advance after
`find_initial` returns `FindNext` with a cursor that was not emitted, use
full-key lower-bound positioning rather than `lower_with_suffix`'s skip-exact
rule.

**Test gap:** No test creates >15 keys under one 8-byte prefix and then starts a
bounded/prefix scan inside a non-leftmost layer leaf. Add the exact 40-key case.

---

### RNG-05 — Values-only scans ignore bounds (empty ranges, far layers, max-length prefixes)

**Severity:** High · **Confidence:** Confirmed (forward and reverse)  
**Legacy IDs:** `BUG-RNG-06`; independent `maxprefix`

**Location:** `src/tree/range/forward_ctx.rs:1287-1331`
(`ValuesOnlyStrategy::emit_initial`/`dup_check_entry` skip the end-bound check),
`:1341-1346` and `src/tree/range/batch_common.rs:514-519` (ikey-only checks),
`src/tree/range/reverse_ctx.rs:1673-1703` (`RevValuesOnlyStrategy` ignores the
start bound), `src/tree/range/batch_common.rs:404-414,502-512` (layer pointer
checked before any bound check).

Values-only scans check only the current layer's ikey and skip the end/start
bounds; `emit_initial` emits the first value unconditionally. A layer pointer
is processed before the bound check, so a values-only scan can descend an
out-of-range layer and emit its first key through the retry/duplicate path.

**Repro A:**

```bash
cd failures/range
cargo run --quiet --bin vals
cargo run --quiet --bin valfar
```

With keys `a`, `zzzzzzzzX`, `zzzzzzzzY` (the last two form a layer at
`zzzzzzzz`):

```text
scan_values(Included("a"), Included("a"))   -> [1, 2]  expected [1]
scan_values(Included("a"), Excluded("b"))   -> [1, 2]  expected [1]
scan_values(Included("b"), Included("c"))   -> [2]     expected []
```

Empty/inverted range (`a`,`b`,`c`):

```text
scan_values(Included("b"), Included("a"))     -> [2]   expected []
scan_values_rev(Included("b"), Included("a")) -> [1]   expected []
```

**Repro B (MAX key length):**

```bash
cd failures/range
cargo run --quiet --bin maxprefix
```

Keys `[a;255]`, `[a;256]`, `[a;255]+b`. `scan_prefix_values([a;256])` returns
two values (`[2,3]`) including the adjacent `[a;255]+b` key, while keyed
`scan_prefix` correctly returns one. This is the same missing start/end bound
check at the maximum-length boundary.

**Impact:** Values-only scans (and `scan_prefix_values`) return values for keys
outside the requested range, including inverted/empty ranges. This is not the
documented “same boundary ikey” approximation: the far-layer examples share no
ikey with the bound.

**Fix:** Give values strategies the bound and perform the same checks as keyed
scans. In `emit_initial`/`dup_check_entry`, test the cursor key against the
end/start bound; in `process_batch_values`, compare `slot_ikey` with the bound
**before** the layer-pointer branch, and either build the full key for equal
ikeys or keep an explicit same-ikey predicate. Reverse values-only must check
the start bound on initial emission. This is a small, low-risk change with an
extra compare per emitted value.

**Test gap:** Values-only tests cover unbounded and ordinary short-key bounds
only. Add empty/inverted ranges, a far layer, and the 256-byte prefix case.

---

### RNG-06 — `DoubleEndedIterator` re-emits the whole range after one direction is exhausted

**Severity:** Medium · **Confidence:** Confirmed (contract violation)  
**Legacy IDs:** `BUG-RNG-07`

**Location:** `src/tree/range/iterator.rs:370,449,628,781,837` (meeting checks
skipped when the opposite side is exhausted), `:365-366,444-446,623-624`
(normal exhaustion marks only the current side).

If one direction reaches the end/start without the other side ever being
initialized, the other direction still points at its initial position and
re-emits the whole range.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin doubleended
```

Output:

```text
forward=[Some([97]), Some([98]), Some([99]), None]
backward_after_exhaustion=[[99], [98], [97]]
assertion failed: backward_after_exhaustion.is_empty()
```

Reverse first behaves symmetrically (`next_back` to exhaustion, then `next`
returns `a,b,c`). Balanced interleaving is correct; existing unit tests only
exercise balanced patterns.

**Impact:** A `DoubleEndedIterator` contract violation and duplicate values for
users who drain one side and then query the other.

**Fix:** When `advance`/`advance_back` exhausts for end-bound/tree-end reasons,
mark both sides exhausted. Early visitor stop does not go through `advance`
(callbacks own `self`), so this is safe. Add a regression test that drains one
side first and asserts the other is `None`.

---

### RNG-07 — Forward multi-layer `find_next` does not validate the cached leaf version

**Severity:** High · **Confidence:** Confirmed (duplicate emission under stress);
stale-value/type-confusion impact Likely  
**Legacy IDs:** `BUG-RNG-08`

**Location:** `src/tree/range/forward_ctx.rs:403-407,472-481`
(`find_next_generic` checks only `is_deleted()` before reading cached
permutation/emitting); contrast `:510-515` (single-layer path),
`src/tree/range/reverse_ctx.rs:575-581`, `forward_ctx.rs:665-667` (advance),
`batch_common.rs:467-469,542-544` (batch end validation).

A physical slot referenced by the cached permutation can be reused for another
ikey between the last version check and the emit. The `slot_ikey < last_ikey`
check only catches backward movement; an equal or forward ikey is emitted.
Concurrently converting a `KSUF` slot to a layer pointer can also be observed
as old `keylenx` + new layer pointer; `leaf.load_value` then wraps the layer
root as `ValuePtr<V>`.

**Repro:**

```bash
cd failures/range
timeout 30s cargo run --quiet --bin race
```

One leaf with keys 0..29, 3 writer threads removing/reinserting, 2 scanners
doing `range(Included(0u64), Unbounded)`. After ~7 rounds the debug ordering
check fires:

```text
=== ORDERING VIOLATION DETECTED (FUSED) ===
Current key: "\0\0\0\0\0\0\0\u{10}"
Last key:    "\0\0\0\0\0\0\0\u{10}"
...
Scan ordering violation: emitted key ... is not > last emitted key
```

The exact interleaving is nondeterministic but repeatable.

**Impact:** Duplicate emissions, stale values, and (in the narrower
layer-pointer interleaving) a value pointer that is not a `V`. This is a
soundness issue rather than just ordering.

**Fix:** Add `leaf.version().has_changed(self.stack.version())` at the top of
`find_next_generic` (and ideally a second check after the snapshot is built,
before returning `Emit`); on mismatch return `Retry` so existing duplicate
suppression runs. Use one validated read helper for single/multi-layer. Add a
start-of-leaf version check to batch paths.

**Follow-up scope:** A precheck alone is insufficient. Reverse initial emits
check the version at `src/tree/range/reverse_ctx.rs:409-413,470-474`, then
load the value. `try_emit_generic` checks at `:706`, then reads suffix/value
at `:717-731`; the forward single-layer pointer path likewise has no final
check after the pointer load. A writer can complete a slot reuse or layer
conversion in that gap even if OCC-02 is fixed. These paths need validation
after the complete payload snapshot and before constructing/exposing a typed
reference. This additional interleaving is source-established (Likely), not
independently reproduced here. Batch callbacks have their own deterministic
proof and stronger emission-before-validation problem in RNG-10.

**Test gap:** Concurrent correctness tests scan during inserts or removals, but
not during remove+reinsert slot reuse, and never combine layer creation with
scanning. Add this stress and a Miri/loom model of the leaf snapshot.

---

### RNG-08 — Reverse scan/`last()` returns nothing after the rightmost leaf becomes empty

**Severity:** Medium · **Confidence:** Confirmed (independent)  
**Legacy IDs:** checkpoint finding (no subsystem ID)

**Location:** `src/tree/range/reverse_ctx.rs:189-192` (empty leaf returns
`(ScanStateBack::Up, None)` before trying `prev`), `src/tree/range/iterator.rs`
reverse initialization/`last_with_guard` path, `src/tree/range/api.rs:161-163`.

`find_initial_reverse` takes a fast path: if the initially reached leaf has
`size == 0`, it returns `(Up, None)` without checking whether the leaf has a
predecessor or whether it is the rightmost leaf. After removing the tail of the
rightmost leaf, the leaf remains linked and empty until maintenance. A full
reverse scan and `last()` therefore see an empty starting leaf and terminate,
while forward iteration sees the prior leaves.

**Repro:**

```bash
cd failures/range
cargo run --quiet --bin empty_right
```

Insert `k0000..k0019`, remove `k0015..k0019`:

```text
len=15 forward=15 reverse=0 last=None
assertion failed: left: 0, right: 15
```

Calling maintenance to unlink the empty leaf fixes the result, which confirms
the leaf state is the trigger.

**Impact:** `last()` and reverse iteration can return no entries even though the
tree contains 15 keys. This is a common “remove the tail, then iterate
backwards” pattern.

**Fix:** On an empty leaf in reverse initialization, follow `prev` before
returning `Up`; if the leaf is the rightmost and empty, move left until a
non-empty leaf or the true beginning is reached. Keep the version/lock protocol
for the moved-to leaf. Add a regression with the exact 20/5 key sequence.

**Test gap:** `tests/range_scan_tests.rs` covers unbounded reverse and
concurrent inserts but not a removal that leaves the rightmost leaf empty.

---

### RNG-09 — Scan traversal is a second, weaker B-link implementation

**Severity:** Medium (systemic root cause) · **Confidence:** Likely  
**Legacy IDs:** `BUG-RNG-09`

**Location:** `src/tree/range/traversal.rs:21-82`,
`src/tree/range/helper.rs:309-347` (`stable_reverse`), `src/tree/range/forward_ctx.rs:665-667`,
`api.rs:503-566`.

`reach_leaf_for_scan` does not use the stable point-read protocol: it does not
climb to the root after promotion, does not check “too far right” (`prev` and
`ikey_bound`), does not follow `next` when a split moved the target right, and
has no `MAX_BLINK_HOPS`/progress bound. `stable_reverse` only walks `next`
(right), never checks `prev`, and has no hop limit. This is the root cause of
RNG-04 and contributes to the remove/split misses (REM-02). It can also start a
forward scan in a leaf to the right of the target and miss keys before the leaf's
first slot.

**Fix:** Share one stable-leaf routine with point reads: climb parent pointers
to the layer root, descend, walk the B-link chain to the leaf whose
`ikey_bound` contains the cursor, with a hop cap and too-far-right restart. Add
a concurrent scan+split test and a scan after promotion test.

---

### RNG-10 — Batch callbacks run before validation and can observe stale slot reuse

**Severity:** High · **Confidence:** Confirmed (six deterministic API variants)

**Location:** `src/tree/range/batch_common.rs:378-471,481-547`
(`process_batch_keyed` / `process_batch_values`),
`src/tree/range/forward_ctx.rs:1045-1059`,
`src/tree/range/reverse_ctx.rs:1494-1514` (retry after callbacks).

The batch loops use a cached permutation and invoke visitors immediately.
Their version check happens only after the leaf's callbacks; stop, layer, and
bound returns can bypass it altogether. A callback can legally insert/remove
through the shared tree. A full leaf's just-freed physical slot can therefore
be reused for a different key before the batch visits it. The visitor observes
out-of-order keys/values before the loop detects any change. Retrying cannot
undo callback side effects; values-only scans additionally replay prior
entries because they do not maintain the same complete cursor state.

This is independent of RNG-07's forward `find_next` path: it reproduces in
both directions, without threads, sleeps, probabilistic races, or bad bounds.
Adding only a check at batch entry is insufficient because mutation occurs
inside the second callback.

**Repro:**

```bash
cd failures/range
cargo test --test batch_callback -- --nocapture
```

All six tests fail: forward/reverse keyed inline batches, forward/reverse
values-only batches, and forward/reverse boxed reference batches. Starting
with keys 1..15, the second forward callback removes 3 and inserts 16:
`[1, 2, 16, 4, ..., 15, 16]`. Reverse removes 13 and inserts 0:
`[15, 14, 0, 12, ..., 1, 1, 0]`. Values-only variants also replay the earlier
part of the leaf. Each assertion checks strict monotonicity of the observed
key-encoded values and callback-count consistency.

**Fix:** Validate each key/value snapshot before invoking a callback, or
buffer a complete validated leaf snapshot before emitting it. Keep retry
progress separate from speculative scan state, and resume strictly beyond
the last delivered full key in both directions. Include callback mutation,
early stop, bounds, slot reuse, and layer conversion in the regression matrix.

---

## 9. Suffix storage

The following findings are in addition to SAFE-06 (external bag data race and
aliasing).

### SUF-01 — Drain does not clear the inline metadata for a replaced slot; an old inline suffix can shadow the new external suffix

**Severity:** Medium (latent corruption) · **Confidence:** Likely (contract
permits it; current tree assignments target free slots)  
**Legacy IDs:** `BUG-SUF-02`

**Location:** `src/policy/suffix_ops.rs:264-274` (`drain_and_rebuild`),
`src/suffix/inline.rs:229-244` (`drain_core` skips `new_slot`),
`src/policy/suffix_ops.rs:305-319` (`install_drained_bag` publishes without
clearing inline), `:28-30` (`get_suffix` checks inline first),
`src/suffix/inline.rs:406-424` (`try_assign` reuse path).

When a suffix is assigned to an already-occupied slot and inline storage cannot
hold it, `drain_core` deliberately skips copying the replaced slot and assigns
the new suffix only to the new external bag. `install_drained_bag` stores the
external pointer but never clears the inline metadata for that slot. Because
`get_suffix` checks inline first, the old inline suffix shadows the new external
one. A later split copies the stale suffix and loses the new one.

Current tree assignments use free slots (`assign_slot_generic` uses `perm.back()`,
split-left uses a freed slot, layer creation uses fresh 0/1, init uses
sequential slots), so this is latent, not currently reachable. But
`SuffixStore::assign` is documented only as “assign a suffix to a slot” with no
free-slot precondition, and the overwrite fast path exists.

**Fix:** In `install_drained_bag`, after storing the external pointer (so a
reader never sees neither store), call `inline.clear(slot)` for the replaced
slot. The new external bag already contains the new bytes; other inline copies
remain live. Add a regression: `assign(0, S_old)` → force a drain with longer
`assign(0, S_new)` → assert `inline.get(0).is_none()` and
`get_suffix(0) == Some(S_new)`.

**Test gap:** `test_inline_drain_replaces_slot` checks only the external bag,
not the inline store, so it cannot catch this.

---

### SUF-02 — `compact` panics on a zero-length suffix sharing an offset with a later suffix

**Severity:** Low (debug/test builds) · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-SUF-03`

**Location:** `src/suffix/compact.rs:30-74,95-160`,
`src/suffix.rs:433-445` (`assign` sets offset before extending),
`src/suffix/unit_tests.rs:57-64` (empty suffixes are supported).

`compact`/`compact_in_place` build `(slot, offset, len)` entries and only
compare `meta.offset < last_offset` to decide whether to sort; equal offsets
are treated as sorted. A zero-length suffix assigned first gets offset 0; the
next non-empty suffix also gets offset 0. During the copy loop,
`write_pos += len` can move past `src_start` for the zero-length entry and the
debug invariant `write_pos <= src_start` fails.

**Repro:**

```bash
cd failures/suffix
cargo run --quiet --bin compact_empty
```

```text
thread 'main' panicked at src/suffix/compact.rs:67:13:
compaction invariant violated: write_pos (3) > src_start (0)
```

In release the arithmetic is benign (zero-byte `copy_within`), so this is a
debug/test panic rather than memory corruption.

**Fix:** Skip `len == 0` entries in the copy loop and repoint their metadata to
`write_pos` without advancing it; or sort zero-length entries last; or treat
`len == 0` as EMPTY and reject empty suffixes. Add an empty-suffix compaction
test.

---

### SUF-03 — Public `SuffixBag` length checks are debug-only and release silently truncates

**Severity:** Low (public standalone API; tree keys are capped at 256)  
**Confidence:** Confirmed  
**Legacy IDs:** `BUG-SUF-04`

**Location:** `src/suffix.rs:190-201,316-327,408-425` (docs say “Panics if
`suffix.len() > u16::MAX`”, checks are `debug_assert!`), `:179-183,395-399`
(`len: suffix.len() as u16`), `:123-124` (`from_vec` debug-only empty check).

In release, a >65,535-byte suffix is stored in full but its metadata length is
truncated to `u16`, so `get` returns only a prefix. `try_assign` also returns
`false` after successfully growing and assigning, contrary to the doc's “does
not fit”. The tree cannot reach this because `Key::new` caps keys at 256 bytes
(max suffix 248), but `SuffixBag` is re-exported and documented as public.

**Fix:** Use `assert!` in all profiles or make the API fallible/return a
`Result`; fix the `try_assign` return doc; make `from_vec` assert or document
the prefix behavior. Add a release test.

---

### SUF-04 — `assign_init` contract and `assign_from_key` ownership are too weak for their actual requirements

**Severity:** Medium (latent) · **Confidence:** Likely/Speculative  
**Legacy IDs:** `BUG-SUF-05`, `BUG-SUF-06`, `SIM-INS-05`

**Location:** `src/policy.rs:465-470`, `src/leaf_trait.rs:535-539`
(`assign_init` documented only as “hold the lock”), `src/policy/suffix_ops.rs:156-164`
(uses growing `bag.assign`), `src/leaf15.rs:2395-2399`
(`assign_from_key` discards the retire pointer and drains with an empty
permutation), `src/tree/generic/layer.rs:48-86` (sets permutation after both
assignments).

`assign_init` may reallocate a published bag; the actual requirement is
“unpublished or exclusively owned”, not merely “locked”. Current call sites are
unpublished (split destination, fresh layer leaf), so this is latent. In the
layer path, `assign_from_key` stores `KSUF` and calls `assign_ksuf` while the
leaf's permutation is still empty; `drain_to_external` copies no inline
siblings, and the returned old-cell pointer is discarded. Today the external
pointer is null on a fresh layer leaf, so it is benign, but it is one refactor
away from a leak or an external bag that permanently lacks inline-resident
siblings.

**Fix:** Tighten `assign_init` to “node must be unpublished (or exclusively
owned)” and add a debug assertion. Use `assign_ksuf_init` for layer entries
(slots 0/1) and handle/retire the returned pointer, or make the init path own
retirement. Add a layer test with two 200-byte suffixes under
`small-suffix-capacity`.

---

### SUF-05 — `drain_core` headroom subtraction can underflow if `copy_count == WIDTH`

**Severity:** Low (defensive) · **Confidence:** Speculative (current callers
cannot produce it)  
**Legacy IDs:** `BUG-SUF-07`

**Location:** `src/suffix/inline.rs:249-255`.

`copy_count` is capped at `WIDTH` by `if copy_count < WIDTH`. If a caller
invokes `drain_to_external(perm, new_slot, ...)` with a 15-entry permutation
that does not contain `new_slot`, all 15 entries are copied and
`(WIDTH - (copy_count + 1))` underflows. In debug this panics; in release it
wraps and the new bag is under-capacity, then grows via `extend_from_slice`.

Current callers prevent it: a 15-active-suffix leaf splits before assignment,
and a free `new_slot` implies `perm.size() < 15`. A future caller (e.g.
reassigning a suffix on a full leaf) could hit it.

**Fix:** Use `WIDTH.saturating_sub(copy_count + 1)` and a debug assertion that
`copy_count < WIDTH` when room is expected.

---

## 10. Batch and public API

Proposal 4 follow-up: the generic batch conversion still leaked previously
boxed outputs if a later key failed validation. The new drop-counter regression
in `batch_insert_tests` failed with 2 of 3 values dropped before the unified
value-owning pipeline and now passes. All batch APIs validate owning entries
before storage conversion; deferred suffix retirement also runs on unwind.
The strengthened `failures/batch-api --bin leak3` allocation probe measures
6 → 3 allocations for 15 preconstructed, long-key write-through updates.
These changes preserve public batch signatures and duplicate input ordering.


### BATCH-01 — Public mutable `BatchEntry.key` with a stale cached ikey corrupts leaf routing

**Severity:** High · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-API-05`

**Location:** `src/tree/generic/batch.rs:61-69` (`pub key`, private `ikey`),
`:98-104` (cache construction), `:784` (sort by cached ikey), `:398,544`
(`try_insert_at` rebuilds the real key), `:973-975` (run bound check uses the
cached value).

`BatchEntry.key` is public and mutable; `ikey` is cached at construction.
`insert_batch_entries` sorts by the cached `ikey`, and the run loop checks only
the cached value against the next leaf's bound, while insertion builds the real
`Key` from `entry.key`. Mutating `entry.key` after construction leaves the
cache stale, so a key with actual ikey above the leaf bound can be inserted into
a lower leaf.

**Repro:**

```bash
cd failures/batch-api
cargo run --quiet
```

The `mutate_ikey` case:

```text
mutate: inserted=1 updated=1 total=2
  get(0x00ff..) = None (expected Some(1111))
  get([0;8]) = None
```

The subsystem's exact fixture saw `get(MAX)=None`, `len=8`, `scan_len=7`: the
key is lost and `len` and an ordered scan disagree.

**Impact:** Safe code can violate the B-link/leaf lower-bound invariant, lose
keys, and produce inconsistent `len`/scan results. This is not an obscure
misuse: `key` is a documented public field and normalization after construction
is plausible.

**Fix:** Make `key` private with a constructor/replacement method that recomputes
`ikey`. At the start of `insert_batch_entries`, recompute/validate every
`ikey` (O(n) 8-byte reads) and sort by `(recomputed_ikey, key, sequence)`.

**Test gap:** No test mutates `BatchEntry.key`; add a regression or a
compile-fail test after making fields private.

---

### BATCH-02 — Write-through `insert_batch_with_guard` leaks one heap Box per updated entry

**Severity:** High · **Confidence:** Confirmed (counting allocator; Miri leak)  
**Legacy IDs:** `BUG-API-03`, `BUG-LEAF-07`

**Location:** `src/tree/generic/batch.rs:705-727` (returns
`BatchInsertResult<P::Output>`, maps old `P::Value` through `P::into_output`),
`src/policy.rs:776-780` (`BoxPolicy::into_output` allocates), `ValuePtr` has no
`Drop`.

For write-through policies the value path computes old values as `P::Value`,
then converts every one to a freshly allocated `Box` behind a `ValuePtr`.
Nothing retires or frees those boxes; the tree already updated in place.

**Repro:**

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin batch_write_through_leak
```

Miri reports a leaked allocation from `batch.rs:724`. The subsystem's measured
run (1,000 keys × 100 rounds = 100,000 updates) observed **98,923 live
allocations / 743,040 bytes** after dropping the guard and tree. Controls
`insert_batch` and `MassTree15Inline` showed no growth; scaling is linear in
updates.

**Impact:** Unbounded memory leak and an extra allocation per update,
defeating the recommended `MassTree15<u64>` write-through path.

**Fix:** Return owned `P::Value` from all batch APIs. Add a public
`insert_batch_values_with_guard` returning `BatchInsertResult<P::Value>` (the
private implementation exists), and deprecate/limit the `P::Output`-returning
guard variant for write-through. Alternatively retire the boxes with the
caller's guard and give `BatchInsertResult` a guard lifetime, but owned values
are simpler. Never manufacture a `ValuePtr` that no owner can free.

**Test gap:** No leak test for `insert_batch_with_guard` on a write-through
policy. Add the Miri test and a counting-allocator regression.

---

### BATCH-03 — Dropping a `BatchEntry` leaks; a panic mid-batch partially applies and leaks remaining outputs

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-API-06`

**Location:** `src/tree/generic/batch.rs:77-82` (`BatchEntry::new` allocates a
Box immediately), no `Drop` for `BatchEntry`, `:773-787`
(`insert_batch_entries` consumes the Vec), `src/key.rs:43` (length panic).

1. Dropping 1,000 `BatchEntry<BoxPolicy<String>>` values leaks 2,000 live
   allocations (one Box + one String buffer each); the user value destructor
   never runs.
2. A key longer than `MAX_KEY_LENGTH` panics during `Key::new` after earlier
   entries have already been inserted. Example: `insert_batch(vec![
   (vec![], "applied"), (vec![b'a';257], "too-long")])` catches the panic but
   leaves `len()==1` and `get(&[]) == Some("applied")`. The 257-byte entry's
   allocated output and later entries' outputs are never retired.
3. The same leak occurs if `insert_batch_entries` panics at `verify_guard`.

**Impact:** Leaks on error/drop paths and user-visible partial mutations. The
architecture's “all entries applied or panic” summary is false.

**Fix:** Store `Option<P::Value>` in `BatchEntry`, implement a normal `Drop`,
and convert to `P::Output` only at insertion. Validate all key lengths before
any mutation (or in `BatchEntry::new`) so invalid-key panics happen before the
tree is touched. Document that internal panics are not atomic.

---

### BATCH-04 — Unstable sort by only the first 8 bytes makes duplicate-key batches nondeterministic

**Severity:** High (API semantics) · **Confidence:** Confirmed  
**Legacy IDs:** `BUG-API-04`

**Location:** `src/tree/generic/batch.rs:741` (`insert_batch_with_guard`),
`:764` (`insert_batch_values_with_guard`), `:784` (`insert_batch_entries`);
`ikey` cached at `:98-104`.

All three paths use `sort_unstable_by_key(...ikey)`. Equal-ikey entries are
reordered by pdqsort for realistic sizes, so the input order of duplicates is
not preserved. “First occurrence inserts, later ones update” does not mean
“last input value wins”. The existing test with four elements passes only
because insertion sort does not reorder equal keys.

**Repro:**

```bash
cd failures/batch-api
cargo run --quiet
```

The `dup_order` case: 400 duplicates of `"a"` interleaved with 400 distinct
`"b<i>"` keys. Observed:

```text
dup: inserted=401 updated=399 final_a=209 expected_last=399 old_len=399
  ** last-wins violated **
old_values match 0..398: false
```

`inserted + updated == 800` is correct and old values are consistent with the
actual processing order, but the final `"a"` value and `old_values` are not
input-order deterministic.

**Impact:** Final values and old values change with batch size or Rust sort
implementation. The repo's semantic test is not robust.

**Fix:** Sort by `(ikey, key_bytes, original_index)` with the original index as
a tiebreaker (unstable sort is fine), or use a stable sort. This preserves
last-wins and clusters same-ikey short/suffix variants. If last-wins is not
intended, change the test and docs explicitly.

**Test gap:** No duplicate test large enough to trigger unstable reordering;
add 400+ duplicates interleaved with other ikeys.

---

### BATCH-05 — `mimalloc` feature installs a library-wide `#[global_allocator]`

**Severity:** Medium · **Confidence:** Confirmed (compile repro)  
**Legacy IDs:** `BUG-API-08`

**Location:** `src/lib.rs:71-73`.

```rust
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

Cargo features are additive. A dependency must not commandeer the final
binary's global allocator. A downstream binary that defines its own allocator
fails to compile when `masstree/mimalloc` is enabled:

```text
error: the `#[global_allocator]` in this crate conflicts with global allocator in: masstree
```

**Repro:**

```bash
cd failures/batch-api/mimalloc-repro
cargo check
```

The repo's own memory benches work around this by defining the allocator in the
binary (`benches/memory_total.rs:13-31`, `memory_dhat.rs:15-31`), confirming the
hazard.

**Fix:** Remove `#[global_allocator]` from the library. Put it in a small
benchmark/example binary or document “enable only from a binary that does not set
its own allocator.” If preserved, make it a non-default, binary-only feature.

**Test gap:** Add a CI check that builds a downstream binary with its own
allocator and `masstree/mimalloc`.

---

### BATCH-06 — Dead `failed` bookkeeping and misleading batch docs

**Severity:** Low · **Confidence:** Confirmed (grep)  
**Legacy IDs:** `SIM-API-01`, `SIM-API-03`

`BatchInsertResult::failed` is documented as “entries that failed and need
individual retry”, but `record_failure` is never called; the fallback panics.
`failed` is always 0, `all_succeeded()` is always true, and tests asserting
`inserted + failed == N` cannot catch a dropped entry. `insert_batch_entries`
claims to support “retrying failed entries from a previous batch”, but entries
are consumed and no failed list is returned.

Other public-surface mismatches:

- `InsertError` has five variants but only `SplitFailed` is constructed; every
  public wrapper panics on every variant.
- `RemoveEntryResult` is public in a private module and not re-exported.
- `BatchInsertResult<'s, O>` docs describe one output type, but
  `insert_batch` returns `P::Value` while `insert_batch_with_guard` returns
  `P::Output`.
- `Extend` says “skip failed inserts” but inserts panic.
- `FromIterator` says it panics on allocation failure; infallible allocation
  aborts.
- `Entry` says “use insert_with_guard for atomic read-modify-write”; a single
  insert is not an atomic RMW.

**Fix:** Remove `failed`/`record_failure` or make failures non-panicking and
populate it; update tests to assert `inserted + updated == input_len` and verify
every key; re-export `RemoveEntryResult`; collapse or hide `InsertError`;
correct the docs. This is low risk but a breaking public-surface cleanup.

---

## 11. Tests, tooling, and benchmarks

The library has a large test suite (610 library tests, 20 leak tests, 12 Loom
tests, 14 doc tests at this commit), but several gates are green while
exercising nothing, and the published benchmark methodology is not controlled.
All tooling repros are in `failures/tooling/`.

### TOOL-01 — `--no-default-features` cannot build all targets; `print_tree` is not feature-gated

**Severity:** High · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-01`, `BUG-API-09`

`debug-print` is a default feature (`Cargo.toml:22`); `Display`/`print_tree`
live under it (`src/tree/generic/print.rs:431-465`). `examples/print_tree.rs`
uses `println!("{tree}")` at lines 17, 29, 41, 54, 68 but has no
`required-features` entry.

**Repro:**

```bash
cd failures/tooling
./01-no-default-features.sh
```

Output: exit 101, five `Display` errors; `cargo test --no-default-features
--no-run` fails identically. Library/tests alone are clean, which hides the
defect from a tests-only matrix.

**Fix:** Add to `Cargo.toml`:

```toml
[[example]]
name = "print_tree"
path = "examples/print_tree.rs"
required-features = ["debug-print"]
```

or gate the example body. Add `cargo check --all-targets --no-default-features`
to CI.

---

### TOOL-02 — `just test-shuttle` is a no-op, and `test-all` depends on it

**Severity:** High · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-02`

`Cargo.toml:63` declares `shuttle`, `:93` declares `cfg(shuttle)`, and
`justfile:152-154` runs `cargo test --lib tree::shuttle_tests`. No such module
exists. `justfile:145` includes it in `test-all`.

**Repro:**

```bash
cd failures/tooling
./02-shuttle-noop.sh
```

```text
running 0 tests
test result: ok. 0 passed; 0 failed; ... 610 filtered out
```

**Impact:** No linearizability/systematic concurrency checking despite the
dependency, cfg, recipe, and “all tests” claim. The dev-dependency adds build
time to every test build.

**Fix:** Delete the dependency/cfg/recipe/doc references, or add a real
`tests/shuttle_tests.rs` (`shuttle::check_random` over insert/get/remove) and
run it with `RUSTFLAGS="--cfg shuttle"`.

---

### TOOL-03 — `miri-leaks` does not run `tests/leak_tests.rs`

**Severity:** High · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-03`

`justfile:211-213` runs `cargo +nightly miri test --lib`.
`tests/leak_tests.rs` is an integration target, so `--lib` cannot compile or
run it.

**Repro:**

```bash
cd failures/tooling
./03-miri-leaks-target.sh
```

Observed: 0 “leak” tests in the lib list; 20 in `--test leak_tests`. The 20
tests pass normally, but normal execution does not perform Miri leak checking.
The only Miri recipes also use `--lib` (TOOL-07).

**Fix:** Run
`MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check"
cargo +nightly miri test --test leak_tests` in `miri-leaks`; optionally keep a
separate `miri-lib` recipe. Install the Miri component in CI.

---

### TOOL-04 — `just lint-strict` (and `check-all`) fails on current stable Clippy

**Severity:** High · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-04`

`justfile:161-162` runs `cargo clippy --all-targets --all-features -- -D
warnings`; `:179` makes `check-all` depend on it.

**Repro:**

```bash
cd failures/tooling
./04-clippy-strict.sh
```

```text
error: used `debug_assert!` to check that a value is not empty
  --> src/tree/coalesce.rs:774
error: used `assert!` with an equality comparison
  --> src/nodeversion/unit_tests.rs:435
error: used `assert!` with an equality comparison
  --> src/nodeversion/unit_tests.rs:460
error: avoid using `collect()` when not needed
  --> src/tree/range/iterator/unit_tests.rs:127
error: could not compile `masstree` (lib test) due to 4 previous errors
```

`cargo fmt --all -- --check` passes. The failure is version-sensitive (new
nursery lints on 1.99); there is no pinned toolchain/CI.

**Fix:** Apply the suggested assertion/iterator changes or explicitly allow the
four lints, and run strict Clippy under a pinned toolchain in CI.

---

### TOOL-05 — C++ benchmark recipe is unrunnable and its log redirection is wrong

**Severity:** High · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-05`

`justfile:32-42` runs `./reference/mttest`; the checkout has no `reference/`
directory. `justfile:70` makes `bench-all` start with `cpp-bench`, so the full
comparison workflow cannot start. `scripts/run_cpp_bench.py:353` expects
`reference/build/mttest`. The redirect in `justfile:40` (and `:57`) is `2>&1 >
cpp_benches/$test.txt`: shell evaluates left-to-right, so stderr stays on the
terminal and stdout goes to the file. The Python driver reads `result.stderr`
and says the JSON is on stderr, so generated files lose the JSON lines.

**Repro:**

```bash
cd failures/tooling
./05-cpp-bench.sh
bash -c 'echo OUT; echo ERR >&2' 2>&1 > /tmp/redir.txt   # ERR stays on terminal
cat /tmp/redir.txt                                      # only OUT
```

**Fix:** Write `> file 2>&1`; align the path (`reference/build/mttest` or a
configurable `MTTEST`); document how to obtain/build the C++ reference; or
remove the recipes and point to `scripts/run_cpp_bench.py --rust-only`.

---

### TOOL-06 — Test tracing writes into the repository and is not ignored

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-06`

`tests/common/mod.rs:114-127` creates `logs/` and appends
`logs/masstree.jsonl` by default (`TracingConfig::default` at 80-88).
`common::init_tracing()` is called by 20+ tests in `stress_tests.rs`,
`concurrent_correctness.rs`, and `concurrent_regression.rs`. `.gitignore:1-2`
only ignores `target/` and `.git/`; the audit workspace has an untracked
`logs/` directory.

**Impact:** Test runs dirty the repository, can fail in a read-only checkout,
and all test processes append to one shared log. The test harness also emits
console output by default.

**Fix:** Default to a temp directory (or `tempfile`), make file/console logging
opt-in, and add `logs/` to `.gitignore` as a backstop.

---

### TOOL-07 — `just test-all` does not run the tests its comment promises

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-07`

`justfile:144-145` says “Run ALL possible tests (unit, doc, integration, loom,
shuttle, miri)” and depends only on `test test-loom test-shuttle miri-strict`.
`test` is a nextest run (`:73-87`), which covers unit and integration targets,
but no recipe invokes `test-doc`, `miri-leaks`, or `miri-strict-int`.
`test-shuttle` is a verified zero-test no-op (TOOL-02).

**Impact:** A green `test-all` can coexist with broken doc tests and an
unexecuted entire leak-test target.

**Fix:** Add `test-doc`, `miri-leaks`, and `miri-strict-int` as dependencies;
remove or implement `test-shuttle`; rename the recipe if “all” is not intended.

---

### TOOL-08 — Miri exclusions are materially broader than the architecture states

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-08`

`ARCHITECTURE.md:1768-1771` mentions only `nodeversion::concurrent_tests`
being excluded. The source has 16 `#[cfg_attr(miri, ignore)]` attributes and 15
`#[cfg(not(miri))]` attributes across six files. Examples include most of
`src/tree/remove/unit_tests.rs` (lines 836, 908, 954, 1001, 1164, 1452, 2244,
2323, 2448, 2601, 2723, 2821), `src/shard_counter/unit_tests.rs`, and
`src/tree/unit_tests.rs`. Many are concurrency/removal tests, precisely the
risky paths.

**Fix:** Enumerate the exclusions and rationale in §11.4, reduce them by making
tests Miri-compatible (shorter loops, deterministic barriers), and report the
Miri test count in CI.

---

### TOOL-09 — Several “parity”/stress tests assert nothing or cannot fail

**Severity:** High (coverage) · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-09`

- `tests/cpp/rwsmall24.rs:20-56,59-115`: all reads are `let _ = tree.get...`;
  no assertion.
- `tests/cpp/rwsep24.rs:21-58`: same discard pattern in the first blocks.
- `tests/cpp/w1.rs:32-48`, `wscale.rs:22-35`, `uscale.rs:36-47`: write loops
  with results discarded; only panics/deadlocks fail.
- `tests/cpp/splitremove.rs:98-158`, `tests/cpp/wd2.rs:113-184`: sleep-driven
  churn followed only by joins; no invariant.
- `tests/deleted_leaf_handling.rs:185-193`: when `total_success == 0`,
  `miss_ratio` is set to 0.0, so a run with zero successful reads passes
  `miss_ratio < 0.5`.
- `tests/concurrent_correctness.rs:1021-1124` (`#[ignore]`): `errors` is
  initialized and never incremented; the periodic verification does nothing
  when the key is missing; `assert_eq!(total_errors, 0)` is vacuous.
- `tests/stress_adaptive.rs:70`: treats `>100` operations in five seconds as
  success, which a severely stalled implementation can satisfy.

**Repro:**

```bash
cd failures/tooling
./07-vacuous-tests.sh
```

**Impact:** A no-op or partially broken tree can pass substantial parts of the
“C++ parity” and stress suite. These tests create false confidence in the most
failure-prone paths.

**Fix:** Add final-state oracles (or a `BTreeMap` model), assert successful-read
counts and value equality, fail when no successful reads occur, and delete or
strengthen the ignored vacuous test. Add mutation testing or a real differential
harness against the C++ binary.

---

### TOOL-10 — Proptest key order is not reproducible, and two scan properties are one-sided

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-10`

`tests/tree_proptests.rs:52-55,103-104` build key vectors with
`prop::collection::hash_set(...).prop_map(|set| set.into_iter().collect())`.
A `std::HashSet` uses `RandomState`; iteration order differs across processes
even with the same proptest seed. A saved failing case can replay a different
insertion order, changing which split/layer path is exercised.

`tests/tree_proptests.rs:758-790` (`scan_respects_bounds`) only checks that
returned keys are inside the range; returning zero keys passes. `:793-831`
(`scan_prefix_filters_correctly`) only checks that returned keys start with the
prefix; it never asserts completeness, so an empty scan passes. The latter also
truncates generated keys to eight bytes, creating overwrites that make an exact
count harder to state but still oracle-computable.

**Fix:** Use `BTreeSet`/sorted vectors (or sort after `HashSet`), and add exact
expected-count or `BTreeMap` comparisons for bound/prefix scans. Add batch,
entry, and remove+scan model tests.

---

### TOOL-11 — `tracing` documentation promises console output that the implementation disables

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-11`, `SIM-API-05`

`src/lib.rs:78` says console output is enabled unless `MASSTREE_LOG_CONSOLE=0`;
`:86` documents the variable; `:109` hardcodes
`let console_enabled: bool = false;`. A grep of `src/` finds no `tracing::`
event calls; only the initializer exists. The test helper in `tests/common`
uses its own subscriber and does not exercise this library path.

**Impact:** Enabling `tracing` adds `tracing-subscriber`/`tracing-appender`
dependencies but produces no events and cannot produce console output; the
documented environment variable is a no-op.

**Fix:** Honor `MASSTREE_LOG_CONSOLE` and add real spans/events, or remove the
console claim and the feature/deps if it is intentionally inert.

---

### TOOL-12 — Release-mode test runs silently lose four debug-only tests and symbols

**Severity:** Low · **Confidence:** Confirmed  
**Legacy ID:** `BUG-TOOL-12`

The test headers recommend release-mode concurrency runs. With a fresh target:

```text
cargo test --lib -- --list              # 610 tests
cargo test --release --lib -- --list    # 606 tests
```

The four missing tests are
`internode::unit_tests::test_invariant_unsorted_keys`,
`tree::unit_tests::test_wrong_guard_panics_on_insert`/`_iter`/`_remove`; they
are compiled behind `#[cfg(debug_assertions)]`. The release binary is stripped,
so failure backtraces are lost.

**Fix:** Run both debug and release jobs in CI; remove `debug_assertions`
gating from tests where meaningful or document the four omissions; build
release tests with debug info.

---

### TOOL-13 — No CI configuration exists

**Severity:** Medium · **Confidence:** Confirmed  
**Legacy ID:** new in merged audit

There is no `.github/` directory. Formatting, strict Clippy, the feature
matrix, Loom, Miri, leak tests, benchmark smoke, and the `--no-default-features`
build are unenforced. This is why TOOL-01/03/04/07 can persist.

**Fix:** Add a minimal CI with:

- pinned stable + nightly/MSRV jobs;
- `fmt --check`, strict Clippy, `check --all-targets` for default/no-default/all;
- `test --lib`, integration tests, doc tests, leak tests;
- Loom under `--cfg loom`;
- Miri for `tests/leak_tests.rs` and the `failures/leaf-values` Miri bins;
- benchmark smoke with `mttest --check --limit` and nonzero exit on check errors.

### TOOL-14 — Debug printing omits promoted-layer internodes and right siblings

**Severity:** Low · **Confidence:** Confirmed (deterministic output test)

**Location:** `src/tree/generic/print.rs:293-308`.

The printer recursively prints the raw layer-slot pointer. After promotion,
that pointer is the original left leaf, so it prints only that leaf; it neither
climbs to the promoted root nor walks the leaf's siblings. The tree header's
count can say 40 while the sublayer table shows only its first 15 values.
This can mislead diagnosis of routing, cleanup, and missing-key failures.

**Repro:**

```bash
cd failures/leaf-values/public
cargo test --test print_promoted -- --nocapture
```

Forty keys under `prefix00` have distinct `VALUE_000_END` through
`VALUE_039_END` values. The printed text contains the first marker and omits
the last. Point lookup still finds the last key.

**Fix:** Resolve a layer-slot pointer to its current root under the guard
before printing the whole layer. Keep diagnostic traversal consistent with
point reads/teardown, and retain a promoted-layer output test. Formatting also
needs the value/suffix snapshot discipline discussed in SAFE-02/RNG-07;
`print_leaf` currently checks its header version before formatting slots.

---

### PERF-TOOL-01 — Divan concurrent benchmarks time warmup, thread spawn/join, and barriers

`benches/bench_utils.rs:878-927` creates a `thread::scope`, spawns workers, runs
each worker's internal warmup, waits on two barriers, executes the measured
loop, and joins all threads inside the divan closure. `ThreadContext` has no
clock; `ctx.begin_measurement()` only synchronizes. The divan counter counts
only workload operations, while the elapsed time includes up to 500 warmup
reads per thread, thread creation/join, and barrier waits. This biases Mitem/s
and scaling curves differently by thread count. Fix: pre-spawn a worker pool
outside the timed closure, use a start/stop barrier whose release is the only
timed work, exclude warmup, and count only measured operations. pbench/custom
timing with explicit phase timestamps is the alternative.

### PERF-TOOL-02 — README tables mix separate runs/binaries and “MT vs Best” is not best-of

`README.md:19` sources masstree from `runs/run208_read_write.txt` and competitors
from a different run, `runs/run184_read_write_reworked.txt`. The same run184
file reports masstree at 34.8 Mitem/s at 12T, while the README uses 53.16 from
run208. High-impact has the same issue (run206 vs run188: 44.53 vs 38.64). The
“MT vs Best” column is also not the minimum competitor: README row 07 reports
2.72 while the best listed competitor gives 6.26; high-impact row 04 reports
3.85 while skipmap gives 4.63; row 03 reports 2.99 while indexset gives 5.14.
The prose claims “2×–3.7×” while the tables show 1.28×/1.65×/1.84× and tail
losses (get 1T p99.9 501 vs 411 ns, insert 8T p99.9 80.3 µs vs 5.70 µs, scan
p50 972 vs 862 ns). Fix: rerun all structures in one binary/run/host, use that
run for every column, compute `min(competitors)`, and update the prose.

### PERF-TOOL-03 — README tables have no reproducible command/metadata path

`justfile:18-19` only provides `cargo bench --bench mttest --features mimalloc`;
there is no recipe for `concurrent_read_write*`, `high_impact*`,
`range_masstree15`, `storage_gap`, or `tail_latency`, yet README:19-105 sources
tables from those binaries. The raw `runs/` files have no CPU model, rustc
version, commit, feature set, allocator, pinning, or seed.
`scripts/compare_benches.py:85-89` and `rust_benches/parse_results.py` hardcode
“12 threads, 10s” regardless of input. Fix: add named recipes per table, emit a
metadata header, teach parsers to read it, and check in a manifest for
`runs/`.

### PERF-TOOL-04 — `mttest --check` reports correctness errors but always exits 0

`benches/mttest.rs:156-159` documents `--check` and says “May report errors due
to known tree bugs at high scale.” Check paths increment local `check_errors`
and only `eprintln!` (`:592-601,715-728,837-845,954-963,1365-1374`). There is no
aggregation into `RunResults`, no `process::exit(1)`, and no assertion; the
default is `check: false`. Fix: accumulate errors, include them in JSON, print
a summary, exit nonzero on any error, and run a small `--check --limit` job in
CI. Remove or track the stale “known tree bugs” note.

---

## 12. Simplification opportunities

These are confirmed dead code, duplicated logic, or misleading API surface.
They are not release blockers, but several are maintenance traps because a
dead variant has different semantics from the live one.

| ID | Area | Location / issue | Recommendation | Effort |
|---|---|---|---|---|
| SIM-01 | Insert errors | `InsertError` has five variants; only `SplitFailed` is constructed; all public wrappers panic | Make split infallible with a fallback; delete or hide the error type | S |
| SIM-02 | Insert search | ~120 duplicated lines between single-layer and generic search (`search.rs:59-133/323-346`, `145-181/263-294`, `185-226/298-319`); binary-search branches dead at `WIDTH=15` | One `const SINGLE_LAYER: bool` implementation; delete dead binary search or bind threshold to `WIDTH` | M |
| SIM-03 | Split API | `InsertTarget`/`insert_target` computed and discarded; `split_into_preallocated`/`split_all_to_right_preallocated` unused | Delete or use one split wrapper; remove dead trait declarations | XS |
| SIM-04 | Remove/GC | Four queues (`chains`/`deferred_chains`, `sublayers`/`deferred_sublayers`), dead `MODSTATE_REMOVE`, unused `kp`, unused allocator/guard args, `Route = Vec<u64>` | Two `ikey_bound`-based queues, `ArrayVec<u64,32>`, delete dead state/args | M |
| SIM-05 | `NodeVersion` | `has_split_no_compiler_fence` alias, `stable_yield`, `try_lock_for`, `acquire_raw`, `is_split_locked`, `Clone` of a live lock word, two `mark_nonroot` paths | Keep one predicate/setter; delete test-only surface | S |
| SIM-06 | Leaf traits | `TreeLeafNode` is ~470 lines of forwarding; most `ValueArray`/`LeafPolicy` methods are dead; `BoxValueArray::is_layer` is semantically wrong; `move_slot` is an ownership trap | Reduce the trait to associated types/constants actually used; delete dead methods; make mutations `unsafe` or private | M |
| SIM-07 | Suffix API | `EmbeddedSuffix` and `SidecarSuffix` duplicate the same algorithm; `SuffixBag::{try_assign, try_assign_in_place, compact, Clone, suffix_equals, suffix_compare}` have no production callers; `suffix_count` RMWs are paid for unused paths | Factor a shared `SuffixCore`; delete or hide dead public API; remove count or gate under debug | M |
| SIM-08 | Batch | `BatchEntry` and `BatchValueEntry`, two `try_insert_at` implementations, two `fallback_insert_at`, `BatchOps` overlapping `InsertStrategy`, `insert_into_empty_leaf_batch` duplicating `insert_into_empty_leaf` (~150 lines) | One value-owning entry type and one locked-leaf insertion helper | M |
| SIM-09 | Range | `ScanState`/`ScanStateBack`, `ForwardFlags`/`ReverseFlags`, `find.rs`/`find_rev.rs` only hold result enums; `InitialPositionResult::Up` never constructed; `debug_last_emitted_key_back` never written; `LayerContext` stores only root/leaf | Collapse states/flags; make reverse lazy; store parent slot in `LayerContext`; delete dead variants | M |
| SIM-10 | Allocation | Dead `alloc_internode_direct`, `InternodeNode::{new,new_root,new_for_split}`, accessor families, `find_insert_position`, `BatchedRetire`; `debug_assert_invariants` only in tests | Delete dead API; keep one in-place initializer; wire invariants into split/teardown debug paths | S |
| SIM-11 | Key ordering | `Key::compare` and `CursorKey` implement the same ikey/keylenx ordering twice | Shared helper/trait; differential test over short/suffix/layer kinds | S |
| SIM-12 | Sentinel tests | `src/inline/sentinel.rs` lacks `#[cfg(test)] mod unit_tests;` although the file exists; `InlineBits` has no tests | Wire the module; test NaN payloads, bool/unit, signed ints, tuples, sentinel identity | S |
| SIM-13 | Batch stats/docs | `estimated_leaves = unique_ikeys` (15× upper bound), `locality_factor` never reaches 1.0, `sequential_keys` can exceed `MAX_KEY_LENGTH`, `zip_into_entries`/`from_iter` thin wrappers, `BTreeSet` analyze | Rename/recompute estimates, validate keys, remove wrappers or use sort+dedup | S |
| SIM-14 | Benchmarks | Three concurrent-read/write and three high-impact binaries duplicate 13/9 groups; four independent JSON parsers disagree on streams/metrics | Shared parameterized bench module and one parser | M |
| SIM-15 | Dependencies | `parking_lot`, `shuttle`, `crossbeam-epoch`, `criterion` declared but unused; `criterion` pulls a large tree | Remove unused dev-deps and `cfg(shuttle)` if no Shuttle tests | S |
| SIM-16 | Tracing | No events/spans; `init_tracing` is the only use; console flag hardcoded off | Remove feature/deps or add real instrumentation | S |

**Estimated reduction:** roughly 1,000–1,500 lines of production/test code and
several dependency edges, without changing behavior. The highest-value cleanup
is SIM-04/06/07 because the dead APIs carry the wrong ownership contracts that
contribute to SAFE-03/SAFE-04/SUF-04.

---

## 13. Performance opportunities

Implementation follow-up (v1.0.0): PERF-02 rebuilds now size from the complete
inline + external live set before copying, reserving at least one live set or
512 bytes of append slack. PERF-01 no longer preallocates a throwaway buffer on
inserts into a leaf that already has external storage. The strengthened
`failures/suffix --bin probe` failed before this change (42 allocations / 71,584
bytes for 20 churn cycles) and passes afterward (6 / 20,560). Value-only updates
remain allocation-free. The historical table below is retained as baseline
context. PERF-09 now reserves split-destination suffix capacity from all moved
suffixes plus the new right-side entry (3 → 0 buffer reallocations in the probe).
PERF-11 writes only the 60-byte sentinel header after leaf zeroing (516 fewer
redundant bytes with the default capacity). `SuffixStore::ensure_external` was
replaced by the exclusively-owned `reserve_external` contract. Proposal 3 / SIM-07
also removes the duplicated Vec owner and cached suffix count: the actual header
is now 144 bytes instead of 168. Removed public APIs: `SuffixBag::{try_assign,
try_assign_in_place, compact_with_permuter, suffix_equals, suffix_compare, count}`,
`Clone for SuffixBag`, and `PermutationProvider`. This is an intentional breaking
low-level API change, approved for the proposal implementation. The removed
compaction implementation was the source of SUF-02; its reproducer now asserts
empty/nonempty suffix preservation through the remaining assignment/growth path.
A real-bag concurrent append/read test runs under Miri; the existing Loom suffix
model still does not model this implementation.

The following are measured or mechanically confirmed; none is a correctness
fix. Values are from the audit host and should be re-measured after correctness
changes.

Proposal 4 update (PERF-06): `FromIterator` and `Extend` now share the sorted
batch pipeline, using bounded chunks that borrow the input keys and discard
unused old-value results. The `failures/batch-api` release `bulk` probe measured
20,000 random keys at 1.92 ms versus 2.12 ms for guarded single-key insertion,
and ascending keys at 0.87 ms versus 1.57 ms (local seven-round medians).
Chunk-boundary duplicate ordering and stable key snapshots are regression tested.
PERF-05: Entry now records the final lookup leaf and attempts a validated local
mutation before falling back to root traversal. The focused regression observed
one redundant descent before the change and zero afterward, for both occupied
and vacant entries. Hints include the layer offset and check full key identity,
slot membership, deletion, and leaf bounds under lock. The stale-vacant update
probe also improves from 3 allocations to 0, closing that PERF-03 bypass.
Native/Miri cover real Entry invalidation and reclamation. A Loom test models the
hint version/lock protocol; its simplified NodeVersion was synchronized with
production's dirty-bit checks and fences. PERF-03 is now closed: guarded
single-key insertion also selects write-through before boxing, reducing 260
long-key boxed primitive updates from 264 allocations to 0. Its return type is
now `Option<ValueRef<'g, P::Value>>` (a user-approved semver break), with guarded
borrows for non-Clone boxed values and owned primitive/inline snapshots.

PERF-07 follow-up: forward and reverse scan contexts now cache the resolved
promoted root. The shared guarded resolver also serves point operations,
printing and GC. A deterministic regression checks that a cached-root retry
takes zero parent hops, including multi-level promotion; it failed on the
original left-leaf cache. Native and Miri tests pass. Parent slots are now also
refreshed opportunistically by point reads, inserts and removals. A try-lock
with no sublayer lock held revalidates slot membership/key/tag/pointer before
publication. Tests verify zero subsequent parent hops, progress when the lock is
unavailable, and reclamation after promotion for both policies. Existing GC is
already root-kind agnostic. RNG-09's duplicate scan descent is now removed: point operations and scans share
`tree::traversal::{reach_leaf, advance_by_bound}` with the same deleted-node,
parent validation and split-sibling handling. Range comparison and focused Miri
cover both directions and promoted multi-height layers.

OCC-04 cached-root follow-up: traversal from an internode root retired by whole
sublayer GC repeatedly retried the same deleted subtree. The focused
`reach_rejects_retired_promoted_root` test reached its sentinel and timed out
after 15 seconds (exit 124). Traversal now reports a restart, with explicit
handling in point readers, write B-link advance, and GC route lookup. The guard
keeps the retired node alive while the old pointer is rejected.

SIM-02 / PERF-18 follow-up: single-layer and suffix insertion now instantiate
one three-way unrolled leaf search. Unreachable binary-search branches are
removed at WIDTH=15. `stable()` and its contended loop use Acquire loads, matching
`try_stable()`; the Loom model mirrors this change. Dirty-publication and
post-read validation fences remain in place.

| ID | Area | Issue | Recommendation | Expected |
|---|---|---|---|---|
| PERF-01 | Suffix insert | `maybe_pre_allocate_suffix` allocates `WIDTH * suffix_len` on every suffix insert, including `Found` updates, conflicts, retries, and splits that drop it | Move to the `NotFound`/`find_usable_slot` path, size from actual suffix usage, pass into split | 1 allocation / 600 B per long-key update removed |
| PERF-02 | Suffix churn | Drain rebuild sizes `15L`, then merges old external suffixes until `used == capacity`; every remove+reinsert reallocates | Add slack (2× used or ≥1 KB) before publishing | 2.1 alloc / 3.5 KB per churn insert → amortized |
| PERF-03 | Write-through API | `insert_with_guard`, Entry, `Extend`, and `insert_batch_entries` bypass the write-through value path | Dispatch on `CAN_WRITE_THROUGH` or add owned-value variants | Removes Box alloc/retire per update |
| PERF-04 | Batch updates | `perm.size() >= WIDTH` precheck breaks the batch before search, so updates on full leaves fall back per key | Remove/move the precheck; `find_usable_slot` already returns `NeedsSplit` | Bulk updates regain one lock per run |
| PERF-05 | Entry API | `entry_with_guard` does a `get_with_guard` descent, then every mutation does a second descent | Single-descent entry classification with fallback on retry | ~2× fewer descents |
| PERF-06 | Bulk construction | `FromIterator`/`Extend` insert one-by-one under one guard | Collect and call `insert_batch_with_guard` for sized/owned inputs | Large win for 10k+ keys |
| PERF-07 | Promoted scans | Layer-root promotion makes scans walk `next`/`stable_reverse` leaf-by-leaf; `reach_leaf_for_scan` ignores the parent | Share stable-leaf climb (RNG-04) | O(N) → O(log N) reposition |
| PERF-08 | Scan batch | Redundant `is_value_empty_relaxed` load before `load_value` for every emitted slot | Remove the precheck; `load_value` already returns `None` | 1–2 atomics/slot |
| PERF-09 | Split suffix | `prepare_split_leaf` allocates an empty 128-B bag, then grows to several KB under the split lock | Add `reserve_external(bytes)` sized from source usage; pass insert prealloc | Fewer reallocs/copies under lock |
| PERF-10 | Suffix prefetch | Embedded prefetch is a no-op; sidecar prefetches metadata/external, not data; leaf prefetch skips suffix field | Prefetch suffix metadata and first two data lines (slot-aware for sidecar) | Hides 1–2 cold misses |
| PERF-11 | Leaf init | `init_at` zeroes 1152 B, then `init_at_zero` rewrites 576 B of inline bag | Header-only sentinel writes; rely on prior zeroing | ~44% fewer init stores (default) |
| PERF-12 | Internode layout | 320 B = 5 cache lines; line 4 holds only `child[15]` + 56 B padding | Evaluate `WIDTH=14` (256 B) or a packed child array | −20% internode memory at −6.7% fanout |
| PERF-13 | Routing | Scalar upper-bound scans up to 15 atomic ikeys with branches | Branchless `sum(key[i] <= target)`/binary search; SIMD only after measurement | Shorter descent |
| PERF-14 | Internode writes | Every child store uses Release; one Release publication would suffice | Relaxed stores inside the locked region + one Release publication | aarch64 store cost |
| PERF-15 | Node pool | 128-block eager refill; 64-B-aligned `posix_memalign` for every block; up to 512 blocks retained per thread | Start small/double; slab per class; measure RSS/latency | First-alloc latency/RSS |
| PERF-16 | Maintenance | `flush(); refresh(); process_all()` flushes before the work that retires, so retirements wait for the next call | `process_all(); flush(); refresh()` | Lower retained peak |
| PERF-17 | Multi-layer remove | `Route = Vec<u64>` allocates per operation, up to 32 entries | `ArrayVec<u64,32>` or inline route | Removes alloc on layered remove |
| PERF-18 | `stable()` | Relaxed load + Acquire fence on the hot descent path; `try_stable` already uses Acquire load | Use `load(Acquire)` | 1 instruction vs `ldr+dmb` on aarch64 |
| PERF-19 | Inline values | Tags and payloads in separate arrays; random slot touches two cache lines | Interleave `{tag, bits}` per slot, keep 240 B | One cache line per point access |
| PERF-20 | Bench timing | Divan closure includes warmup/spawn/join/barriers | Pre-spawn pool, barrier release as timed region | Correct Mitem/s |
| PERF-21 | README tables | Mixed runs/binary/value types; “MT vs Best” not minimum | One controlled run; compute min competitor | Reproducible claims |
| PERF-22 | Bench metadata | No command/CPU/commit/feature/allocator/seed metadata; parsers hardcode 12T/10s | Emit/record metadata; parser reads it | Auditable results |
| PERF-23 | `mttest --check` | Errors printed but exit 0; default unchecked | Aggregate + nonzero exit; CI smoke | Correctness gate |
| PERF-24 | Batch result | `with_capacity(len/4)` allocates unused old-values capacity for all-insert batches | Start empty/small; grow on update | Saves 25% batch-size capacity |
| PERF-25 | Layer creation | Holds parent leaf lock across up to 31 twig allocations, value clone, suffix writes | Speculative layer build outside lock, retry on version change | Lower conflict stall |

The largest measurable wins in the current workloads are PERF-01/02 (suffix
allocation storms), PERF-03/04 (write-through/batch update bypass), PERF-07
(promoted-layer scan walks), and PERF-20/21 (benchmark methodology). The
suffix measurements are in `failures/suffix`; the benchmark methodology is
documented in PERF-TOOL-01..04.

---

## 14. Recommended fix order

The findings have dependencies. This order minimizes rework and gets the
highest-risk issues behind tests first.

### Phase 0 — Make the public surface honest (major version)

1. **SAFE-01/02:** lifetime-tie `ValuePtr`/`ScanEntry`/Entry/batch outputs, or
   return owned/`ValueRef` values. For write-through, never expose a plain
   `&V`.
2. **SAFE-03:** make `BatchEntry` own `P::Value`; remove `from_output`; make
   fields private.
3. **SAFE-04:** make `TreeAllocator` unsafe/sealed, hide `debug_root_ptr`, and
   fix or hide the safe `LeafNode15`/`InternodeNode` ownership APIs.
4. **SAFE-05:** replace the `CAN_WRITE_THROUGH` arithmetic gate with a sealed
   marker.
5. **SAFE-06:** atomics for external suffix metadata and a non-aliasing writer
   API.
6. **BATCH-05:** remove the library-wide `#[global_allocator]`.

Until these land, downstream users should avoid `MassTree15<BoxPolicy>` output
APIs, custom allocators, `mimalloc`, and direct `LeafNode15`/`InternodeNode`
use.

### Phase 1 — Repair OCC and remove identity

1. **OCC-01:** unconditional `mark_insert` on every new-slot/batch/reuse path;
   Release publication.
2. **OCC-02:** `has_changed > LOCK_BIT` (or a dedicated structural bit) and
   null-layer retries.
3. **OCC-03:** fix marker/validation fences for aarch64.
   **OCC-06:** audit reader pointer loads against seize's protected-load contract.
4. **REM-01:** full key identity and post-lock version/permutation validation.
5. **REM-02:** B-link advance and deleted-leaf handling in remove.
6. **RNG-07:** validate complete snapshots in forward/reverse scan paths.

Add the deterministic layer-read test, the same-ikey remove test, and the
lost-remove split test before changing the protocol.

### Phase 2 — Stabilize maintenance and reclamation

1. **GC-01:** replace raw chain pointers with ikey-bound re-traversal or a
   generation token; check deleted under lock.
2. **GC-02:** root-aware whole-sublayer GC, with a counting-allocator test.
3. **GC-03/OCC-04/GC-04:** preserve non-root live unary levels; retire only
   after unlink/reparent; never abandon; deleted-child check in reach.
   **GC-05:** defer each failed entry to the next call instead of consuming
   its entire retry budget immediately.
4. **ALLOC-01/02:** `POOL.try_with` fallbacks and unwind-safe reclamation.
5. **ALLOC-03/INS-04:** keep allocations and user encoding out of the
   non-abandonable phase, or provide transactional unwind handling.

### Phase 3 — Fix insert/split and range semantics

1. **INS-01/02:** strict split boundaries and an infallible split-point
   fallback; stop panicking.
2. **INS-03:** deleted+null rightmost leaf restart.
3. **RNG-01..05:** full-key bounds, layer-root climb, values-only bounds.
4. **RNG-06/08:** DoubleEnded exhaustion and empty-rightmost reverse.
5. **RNG-09:** share one stable-leaf traversal with point reads.
6. **RNG-10:** validate before callbacks and preserve full-key progress across retries.

### Phase 4 — Performance and workflow

1. **PERF-01/02/03/04:** suffix preallocation/drain slack and write-through
   dispatch; batch full-leaf precheck.
2. **PERF-07/09/10/11:** promoted scan climb, split reservation, suffix
   prefetch, leaf init.
3. **TOOL-01..14:** CI, feature gate, Miri target, Clippy, test oracles,
   benchmark methodology, and complete promoted-layer diagnostics.
4. **SIM-01..16:** delete dead code after correctness tests exist, not before.

---

## 15. Test-gap summary

The current suite is broad but misses the exact sequences that break. Add these
regressions before or with the fixes:

| Area | Required tests |
|---|---|
| OCC | version changes after new-key insert; snapshot + `mark_insert` + cleared value must be `has_changed`; layer creation vs `get`/`get_ref`/scan; aarch64 litmus (or documented x86-only gate) |
| Remove | deterministic staged same-ikey wrong-key removal; public 8-thread same-ikey churn; lost remove during a 15-key split; retry-limit test with a hot leaf |
| Coalesce/GC | two immediate entries for one leaf; `process_coalesce_batch(1)` + epoch advance; counting-allocator promoted-layer drain; cascade with `MAX_PARENT_RETRIES` forced; `with_batch_size(1)` teardown |
| Insert/split | prepend split between equal ikeys; both equal-run split panics; randomized short/NUL oracle; insert/coalesce rightmost livelock with watchdog |
| Range | the four NUL keys; `abcdefgh`/suffix bounds; 40-key promoted layer starts 0/15/39; reverse layer-chunk end; empty rightmost `last`/rev; DoubleEnded drain-then-other; values-only empty/far/max-prefix; remove+reinsert scan ordering |
| Suffix | occupied-slot drain shadow; empty-suffix compaction; >u16 release behavior; real-bag Miri/TSan; allocation-count regression for updates/churn |
| Batch/API | alias + double retire; Drop leak; partial-panic; duplicate 400+; mutated `BatchEntry.key`; custom allocator teardown; padded write-through under Miri |
| Tooling | CI matrix, zero-test detection (Shuttle), `--all-targets --no-default-features`, `miri --test leak_tests`, `mttest --check` nonzero, benchmark metadata |

Several of these are already in `failures/`; they should be promoted into CI,
not left as an external corpus.

---

## Appendix A — Canonical finding index

**Counts:** 10 Critical, 29 High, 16 Medium, 9 Low (64 code/tooling findings:
58 original IDs plus 6 follow-up IDs), plus 16 simplification groups and 25
performance opportunities. OCC-05's index severity is corrected to Low to
match its finding body.

| ID | Severity | Confidence | Short description |
|---|---|---|---|
| SAFE-01 | Critical | Confirmed | Guard-scoped outputs escape the guard/tree; safe UAF |
| SAFE-02 | Critical | Confirmed | `ValuePtr::deref` races write-through atomic updates |
| SAFE-03 | Critical | Confirmed | `BatchEntry` aliases tree-owned values; double-free/corruption |
| SAFE-04 | High | Confirmed | Safe allocator/leaf/internode APIs permit invalid dereferences/double-free/type confusion |
| SAFE-05 | High | Confirmed | Write-through admits padding/non-Copy/reference representations; invalid integer-atomic reads |
| SAFE-06 | High | Confirmed | External `SuffixBag` metadata/Vec races and `&mut` aliasing |
| OCC-01 | Critical | Confirmed | New-key inserts do not mark/bump the version |
| OCC-02 | Critical | Confirmed | `has_changed` ignores `INSERTING`; layer conversion false miss/type confusion |
| OCC-03 | High | Likely | Weak-memory dirty/validation fences are reversed |
| OCC-04 | High | Likely | Descent into a deleted/empty internode has no escape |
| OCC-05 | Low | Likely | Separator redirects bypass the version protocol |
| OCC-06 | High | Likely | Reader pointer loads bypass seize protected-load ordering |
| INS-01 | Critical | Confirmed | Prepend split makes a key equal to the separator unreachable |
| INS-02 | High | Confirmed | Equal-ikey split point returns `None`; insert panics |
| INS-03 | High | Confirmed | Insert livelocks on a coalesced rightmost leaf |
| INS-04 | High | Confirmed | Safe inline encoder panic during split leaves an unusable root |
| REM-01 | Critical | Confirmed | Remove deletes a different same-ikey key |
| REM-02 | High | Confirmed | Remove loses a present key during a split |
| REM-03 | Medium | Confirmed code | Retry cap does not bound the inner remove/reach loops |
| GC-01 | Critical | Confirmed | Duplicate raw chain entries: maintenance hang/UAF |
| GC-02 | High | Confirmed | Promoted/multi-leaf sublayers are never reclaimed |
| GC-03 | High | Likely | Cascade retires before unlink; stale child/UAF |
| GC-04 | High | Confirmed | Non-root unary collapse breaks height; later leaf split hangs |
| GC-05 | Medium | Confirmed | Deferred chain consumes all cleanup retries in one call |
| ALLOC-01 | Critical | Confirmed | TLS pool teardown aborts the process |
| ALLOC-02 | High | Confirmed | Panicking value Drop corrupts seize batch state |
| ALLOC-03 | Medium | Likely | Allocator panic after `mark_split` leaves a split-locked sibling |
| PRIM-01 | Low | Likely | `ShardedCounter::load` debug assertion on valid snapshot |
| PRIM-02 | Low | Confirmed | `Permuter<WIDTH<15>` raw bits above the mask |
| RNG-01 | Critical | Confirmed | Forward lower bound is ikey-only; emits out-of-range keys |
| RNG-02 | High | Confirmed | Reverse end-bound descends a layer above the bound |
| RNG-03 | High | Confirmed | Reverse upper bound is not full-key |
| RNG-04 | High | Confirmed | Promoted-layer bounded scan skips the start key |
| RNG-05 | High | Confirmed | Values-only scans ignore bounds |
| RNG-06 | Medium | Confirmed | DoubleEnded re-emits after one-side exhaustion |
| RNG-07 | High | Confirmed | Forward multi-layer `find_next` lacks a version check |
| RNG-08 | Medium | Confirmed | Empty rightmost leaf breaks reverse/`last()` |
| RNG-09 | Medium | Likely | Scan traversal is a weaker B-link implementation |
| RNG-10 | High | Confirmed | Callbacks observe stale batches before version validation |
| SUF-01 | Medium | Likely | Drain leaves stale inline metadata shadowing new external suffix |
| SUF-02 | Low | Confirmed | Empty-suffix compaction debug panic |
| SUF-03 | Low | Confirmed | Public `SuffixBag` length checks are debug-only |
| SUF-04 | Medium | Likely/Speculative | `assign_init`/`assign_from_key` ownership contract gaps |
| SUF-05 | Low | Speculative | `drain_core` headroom underflow |
| BATCH-01 | High | Confirmed | Mutable `BatchEntry.key` + stale cached ikey corrupts routing |
| BATCH-02 | High | Confirmed | Write-through batch old values leak one Box per update |
| BATCH-03 | Medium | Confirmed | `BatchEntry` drop/partial-panic leaks |
| BATCH-04 | High | Confirmed | Unstable ikey-only sort makes duplicate batches nondeterministic |
| BATCH-05 | Medium | Confirmed | `mimalloc` installs a library-wide global allocator |
| BATCH-06 | Low | Confirmed | Dead `failed` bookkeeping and misleading docs |
| TOOL-01 | High | Confirmed | `--no-default-features --all-targets` fails on `print_tree` |
| TOOL-02 | High | Confirmed | `test-shuttle` is a zero-test no-op |
| TOOL-03 | High | Confirmed | `miri-leaks` runs no leak tests |
| TOOL-04 | High | Confirmed | Strict Clippy fails |
| TOOL-05 | High | Confirmed | C++ benchmark recipe is missing/wrong redirect |
| TOOL-06 | Medium | Confirmed | Tests write untracked `logs/` |
| TOOL-07 | Medium | Confirmed | `test-all` does not run doc/leak/Miri |
| TOOL-08 | Medium | Confirmed | Miri exclusions are much broader than documented |
| TOOL-09 | High | Confirmed | Vacuous parity/stress tests |
| TOOL-10 | Medium | Confirmed | Non-reproducible proptests; one-sided scan properties |
| TOOL-11 | Medium | Confirmed | `tracing` console docs are false; no events |
| TOOL-12 | Low | Confirmed | Release tests silently lose four tests and symbols |
| TOOL-13 | Medium | Confirmed | No CI configuration exists |
| TOOL-14 | Low | Confirmed | Printer omits promoted-layer root and right siblings |

## Appendix B — Reproduction quick reference

| Finding group | Command |
|---|---|
| SAFE-01 | `cd failures/leaf-values/public && cargo +nightly miri run --bin valueptr_escape` |
| SAFE-02 | `... && cargo +nightly miri run --bin valueptr_race` |
| SAFE-03 | `cd failures/batch-api && cargo run --bin alias` / `--bin doublefree` |
| SAFE-04 | `cd failures/leaf-values/public && cargo +nightly miri run --bin move_value_same_leaf`; `cd failures/batch-api && cargo run --bin teardown_safe` |
| SAFE-05 | `cd failures/leaf-values/public && cargo +nightly miri run --bin padded_write_through` |
| SAFE-05 follow-up | `... && cargo +nightly miri run --bin reference_write_through` |
| SAFE-04 correction | `... && cargo test --release --test internode_bounds` (2 passing controls) |
| SAFE-06 | source-level; `failures/suffix` allocation probe plus the audit interleaving |
| OCC-01/02 | `cd failures/concurrency/vercheck && cargo run`; `cd ../typeconf && cargo run --release`; `cd failures/remove-coalesce && cargo test --release --test version_probe` |
| OCC-03 | no hardware repro on x86; aarch64 litmus required |
| OCC-06 | pinned seize 0.5.1 protected-load/retirement contract; no hardware repro |
| OCC-04/GC-03 | `cd failures/internode-alloc && cargo test --release --test stuck -- --nocapture` |
| INS-01/02 | `cd failures/insert-split && cargo test --test prepend -- --nocapture`; `cargo test --test equal_run -- --nocapture` |
| INS-03 | `cd failures/insert-split && timeout 120s cargo run --release --bin livelock2` |
| INS-04 | `cd failures/insert-split && cargo build --bin inline_encode_panic`, then `timeout 3s "$CARGO_TARGET_DIR/debug/inline_encode_panic"` (124) |
| REM-01 | `cd failures/remove-coalesce && cargo test --release --test wrong_key -- --nocapture` |
| REM-02 | `... && cargo test --release --test lost_remove -- --nocapture` |
| GC-01 | `... && cargo test --release --test double_unlink_deterministic -- --nocapture` (exit 42) |
| GC-02 | `... && cargo test --release --test sublayer_leak -- --nocapture`; `cd failures/independent && cargo run --quiet --bin gc` |
| GC-04 | `... && cargo test --test cascade_height`; build `cascade_reinsert`, then run it with `timeout 4s` (124) |
| GC-05 | `... && cargo test --test requeue_budget -- --nocapture` |
| ALLOC-01 | `cd failures/internode-alloc && cargo test --release --test tls_abort -- --nocapture` |
| ALLOC-02 | `... && cargo test --release --test panic_value -- --nocapture` |
| PRIM-01 | `cd failures/primitives/shard && cargo run --quiet` |
| PRIM-02 | `cd failures/primitives/nodeversion && cargo run --quiet --bin perm_raw` |
| RNG-01 | `cd failures/range && cargo run --quiet --bin nul` / `--bin nulpfx` |
| RNG-02 | `... && cargo run --quiet --bin layerend` |
| RNG-03 | `... && cargo run --quiet --bin nul` / `--bin apis` |
| RNG-04 | `... && cargo run --quiet --bin promoted` |
| RNG-05 | `... && cargo run --quiet --bin vals` / `--bin valfar` / `--bin maxprefix` |
| RNG-06 | `... && cargo run --quiet --bin doubleended` |
| RNG-07 | `... && timeout 30s cargo run --quiet --bin race` |
| RNG-08 | `... && cargo run --quiet --bin empty_right` |
| RNG-10 | `... && cargo test --test batch_callback -- --nocapture` (6 failures) |
| SUF-02 | `cd failures/suffix && cargo run --quiet --bin compact_empty` |
| BATCH-01/04 | `cd failures/batch-api && cargo run --quiet` |
| BATCH-02 | `cd failures/leaf-values/public && cargo +nightly miri run --bin batch_write_through_leak` |
| BATCH-05 | `cd failures/batch-api/mimalloc-repro && cargo check` (expected conflict) |
| TOOL-01..07 | `cd failures/tooling && ./run-all.sh` |
| TOOL-14 | `cd failures/leaf-values/public && cargo test --test print_promoted` |

## Appendix C — Environment, limitations, and assumptions

- Host: x86_64 Linux, AMD Ryzen 5 3600 (6C/12T), Rust 1.99.1. The weak-memory
  findings (OCC-03) are not observable on x86 TSO and were not tested on
  aarch64 hardware; they are labelled Likely.
- The repository had pre-existing uncommitted changes before the audit
  (`Cargo.toml` modified, `rust-toolchain.toml` deleted, untracked `logs/`).
  During the session `README.md` also acquired an unrelated formatting and
  disclaimer edit. None of these were treated as audit findings or modified by
  the audit. The audit added only `ARCHITECTURE.md`, `SystemAudit.md`,
  `AGENTS.md`, and `failures/`.
- The full integration suite was not run in one shot. Targeted suites,
  repeated stress tests, Loom, and focused Miri tests were run. See each
  finding for its exact command.
- Miri was available for this audit but is not in the checked-in toolchain.
  Some concurrency tests are excluded from Miri by the repository (TOOL-08);
  those exclusions were not treated as passing coverage.
- Stress tests are stochastic. Counts quoted in this report are from a single
  representative run and are evidence of reachability, not a performance
  benchmark.
- Line numbers refer to `eb8bee2`; the repository was read-only during the
  audit.
- The `failures/` corpus is intentionally allowed to fail. It is not part of
  the library build and should be run in a disposable process.

---

## Appendix D — Files reviewed

The full read list is in the ten subsystem reports under
`/tmp/opencode/audit/*.md` for this session. The merged review covered:

`src/{lib,tree,leaf15,leaf_trait,internode,key,nodeversion,permuter,ksearch,link,ordering,prefetch,hints,shard_counter,retirement,alloc_trait,alloc15,node_pool,suffix,policy}.rs`
and their submodules; `src/policy/{box_values,inline_values,embedded_suffix,sidecar_suffix,suffix_ops}.rs`;
`src/suffix/{inline,compact,clone,sidecar}.rs`; `src/tree/generic/{search,insert,insert_strategy,layer,split,batch,entry,optimistic_reads,optimistic_reads/get_guarded,print,batch_utils}.rs`;
`src/tree/split/{propagation,propagation_context,parent_locking,root_creation}.rs`;
`src/tree/range/{api,api_ref,batch_common,cursor_key,find,find_rev,forward_ctx,reverse_ctx,helper,scan_state,traversal,iterator,iterator/scan_entry}.rs`;
`src/internode/{accessors,layout}.rs`; `src/leaf15/layout.rs`;
all `tests/*.rs`, `tests/cpp/*.rs`, `benches/*.rs`, `examples/*.rs`,
`scripts/*.py`, `Cargo.toml`, `justfile`, `README.md`, and `ARCHITECTURE.md`.

**End of audit.**



