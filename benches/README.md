# Benchmarks

Comparative benchmarks for MassTree24 against other concurrent ordered maps.

## Structures Compared

| Structure | Crate | Type | Notes |
|-----------|-------|------|-------|
| `masstree24` | this crate | Trie of B+trees | Values stored as `Arc<V>` |
| `masstree24_inline` | this crate | Trie of B+trees | Values stored inline (Copy types) |
| `skipmap` | `crossbeam-skiplist` | Lock-free skip list | Implicit epoch reclamation |
| `indexset` | `indexset` | Concurrent B-tree | Fine-grained locking |
| `tree_index` | `scc` | Lock-free B+tree | No native upsert (requires remove+insert) |
| `dashmap` | `dashmap` | Sharded hash map | O(1) lookup, no ordering |

## Running Benchmarks

```bash
# Full suite (recommended: use mimalloc for consistent allocation)
cargo bench --bench concurrent_maps24 --features mimalloc

# Specific group
cargo bench --bench concurrent_maps24 -- 01_concurrent_writes

# Multiple groups
cargo bench --bench concurrent_maps24 -- "10a_read\|10b_read"
```

---

## Key Generation (`bench_utils.rs`)

### `keys<K>(n)` — Default Unique Keys

```rust
// Each 8-byte chunk: i * MULTIPLIER[chunk_index]
// Chunk 0: i * 1 (sequential)
// Chunk 1: i * 0x517cc1b727220a95
// Chunk 2: i * 0x9e3779b97f4a7c15
// ...
```

**Properties:**

- Every key is unique (no prefix collisions)
- No special advantage for trie structures
- Sequential chunk 0 is neutral: B-trees handle sorted data well

**Used by:** Groups 01-11 (main benchmarks)

### `keys_shared_prefix<K>(n, buckets)` — Forced Prefix Collisions

```rust
// First 8 bytes: i % buckets (forces collisions)
// Remaining chunks: unique per key
```

**Properties:**

- Forces MassTree layering (multiple keys share first ikey)
- Tests trie prefix sharing efficiency
- Explicitly labeled as "MassTree-favoring"

**Used by:** Groups 12, 13, 14 (prefix-sharing benchmarks)

### `keys_shared_prefix_chunks<K>(n, prefix_chunks, buckets)` — Aggressive Sharing

```rust
// First N chunks: each drawn from small bucket space
// Remaining chunks: unique
```

**Properties:**

- Simulates hierarchical keys (e.g., `/users/alice/profile/setting1`)
- Maximum trie advantage scenario

**Used by:** Group 14 (aggressive prefix benchmarks)

### `uniform_indices(n, count, seed)` — Random Access Pattern

```rust
// LCG-based pseudo-random indices into key array
```

**Properties:**

- Eliminates sequential access locality benefits
- Fair comparison against hash maps

---

## Benchmark Groups — Detailed Description

### Group 01: `concurrent_writes_disjoint`

**What it tests:** Concurrent insert throughput with no contention.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(n)` — unique 8-byte keys |
| Ops/thread | 50,000 |
| Threads | 1-6 |
| Pattern | Each thread writes to disjoint range: `[t*50k, (t+1)*50k)` |

**Why this distribution:** Sequential keys within each thread's range. This is actually *favorable to B-trees* (sorted insertion is their best case). MassTree treats it as normal insertion.

**Fairness:** Neutral to slightly favoring competitors.

---

### Group 02: `concurrent_writes_contention`

**What it tests:** Update throughput on a shared, pre-populated key space.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(1000)` — 1,000 unique keys |
| Ops/thread | 10,000 |
| Threads | 1-6 |
| Pattern | All threads update same 1,000 keys randomly |

**Why this distribution:** Small key space forces lock contention. Tests how well each structure handles concurrent updates to overlapping keys.

**Note:** TreeIndex lacks native upsert; uses `remove_sync + insert_sync` which adds overhead. This reflects real API limitations.

**Fairness:** Fair — tests actual library capabilities.

---

### Group 03: `single_threaded_insert`

**What it tests:** Baseline single-threaded insert throughput.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(100_000)` — sequential |
| Pattern | Insert all keys in order |

**Why this distribution:** Measures raw insertion speed without concurrency overhead. Sequential insertion is B-tree's best case.

**Fairness:** Slightly favors B-trees.

---

### Group 04: `read_after_write`

**What it tests:** Read throughput on a pre-populated tree.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(50_000)` |
| Threads | 1-6 |
| Pattern | Each thread reads its own range sequentially |

**Why this distribution:** Sequential reads within thread's range tests cache locality. All structures benefit similarly.

---

### Group 05: `get_by_key_size`

**What it tests:** Single-threaded point lookup with varying key sizes.

| Key Sizes | 8B, 16B, 24B, 32B |
|-----------|-------------------|
| Keys | 10,000 per size |
| Lookups | 1,000 random indices |

**Why this distribution:** Tests how key length affects lookup. MassTree's trie handles longer keys by layering; B-trees compare full keys.

**Expected behavior:** MassTree should show consistent performance across sizes due to trie structure. B-trees may slow down with longer keys.

---

### Group 06: `insert_by_key_size`

**What it tests:** Single-threaded insert with varying key sizes.

| Key Sizes | 8B, 16B, 24B, 32B |
|-----------|-------------------|
| Keys | 1,000 per size |

**Why this distribution:** Same rationale as Group 05 but for inserts.

---

### Group 07: `concurrent_reads_scaling`

**What it tests:** Read throughput scaling with thread count.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(10_000_000)` — 10M keys |
| Ops/thread | 50,000 |
| Pattern | Random access with prime offset per thread |

**Why this distribution:** Large dataset ensures cache misses. Tests true concurrent read scalability.

---

### Group 08: `concurrent_reads_long_keys`

**What it tests:** Same as Group 07 but with 32-byte keys.

| Parameter | Value |
|-----------|-------|
| Keys | `keys<32>(10_000_000)` |

**Why this distribution:** Tests multi-layer traversal in MassTree (32B = 4 layers). Other structures compare full 32-byte keys.

---

### Group 09: `mixed_uniform`

**What it tests:** Mixed read/write workload (90% reads, 10% writes).

| Parameter | Value |
|-----------|-------|
| Keys | `keys<8>(100_000)` |
| Threads | 1-6 |
| Write ratio | 10% |

**Why this distribution:** Realistic workload simulation. Tests how structures handle concurrent reads during writes.

---

### Group 10a: `read_scaling_8B`

**What it tests:** Read throughput scaling with 8-byte keys.

Reports items/second for throughput comparison.

---

### Group 10b: `read_scaling_32B`

**What it tests:** Read throughput scaling with 32-byte keys.

Same as 10a but tests multi-layer MassTree performance.

---

### Group 10c: `write_scaling_32B`

**What it tests:** Write throughput scaling with 32-byte keys.

| Parameter | Value |
|-----------|-------|
| Keys | Pre-populated with N/2 keys |
| Pattern | Mixed inserts and updates |

---

### Group 11: `single_hot_key`

**What it tests:** Maximum contention — all threads access ONE key.

| Parameter | Value |
|-----------|-------|
| Threads | 2, 4, 8, 16, 32 |
| Pattern | 90% reads, 10% writes to single hot key |

**Why this distribution:** Stress test for lock contention. Shows how each structure handles pathological contention.

---

### Group 11a/11b: `random_read_8B` / `random_read_32B`

**What it tests:** True random access reads (includes DashMap).

| Parameter | Value |
|-----------|-------|
| Keys | 1M keys |
| Pattern | Pre-computed random indices per thread |

**Why this distribution:** Eliminates any sequential access locality. Fair comparison against hash maps which have O(1) lookup but no ordering.

---

### Group 12a/12b: `string_values_read` / `string_values_write`

**What it tests:** Performance with `String` values (non-Copy types).

| Parameter | Value |
|-----------|-------|
| Keys | 16-byte keys |
| Values | Heap-allocated strings |

**Why this distribution:** Realistic use case. MassTree stores values as `Arc<V>`, so this tests the Arc overhead.

---

### Group 12 (legacy): `get_by_key_size_shared_prefix`

**What it tests:** Lookups with shared prefixes.

| Parameter | Value |
|-----------|-------|
| Keys | `keys_shared_prefix<K>(10_000, 256)` |

**Why this distribution:** **Explicitly MassTree-favoring.** First 8 bytes drawn from 256 buckets, forcing trie layer sharing.

**Label:** Clearly named `_shared_prefix` to indicate bias.

---

### Group 13: `concurrent_reads_long_keys_shared_prefix`

**What it tests:** Concurrent reads with shared prefix 32-byte keys.

**Why this distribution:** **MassTree-favoring.** Tests trie prefix sharing under concurrency.

---

### Group 14a/14b: `aggressive_shared_prefix_read` / `_write`

**What it tests:** Extreme prefix sharing scenario.

| Parameter | Value |
|-----------|-------|
| Keys | `keys_shared_prefix_chunks<32>(n, 3, 16)` |
| Pattern | First 24 bytes shared across 16 buckets |

**Why this distribution:** **Maximum MassTree advantage.** Simulates hierarchical keys like file paths or user namespaces. Layers 0-2 are heavily reused.

**Label:** Clearly named to indicate this tests MassTree's theoretical strength.

---

## Fairness Summary

### Main Benchmarks (Groups 01-11)

| Aspect | Fair? | Notes |
|--------|-------|-------|
| Key distribution | ✅ | Unique keys, no prefix sharing |
| Sequential insertion | ⚠️ | Slightly favors B-trees (their best case) |
| Guard handling | ✅ | All structures amortize per-thread |
| TreeIndex upsert | ✅ | Uses remove+insert (library limitation) |
| Random access | ✅ | Pre-computed indices, same for all |

### Prefix-Sharing Benchmarks (Groups 12-14)

| Aspect | Fair? | Notes |
|--------|-------|-------|
| Key distribution | ⚠️ | Explicitly MassTree-favoring |
| Labeling | ✅ | Names include `shared_prefix` |
| Purpose | ✅ | Tests trie design advantage |

### Overall Assessment

**No hidden unfair advantages.** Main benchmarks use neutral key distributions. Prefix-sharing benchmarks are clearly labeled and exist to demonstrate where MassTree's trie design excels (hierarchical keys, shared prefixes).

---

## Interpreting Results

### What "Faster" Means

- **Lower time** = better (benchmarks measure total time for all ops)
- **Higher items/sec** = better (throughput benchmarks)

### Expected Patterns

| Structure | Strength | Weakness |
|-----------|----------|----------|
| MassTree | Long keys, shared prefixes, concurrent reads | Single-threaded overhead |
| SkipMap | Simple API, good scaling | Higher memory, slower than B-trees |
| IndexSet | Single-threaded performance | Coarse locking under contention |
| TreeIndex | Lock-free reads | No native upsert, complex API |
| DashMap | O(1) point lookup | No ordering, no range scans |

### Variance Considerations

- Run multiple times; concurrent benchmarks have inherent variance
- Use `--features mimalloc` for consistent allocation behavior
- Watch for outliers in `slowest` column
