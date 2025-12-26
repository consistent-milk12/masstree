# JAQ Guide for MassTree Log Analysis

`jaq` is a Rust implementation of `jq` - a powerful JSON query language. This guide covers analyzing MassTree tracing logs.

## Log Format

MassTree logs are newline-delimited JSON (NDJSON):

```json
{
  "timestamp": "2025-12-25T18:10:59.624868Z",
  "level": "WARN",
  "fields": {
    "message": "FAR_LANDING: reach_leaf returned leaf far from target",
    "ikey": "0000000000061a83",
    "leaf_ptr": "0x2f026032a00",
    "gap": 299820
  },
  "target": "masstree::tree::generic",
  "filename": "src/tree/generic.rs",
  "line_number": 1389,
  "threadId": "ThreadId(7)"
}
```

## Basic Syntax

```bash
# Process file (use -s to slurp into array)
jaq -s 'EXPRESSION' logs/lock_contention.json

# Pretty print
jaq -s '.' logs/file.json

# Compact output (no pretty print)
jaq -c 'EXPRESSION' logs/file.json
```

## Essential Operations

### Counting

```bash
# Total entries
jaq -s 'length' logs/lock_contention.json

# Count by message type
jaq -s 'group_by(.fields.message) | map({msg: .[0].fields.message, count: length})' logs/file.json

# Count with sorting
jaq -s 'group_by(.fields.message) | map({msg: .[0].fields.message, count: length}) | sort_by(.count) | reverse' logs/file.json
```

### Filtering

```bash
# Filter by level
jaq -s '[.[] | select(.level == "ERROR")]' logs/file.json

# Filter by message contains
jaq -s '[.[] | select(.fields.message | contains("BLINK"))]' logs/file.json

# Filter by numeric field
jaq -s '[.[] | select(.fields.gap > 100000)]' logs/file.json

# Multiple conditions (AND)
jaq -s '[.[] | select(.level == "ERROR" and .fields.gap > 50000)]' logs/file.json

# Multiple conditions (OR)
jaq -s '[.[] | select(.level == "ERROR" or .level == "WARN")]' logs/file.json

# Negation
jaq -s '[.[] | select(.fields.message | contains("SLOW") | not)]' logs/file.json
```

### Limiting Results

```bash
# First N entries
jaq -s '.[0:10]' logs/file.json

# Last N entries
jaq -s '.[-10:]' logs/file.json

# First N after filter
jaq -s '[.[] | select(.level == "ERROR")] | .[0:5]' logs/file.json
```

### Field Extraction

```bash
# Extract single field
jaq -s '[.[] | .fields.message]' logs/file.json

# Extract multiple fields
jaq -s '[.[] | {msg: .fields.message, thread: .threadId}]' logs/file.json

# Unique values
jaq -s '[.[] | .fields.message] | unique' logs/file.json

# Unique with counts
jaq -s 'group_by(.threadId) | map({thread: .[0].threadId, count: length})' logs/file.json
```

### Grouping and Aggregation

```bash
# Group by field
jaq -s 'group_by(.threadId)' logs/file.json

# Group and count
jaq -s 'group_by(.fields.ikey) | map({ikey: .[0].fields.ikey, count: length})' logs/file.json

# Group, count, and sort descending
jaq -s 'group_by(.fields.ikey) | map({ikey: .[0].fields.ikey, count: length}) | sort_by(.count) | reverse' logs/file.json

# Top N after grouping
jaq -s 'group_by(.fields.ikey) | map({ikey: .[0].fields.ikey, count: length}) | sort_by(.count) | reverse | .[0:10]' logs/file.json
```

### Numeric Operations

```bash
# Sum a field
jaq -s '[.[] | .fields.gap] | add' logs/file.json

# Average
jaq -s '[.[] | .fields.gap] | add / length' logs/file.json

# Min/Max
jaq -s '[.[] | .fields.gap] | min' logs/file.json
jaq -s '[.[] | .fields.gap] | max' logs/file.json

# Filter then aggregate
jaq -s '[.[] | select(.fields.gap != null) | .fields.gap] | {min: min, max: max, avg: (add/length)}' logs/file.json
```

### String Operations

```bash
# Starts with
jaq -s '[.[] | select(.fields.message | startswith("SLOW"))]' logs/file.json

# Ends with
jaq -s '[.[] | select(.fields.message | endswith("retries"))]' logs/file.json

# Split and access
jaq -s '[.[] | .threadId | split("(") | .[1] | split(")") | .[0]]' logs/file.json
```

---

## MassTree-Specific Queries

### Message Type Summary

```bash
# Get overview of all message types with counts
jaq -s 'group_by(.fields.message) | map({message: .[0].fields.message, count: length}) | sort_by(.count) | reverse' logs/lock_contention.json
```

### Anomaly Analysis

```bash
# All INSERT_ABORT events
jaq -s '[.[] | select(.fields.message | contains("INSERT_ABORT"))]' logs/lock_contention.json

# BLINK_LIMIT events (B-link traversal hit limit)
jaq -s '[.[] | select(.fields.message | contains("BLINK_LIMIT"))]' logs/lock_contention.json

# FAR_LANDING events (reach_leaf returned wrong leaf)
jaq -s '[.[] | select(.fields.message | contains("FAR_LANDING"))]' logs/lock_contention.json

# All ERROR level events
jaq -s '[.[] | select(.level == "ERROR")]' logs/lock_contention.json
```

### Performance Analysis

```bash
# All SLOW events
jaq -s '[.[] | select(.fields.message | startswith("SLOW"))]' logs/lock_contention.json

# SLOW_SPLIT with timing details
jaq -s '[.[] | select(.fields.message | contains("SLOW_SPLIT"))] | .[0:5]' logs/lock_contention.json

# SLOW_STABLE - version stabilization delays
jaq -s '[.[] | select(.fields.message | contains("SLOW_STABLE"))] | .[0:5]' logs/lock_contention.json

# Average split time (in microseconds)
jaq -s '[.[] | select(.fields.total_elapsed_us != null) | .fields.total_elapsed_us] | add / length' logs/lock_contention.json

# Max split time
jaq -s '[.[] | select(.fields.total_elapsed_us != null) | .fields.total_elapsed_us] | max' logs/lock_contention.json
```

### Parent Wait Analysis

```bash
# Parent wait events (NULL parent during split)
jaq -s '[.[] | select(.fields.message | contains("PARENT_WAIT"))]' logs/lock_contention.json

# Parent wait durations
jaq -s '[.[] | select(.fields.message | contains("PARENT_WAIT_END"))] | map({ptr: .fields.left_leaf_ptr, spins: .fields.spins, wait_us: .fields.wait_us})' logs/lock_contention.json

# Average parent wait time
jaq -s '[.[] | select(.fields.wait_us != null) | .fields.wait_us] | add / length' logs/lock_contention.json
```

### Thread Analysis

```bash
# Events per thread
jaq -s 'group_by(.threadId) | map({thread: .[0].threadId, count: length}) | sort_by(.count) | reverse' logs/lock_contention.json

# Errors by thread
jaq -s '[.[] | select(.level == "ERROR")] | group_by(.threadId) | map({thread: .[0].threadId, errors: length}) | sort_by(.errors) | reverse' logs/lock_contention.json

# Events for specific thread
jaq -s '[.[] | select(.threadId == "ThreadId(7)")]' logs/lock_contention.json
```

### Key (ikey) Analysis

```bash
# Unique ikeys with issues
jaq -s '[.[] | select(.fields.ikey != null) | .fields.ikey] | unique | length' logs/lock_contention.json

# Most problematic ikeys (most events)
jaq -s '[.[] | select(.fields.ikey != null)] | group_by(.fields.ikey) | map({ikey: .[0].fields.ikey, count: length}) | sort_by(.count) | reverse | .[0:10]' logs/lock_contention.json

# All events for specific ikey
jaq -s '[.[] | select(.fields.ikey == "0000000000061a83")]' logs/lock_contention.json
```

### Pointer Analysis

```bash
# Unique leaf pointers in BLINK_LIMIT
jaq -s '[.[] | select(.fields.message | contains("BLINK_LIMIT"))] | map(.fields.start_ptr) | unique' logs/lock_contention.json

# Group BLINK_LIMIT by starting pointer
jaq -s '[.[] | select(.fields.message | contains("BLINK_LIMIT"))] | group_by(.fields.start_ptr) | map({ptr: .[0].fields.start_ptr, count: length}) | sort_by(.count) | reverse' logs/lock_contention.json
```

### Timeline Analysis

```bash
# First 10 events (chronological)
jaq -s '.[0:10]' logs/lock_contention.json

# Last 10 events
jaq -s '.[-10:]' logs/lock_contention.json

# Events in time range (string comparison works for ISO timestamps)
jaq -s '[.[] | select(.timestamp > "2025-12-25T18:11:00" and .timestamp < "2025-12-25T18:11:01")]' logs/lock_contention.json

# First error
jaq -s '[.[] | select(.level == "ERROR")] | .[0]' logs/lock_contention.json

# Time between first and last event
jaq -s '{first: .[0].timestamp, last: .[-1].timestamp}' logs/lock_contention.json
```

### Gap Analysis (FAR_LANDING)

```bash
# Gap statistics
jaq -s '[.[] | select(.fields.gap != null) | .fields.gap] | {min: min, max: max, count: length}' logs/lock_contention.json

# Largest gaps
jaq -s '[.[] | select(.fields.gap != null)] | sort_by(.fields.gap) | reverse | .[0:5] | map({ikey: .fields.ikey, gap: .fields.gap, leaf: .fields.leaf_ptr})' logs/lock_contention.json
```

### Advance Count Analysis (BLINK_LIMIT)

```bash
# Advance count distribution
jaq -s '[.[] | select(.fields.advance_count != null)] | group_by(.fields.advance_count) | map({count: .[0].fields.advance_count, occurrences: length})' logs/lock_contention.json

# Events with highest advance counts
jaq -s '[.[] | select(.fields.advance_count != null)] | sort_by(.fields.advance_count) | reverse | .[0:5]' logs/lock_contention.json
```

---

## Common Patterns

### Pipeline Pattern

Build complex queries step by step:

```bash
# Step 1: Filter
jaq -s '[.[] | select(.level == "ERROR")]' logs/file.json

# Step 2: Add grouping
jaq -s '[.[] | select(.level == "ERROR")] | group_by(.fields.message)' logs/file.json

# Step 3: Add counting
jaq -s '[.[] | select(.level == "ERROR")] | group_by(.fields.message) | map({msg: .[0].fields.message, count: length})' logs/file.json

# Step 4: Sort
jaq -s '[.[] | select(.level == "ERROR")] | group_by(.fields.message) | map({msg: .[0].fields.message, count: length}) | sort_by(.count) | reverse' logs/file.json
```

### Correlation Pattern

Find related events:

```bash
# Find all events for a pointer that appeared in BLINK_LIMIT
# First, get the pointer:
jaq -s '[.[] | select(.fields.message | contains("BLINK_LIMIT"))] | .[0].fields.start_ptr' logs/file.json
# Output: "0x2f03a092e00"

# Then find all events mentioning that pointer:
jaq -s '[.[] | select(.fields.leaf_ptr == "0x2f03a092e00" or .fields.left_leaf_ptr == "0x2f03a092e00" or .fields.start_ptr == "0x2f03a092e00")]' logs/file.json
```

### Debugging Pattern

Trace a specific key's journey:

```bash
# All events for a specific ikey
jaq -s '[.[] | select(.fields.ikey == "0000000000061a83")] | sort_by(.timestamp)' logs/file.json
```

---

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Slurp to array | `jaq -s` |
| Filter | `select(.field == "value")` |
| Contains | `select(.field \| contains("text"))` |
| Array wrap | `[.[] \| ...]` |
| First N | `.[0:N]` |
| Last N | `.[-N:]` |
| Group by | `group_by(.field)` |
| Sort | `sort_by(.field)` |
| Reverse | `reverse` |
| Count | `length` |
| Sum | `add` |
| Unique | `unique` |
| Map/Transform | `map({...})` |
| Field access | `.field` or `.fields.subfield` |
| Null check | `select(.field != null)` |
| AND | `and` |
| OR | `or` |
| NOT | `\| not` |

## Generating Logs

```bash
# Run with tracing enabled
RUST_LOG=masstree=debug cargo test --features tracing TEST_NAME -- --nocapture

# Run lock_contention benchmark with tracing
RUST_LOG=masstree=warn,lock_contention=warn cargo run --release --bin lock_contention --features tracing

# Logs are written to logs/*.json
```

## Tips

1. **Always use `-s`** (slurp) for log files - it reads all lines into an array
2. **Wrap filters in `[.[] | ...]`** to maintain array output
3. **Build incrementally** - start simple, add complexity
4. **Use `.[0:5]`** to preview results before processing entire file
5. **Pipe to `less`** for large outputs: `jaq -s '...' file.json | less`
6. **Save complex queries** in shell scripts for reuse

---

## Advanced Techniques for Large Logs

### Handling Multi-GB Log Files

For logs > 1GB, avoid loading everything into memory:

```bash
# Sample first N lines only
head -10000 logs/lock_contention.json | jaq -s 'QUERY'

# Sample last N lines
tail -10000 logs/lock_contention.json | jaq -s 'QUERY'

# Sample middle section
head -50000 logs/lock_contention.json | tail -10000 | jaq -s 'QUERY'

# Count message types (streaming approach - get counts first)
head -50000 logs/lock_contention.json | jaq -s '
  [.[] | .fields.message] | group_by(.) | map({msg: .[0], count: length}) | sort_by(.count) | reverse'
```

### Multi-Stage Filtering

Filter by multiple conditions with parentheses:

```bash
# AND with contains (note parentheses)
jaq -s '[.[] | select((.fields.message | contains("DESCENT")) and (.fields.target_ikey == "0000000000030d40"))]'

# Complex OR conditions
jaq -s '[.[] | select(
  (.fields.message | contains("FAR_LANDING")) or
  (.fields.message | contains("BLINK_LIMIT")) or
  (.fields.message | contains("INSERT_ABORT"))
)]'
```

### Tracing Event Sequences for Specific Keys

```bash
# Full trace for one ikey
head -20000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.target_ikey == "0000000000030d40")]
| map({
  msg: .fields.message,
  retry: .fields.retry,
  nkeys: .fields.nkeys,
  child_idx: .fields.child_idx
})'

# Group events by ikey and list event types
head -10000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.target_ikey != null)]
| group_by(.fields.target_ikey)
| map({ikey: .[0].fields.target_ikey, events: [.[].fields.message]})'
```

### Tracking State Changes Across Retries

```bash
# Version changes between retries for FAR_LANDING
head -10000 logs/lock_contention.json | jaq -s '
[.[] | select((.fields.message | contains("INTERNODE_DESCENT")) and (.fields.target_ikey == "0000000000030d40"))]
| map({retry: .fields.retry, version: .fields.inode_version, nkeys: .fields.nkeys})'

# Check if same root returned on retry
head -10000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("FRESH_ROOT"))]
| map({ikey: .fields.target_ikey, retry: .fields.retry, same_as_start: .fields.same_as_start})
| .[0:10]'
```

### Correlating Splits with Propagations

```bash
# Count splits vs propagations
head -100000 logs/lock_contention.json | jaq -s '
{
  splits: ([.[] | select(.fields.message | contains("SPLIT_START"))] | length),
  propagate_start: ([.[] | select(.fields.message | contains("PROPAGATE_START"))] | length),
  propagate_complete: ([.[] | select(.fields.message | contains("PROPAGATE_COMPLETE"))] | length),
  delta: (
    ([.[] | select(.fields.message | contains("PROPAGATE_START"))] | length) -
    ([.[] | select(.fields.message | contains("PROPAGATE_COMPLETE"))] | length)
  )
}'
```

### Analyzing Internode Growth Over Time

```bash
# nkeys distribution in internodes
head -50000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("INTERNODE_DESCENT")) | .fields.nkeys]
| group_by(.)
| map({nkeys: .[0], count: length})
| sort_by(.nkeys)'

# Sample internode state at different times
head -50000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("INTERNODE_DESCENT"))]
| [.[0], .[100], .[500], .[-1]]
| map({timestamp: .timestamp, nkeys: .fields.nkeys, k0: .fields.k0, k_last: .fields.k_last})'
```

### Timeline Correlation Across Event Types

```bash
# Compare timing of different event types
head -50000 logs/lock_contention.json | jaq -s '
{
  first_far_landing: ([.[] | select(.fields.message | contains("FAR_LANDING_RETRY"))] | .[0].timestamp),
  last_far_landing: ([.[] | select(.fields.message | contains("FAR_LANDING_RETRY"))] | .[-1].timestamp),
  first_propagate: ([.[] | select(.fields.message | contains("PROPAGATE_COMPLETE"))] | .[0].timestamp),
  last_propagate: ([.[] | select(.fields.message | contains("PROPAGATE_COMPLETE"))] | .[-1].timestamp)
}'
```

### Extracting Internode Keys for Analysis

```bash
# All keys in internodes during FAR_LANDING
head -5000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("INTERNODE_DESCENT"))]
| map({
  target: .fields.target_ikey,
  k0: .fields.k0,
  k1: .fields.k1,
  k2: .fields.k2,
  k_last: .fields.k_last
})
| .[0:10]'

# Largest k_last values (how far tree has grown)
head -100000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("INTERNODE_DESCENT"))]
| map(.fields.k_last)
| unique
| sort
| .[-10:]'
```

### Thread Hotspot Analysis

```bash
# Which threads have most FAR_LANDING events
head -100000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("FAR_LANDING_RETRY")) | .threadId]
| group_by(.)
| map({thread: .[0], count: length})
| sort_by(.count)
| reverse
| .[0:10]'
```

### Direction Analysis for FAR_LANDING

```bash
# Check if landing too far LEFT (key > bound) or RIGHT (bound > key)
# This requires hex comparison which jaq supports
head -10000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("FAR_LANDING_RETRY")) |
  {direction: (if (.fields.ikey > .fields.leaf_bound) then "LEFT" else "RIGHT" end)}]
| group_by(.direction)
| map({direction: .[0].direction, count: length})'
```

### Combined Statistics Report

```bash
# Generate a full analysis report
head -100000 logs/lock_contention.json | jaq -s '
{
  total_events: length,

  message_counts: (
    [.[] | .fields.message] | group_by(.) |
    map({msg: .[0], count: length}) | sort_by(.count) | reverse | .[0:10]
  ),

  gap_stats: (
    [.[] | select(.fields.gap != null) | .fields.gap] |
    {min: min, max: max, avg: (add / length), count: length}
  ),

  nkeys_distribution: (
    [.[] | select(.fields.nkeys != null) | .fields.nkeys] |
    group_by(.) | map({nkeys: .[0], count: length}) | sort_by(.nkeys)
  ),

  thread_hotspots: (
    [.[] | select(.level == "WARN" or .level == "ERROR") | .threadId] |
    group_by(.) | map({thread: .[0], count: length}) |
    sort_by(.count) | reverse | .[0:5]
  )
}'
```

### Debugging Specific Anomalies

```bash
# Full context around an INSERT_ABORT
# First find the timestamp:
head -100000 logs/lock_contention.json | jaq -s '
[.[] | select(.fields.message | contains("INSERT_ABORT"))] | .[0].timestamp'
# Output: "2025-12-26T08:09:24.916494Z"

# Then get events around that time:
head -100000 logs/lock_contention.json | jaq -s '
[.[] | select(.timestamp > "2025-12-26T08:09:24.91" and .timestamp < "2025-12-26T08:09:24.92")]'
```
