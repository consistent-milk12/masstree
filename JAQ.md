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
| NOT | `| not` |

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
