#!/usr/bin/env bash
# BUG-TOOL-09 / BUG-TOOL-10: several "parity"/stress tests cannot fail, and
# the scan proptests only check one direction (no completeness).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0
check() {
  if eval "$2"; then
    echo "REPRODUCED: $1"
  else
    echo "NOT REPRODUCED: $1"
    status=1
  fi
}
check "rwsmall24 discards every get result" \
  "! grep -q 'assert' tests/cpp/rwsmall24.rs"
check "deleted_leaf_handling has a zero-success escape" \
  "grep -q 'if total_success > 0' tests/deleted_leaf_handling.rs"
check "scan_respects_bounds does not assert completeness" \
  "! sed -n '/fn scan_respects_bounds/,/^}/p' tests/tree_proptests.rs | grep -Eq 'expected|len\(\)'"
check "unique_keys uses HashSet iteration order" \
  "grep -q 'hash_set' tests/tree_proptests.rs"
exit "$status"
