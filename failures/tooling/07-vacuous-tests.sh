#!/usr/bin/env bash
# TOOL-09/TOOL-10 regression: parity/stress tests assert real state and the
# scan properties are complete and deterministic.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0

if ! grep -q 'assert' tests/cpp/rwsmall24.rs; then
  echo "REGRESSION: rwsmall24 still discards read results"
  status=1
fi

if grep -q 'if total_success > 0' tests/deleted_leaf_handling.rs; then
  echo "REGRESSION: deleted_leaf_handling still has a zero-success escape"
  status=1
fi

if ! grep -q 'total_success > 0' tests/deleted_leaf_handling.rs; then
  echo "REGRESSION: deleted_leaf_handling does not require a successful read"
  status=1
fi

if ! sed -n '/fn scan_respects_bounds/,/^    }/p' tests/tree_proptests.rs | grep -q 'expected'; then
  echo "REGRESSION: scan_respects_bounds does not assert completeness"
  status=1
fi

if grep -q 'prop::collection::hash_set' tests/tree_proptests.rs; then
  echo "REGRESSION: proptest key strategies still use HashSet order"
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "FIXED: parity/stress tests assert values and scan properties are exact"
fi
exit "$status"
