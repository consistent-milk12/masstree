#!/usr/bin/env bash
# TOOL-05 regression: the C++ benchmark recipe must not hide stderr and must
# use a configurable/consistent reference path.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0

if grep -q '2>&1 > cpp_benches/' justfile; then
  echo "REGRESSION: justfile still uses '2>&1 > file' for cpp-bench"
  status=1
fi

if ! sed -n '/^cpp-bench:/,/^$/p' justfile | grep -q 'MTTEST'; then
  echo "REGRESSION: cpp-bench does not use the configurable MTTEST path"
  status=1
fi

if ! sed -n '/^cpp-bench:/,/^$/p' justfile | grep -q '> "cpp_benches/\$test.txt" 2>&1'; then
  echo "REGRESSION: cpp-bench does not redirect stdout+stderr to the result file"
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "FIXED: cpp-bench uses a configurable path and correct redirection"
fi
exit "$status"
