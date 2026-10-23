#!/usr/bin/env bash
# BUG-TOOL-05: the C++ recipe references a missing binary, uses a path that
# disagrees with the Python driver, and redirects stderr away from the JSON log.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
status=0
if [ -e reference/mttest ]; then
  echo "UNEXPECTED: reference/mttest exists"
  status=1
else
  echo "REPRODUCED: reference/mttest is absent (just cpp-bench exits 127)"
fi
if grep -q '2>&1 > cpp_benches/' justfile; then
  echo "REPRODUCED: justfile uses '2>&1 > file', which leaves stderr on the terminal"
else
  echo "NOT REPRODUCED: redirect ordering changed"
  status=1
fi
if grep -q '"reference" / "build" / "mttest"' scripts/run_cpp_bench.py; then
  echo "REPRODUCED: run_cpp_bench.py expects reference/build/mttest"
else
  echo "NOT REPRODUCED: Python driver path changed"
  status=1
fi
exit "$status"
