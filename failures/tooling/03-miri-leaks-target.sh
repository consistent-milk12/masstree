#!/usr/bin/env bash
# TOOL-03 regression: `miri-leaks` must run the integration leak target.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if sed -n '/^miri-leaks:/,/^$/p' justfile | grep -q -- '--test leak_tests'; then
  echo "FIXED: miri-leaks runs --test leak_tests"
  exit 0
fi
echo "REGRESSION: miri-leaks does not run tests/leak_tests.rs"
sed -n '/^miri-leaks:/,/^$/p' justfile
exit 1
