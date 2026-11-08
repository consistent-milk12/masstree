#!/usr/bin/env bash
# Run the tooling regression checks. Each script exits 0 when the fixed
# behavior is present and non-zero when the defect has regressed.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
failed=0
for script in \
  01-no-default-features.sh \
  02-shuttle-noop.sh \
  03-miri-leaks-target.sh \
  04-clippy-strict.sh \
  05-cpp-bench.sh \
  06-test-logs.sh \
  07-vacuous-tests.sh
do
  echo
  echo "===== $script ====="
  if ! "$DIR/$script"; then
    failed=$((failed + 1))
  fi
done
echo
if [ "$failed" -eq 0 ]; then
  echo "all tooling regression checks passed"
  exit 0
fi
echo "$failed tooling regression check(s) failed"
exit 1
