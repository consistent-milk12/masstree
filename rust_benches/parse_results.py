#!/usr/bin/env python3
"""
Parse and verify Rust mttest benchmark results.
"""

import re
from pathlib import Path
from typing import Dict


def parse_rust_output(filepath: Path) -> Dict:
    """Parse a Rust mttest output file and extract benchmark metrics."""
    test_name = filepath.stem

    total_puts = 0
    total_puts_per_sec = 0
    total_gets = 0
    total_gets_per_sec = 0
    total_ops = 0
    total_ops_per_sec = 0
    threads = 0

    with open(filepath, "r") as f:
        content = f.read()

    # Find the TOTAL line
    # Format: "   TOTAL     68580864        6856327     68580864        9171173    137161728        7843288"
    total_pattern = re.compile(
        r"^\s*TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    match = total_pattern.search(content)
    if match:
        total_puts = int(match.group(1))
        total_puts_per_sec = int(match.group(2))
        total_gets = int(match.group(3))
        total_gets_per_sec = int(match.group(4))
        total_ops = int(match.group(5))
        total_ops_per_sec = int(match.group(6))

    # Count thread lines (lines starting with a number after the header)
    thread_pattern = re.compile(r"^\s+(\d+)\s+\d+\s+\d+", re.MULTILINE)
    threads = len(thread_pattern.findall(content))

    # Determine which metric to use (similar to C++ logic)
    # For multi-phase tests: use ops_per_sec
    # For single-phase write tests: puts_per_sec == ops_per_sec
    if total_gets > 0:
        # Multi-phase test (has both puts and gets)
        metric_type = "ops_per_sec"
        primary_metric = total_ops_per_sec
    else:
        # Single-phase write test
        metric_type = "puts_per_sec"
        primary_metric = total_puts_per_sec

    return {
        "test": test_name,
        "threads": threads,
        "metric_type": metric_type,
        "total_ops_per_sec": total_ops_per_sec,
        "mops": primary_metric / 1e6,
        "puts": total_puts,
        "puts_per_sec": total_puts_per_sec,
        "gets": total_gets,
        "gets_per_sec": total_gets_per_sec,
        "ops": total_ops,
        "ops_per_sec": total_ops_per_sec,
    }


def main():
    script_dir = Path(__file__).parent

    tests = ["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]

    print("=" * 70)
    print("Rust mttest Benchmark Results (12 threads, 10s)")
    print("=" * 70)
    print()
    print(f"{'Test':<10} {'Threads':>8} {'Metric Type':>15} {'Total Mops/s':>14}")
    print("-" * 50)

    results = []
    for test in tests:
        filepath = script_dir / f"{test}.txt"
        if not filepath.exists():
            print(f"{test:<10} NOT FOUND")
            continue

        result = parse_rust_output(filepath)
        results.append(result)
        print(
            f"{result['test']:<10} {result['threads']:>8} {result['metric_type']:>15} {result['mops']:>14.2f}"
        )

    print("-" * 50)
    print()

    # Summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("| Benchmark | Mops/s |")
    print("|-----------|--------|")
    for r in results:
        print(f"| {r['test']:<9} | {r['mops']:>6.2f} |")

    print()
    print("Detailed breakdown:")
    for r in results:
        print(f"\n{r['test']}:")
        print(f"  puts: {r['puts']:,} ({r['puts_per_sec']:,}/s)")
        print(f"  gets: {r['gets']:,} ({r['gets_per_sec']:,}/s)")
        print(f"  ops:  {r['ops']:,} ({r['ops_per_sec']:,}/s)")


if __name__ == "__main__":
    main()
