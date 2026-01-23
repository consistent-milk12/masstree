#!/usr/bin/env python3
"""
Parse and verify C++ mttest benchmark results.
This script validates that we can correctly extract ops/sec from the raw output.
"""

import json
from pathlib import Path
from typing import List, Dict


def parse_cpp_output(filepath: Path) -> Dict:
    """Parse a C++ mttest output file and extract benchmark metrics."""
    test_name = filepath.stem

    ops_per_sec_values: List[float] = []
    puts_per_sec_values: List[float] = []
    gets_per_sec_values: List[float] = []
    thread_data: List[Dict] = []

    with open(filepath, "r") as f:
        for line in f:
            brace_idx = line.find("{")
            if brace_idx == -1:
                continue
            try:
                data = json.loads(line[brace_idx:])
                if data.get("test") != test_name:
                    continue

                thread_info = {
                    "thread": data.get("thread"),
                    "puts": data.get("puts"),
                    "gets": data.get("gets"),
                    "ops": data.get("ops"),
                }

                if "ops_per_sec" in data:
                    ops_per_sec_values.append(float(data["ops_per_sec"]))
                    thread_info["ops_per_sec"] = data["ops_per_sec"]
                if "puts_per_sec" in data:
                    puts_per_sec_values.append(float(data["puts_per_sec"]))
                    thread_info["puts_per_sec"] = data["puts_per_sec"]
                if "gets_per_sec" in data:
                    gets_per_sec_values.append(float(data["gets_per_sec"]))
                    thread_info["gets_per_sec"] = data["gets_per_sec"]

                thread_data.append(thread_info)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    # Determine total throughput
    if ops_per_sec_values:
        total_ops = sum(ops_per_sec_values)
        metric_type = "ops_per_sec"
    elif puts_per_sec_values:
        total_ops = sum(puts_per_sec_values)
        metric_type = "puts_per_sec"
    elif gets_per_sec_values:
        total_ops = sum(gets_per_sec_values)
        metric_type = "gets_per_sec"
    else:
        total_ops = 0.0
        metric_type = "none"

    return {
        "test": test_name,
        "threads": len(thread_data),
        "metric_type": metric_type,
        "total_ops": total_ops,
        "mops": total_ops / 1e6,
        "ops_per_sec_sum": sum(ops_per_sec_values) if ops_per_sec_values else None,
        "puts_per_sec_sum": sum(puts_per_sec_values) if puts_per_sec_values else None,
        "gets_per_sec_sum": sum(gets_per_sec_values) if gets_per_sec_values else None,
    }


def main():
    script_dir = Path(__file__).parent

    tests = ["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]

    print("=" * 70)
    print("C++ mttest Benchmark Results (12 threads, 10s)")
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

        result = parse_cpp_output(filepath)
        results.append(result)
        print(
            f"{result['test']:<10} {result['threads']:>8} {result['metric_type']:>15} {result['mops']:>14.2f}"
        )

    print("-" * 50)
    print()

    # Summary table for comparison
    print("=" * 70)
    print("Summary (for comparison with run_cpp_bench.py)")
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
        print(f"  Total ops/s: {r['total_ops']:,.0f}")
        if r["ops_per_sec_sum"]:
            print(f"  ops_per_sec sum: {r['ops_per_sec_sum']:,.0f}")
        if r["puts_per_sec_sum"]:
            print(f"  puts_per_sec sum: {r['puts_per_sec_sum']:,.0f}")
        if r["gets_per_sec_sum"]:
            print(f"  gets_per_sec sum: {r['gets_per_sec_sum']:,.0f}")


if __name__ == "__main__":
    main()
