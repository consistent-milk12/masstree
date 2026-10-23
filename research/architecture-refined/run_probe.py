#!/usr/bin/env python3
"""Build once per layout, then run interleaved, pinned trials without builds."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    variants = [("embedded512", []), ("embedded256", ["small-suffix-capacity"]),
                ("sidecar", ["sidecar-suffix"])]
    manifest = here / "Cargo.toml"
    metadata = {
        "cpu": args.cpu, "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "rust_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip(),
        "note": "Current worktree, fixed 8-byte keys/u64, forward scans; no concurrent writers or structural churn.",
    }
    (args.out / "probe-environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="masstree-layout-probe-") as scratch:
        programs = {}
        for name, features in variants:
            command = ["cargo", "build", "--offline", "--locked", "--release", "--manifest-path", str(manifest), "--target-dir", str(here / "target"), "--bin", "layout_probe"]
            if features:
                command += ["--features", ",".join(features)]
            with (args.out / f"build-{name}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            binary = Path(scratch) / name
            shutil.copy2(here / "target/release/layout_probe", binary)
            programs[name] = binary
        schedule = [name for name, _ in variants] * 3
        random.Random(42).shuffle(schedule)
        (args.out / "schedule.json").write_text(json.dumps(schedule) + "\n")
        with (args.out / "layout-probe.csv").open("w", newline="") as stream:
            writer = None
            for block, name in enumerate(schedule):
                result = subprocess.run(["taskset", "-c", str(args.cpu), str(programs[name])],
                                        check=True, capture_output=True, text=True, timeout=120)
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=["block", *rows[0].keys()])
                    writer.writeheader()
                for row in rows:
                    writer.writerow({"block": block, **row})
                stream.flush()
                print(f"finished block {block}: {name}", flush=True)


if __name__ == "__main__":
    main()
