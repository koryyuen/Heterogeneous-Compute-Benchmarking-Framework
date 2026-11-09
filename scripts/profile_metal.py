#!/usr/bin/env python3
"""Record an Instruments trace separately from uninstrumented benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/release/heterobench"))
    parser.add_argument("--backend", choices=["metal", "metal-tiled", "mps"], default="metal")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--template", default="Metal System Trace")
    parser.add_argument("--output", type=Path, required=True, help="New .trace directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".log.txt").exists():
        parser.error("Use a new trace output path")
    if min(args.size, args.iterations, args.seconds) <= 0:
        parser.error("size, iterations, and seconds must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["xcrun", "xctrace", "record", "--template", args.template,
               "--no-prompt", "--time-limit", f"{args.seconds}s", "--output", str(output),
               "--target-stdout", str(output) + ".target.stdout.txt", "--launch", "--",
               str(args.binary.resolve()), "--backend", args.backend,
               "--size", str(args.size), "--warmup", "10", "--iterations", str(args.iterations)]
    completed = subprocess.run(command, capture_output=True, text=True)
    log = completed.stdout + completed.stderr
    Path(str(output) + ".log.txt").write_text(log)
    Path(str(output) + ".manifest.json").write_text(json.dumps(
        {"argv": command, "returncode": completed.returncode,
         "instrumented": True, "not_for_latency_comparison": True}, indent=2) + "\n")
    print(log, end="")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
