#!/usr/bin/env python3
"""Run a reproducible size/repetition sweep; preserve argv and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root):
    paths = [root / "CMakeLists.txt"]
    for folder in ["include", "src", "shaders", "scripts"]:
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/release/heterobench"))
    parser.add_argument("--backend", default="all")
    parser.add_argument("--reference", choices=["cpu", "accelerate"], default="cpu")
    parser.add_argument("--sizes", default="64,128,256,512")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="New output prefix")
    args = parser.parse_args()
    if args.runs <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("runs/iterations must be positive; warmup may be zero")
    binary = args.binary.resolve()
    prefix = args.output.resolve()
    manifest_path = Path(str(prefix) + ".manifest.json")
    for suffix in [".manifest.json", ".samples.csv", ".summary.json", ".stdout.txt"]:
        if Path(str(prefix) + suffix).exists():
            parser.error(f"Output already exists: {prefix}{suffix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--backend", args.backend, "--reference", args.reference, "--sizes", args.sizes,
               "--runs", str(args.runs), "--iterations", str(args.iterations),
               "--warmup", str(args.warmup), "--output", str(prefix)]
    root = Path(__file__).resolve().parents[1]
    manifest = {"argv": command, "binary_sha256": sha256(binary), "source_sha256": source_manifest(root)}
    completed = subprocess.run(command, capture_output=True, text=True)
    manifest["returncode"] = completed.returncode
    manifest["stderr"] = completed.stderr
    Path(str(prefix) + ".stdout.txt").write_text(completed.stdout)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
