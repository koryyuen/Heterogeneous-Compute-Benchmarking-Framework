"""Exercise the real CPU CLI and parse output with independent standard libraries."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="heterobench-export-") as directory:
    prefix = Path(directory) / 'run with spaces,"quotes"'
    command = [binary, "--backend", "cpu", "--sizes", "2,3", "--runs", "2",
               "--warmup", "0", "--iterations", "3", "--output", str(prefix)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads(Path(str(prefix) + ".summary.json").read_text())
    with Path(str(prefix) + ".samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and len(summary["experiments"]) == 4
    assert summary["schema_version"] == 1
    assert summary["metadata"]["validation_reference"] == "cpu"
    assert '"quotes"' in summary["metadata"]["command"]
    for experiment in summary["experiments"]:
        selected = [row for row in rows if int(row["dimension"]) == experiment["dimension"]
                    and int(row["repetition"]) == experiment["repetition"]]
        values = [float(row["host_ms"]) for row in selected]
        assert len(values) == 3 and all(math.isfinite(v) and v >= 0 for v in values)
        assert all(row["gpu_ms"] == "" for row in selected)
        assert experiment["gpu"] is None and experiment["gpu_sample_count"] == 0
        assert math.isclose(experiment["host"]["median_ms"], statistics.median(values))
        assert math.isclose(experiment["host"]["variance_ms2"], statistics.pvariance(values), abs_tol=1e-18)
        assert experiment["host"]["p95_ms"] == max(values)
    before = Path(str(prefix) + ".summary.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert Path(str(prefix) + ".summary.json").read_bytes() == before
    for arguments in [["--size", "0"], ["--size", "-1"], ["--sizes", "2,"],
                      ["--sizes", "2,2"], ["--iterations", "0"], ["--runs", "0"], ["--backend", "unknown"], ["--backend", "cpu,cpu"],
                      ["--backend", "cpu,"], ["--reference", "mps"]]:
        assert subprocess.run([binary] + arguments, capture_output=True).returncode != 0
    assert subprocess.run([binary, "--help"], capture_output=True).returncode == 0
print("PASS: CLI, JSON/CSV round trip, metadata escaping, missing GPU values, and overwrite protection")

# An optional optimized reference must remain numerically anchored to known results.
probe = subprocess.run([binary, "--backend", "accelerate", "--reference", "accelerate",
                        "--size", "3", "--iterations", "1"], capture_output=True, text=True)
if "no Accelerate" not in probe.stderr and "unavailable backend" not in probe.stderr:
    assert probe.returncode == 0, probe.stderr
    subset = subprocess.run([binary, "--backend", "cpu,accelerate", "--size", "3",
                             "--reference", "accelerate", "--iterations", "1"],
                            capture_output=True, text=True)
    assert subset.returncode == 0, subset.stderr
    assert "cpu N=3" in subset.stdout and "accelerate N=3" in subset.stdout
