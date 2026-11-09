#!/usr/bin/env python3
"""Aggregate independent-run medians, export plots and a measured summary."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / ".matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((args.output / ".cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    document = json.loads(args.summary.read_text())
    groups = defaultdict(list)
    for experiment in document["experiments"]:
        groups[(experiment["backend"], experiment["dimension"])].append(experiment)
    rows = []
    for (backend, dimension), experiments in sorted(groups.items()):
        host = [entry["host"]["median_ms"] for entry in experiments]
        gpu = [entry["gpu"]["median_ms"] for entry in experiments if entry["gpu"] is not None]
        rows.append({"backend": backend, "dimension": dimension, "runs": len(host),
                     "host_median_of_run_medians_ms": statistics.median(host),
                     "host_run_medians_stddev_ms": statistics.pstdev(host),
                     "host_run_medians_variance_ms2": statistics.pvariance(host),
                     "gpu_runs_available": len(gpu),
                     "gpu_median_of_run_medians_ms": statistics.median(gpu) if gpu else None,
                     "gpu_run_medians_stddev_ms": statistics.pstdev(gpu) if gpu else None,
                     "gpu_run_medians_variance_ms2": statistics.pvariance(gpu) if gpu else None})
    with (args.output / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for throughput in [False, True]:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for axis, metric in zip(axes, ["host", "gpu"]):
            for backend in sorted({row["backend"] for row in rows}):
                selected = [row for row in rows if row["backend"] == backend
                            and row[f"{metric}_median_of_run_medians_ms"] is not None
                            and row[f"{metric}_median_of_run_medians_ms"] > 0]
                selected.sort(key=lambda row: row["dimension"])
                if not selected:
                    continue
                x = [row["dimension"] for row in selected]
                latency = [row[f"{metric}_median_of_run_medians_ms"] for row in selected]
                if throughput:
                    values = [2 * n**3 / (ms * 1e6) for n, ms in zip(x, latency)]
                    axis.plot(x, values, marker="o", label=backend)
                else:
                    deviations = [row[f"{metric}_run_medians_stddev_ms"] for row in selected]
                    axis.errorbar(x, latency, yerr=deviations, marker="o", capsize=3, label=backend)
            axis.set_title("Host completed-operation" if metric == "host" else "GPU command-buffer")
            axis.set_xlabel("Square matrix dimension N")
            axis.set_ylabel("GFLOP/s from median latency" if throughput else "Median latency (ms)")
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend()
        name = "throughput" if throughput else "latency"
        figure.savefig(args.output / f"{name}.png", dpi=180)
        figure.savefig(args.output / f"{name}.svg")
        plt.close(figure)
    lines = ["# Measured experiment summary", "",
             f"Input: `{args.summary.name}`", "",
             f"SHA256: `{hashlib.sha256(args.summary.read_bytes()).hexdigest()}`", "",
             "Each cell is the median of independent run medians. Error bars show population",
             "standard deviation of those run medians, not confidence intervals.", "",
             "| Backend | N | Runs | Host ms | GPU command buffer ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        gpu = row["gpu_median_of_run_medians_ms"]
        gpu_text = f"{gpu:.6g}" if gpu is not None else "unavailable"
        lines.append(f'| {row["backend"]} | {row["dimension"]} | {row["runs"]} | '
                     f'{row["host_median_of_run_medians_ms"]:.6g} | {gpu_text} |')
    lines += ["", "## Naive and tiled comparison", ""]
    by_key = {(row["backend"], row["dimension"]): row for row in rows}
    for row in rows:
        other = by_key.get(("metal-tiled", row["dimension"]))
        if row["backend"] == "metal" and other:
            host_ratio = row["host_median_of_run_medians_ms"] / other["host_median_of_run_medians_ms"]
            lines.append(f'- N={row["dimension"]}: naive/tiled host latency ratio {host_ratio:.3f}x.')
            naive_gpu = row["gpu_median_of_run_medians_ms"]
            tiled_gpu = other["gpu_median_of_run_medians_ms"]
            if naive_gpu and tiled_gpu:
                lines.append(f'  Naive/tiled GPU command-buffer ratio {naive_gpu / tiled_gpu:.3f}x.')
    lines += ["", "Ratios above one mean the tiled variant had lower latency in this experiment.",
              "The host/GPU difference does not isolate exact dispatch overhead. Workloads use",
              "fixed uploaded inputs, output readback, warmup, and sequential backend runs.",
              "Thermals and background activity were not controlled; this is one machine and",
              "one experiment, not a universal speedup or an accelerator power measurement.", ""]
    (args.output / "analysis.md").write_text("\n".join(lines))
    print(f"Wrote {args.output}/analysis.md, aggregate.csv, and PNG/SVG plots")


if __name__ == "__main__":
    main()
