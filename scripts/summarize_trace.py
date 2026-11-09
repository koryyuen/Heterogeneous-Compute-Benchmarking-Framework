#!/usr/bin/env python3
"""Summarize only labeled HeteroBench compute intervals from xctrace XML."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = ET.parse(args.xml).getroot()
    identifiers = {node.get("id"): node for node in tree.iter() if node.get("id")}
    def resolve(node):
        return identifiers[node.get("ref")] if node.get("ref") else node
    columns = [column.findtext("mnemonic") for column in tree.findall(".//schema/col")]
    required = ["start", "duration", "process", "channel-name", "event-label", "event-depth", "state"]
    if not all(name in columns for name in required):
        raise SystemExit("Unexpected schema: inspect xctrace --toc for this Xcode version")
    selected = []
    for row in tree.findall(".//row"):
        cells = {name: resolve(cell) for name, cell in zip(columns, row)}
        if "heterobench" not in cells["process"].get("fmt", ""):
            continue
        if cells["channel-name"].get("fmt") != "Compute" or cells["state"].get("fmt") != "Active":
            continue
        if "HeteroBench" not in cells["event-label"].get("fmt", ""):
            continue
        selected.append({"start": int(cells["start"].text), "duration": int(cells["duration"].text),
                         "depth": int(cells["event-depth"].text),
                         "start_latency": int(cells["start-latency"].text) if "start-latency" in cells else 0})
    if not selected:
        raise SystemExit("No labeled HeteroBench compute intervals; capture is not evidence of target GPU execution")
    depth = min(row["depth"] for row in selected)
    selected = sorted((row for row in selected if row["depth"] == depth), key=lambda row: row["start"])
    # Union intervals rather than double-counting any overlap in the trace.
    busy = 0
    end = selected[0]["start"]
    gaps = []
    for row in selected:
        start = row["start"]
        if start > end and end != selected[0]["start"]:
            gaps.append(start - end)
        finish = start + row["duration"]
        busy += max(0, finish - max(end, start))
        end = max(end, finish)
    span = end - selected[0]["start"]
    result = {"source_xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
              "filter": "heterobench process, labeled HeteroBench, Active Compute, minimum nesting depth",
              "interval_count": len(selected), "nesting_depth": depth,
              "median_compute_interval_ms": statistics.median(row["duration"] for row in selected) / 1e6,
              "compute_union_ms": busy / 1e6, "compute_span_ms": span / 1e6,
              "target_compute_busy_fraction_within_span": busy / span if span else None,
              "median_gap_ms": statistics.median(gaps) / 1e6 if gaps else None,
              "interpretation": "Instrumented target-only intervals; not system GPU utilization or benchmark latency"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
