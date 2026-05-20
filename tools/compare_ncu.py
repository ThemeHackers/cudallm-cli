#!/usr/bin/env python3
"""
Simple comparator for two Nsight Compute CSV exports.
This is a heuristic tool: it finds a numeric column in each CSV and compares means.
Usage: python tools/compare_ncu.py baseline.csv current.csv
"""
import argparse
import csv
import sys
from statistics import mean


def numeric_columns(path, column=None):
    try:
        with open(path, newline="", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.reader(handle))
    except Exception as exc:
        print("Error reading", path, exc)
        return None

    if not rows:
        return None

    header_idx = 0
    for idx, r in enumerate(rows):
        if not r:
            continue
        r_joined = ",".join(r)
        if r_joined.startswith("#") or r_joined.startswith("##"):
            continue
        if any(k in [col.lower().strip() for col in r] for k in ["id", "kernel name", "kernel", "metric name", "metric value", "device", "process ID", "process name", "host name"]):
            header_idx = idx
            break

    headers = rows[header_idx]
    data_rows = rows[header_idx + 1:]

    idx = None
    if column is not None:
        try:
            idx = int(column)
        except Exception:
            for i, header in enumerate(headers):
                if header.strip() == column:
                    idx = i
                    break
            if idx is None:
                for i, header in enumerate(headers):
                    lowered = header.lower()
                    if column.lower() in lowered or lowered.endswith(column.lower()):
                        idx = i
                        break

    if idx is not None:
        values = []
        for row in data_rows:
            if idx < len(row):
                try:
                    values.append(float(row[idx]))
                except Exception:
                    pass
        return (idx, values) if values else None

    keywords = ["cycles", "throughput", "inst", "time", "duration"]
    candidates = []
    for i, header in enumerate(headers):
        lowered = header.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > 0:
            candidates.append((score, i))

    candidates.sort(reverse=True)
    for _, i in candidates:
        values = []
        for row in data_rows:
            if i < len(row):
                try:
                    values.append(float(row[i]))
                except Exception:
                    pass
        if values:
            return (i, values)

    for row in data_rows[:20]:
        for i, value in enumerate(row):
            try:
                float(value)
            except Exception:
                continue

            values = []
            for candidate_row in data_rows:
                if i < len(candidate_row):
                    try:
                        values.append(float(candidate_row[i]))
                    except Exception:
                        pass
            if values:
                return (i, values)
            break

    return None


def main():
    parser = argparse.ArgumentParser(description="Compare two ncu CSVs")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("-c", "--column", help="Column name or index to compare")
    parser.add_argument("--threshold", type=float, default=0.02, help="Relative regression threshold (default 0.02)")
    args = parser.parse_args()

    base = args.baseline
    cur = args.current
    col = args.column

    common_map = {
        "cycles": ["sm__cycles_elapsed.avg", "sm__cycles_elapsed", "sm__cycles_elapsed.sum"],
        "dram_throughput": ["dram__throughput.avg", "dram__throughput"],
        "inst_executed": ["sm__sass_thread_inst_executed_avg", "inst_executed"],
    }

    if col and not col.isdigit():
        baseline_match = None
        current_match = None
        for key, variants in common_map.items():
            if col.lower() == key or col.lower() in key:
                for variant in variants:
                    baseline_match = numeric_columns(base, variant)
                    current_match = numeric_columns(cur, variant)
                    if baseline_match and current_match:
                        col = variant
                        break
                if baseline_match and current_match:
                    break

    baseline_match = numeric_columns(base, col)
    current_match = numeric_columns(cur, col)
    if not baseline_match or not current_match:
        print("Could not find numeric columns in one of the files")
        sys.exit(2)

    baseline_index, baseline_values = baseline_match
    current_index, current_values = current_match
    baseline_mean = mean(baseline_values) if baseline_values else 0
    current_mean = mean(current_values) if current_values else 0
    print(f"Baseline mean (col {baseline_index}): {baseline_mean}")
    print(f"Current mean  (col {current_index}): {current_mean}")

    if baseline_mean == 0:
        print("Baseline zero, cannot compare ratio")
        sys.exit(1 if current_mean > 0 else 0)

    ratio = (current_mean - baseline_mean) / baseline_mean
    print(f"Relative change: {ratio * 100:.2f}%")

    if ratio > args.threshold:
        print(f"Performance regression detected (> {args.threshold * 100:.2f}%)")
        sys.exit(3)

    sys.exit(0)


if __name__ == "__main__":
    main()
