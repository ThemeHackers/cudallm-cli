import csv
import sqlite3
import os
from typing import Optional, Dict, Any


def parse_ncu_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return None
            header_idx = 0
            for i, r in enumerate(rows[:20]):
                if any('kernel' in c.lower() or 'name' in c.lower() for c in r):
                    header_idx = i
                    break
            headers = [h.strip() for h in rows[header_idx]]
            data = rows[header_idx+1: header_idx+1+50]
            time_idx = None
            for i, h in enumerate(headers):
                hl = h.lower()
                if 'time' in hl or 'elapsed' in hl or 'duration' in hl or 'cycles' in hl:
                    time_idx = i
                    break
            name_idx = None
            for i, h in enumerate(headers):
                hl = h.lower()
                if 'kernel' in hl or 'name' in hl:
                    name_idx = i
                    break
            if time_idx is None or name_idx is None:
                return {"headers": headers, "rows_sample": data}
            best = None
            best_val = -1.0
            for r in data:
                if len(r) <= max(time_idx, name_idx):
                    continue
                try:
                    val = float(r[time_idx])
                except Exception:
                    continue
                if val > best_val:
                    best_val = val
                    best = r[name_idx]
            return {"hotspot_kernel": best, "hotspot_value": best_val, "headers": headers}
    except Exception:
        return None


def extract_metrics_from_ncu_csv(csv_path: str, metrics_list: list) -> Optional[Dict[str, float]]:
    """Extract numeric metric columns from an NCU CSV and return aggregated values.

    For each metric name in metrics_list, this function searches header columns
    for a matching name (case-insensitive substring) and returns the max value
    observed across rows for that column. Returns dict metric->value.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return None
            header_idx = 0
            for i, r in enumerate(rows[:20]):
                if r and any(c.strip() for c in r):
                    header_idx = i
                    break
            headers = [h.strip() for h in rows[header_idx]]
            data_rows = rows[header_idx+1:]
            result = {}
            for metric in metrics_list:
                col_idx = None
                for i, h in enumerate(headers):
                    if metric.lower() in h.lower():
                        col_idx = i
                        break
                if col_idx is None:
                    result[metric] = None
                    continue
                best = None
                for r in data_rows:
                    if len(r) <= col_idx:
                        continue
                    try:
                        v = float(r[col_idx])
                    except Exception:
                        continue
                    if best is None or v > best:
                        best = v
                result[metric] = best
            return result
    except Exception:
        return None


def parse_sqlite_rep(rep_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(rep_path):
        return None
    try:
        conn = sqlite3.connect(rep_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        info = {"tables": tables}
        for candidate in ['metrics', 'summary', 'ncu_summary', 'details', 'kernel_summary']:
            if candidate in tables:
                try:
                    cur.execute(f"SELECT * FROM {candidate} LIMIT 10")
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    info[candidate] = {"columns": cols, "rows_sample": rows}
                except Exception:
                    pass
        conn.close()
        return info
    except Exception:
        return None


def parse_nsys_rep(rep_path: str) -> Optional[Dict[str, Any]]:
    return parse_sqlite_rep(rep_path)


def parse_ncu_rep(rep_path: str) -> Optional[Dict[str, Any]]:
    if rep_path.endswith('.csv') and os.path.exists(rep_path):
        return parse_ncu_csv(rep_path)
    return parse_sqlite_rep(rep_path)
