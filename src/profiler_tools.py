import csv
import os
import subprocess
from datetime import datetime

def run_nsys(exe, output_base='nsys_expert', code=False, timeout=900):
    nsys = None
    try:
        from .discover import find_nsys_path
        nsys = find_nsys_path()
    except Exception:
        pass
    if not nsys:
        return {"error": "nsys not found"}
    cmd = [nsys, 'profile', '--output', output_base, '--trace', 'cuda,cudnn,nvtx']
    if code:
        cmd += ['--capture-range=cudaProfilerApi']
    cmd += [exe]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    out = res.stdout + res.stderr
    return {"out": out, "basename": output_base}

def run_ncu_broad(exe, output_base=None, metrics=None, timeout=600):
    from .discover import find_ncu_path
    ncu = find_ncu_path()
    if not ncu:
        return {"error": "ncu not found"}
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = output_base or f'ncu_expert_{ts}'
    cmd = [ncu, '--csv', '-o', base, exe]
    if metrics:
        cmd = [ncu, '--metrics', metrics, '--csv', '-o', base, exe]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    out = res.stdout + res.stderr
    csv_path = f"{base}.csv"
    rep_path = f"{base}.ncu-rep"

    if os.path.exists(rep_path):
        export_cmd = [ncu, '--import', rep_path, '--csv']
        try:
            exp_res = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if exp_res.returncode == 0 and exp_res.stdout.strip():
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(exp_res.stdout)
        except Exception:
            pass

    return {"out": out, "csv": csv_path, "basename": base}

def parse_ncu_csv_for_hotspot(csv_path, target_metric=None):
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
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
            if header_idx >= len(rows):
                return None
            headers = [h.strip() for h in rows[header_idx]]
            data_rows = rows[header_idx + 1:]
         
            name_idx = None
            for i, h in enumerate(headers):
                if 'kernel' in h.lower() or 'name' == h.lower() or 'kernel name' in h.lower():
                    name_idx = i
                    break
          
            time_idx = None
            if target_metric and target_metric.lower() != 'latency':
                for i, h in enumerate(headers):
                    if h.lower() == target_metric.lower() or target_metric.lower() in h.lower():
                        time_idx = i
                        break

            if time_idx is None:
                for i, h in enumerate(headers):
                    hl = h.lower()
                    if 'time' in hl or 'cycles' in hl or 'duration' in hl or 'elapsed' in hl:
                        time_idx = i
                        break
          
            if time_idx is None and data_rows:
                for i in range(len(headers)):
                    try:
                        float(data_rows[0][i])
                        time_idx = i
                        break
                    except Exception:
                        continue
            if name_idx is None or time_idx is None:
                return None
            best = None
            best_val = -1.0
            for r in data_rows:
                if len(r) <= max(name_idx, time_idx):
                    continue
                name = r[name_idx]
                try:
                    val = float(r[time_idx].replace(',', ''))
                except Exception:
                    continue
                if val > best_val:
                    best_val = val
                    best = name
            return {"kernel": best, "value": best_val, "name_idx": name_idx, "time_idx": time_idx}
    except Exception:
        return None

def summarize_profile_outputs(nsys_out, ncu_csv_path, max_lines=200):
    parts = []
    if nsys_out:
        lines = nsys_out.splitlines()
        parts.append("NSYS Output (top lines):")
        parts.extend(lines[:min(len(lines), max_lines)])
    if ncu_csv_path and os.path.exists(ncu_csv_path):
        parts.append("\nNCU CSV Preview:")
        try:
            with open(ncu_csv_path, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    parts.append(','.join(rows[0]))
                    for r in rows[1: min(1+10, len(rows))]:
                        parts.append(','.join(r))
        except Exception:
            parts.append("(failed to read CSV)")

    return "\n".join(parts)
