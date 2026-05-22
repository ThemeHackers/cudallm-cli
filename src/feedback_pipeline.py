import os
import json
import tempfile
import re
from typing import Optional, Dict, Any

from .langchain_agent import compile_cuda
from .profiler_tools import parse_ncu_csv_for_hotspot, summarize_profile_outputs
from .discover import find_nsys_path, find_ncu_path
from .sandbox_security import run_sandboxed
from .profile_parsers import extract_metrics_from_ncu_csv
import shlex


def _parse_nsys_total_time(nsys_out: str) -> Optional[float]:
    if not nsys_out:
        return None

    m = re.search(r"Total\s+Time[:\s]+([0-9\.]+)\s*(s|ms|us)?", nsys_out, re.IGNORECASE)
    if not m:
        m = re.search(r"Duration[:\s]+([0-9\.]+)\s*(s|ms|us)?", nsys_out, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == 'ms':
        return val / 1000.0
    if unit == 'us':
        return val / 1_000_000.0
    return val


def compute_reward(metrics: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
    default_weights = {
        "nsys_total_time": -1.0,
        "sm__throughput": 0.30,
        "mem__throughput": 0.20,
        "sm__efficiency": 0.35,
        "ncu_hotspot_value": -0.10,
    }
    if weights:
        default_weights.update(weights)

    score = 0.0
    t = metrics.get('nsys_total_time')
    if t is not None:
        score += default_weights["nsys_total_time"] * float(t)

    for key in ["sm__throughput", "mem__throughput", "sm__efficiency"]:
        val = metrics.get(key)
        if val is not None:
            score += default_weights[key] * float(val)

    hotspot = metrics.get('ncu_hotspot_value')
    if hotspot is not None:
        score += default_weights["ncu_hotspot_value"] * float(hotspot)

    return float(score)


def run_feedback_pipeline(source_path: str, verify_cmd: str = 'python verify.py', metrics: Optional[str] = None, work_dir: Optional[str] = None) -> Dict:
    wd = work_dir or tempfile.mkdtemp(prefix='cudallm_pipeline_')
    report: Dict[str, Any] = {"work_dir": wd}
    compile_res = compile_cuda(source_path, out_path=os.path.join(wd, 'module.so'))
    if compile_res.get('rc', 0) != 0:
        report.update({"compile_ok": False, "compile_res": compile_res})
        return report
    report['compile_ok'] = True
    report['compile_res'] = compile_res

    nsys_out = None
    try:
        nsys = find_nsys_path()
        if nsys:
            parts = shlex.split(verify_cmd) if isinstance(verify_cmd, str) else verify_cmd
            cmd = [nsys, 'profile', '--output', os.path.join(wd, 'nsys_out'), '--trace', 'cuda,cudnn,nvtx'] + parts
            nsys_run = run_sandboxed(cmd, timeout=900, mem_limit_mb=4096)
            report['nsys_res'] = nsys_run
            nsys_out = (nsys_run.get('stdout') or '') + (nsys_run.get('stderr') or '')
            report['nsys_report_base'] = os.path.join(wd, 'nsys_out')
        else:
            report['nsys_error'] = 'nsys not found'
    except Exception as e:
        report['nsys_error'] = str(e)
    try:
        ncu = find_ncu_path()
        if ncu:
            parts = shlex.split(verify_cmd) if isinstance(verify_cmd, str) else verify_cmd
            cmd = [ncu, '--csv']
            if metrics:
                cmd += ['--metrics', metrics]
            cmd += ['-o', os.path.join(wd, 'ncu_out')] + parts
            ncu_run = run_sandboxed(cmd, timeout=600, mem_limit_mb=4096)
            report['ncu_res'] = ncu_run
            report['ncu_csv'] = os.path.join(wd, 'ncu_out.csv')
        else:
            report['ncu_error'] = 'ncu not found'
    except Exception as e:
        report['ncu_error'] = str(e)
    nsys_time = _parse_nsys_total_time(nsys_out) if nsys_out else None
    report['nsys_total_time'] = nsys_time

    hotspot = None
    extracted_metrics = {}
    if report.get('ncu_csv'):
        try:
            hp = parse_ncu_csv_for_hotspot(report['ncu_csv'])
            if hp:
                hotspot = hp.get('value')
                report['ncu_hotspot'] = hp
            extracted = extract_metrics_from_ncu_csv(report['ncu_csv'], ["sm__throughput", "mem__throughput", "sm__efficiency"])
            if extracted:
                extracted_metrics = extracted
                report['ncu_metrics'] = extracted
        except Exception:
            report['ncu_hotspot'] = None

    report['ncu_hotspot_value'] = hotspot

    report['summary'] = summarize_profile_outputs(nsys_out or report.get('nsys_res', {}).get('out', ''), report.get('ncu_csv'))

    report['reward'] = compute_reward({
        'nsys_total_time': nsys_time,
        'ncu_hotspot_value': hotspot,
        'sm__throughput': extracted_metrics.get('sm__throughput'),
        'mem__throughput': extracted_metrics.get('mem__throughput'),
        'sm__efficiency': extracted_metrics.get('sm__efficiency'),
    })

    return report
