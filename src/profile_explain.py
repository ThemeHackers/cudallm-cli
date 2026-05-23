import os
import json
from typing import Optional, Dict, Any

from .profiler_tools import parse_ncu_csv_for_hotspot
from .profile_parsers import extract_metrics_from_ncu_csv


def _short(val: float) -> str:
    try:
        return f"{val:.4g}"
    except Exception:
        return str(val)


def explain_nsys(nsys_out: str) -> Dict[str, Any]:
    """Produce a lightweight diagnosis from nsys CLI output."""
    if not nsys_out:
        return {"status": "no_data", "message": "No NSYS output provided."}

    lines = nsys_out.splitlines()
    joined = nsys_out.lower()
    issues = []
    recommendations = []

    if 'cudaMemcpy' in joined or 'memcpy' in joined:
        issues.append('host_device_transfers')
        recommendations.append('Reduce host<->device transfers; use async/streams if possible.')

    if 'cuda devicesynchronize' in joined or 'device synchronize' in joined or 'synchroniz' in joined:
        issues.append('synchronization_overhead')
        recommendations.append('Investigate synchronization points; move work off critical path or batch operations.')

    if 'launch' in joined and 'overhead' in joined:
        issues.append('kernel_launch_overhead')
        recommendations.append('Consider kernel fusion or batching small kernels.')

   
    for l in lines[:40]:
        if 'total time' in l.lower() or 'duration' in l.lower():
            recommendations.append(f"Observed timeline line: '{l.strip()}'")
            break

    if not issues:
        issues.append('unknown')
        recommendations.append('No obvious high-level issue detected. Use NCU to inspect hotspot kernels.')

    return {
        "status": "ok",
        "issues": issues,
        "recommendations": recommendations,
        "summary_lines": lines[:20],
    }


def explain_ncu(ncu_csv_path: str) -> Dict[str, Any]:
    """Produce a lightweight diagnosis from an NCU CSV export using heuristics."""
    if not ncu_csv_path or not os.path.exists(ncu_csv_path):
        return {"status": "no_data", "message": "NCU CSV missing or path invalid."}

    hp = parse_ncu_csv_for_hotspot(ncu_csv_path)
    metrics = extract_metrics_from_ncu_csv(ncu_csv_path, ['sm__throughput', 'mem__throughput', 'sm__efficiency']) or {}

    issues = []
    recommendations = []
    hotspot_name = hp.get('kernel') if hp else None

  
    sm = None
    mem = None
    eff = None
    try:
        sm = float(metrics.get('sm__throughput') or 0.0)
    except Exception:
        sm = None
    try:
        mem = float(metrics.get('mem__throughput') or 0.0)
    except Exception:
        mem = None
    try:
        eff = float(metrics.get('sm__efficiency') or 0.0)
    except Exception:
        eff = None

    if mem and sm:
    
        if mem > sm * 0.8:
            issues.append('memory_bound')
            recommendations.append('Kernel appears memory-bound: improve memory coalescing, use shared memory or reduce traffic.')

    if eff is not None and eff < 0.3:
        issues.append('low_sm_efficiency')
        recommendations.append('SM efficiency is low; check occupancy, divergence, and branch imbalance.')

    if not issues:
        issues.append('compute_bound_or_unknown')
        recommendations.append('No clear metric dominance found; consider inspecting launch parameters and instruction mix.')

    return {
        "status": "ok",
        "hotspot": hotspot_name,
        "metrics": {k: _short(v) for k, v in metrics.items()},
        "issues": issues,
        "recommendations": recommendations,
    }


def explain_combined(nsys_out: Optional[str], ncu_csv_path: Optional[str]) -> Dict[str, Any]:
    """Combine NSYS + NCU explanations into a single diagnosis and action list."""
    result = {"nsys": None, "ncu": None, "combined": {}}
    if nsys_out:
        result['nsys'] = explain_nsys(nsys_out)
    if ncu_csv_path:
        result['ncu'] = explain_ncu(ncu_csv_path)

    combined_issues = set()
    combined_recs = []
    if result['nsys'] and result['nsys'].get('issues'):
        combined_issues.update(result['nsys']['issues'])
        combined_recs.extend(result['nsys'].get('recommendations', []))
    if result['ncu'] and result['ncu'].get('issues'):
        combined_issues.update(result['ncu']['issues'])
        combined_recs.extend(result['ncu'].get('recommendations', []))

    if not combined_issues:
        combined_issues.add('no_data')

    result['combined']['issues'] = list(combined_issues)
    result['combined']['recommendations'] = combined_recs
   
    result['nsys_nvtx_ranges'] = detect_nvtx_ranges(nsys_out) if nsys_out else []

    result['generated_prompt'] = generate_prompt(result)
    return result


def detect_nvtx_ranges(nsys_out: str) -> list:
    """Heuristically detect NVTX ranges from NSYS stdout/stderr text.
    Returns a list of range identifiers suitable for passing to `ncu --nvtx-include`.
    """
    if not nsys_out:
        return []
    ranges = []
    import re
  
    for m in re.finditer(r'"?([\w \-\.:/\\]+)@([\w \-\.:/\\]+)"?', nsys_out):
        dom = m.group(1).strip()
        rng = m.group(2).strip()
        token = f"{dom}@{rng}" if dom else rng
        if token not in ranges:
            ranges.append(token)

  
    for line in nsys_out.splitlines():
        if 'nvtx' in line.lower() and '@' in line:
          
            q = re.findall(r'"([^\"]+)"', line)
            for s in q:
                if s not in ranges:
                    ranges.append(s)


    return ranges


def generate_prompt(explanation: Dict[str, Any]) -> str:
    """Create an LLM-ready prompt from the combined explanation structure."""
    import pathlib

    header = "You are a CUDA performance engineer assistant. Analyze the profiling findings and propose concrete, minimal code edits or harness changes to improve performance. Always preserve numerical correctness and include a test-case adjustment if needed."
    ncu_section = explanation.get('ncu') or {}
    hotspot = ncu_section.get('hotspot') or '(unknown)'
    issues = (explanation.get('combined') or {}).get('issues', [])
    nvtx = explanation.get('nsys_nvtx_ranges') or []

    tmpl_dir = pathlib.Path(__file__).resolve().parents[1] / 'config' / 'prompt_templates'
    issue_to_template = {
        'memory_bound': 'memory_bound.txt',
        'host_device_transfers': 'memory_bound.txt',
        'low_sm_efficiency': 'compute_bound.txt',
        'compute_bound_or_unknown': 'compute_bound.txt',
        'synchronization_overhead': 'sync_bound.txt',
        'kernel_launch_overhead': 'sync_bound.txt',
    }

    selected_template = None
    for issue in issues:
        template_name = issue_to_template.get(issue)
        if not template_name:
            continue
        candidate = tmpl_dir / template_name
        if candidate.exists():
            selected_template = candidate
            break

    if selected_template is not None:
        try:
            prompt = selected_template.read_text(encoding='utf-8').replace('{{hotspot}}', hotspot)
            if nvtx:
                prompt += "\n\nNVTX ranges to focus on:\n"
                for range_name in nvtx:
                    prompt += f"- {range_name}\n"
            return prompt
        except Exception:
            pass

    parts = [header, f"Hotspot kernel: {hotspot}", "Detected issues:"]
    for issue in issues:
        parts.append(f"- {issue}")
    if nvtx:
        parts.append("NVTX ranges detected:")
        for range_name in nvtx:
            parts.append(f"- {range_name}")
    parts.append("For each suggested code edit: 1) show a minimal unified diff patch, 2) explain why it helps with the detected metric, 3) provide the validation command(s) and expected metric to check.")
    return "\n".join(parts)
