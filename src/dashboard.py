import os
import json
import threading
import time
import urllib.parse
import difflib
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
from copy import deepcopy
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from .discover import check_environment, find_nvcc_path, find_ncu_path, find_nsys_path
from .cli import load_config, create_llm_client, collect_cuda_files
from .sandbox import CUDASandbox
from .roofline import RooflineAnalyzer
from . import platform_info


PROFILING_FAILURE_LATENCY = 99999.0
BENCHMARK_HISTORY_LIMIT = 25
BENCHMARK_HISTORY_PATH = os.path.join(str(platform_info.get_config_dir()), "benchmark_history.json")
MAX_LIVE_SERIES_POINTS = 240

OPTIMIZATION_PRESETS = {
    "balanced": {
        "label": "Balanced",
        "description": "General-purpose optimization with conservative defaults.",
        "target": "latency",
        "default_fast_math": False,
        "default_profile_mode": "auto",
        "extra_flags": [],
        "profile_metrics": "",
        "prompt_hint": "Optimize conservatively for balanced throughput, maintainability, and correctness.",
    },
    "memory_bound": {
        "label": "Memory-bound",
        "description": "Favor coalescing, shared memory, and bandwidth reduction.",
        "target": "mem__throughput",
        "default_fast_math": False,
        "default_profile_mode": "ncu",
        "extra_flags": ["-lineinfo"],
        "profile_metrics": "dram__bytes.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
        "prompt_hint": "Prioritize memory coalescing, shared memory tiling, fewer global transactions, and reduced bandwidth waste.",
    },
    "compute_bound": {
        "label": "Compute-bound",
        "description": "Push instruction throughput and math utilization.",
        "target": "sm__throughput",
        "default_fast_math": True,
        "default_profile_mode": "ncu",
        "extra_flags": ["-use_fast_math"],
        "profile_metrics": "sm__throughput.avg.pct_of_peak_sustained_active",
        "prompt_hint": "Prioritize arithmetic throughput, instruction-level parallelism, and reduced control-flow overhead.",
    },
    "low_register_pressure": {
        "label": "Low register pressure",
        "description": "Favor occupancy and lower register usage.",
        "target": "latency",
        "default_fast_math": False,
        "default_profile_mode": "auto",
        "extra_flags": ["-lineinfo", "-maxrregcount=64"],
        "profile_metrics": "register_spill_count,registers_per_thread",
        "prompt_hint": "Reduce register pressure, spills, and live range pressure even if it means smaller unroll factors.",
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Maximize performance with higher optimization risk.",
        "target": "latency",
        "default_fast_math": True,
        "default_profile_mode": "auto",
        "extra_flags": ["-use_fast_math", "-lineinfo"],
        "profile_metrics": "sm__throughput.avg.pct_of_peak_sustained_active,dram__bytes.sum",
        "prompt_hint": "Apply aggressive loop unrolling, specialization, and throughput-oriented rewrites while preserving correctness.",
    },
}


def get_optimizer_preset(preset_name):
    return OPTIMIZATION_PRESETS.get(preset_name or "balanced", OPTIMIZATION_PRESETS["balanced"])


def load_benchmark_history():
    try:
        if os.path.exists(BENCHMARK_HISTORY_PATH):
            with open(BENCHMARK_HISTORY_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def save_benchmark_history(history_items):
    try:
        os.makedirs(os.path.dirname(BENCHMARK_HISTORY_PATH), exist_ok=True)
        with open(BENCHMARK_HISTORY_PATH, "w", encoding="utf-8") as handle:
            json.dump(history_items[-BENCHMARK_HISTORY_LIMIT:], handle, indent=2)
    except Exception:
        pass


benchmark_history = load_benchmark_history()
benchmark_history_lock = threading.Lock()


def is_profiling_failure(latency):
    return latency == PROFILING_FAILURE_LATENCY


def format_latency_value(latency):
    if latency is None or is_profiling_failure(latency):
        return "Profiling Failed"
    return f"{latency:.4f} ms"


def record_profile_result(run_state, key, prof_res):
    latency = prof_res.get("latency")
    raw_output = prof_res.get("raw_output", "")
    profile_failed = is_profiling_failure(latency)

    run_state[key] = latency
    run_state[f"{key}_profiling_failed"] = profile_failed
    run_state[f"{key}_raw_output"] = raw_output
    return latency, profile_failed, raw_output


def _new_triage_summary():
    return {
        "baseline": 0,
        "success": 0,
        "compile_failure": 0,
        "verification_failure": 0,
        "profiling_failure": 0,
        "regression_rejected": 0,
        "generated_only": 0,
        "other": 0,
    }


def _classify_history_entry(entry):
    status = str(entry.get("status", "")).lower()
    compile_success = bool(entry.get("compile_success"))
    profiling_failed = bool(entry.get("profiling_failed"))

    if status == "baseline":
        return "baseline"
    if status == "generated_only":
        return "generated_only"
    if status == "regression_guard_rejected":
        return "regression_rejected"
    if not compile_success:
        if status == "verification_failed":
            return "verification_failure"
        return "compile_failure"
    if profiling_failed:
        return "profiling_failure"
    if status in ("success", "replay_success"):
        return "success"
    return "other"


def _append_live_series_point(run_state, entry):
    latency = entry.get("latency")
    if not isinstance(latency, (int, float)) or is_profiling_failure(latency):
        return

    baseline = run_state.get("original_latency")
    speedup = None
    if isinstance(baseline, (int, float)) and baseline not in (0, float("inf")) and not is_profiling_failure(baseline):
        speedup = baseline / latency

    series = run_state.setdefault("live_series", [])
    series.append({
        "iteration": entry.get("iteration", 0),
        "latency": latency,
        "speedup": speedup,
        "status": entry.get("status", "unknown"),
        "triage": entry.get("triage", "other"),
        "timestamp": time.time(),
    })
    if len(series) > MAX_LIVE_SERIES_POINTS:
        del series[:-MAX_LIVE_SERIES_POINTS]


def append_history_entry(run_state, entry):
    triage = entry.get("triage") or _classify_history_entry(entry)
    entry["triage"] = triage

    run_state.setdefault("history", []).append(entry)

    triage_summary = run_state.setdefault("triage_summary", _new_triage_summary())
    if triage not in triage_summary:
        triage_summary[triage] = 0
    triage_summary[triage] += 1

    _append_live_series_point(run_state, entry)
    return entry


active_run = {
    "status": "idle",
    "input_file": None,
    "current_iteration": 0,
    "total_iterations": 0,
    "stage": "Idle",
    "preset": "balanced",
    "target": "latency",
    "profile_mode": "auto",
    "profile_metrics": "",
    "flags": [],
    "logs": [],
    "history": [],
    "best_time": float('inf'),
    "original_latency": None,
    "original_latency_profiling_failed": False,
    "original_latency_raw_output": "",
    "benchmark_alerts": [],
    "benchmark_summary": None,
    "best_code": None,
    "original_code": None,
    "current_code": None,
    "compiler_errors": [],
    "triage_summary": _new_triage_summary(),
    "live_series": [],
    "regression_guard": {
        "enabled": True,
        "max_regression_pct": 5.0,
        "auto_rollback": True,
    },
    "regression_guard_events": [],
    "run_id": None
}

active_run_lock = threading.Lock()
stop_requested = False

def _strip_markup(text):
    """Strip Rich-style markup tags and ANSI escape sequences from text."""
    import re

    text = re.sub(r'\[/?[a-zA-Z_ ]+\]', '', text)
   
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    return text

def log_event(message):
    message = _strip_markup(message)
    with active_run_lock:
        active_run["logs"].append({
            "timestamp": time.time(),
            "message": message
        })

def update_stage(stage_name):
    with active_run_lock:
        active_run["stage"] = stage_name
    log_event(f"Stage changed to: {stage_name}")

def ensure_optimized_path(file_path):
    workspace = os.path.abspath(os.getcwd())
    if os.path.isabs(file_path):
        abs_path = os.path.abspath(file_path)
        if abs_path.startswith(workspace):
            return abs_path
        return os.path.join(workspace, "examples", os.path.basename(file_path))
    
    normalized = file_path.replace('\\', '/')
    parts = [p for p in normalized.split('/') if p]
    if parts and parts[0] in ('examples', 'optimized'):
        return os.path.join(workspace, normalized)
    return os.path.join(workspace, "examples", file_path)


def find_latest_ncu_csv_for_file(target_rel_path):
    target = (target_rel_path or "").replace('\\', '/')

    with active_run_lock:
        current = deepcopy(active_run)
    candidates = [current]

    with benchmark_history_lock:
        candidates.extend(deepcopy(benchmark_history[-BENCHMARK_HISTORY_LIMIT:]))

    for run in reversed(candidates):
        if (run.get("input_file") or "").replace('\\', '/') != target:
            continue
        for item in reversed(run.get("history", [])):
            csv_path = item.get("ncu_csv")
            if csv_path and os.path.exists(csv_path):
                return csv_path
    return None


def build_benchmark_summary(run_state):
    history = run_state.get("history", [])
    profiling_failures = sum(1 for item in history if item.get("profiling_failed"))
    compile_failures = sum(1 for item in history if not item.get("compile_success"))
    successful_items = [item for item in history if item.get("compile_success") and not item.get("profiling_failed") and isinstance(item.get("latency"), (int, float)) and not is_profiling_failure(item.get("latency"))]

    original_latency = run_state.get("original_latency")
    best_time = run_state.get("best_time")
    speedup = None
    if isinstance(original_latency, (int, float)) and isinstance(best_time, (int, float)) and best_time not in (0, float("inf")) and not is_profiling_failure(original_latency) and not is_profiling_failure(best_time):
        speedup = original_latency / best_time

    regression_guard = run_state.get("regression_guard") or {}
    regression_threshold_pct = float(regression_guard.get("max_regression_pct", 5.0))

    triage_summary = _new_triage_summary()
    for item in history:
        triage = item.get("triage") or _classify_history_entry(item)
        triage_summary[triage] = triage_summary.get(triage, 0) + 1

    alerts = []
    if profiling_failures >= 2:
        alerts.append(f"Repeated profiling failures detected ({profiling_failures}).")
    if compile_failures >= 2:
        alerts.append(f"Repeated compile/verification failures detected ({compile_failures}).")
    if successful_items and isinstance(best_time, (int, float)) and not is_profiling_failure(best_time):
        best_success = min(item["latency"] for item in successful_items)
        previous_entry = None
        with benchmark_history_lock:
            for candidate in reversed(benchmark_history):
                if candidate.get("input_file") == run_state.get("input_file") and candidate.get("best_time") not in (None, float("inf")):
                    previous_entry = candidate
                    break
        if previous_entry:
            previous_best = previous_entry.get("best_time")
            threshold_scale = 1.0 + (regression_threshold_pct / 100.0)
            if isinstance(previous_best, (int, float)) and previous_best not in (0, float("inf")) and best_success > previous_best * threshold_scale:
                delta_pct = ((best_success - previous_best) / previous_best) * 100.0
                alerts.append(f"Regression alert: best latency is {delta_pct:.1f}% slower than the previous recorded run.")

    summary = {
        "run_id": run_state.get("run_id"),
        "timestamp": time.time(),
        "status": run_state.get("status"),
        "stage": run_state.get("stage"),
        "input_file": run_state.get("input_file"),
        "target": run_state.get("target"),
        "preset": run_state.get("preset"),
        "profile_mode": run_state.get("profile_mode"),
        "profile_metrics": run_state.get("profile_metrics", ""),
        "flags": run_state.get("flags", []),
        "iters": run_state.get("total_iterations", 0),
        "best_time": best_time,
        "original_latency": original_latency,
        "speedup": speedup,
        "profiling_failures": profiling_failures,
        "compile_failures": compile_failures,
        "triage_summary": triage_summary,
        "live_series": deepcopy(run_state.get("live_series", [])),
        "regression_guard": deepcopy(regression_guard),
        "regression_guard_events": deepcopy(run_state.get("regression_guard_events", [])),
        "alerts": alerts,
        "history": deepcopy(history),
        "original_code": run_state.get("original_code"),
        "best_code": run_state.get("best_code"),
        "current_code": run_state.get("current_code"),
    }
    return summary


def persist_benchmark_summary(run_state):
    summary = build_benchmark_summary(run_state)
    with benchmark_history_lock:
        benchmark_history.append(summary)
        save_benchmark_history(benchmark_history)
    with active_run_lock:
        active_run["benchmark_alerts"] = summary["alerts"]
        active_run["benchmark_summary"] = summary
    return summary


def replay_benchmark_iteration(run_record, iteration):
    history = run_record.get("history", [])
    selected = next((item for item in history if item.get("iteration") == iteration), None)
    if not selected:
        raise ValueError("Iteration not found in benchmark history")

    code_snapshot = selected.get("code_snapshot")
    if not code_snapshot:
        raise ValueError("No code snapshot stored for that iteration")

    flags = run_record.get("flags", [])
    profile_mode = run_record.get("profile_mode", "auto")
    profile_metrics = run_record.get("profile_metrics", "")
    target = run_record.get("target", "latency")

    temp_name = f"replay_{run_record.get('run_id', 'run')}_{iteration}.cu"
    with open(temp_name, "w", encoding="utf-8") as handle:
        handle.write(code_snapshot)

    sandbox = CUDASandbox(
        temp_name,
        flags=flags,
        profile_mode=profile_mode,
        profile_metrics=profile_metrics,
    )
    try:
        compile_result = sandbox.compile()
        replay_result = {
            "iteration": iteration,
            "compile_success": compile_result.get("success", False),
            "error_log": compile_result.get("error_log", ""),
            "code_snapshot": code_snapshot,
        }
        if compile_result.get("success"):
            prof_res = sandbox.profile_latency(target_metric=target)
            replay_result.update({
                "latency": prof_res.get("latency"),
                "raw_output": prof_res.get("raw_output", ""),
                "profiling_failed": is_profiling_failure(prof_res.get("latency")),
            })
        else:
            replay_result.update({
                "latency": PROFILING_FAILURE_LATENCY,
                "raw_output": compile_result.get("error_log", ""),
                "profiling_failed": False,
            })
        return replay_result
    finally:
        try:
            sandbox.cleanup()
        except Exception:
            pass
        try:
            os.remove(temp_name)
        except Exception:
            pass

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
     
        pass

    def send_json(self, data, status=200):
        import math
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(x) for x in obj]
            elif isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return None
            return obj

        sanitized_data = sanitize(data)
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(sanitized_data).encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

     
        if path == '/' or path == '/index.html':
            self.serve_static('index.html', 'text/html')
            return
        elif path == '/favicon.ico':
            self.serve_static('favicon.ico', 'image/x-icon')
            return
        elif path == '/favicon.png':
            self.serve_static('favicon.png', 'image/png')
            return
        elif path == '/favicon.svg':
            self.serve_static('favicon.svg', 'image/svg+xml')
            return
        
       
        if path == '/api/status':
            self.handle_api_status()
        elif path == '/api/files':
            self.handle_api_files()
        elif path == '/api/file':
            self.handle_api_file_get(query)
        elif path == '/api/optimize/status':
            self.handle_api_optimize_status()
        elif path == '/api/benchmark/history':
            self.handle_api_benchmark_history()
        elif path == '/api/optimizer/presets':
            self.handle_api_optimizer_presets()
        elif path == '/api/roofline':
            self.handle_api_roofline(query)
        else:
           
            self.serve_static('index.html', 'text/html')

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b''
        
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            self.send_json({"error": "Invalid JSON body"}, 400)
            return

        if path == '/api/file':
            self.handle_api_file_post(body)
        elif path == '/api/file/delete':
            self.handle_api_file_delete(body)
        elif path == '/api/optimize':
            self.handle_api_optimize_post(body)
        elif path == '/api/optimize/stop':
            self.handle_api_optimize_stop()
        elif path == '/api/benchmark/replay':
            self.handle_api_benchmark_replay(body)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_static(self, filename, content_type):
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        filepath = os.path.join(static_dir, filename)
        try:
            if not os.path.exists(filepath):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Static assets not found. Build or place them in src/static/")
                return

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass

    def handle_api_status(self):
        env = check_environment()
        
    
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        
        gpu_name = env.get("gpu_model", "Unknown")
        vram_total_gb = 0.0
        vram_used_gb = 0.0
        gpu_util = 0

        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = util.gpu
                vram_total_gb = mem.total / (1024 ** 3)
                vram_used_gb = mem.used / (1024 ** 3)
            except Exception:
                pass

        config = load_config()

        status_data = {
            "gpu_model": gpu_name,
            "compute_capability": env.get("compute_capability", "N/A"),
            "cuda_version": env.get("cuda_version", "N/A"),
            "cpu_usage": cpu,
            "ram_total": ram.total / (1024 ** 3),
            "ram_used": ram.used / (1024 ** 3),
            "gpu_usage": gpu_util,
            "vram_total": vram_total_gb,
            "vram_used": vram_used_gb,
            "tools": {
                "nvcc": env.get("nvcc_found", False),
                "ncu": env.get("ncu_found", False),
                "nsys": env.get("nsys_found", False),
            },
            "config": {
                "llm_url": config.get("llm_url"),
                "llm_verify_tls": config.get("llm_verify_tls", True),
                "llm_allow_insecure_remote": config.get("llm_allow_insecure_remote", False)
            }
        }
        self.send_json(status_data)

    def handle_api_files(self):
        workspace = os.getcwd()
        examples_dir = os.path.join(workspace, "examples")
        optimized_dir = os.path.join(workspace, "optimized")
        
        os.makedirs(examples_dir, exist_ok=True)
        os.makedirs(optimized_dir, exist_ok=True)
        
        files = collect_cuda_files(examples_dir, recursive=True) + collect_cuda_files(optimized_dir, recursive=True)
        
        rel_files = []
        seen = set()
        for f in files:
            rel = os.path.relpath(f, workspace)
            if rel not in seen:
                seen.add(rel)
                rel_files.append(rel)
                
        self.send_json({"files": sorted(rel_files)})

    def handle_api_file_get(self, query):
        file_path = query.get('path', [None])[0]
        if not file_path:
            self.send_json({"error": "Path parameter is required"}, 400)
            return

        file_path = ensure_optimized_path(file_path)
        abs_path = os.path.abspath(file_path)
        workspace = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace):
            self.send_json({"error": "Access denied: Path lies outside workspace"}, 403)
            return

        if not os.path.exists(abs_path):
            self.send_json({"error": "File not found"}, 404)
            return

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_json({"path": file_path, "content": content})
        except Exception as e:
            self.send_json({"error": f"Failed to read file: {str(e)}"}, 500)

    def handle_api_file_post(self, body):
        file_path = body.get('path')
        content = body.get('content')
        if not file_path or content is None:
            self.send_json({"error": "path and content are required fields"}, 400)
            return

        file_path = ensure_optimized_path(file_path)
        abs_path = os.path.abspath(file_path)
        workspace = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace):
            self.send_json({"error": "Access denied: Path lies outside workspace"}, 403)
            return

        try:
            dir_name = os.path.dirname(abs_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            rel_path = os.path.relpath(abs_path, workspace).replace('\\', '/')
            self.send_json({"success": True, "path": rel_path})
        except Exception as e:
            self.send_json({"error": f"Failed to write file: {str(e)}"}, 500)

    def handle_api_file_delete(self, body):
        file_path = body.get('path')
        if not file_path:
            self.send_json({"error": "path is a required field"}, 400)
            return

        file_path = ensure_optimized_path(file_path)
        abs_path = os.path.abspath(file_path)
        workspace = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace):
            self.send_json({"error": "Access denied: Path lies outside workspace"}, 403)
            return

        if not os.path.exists(abs_path):
            self.send_json({"error": "File not found"}, 404)
            return

        try:
            os.remove(abs_path)
            self.send_json({"success": True, "message": "File deleted successfully"})
        except Exception as e:
            self.send_json({"error": f"Failed to delete file: {str(e)}"}, 500)

    def handle_api_optimize_status(self):
        with active_run_lock:
            self.send_json(active_run)

    def handle_api_benchmark_history(self):
        with benchmark_history_lock:
            self.send_json({"items": benchmark_history[-BENCHMARK_HISTORY_LIMIT:]})

    def handle_api_optimizer_presets(self):
        self.send_json({
            "presets": [
                {"name": key, **value} for key, value in OPTIMIZATION_PRESETS.items()
            ]
        })

    def handle_api_benchmark_replay(self, body):
        run_id = body.get("run_id")
        iteration = int(body.get("iteration", 0))
        if not run_id or iteration <= 0:
            self.send_json({"error": "run_id and iteration are required"}, 400)
            return

        record = None
        with active_run_lock:
            if active_run.get("run_id") == run_id:
                record = deepcopy(active_run)

        if record is None:
            with benchmark_history_lock:
                for item in reversed(benchmark_history):
                    if item.get("run_id") == run_id:
                        record = deepcopy(item)
                        break

        if record is None:
            self.send_json({"error": "Benchmark run not found"}, 404)
            return

        try:
            replay_result = replay_benchmark_iteration(record, iteration)
            self.send_json({"success": True, **replay_result})
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def handle_api_optimize_stop(self):
        global stop_requested
        with active_run_lock:
            if active_run["status"] == "running":
                stop_requested = True
                log_event("User requested to stop the optimization run.")
                self.send_json({"success": True, "message": "Stop requested"})
            else:
                self.send_json({"error": "No active optimization run to stop"}, 400)

    def handle_api_optimize_post(self, body):
        global stop_requested
        file_path = body.get('path')
        if not file_path:
            self.send_json({"error": "File path is required"}, 400)
            return

        file_path = ensure_optimized_path(file_path)
        abs_path = os.path.abspath(file_path)
        workspace = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace):
            self.send_json({"error": "Access denied: Path lies outside workspace"}, 403)
            return

        if not os.path.exists(abs_path):
            self.send_json({"error": "File not found"}, 404)
            return

        
        with active_run_lock:
            if active_run["status"] == "running":
                self.send_json({"error": "An optimization task is already running"}, 409)
                return

     
        stop_requested = False
        preset = body.get('preset', 'balanced')
        preset_config = get_optimizer_preset(preset)
        iters = int(body.get('iters', 3))
        target = body.get('target', preset_config.get('target', 'latency'))
        retries = int(body.get('retries', 3))
        fast_math = bool(body.get('fast_math', False) or preset_config.get('default_fast_math', False))
        opt_level = body.get('opt_level', '3')
        profile_mode = body.get('profile_mode', preset_config.get('default_profile_mode', 'auto'))
        nvtx = bool(body.get('nvtx', False))
        apply_nvtx = bool(body.get('apply_nvtx', False))
        regression_guard_enabled = bool(body.get('regression_guard_enabled', True))
        try:
            max_regression_pct = float(body.get('max_regression_pct', 5.0))
        except (TypeError, ValueError):
            max_regression_pct = 5.0
        max_regression_pct = max(0.0, min(max_regression_pct, 200.0))
        auto_rollback_on_regression = bool(body.get('auto_rollback_on_regression', True))

        regression_guard = {
            "enabled": regression_guard_enabled,
            "max_regression_pct": max_regression_pct,
            "auto_rollback": auto_rollback_on_regression,
        }

        t = threading.Thread(
            target=run_optimization_background,
            args=(abs_path, iters, target, retries, fast_math, opt_level, profile_mode, nvtx, apply_nvtx, preset, preset_config, regression_guard)
        )
        t.daemon = True
        t.start()

        self.send_json({"success": True, "message": "Optimization started"})

    def handle_api_roofline(self, query):
        file_path = query.get('path', [None])[0]
        if not file_path:
            self.send_json({"error": "Path parameter is required"}, 400)
            return

        file_path = ensure_optimized_path(file_path)
        abs_path = os.path.abspath(file_path)
        workspace = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace):
            self.send_json({"error": "Access denied"}, 403)
            return

        if not os.path.exists(abs_path):
            self.send_json({"error": "File not found"}, 404)
            return

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                code = f.read()

            static_metrics = RooflineAnalyzer.analyze_static(code)
            rel_path = os.path.relpath(abs_path, workspace).replace('\\', '/')
            ncu_csv_hint = query.get('ncu_csv', [None])[0]
            dynamic_metrics = None

            if ncu_csv_hint:
                hinted = os.path.abspath(ncu_csv_hint)
                if hinted.startswith(workspace) and os.path.exists(hinted):
                    dynamic_metrics = RooflineAnalyzer.analyze_ncu_csv(hinted)

            if dynamic_metrics is None:
                latest_csv = find_latest_ncu_csv_for_file(rel_path)
                if latest_csv:
                    dynamic_metrics = RooflineAnalyzer.analyze_ncu_csv(latest_csv)

            blended_metrics = RooflineAnalyzer.blend_static_dynamic(static_metrics, dynamic_metrics)

            self.send_json({
                "static": static_metrics,
                "dynamic": dynamic_metrics,
                "blended": blended_metrics
            })
        except Exception as e:
            self.send_json({"error": f"Failed to compute roofline: {str(e)}"}, 500)


def run_optimization_background(input_file, iters, target, retries, fast_math, opt_level, profile_mode, use_nvtx, apply_nvtx, preset_name, preset_config, regression_guard):
    global stop_requested
    import uuid
    run_id = uuid.uuid4().hex[:8]

   
    with active_run_lock:
        active_run["status"] = "running"
        active_run["input_file"] = os.path.relpath(input_file, os.getcwd())
        active_run["current_iteration"] = 0
        active_run["total_iterations"] = iters
        active_run["stage"] = "Initializing"
        active_run["preset"] = preset_name
        active_run["target"] = target
        active_run["profile_mode"] = profile_mode
        active_run["profile_metrics"] = preset_config.get("profile_metrics", "")
        active_run["flags"] = []
        active_run["logs"] = []
        active_run["history"] = []
        active_run["best_time"] = float('inf')
        active_run["original_latency"] = None
        active_run["original_latency_profiling_failed"] = False
        active_run["original_latency_raw_output"] = ""
        active_run["benchmark_alerts"] = []
        active_run["benchmark_summary"] = None
        active_run["best_code"] = None
        active_run["original_code"] = None
        active_run["current_code"] = None
        active_run["compiler_errors"] = []
        active_run["triage_summary"] = _new_triage_summary()
        active_run["live_series"] = []
        active_run["regression_guard"] = {
            "enabled": bool(regression_guard.get("enabled", True)),
            "max_regression_pct": float(regression_guard.get("max_regression_pct", 5.0)),
            "auto_rollback": bool(regression_guard.get("auto_rollback", True)),
        }
        active_run["regression_guard_events"] = []
        active_run["run_id"] = run_id

    log_event(f"Starting optimization run ID: {run_id} for file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()

        with active_run_lock:
            active_run["original_code"] = original_code
            active_run["current_code"] = original_code
            active_run["best_code"] = original_code

        flags = [f"-O{opt_level}"]
        flags.extend(preset_config.get("extra_flags", []))
        if fast_math:
            if "-use_fast_math" not in flags:
                flags.append("-use_fast_math")

        with active_run_lock:
            active_run["flags"] = flags[:]

        compile_enabled = os.path.splitext(input_file)[1].lower() == ".cu"
        if not compile_enabled:
            log_event("Non-.cu file detected. Running in code-only mode.")
            iters = 1
            sandbox = None
        else:
            sandbox = CUDASandbox(
                input_file,
                flags=flags,
                profile_mode=profile_mode,
                use_nvtx=use_nvtx,
                profile_metrics=preset_config.get("profile_metrics", ""),
                apply_nvtx_suggestion=apply_nvtx
            )

        config = load_config()
        llm = create_llm_client(config)
        env_info = check_environment()

        best_time = float('inf')
        best_code = original_code
        current_code = original_code

       
        if compile_enabled:
            update_stage("Baseline Compilation")
            log_event("Compiling original baseline kernel...")
            sandbox.file_path = input_file
            compile_res = sandbox.compile()
            if not compile_res['success']:
                log_event(f"Error compiling baseline kernel: {compile_res['error_log']}")
                with active_run_lock:
                    active_run["status"] = "failed"
                    active_run["stage"] = "Baseline Compilation Failed"
                return

            log_event("Profiling baseline kernel...")
            prof_res = sandbox.profile_latency(target_metric=target)
            if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                log_event("Verification failure running baseline kernel!")
                with active_run_lock:
                    active_run["status"] = "failed"
                    active_run["stage"] = "Baseline Verification Failed"
                return

            baseline_time = prof_res["latency"]
            best_time = baseline_time
            with active_run_lock:
                _, baseline_failed, baseline_raw_output = record_profile_result(active_run, "original_latency", prof_res)
                active_run["best_time"] = baseline_time
                append_history_entry(active_run, {
                    "iteration": 0,
                    "latency": baseline_time,
                    "profiling_failed": baseline_failed,
                    "raw_output": baseline_raw_output,
                    "ncu_csv": prof_res.get("ncu_csv"),
                    "compile_success": True,
                    "gen_time": 0.0,
                    "status": "baseline",
                    "code_snapshot": original_code,
                    "parent_code_snapshot": original_code,
                })
            if baseline_failed:
                log_event(f"Baseline profiling failed: {baseline_raw_output or 'No profiler output captured.'}")
            else:
                log_event(f"Baseline latency obtained: {baseline_time:.4f} ms")

        for i in range(iters):
            if stop_requested:
                log_event("Optimization stopped by user request.")
                break

            with active_run_lock:
                active_run["current_iteration"] = i + 1
            log_event(f"--- Starting Iteration {i+1}/{iters} ---")
            
            update_stage(f"Generating Prompt (Iter {i+1})")
            ncu_csv_path = prof_res.get("ncu_csv") if (i > 0 and 'prof_res' in locals()) else None
            prompt = llm.create_optimization_prompt(
                current_code,
                env_info,
                target,
                best_time,
                flags,
                ncu_csv=ncu_csv_path,
                history=active_run["history"],
                preset={"name": preset_name, **preset_config}
            )

            update_stage(f"LLM Reasoning & Generation (Iter {i+1})")
            def llm_callback(msg):
                log_event(f"LLM Status: {msg}")

            new_code, gen_time = llm.generate_code(
                prompt,
                status_callback=llm_callback,
                prefill=True
            )

            if stop_requested:
                break

            if not new_code:
                log_event("LLM did not return any CUDA code. Skipping iteration.")
                continue

            if not compile_enabled:
                parent_snapshot = current_code
                best_code = new_code
                current_code = new_code
                with active_run_lock:
                    active_run["best_code"] = new_code
                    active_run["current_code"] = new_code
                    append_history_entry(active_run, {
                        "iteration": i + 1,
                        "latency": None,
                        "compile_success": False,
                        "gen_time": gen_time,
                        "status": "generated_only",
                        "code_snapshot": new_code,
                        "parent_code_snapshot": parent_snapshot,
                    })
                log_event("Finished code iteration (No compilation).")
                continue

            update_stage(f"Compiling Optimized Kernel (Iter {i+1})")
            temp_file = f"temp_kernel_{run_id}.cu"
            parent_snapshot = current_code
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(new_code)
            
            sandbox.file_path = temp_file
            compile_res = sandbox.compile()
            
            verification_failed = False
            prof_res = {"latency": float('inf'), "raw_output": ""}
            iteration_status = "compile_failed"
            iteration_error_log = ""

            if compile_res['success']:
                update_stage(f"Profiling & Verifying (Iter {i+1})")
                log_event("Optimized code compiled successfully! Profiling and comparing output...")
                iteration_status = "success"
                prof_res = sandbox.profile_latency(target_metric=target)
                if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                    verification_failed = True
                    compile_res['success'] = False
                    compile_res['error_log'] = prof_res["raw_output"]
                    log_event("Mathematical verification FAILED against baseline output.")
                    iteration_status = "verification_failed"
                    iteration_error_log = prof_res["raw_output"]

            heal_attempts = 0
            while not compile_res['success'] and heal_attempts < retries:
                if stop_requested:
                    break

                with active_run_lock:
                    active_run["compiler_errors"].append({
                        "iteration": i + 1,
                        "attempt": heal_attempts + 1,
                        "log": compile_res['error_log']
                    })

                if verification_failed:
                    update_stage(f"Healing Verification Failure (Attempt {heal_attempts+1})")
                    log_event("Requesting LLM to correct mathematical correctness mismatch...")
                    repair_prompt = llm.create_healing_prompt(new_code, compile_res['error_log'])
                else:
                    update_stage(f"Healing Compilation Failure (Attempt {heal_attempts+1})")
                    log_event("Requesting LLM to resolve syntax/compilation issues...")
                    repair_prompt = llm.create_compile_repair_prompt(
                        new_code,
                        compile_res['error_log'],
                        attempt_index=heal_attempts + 1,
                        max_attempts=retries
                    )

                new_code, gen_time = llm.generate_code(
                    repair_prompt,
                    status_callback=llm_callback,
                    prefill=True
                )

                if stop_requested:
                    break

                if new_code:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(new_code)
                    
                    update_stage(f"Re-compiling Healed Kernel (Attempt {heal_attempts+1})")
                    compile_res = sandbox.compile()
                    if compile_res['success']:
                        update_stage(f"Re-profiling & Verifying (Attempt {heal_attempts+1})")
                        prof_res = sandbox.profile_latency(target_metric=target)
                        if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                            verification_failed = True
                            compile_res['success'] = False
                            compile_res['error_log'] = prof_res["raw_output"]
                            log_event("Re-compiled healed kernel FAILED mathematical verification.")
                            iteration_status = "verification_failed"
                            iteration_error_log = prof_res["raw_output"]
                        else:
                            verification_failed = False
                            log_event("Healed kernel compiled and verified successfully!")

                heal_attempts += 1

            if not compile_res['success']:
                log_event(f"Iteration {i+1} failed optimization. Max healing attempts reached.")
                if not iteration_error_log:
                    iteration_error_log = compile_res.get("error_log", "")
                with active_run_lock:
                    append_history_entry(active_run, {
                        "iteration": i + 1,
                        "latency": PROFILING_FAILURE_LATENCY,
                        "profiling_failed": iteration_status == "verification_failed",
                        "raw_output": prof_res.get("raw_output", compile_res.get("error_log", "")),
                        "compile_success": False,
                        "gen_time": gen_time,
                        "status": iteration_status,
                        "error_log": iteration_error_log,
                        "code_snapshot": new_code,
                        "parent_code_snapshot": parent_snapshot,
                    })
                continue

            latency = prof_res["latency"]
            profile_failed = is_profiling_failure(latency)
            if profile_failed:
                log_event(f"Iteration {i+1} profiling failed: {prof_res.get('raw_output', 'No profiler output captured.')}")
            else:
                log_event(f"Iteration {i+1} verified latency: {latency:.4f} ms")

            regression_cfg = active_run.get("regression_guard", {})
            guard_enabled = bool(regression_cfg.get("enabled", True))
            guard_threshold = float(regression_cfg.get("max_regression_pct", 5.0))
            auto_rollback = bool(regression_cfg.get("auto_rollback", True))

            baseline_valid = isinstance(baseline_time, (int, float)) and not is_profiling_failure(baseline_time) and baseline_time > 0
            rejected_by_guard = False
            guard_message = ""
            if guard_enabled and baseline_valid and not profile_failed:
                max_allowed = baseline_time * (1.0 + (guard_threshold / 100.0))
                if latency > max_allowed:
                    rejected_by_guard = True
                    delta_pct = ((latency - baseline_time) / baseline_time) * 100.0
                    guard_message = (
                        f"Iteration {i+1} rejected by regression guard: "
                        f"latency {latency:.4f} ms is {delta_pct:.1f}% slower than baseline "
                        f"(threshold {guard_threshold:.1f}%)."
                    )
                    log_event(guard_message)

                    with active_run_lock:
                        active_run.setdefault("regression_guard_events", []).append({
                            "iteration": i + 1,
                            "latency": latency,
                            "baseline_latency": baseline_time,
                            "delta_pct": delta_pct,
                            "threshold_pct": guard_threshold,
                            "timestamp": time.time(),
                            "message": guard_message,
                        })
                        if len(active_run["regression_guard_events"]) > 120:
                            del active_run["regression_guard_events"][:-120]
            
            with active_run_lock:
                append_history_entry(active_run, {
                    "iteration": i + 1,
                    "latency": latency,
                    "profiling_failed": profile_failed,
                    "raw_output": prof_res.get("raw_output", ""),
                    "ncu_csv": prof_res.get("ncu_csv"),
                    "compile_success": True,
                    "gen_time": gen_time,
                    "status": "regression_guard_rejected" if rejected_by_guard else "success",
                    "error_log": guard_message,
                    "code_snapshot": new_code,
                    "parent_code_snapshot": parent_snapshot,
                })

            if rejected_by_guard:
                if auto_rollback and best_code:
                    current_code = best_code
                    log_event(f"Regression guard rollback applied. Restored best-known candidate at iteration {i+1}.")
                else:
                    current_code = new_code
                with active_run_lock:
                    active_run["current_code"] = current_code
                continue

            if not profile_failed and latency < best_time:
                best_time = latency
                best_code = new_code
                log_event(f"New best latency achieved! Speedup: {baseline_time / latency:.2f}x")
                with active_run_lock:
                    active_run["best_time"] = latency
                    active_run["best_code"] = best_code

            current_code = new_code
            with active_run_lock:
                active_run["current_code"] = current_code

     
        if sandbox:
            sandbox.cleanup()
        
      
        optimized_dir = os.path.join(os.getcwd(), "optimized")
        os.makedirs(optimized_dir, exist_ok=True)
        base_name = os.path.basename(input_file)
        if not base_name.startswith("optimized_"):
            base_name = f"optimized_{base_name}"
        output_file = os.path.join(optimized_dir, base_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(best_code)
        
        log_event(f"Optimization finished. Best code written to: {output_file}")
        
        with active_run_lock:
            active_run["status"] = "completed"
            active_run["stage"] = "Completed"

    except Exception as e:
        err_msg = traceback.format_exc()
        log_event(f"Fatal error in optimization thread: {err_msg}")
        with active_run_lock:
            active_run["status"] = "failed"
            active_run["stage"] = f"Fatal Error: {str(e)}"
    finally:
        temp_file = f"temp_kernel_{run_id}.cu"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        try:
            with active_run_lock:
                if active_run.get("run_id") == run_id:
                    persist_benchmark_summary(deepcopy(active_run))
        except Exception:
            pass


def start_dashboard_server(port=8000):
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    server_address = ('', port)
    class ThreadingHTTPServer(ThreadingTCPServer, HTTPServer):
        pass

    httpd = ThreadingHTTPServer(server_address, DashboardHTTPHandler)
    print(f"CUDA LLM Dashboard running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()
