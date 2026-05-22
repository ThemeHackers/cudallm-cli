import os
import json
import threading
import time
import urllib.parse
import difflib
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
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


active_run = {
    "status": "idle",
    "input_file": None,
    "current_iteration": 0,
    "total_iterations": 0,
    "stage": "Idle",
    "logs": [],
    "history": [],
    "best_time": float('inf'),
    "original_latency": None,
    "best_code": None,
    "original_code": None,
    "current_code": None,
    "compiler_errors": [],
    "run_id": None
}

active_run_lock = threading.Lock()
stop_requested = False

def log_event(message):
    with active_run_lock:
        active_run["logs"].append({
            "timestamp": time.time(),
            "message": message
        })

def update_stage(stage_name):
    with active_run_lock:
        active_run["stage"] = stage_name
    log_event(f"Stage changed to: {stage_name}")

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
     
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

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
        
      
        if path == '/api/status':
            self.handle_api_status()
        elif path == '/api/files':
            self.handle_api_files()
        elif path == '/api/file':
            self.handle_api_file_get(query)
        elif path == '/api/optimize/status':
            self.handle_api_optimize_status()
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
        elif path == '/api/optimize':
            self.handle_api_optimize_post(body)
        elif path == '/api/optimize/stop':
            self.handle_api_optimize_stop()
        else:
            self.send_response(404)
            self.end_headers()

    def serve_static(self, filename, content_type):
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        filepath = os.path.join(static_dir, filename)
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

    def handle_api_status(self):
        env = check_environment()
        
     
        cpu = psutil.cpu_percent()
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
        files = collect_cuda_files(workspace, recursive=True)
     
        rel_files = [os.path.relpath(f, workspace) for f in files]
        self.send_json({"files": rel_files})

    def handle_api_file_get(self, query):
        file_path = query.get('path', [None])[0]
        if not file_path:
            self.send_json({"error": "Path parameter is required"}, 400)
            return

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
            self.send_json({"success": True})
        except Exception as e:
            self.send_json({"error": f"Failed to write file: {str(e)}"}, 500)

    def handle_api_optimize_status(self):
        with active_run_lock:
            self.send_json(active_run)

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
        iters = int(body.get('iters', 3))
        target = body.get('target', 'latency')
        retries = int(body.get('retries', 3))
        fast_math = bool(body.get('fast_math', False))
        opt_level = body.get('opt_level', '3')
        profile_mode = body.get('profile_mode', 'auto')
        nvtx = bool(body.get('nvtx', False))
        apply_nvtx = bool(body.get('apply_nvtx', False))

        t = threading.Thread(
            target=run_optimization_background,
            args=(abs_path, iters, target, retries, fast_math, opt_level, profile_mode, nvtx, apply_nvtx)
        )
        t.daemon = True
        t.start()

        self.send_json({"success": True, "message": "Optimization started"})

    def handle_api_roofline(self, query):
        file_path = query.get('path', [None])[0]
        if not file_path:
            self.send_json({"error": "Path parameter is required"}, 400)
            return

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
            
         
            dynamic_metrics = None
          
            self.send_json({
                "static": static_metrics,
                "dynamic": dynamic_metrics
            })
        except Exception as e:
            self.send_json({"error": f"Failed to compute roofline: {str(e)}"}, 500)


def run_optimization_background(input_file, iters, target, retries, fast_math, opt_level, profile_mode, use_nvtx, apply_nvtx):
    global stop_requested
    import uuid
    run_id = uuid.uuid4().hex[:8]

   
    with active_run_lock:
        active_run["status"] = "running"
        active_run["input_file"] = os.path.relpath(input_file, os.getcwd())
        active_run["current_iteration"] = 0
        active_run["total_iterations"] = iters
        active_run["stage"] = "Initializing"
        active_run["logs"] = []
        active_run["history"] = []
        active_run["best_time"] = float('inf')
        active_run["original_latency"] = None
        active_run["best_code"] = None
        active_run["original_code"] = None
        active_run["current_code"] = None
        active_run["compiler_errors"] = []
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
        if fast_math:
            flags.append("-use_fast_math")

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
            log_event(f"Baseline latency obtained: {baseline_time:.4f} ms")
            with active_run_lock:
                active_run["original_latency"] = baseline_time
                active_run["best_time"] = baseline_time
                active_run["history"].append({
                    "iteration": 0,
                    "latency": baseline_time,
                    "compile_success": True,
                    "gen_time": 0.0
                })

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
                history=active_run["history"]
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
                best_code = new_code
                current_code = new_code
                with active_run_lock:
                    active_run["best_code"] = new_code
                    active_run["current_code"] = new_code
                    active_run["history"].append({
                        "iteration": i + 1,
                        "latency": None,
                        "compile_success": False,
                        "gen_time": gen_time
                    })
                log_event("Finished code iteration (No compilation).")
                continue

            update_stage(f"Compiling Optimized Kernel (Iter {i+1})")
            temp_file = f"temp_kernel_{run_id}.cu"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(new_code)
            
            sandbox.file_path = temp_file
            compile_res = sandbox.compile()
            
            verification_failed = False
            prof_res = {"latency": float('inf'), "raw_output": ""}

            if compile_res['success']:
                update_stage(f"Profiling & Verifying (Iter {i+1})")
                log_event("Optimized code compiled successfully! Profiling and comparing output...")
                prof_res = sandbox.profile_latency(target_metric=target)
                if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                    verification_failed = True
                    compile_res['success'] = False
                    compile_res['error_log'] = prof_res["raw_output"]
                    log_event("Mathematical verification FAILED against baseline output.")

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
                        else:
                            verification_failed = False
                            log_event("Healed kernel compiled and verified successfully!")

                heal_attempts += 1

            if not compile_res['success']:
                log_event(f"Iteration {i+1} failed optimization. Max healing attempts reached.")
                continue

            latency = prof_res["latency"]
            log_event(f"Iteration {i+1} verified latency: {latency:.4f} ms")
            
            with active_run_lock:
                active_run["history"].append({
                    "iteration": i + 1,
                    "latency": latency,
                    "compile_success": True,
                    "gen_time": gen_time
                })

            if latency < best_time:
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
        
      
        output_file = os.path.join(
            os.path.dirname(input_file),
            f"optimized_{os.path.basename(input_file)}"
        )
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
