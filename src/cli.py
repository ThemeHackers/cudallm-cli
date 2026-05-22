import click
import re
import json
import os
import difflib
import sys
import subprocess
import glob
import time
import shutil
import requests
from datetime import datetime
from .discover import check_environment, discover_tool_paths, find_ncu_path, find_nsys_path
from .sandbox import CUDASandbox
from .llm_client import LLMClient
from .profiler_tools import run_nsys, run_ncu_broad, parse_ncu_csv_for_hotspot, summarize_profile_outputs
from .network_security import validate_llm_endpoint
from .docker_sandbox import run_in_docker
from .terminal_manager import TerminalManager
from . import platform_info
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.console import Group
from rich.spinner import Spinner
from rich.text import Text

console = Console()

CONFIG_PATH = str(platform_info.get_config_path())
_LEGACY_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

DEFAULT_CONFIG = {
    "llm_url": f"http://127.0.0.1:{platform_info.get_default_llm_port()}/v1/completions",
    "llm_api_key": None,
    "llm_api_key_file": None,
    "llm_verify_tls": True,
    "llm_allow_insecure_remote": False,
    "max_stream_chunks": 2000,
}

CUDA_EXTENSIONS = {".cu", ".cuh"}
IGNORE_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "node_modules"}




def _probe_llm_server(test_url, headers, verify_tls):
    return requests.get(test_url, headers=headers, timeout=3, verify=verify_tls)

def collect_cuda_files(input_path, recursive=True, exclude_dirs=None):
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        return []

    excluded = set(exclude_dirs or [])
    files = []
    for root, dirs, filenames in os.walk(input_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and d not in excluded]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in CUDA_EXTENSIONS:
                files.append(os.path.join(root, name))
        if not recursive:
            break
    return sorted(files)

def locate_and_setup_msvc():
    if sys.platform != "win32":
        return True

    if shutil.which("cl") and "INCLUDE" in os.environ:
        return True

    bat_paths = [
        "C:\\Program Files\\Microsoft Visual Studio\\*\\*\\VC\\Auxiliary\\Build\\vcvars64.bat",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\*\\VC\\Auxiliary\\Build\\vcvars64.bat"
    ]
    found_bats = []
    for pattern in bat_paths:
        found_bats.extend(glob.glob(pattern))

    if found_bats:
        found_bats.sort(reverse=True)
        bat_path = found_bats[0]
        cmd = f'call "{bat_path}" && set'
        try:
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key_upper = key.upper()
                        if key_upper in {"PATH", "INCLUDE", "LIB", "LIBPATH"}:
                            os.environ[key_upper] = val
                return True
        except Exception:
            pass

    search_paths = [
        "C:\\Program Files\\Microsoft Visual Studio\\*\\*\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\*\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"
    ]
    found_paths = []
    for pattern in search_paths:
        found_paths.extend(glob.glob(pattern))

    if found_paths:
        found_paths.sort(reverse=True)
        msvc_path = found_paths[0]
        os.environ["PATH"] = msvc_path + os.pathsep + os.environ["PATH"]
        return True

    return shutil.which("cl") is not None

def migrate_legacy_config():
    """Copy project-relative config/config.json to ~/.cudallm/config.json if needed."""
    if os.path.exists(CONFIG_PATH):
        return
    if os.path.exists(_LEGACY_CONFIG_PATH):
        config_dir = os.path.dirname(CONFIG_PATH)
        os.makedirs(config_dir, exist_ok=True)
        try:
            shutil.copy2(_LEGACY_CONFIG_PATH, CONFIG_PATH)
        except Exception:
            pass


def load_config():
    migrate_legacy_config()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            loaded = json.load(f)
            return {**DEFAULT_CONFIG, **loaded}
    return DEFAULT_CONFIG.copy()

def save_config(config_data):
    config_dir = os.path.dirname(CONFIG_PATH)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f, indent=2)
    try:
        legacy_dir = os.path.dirname(_LEGACY_CONFIG_PATH)
        if legacy_dir and os.path.isdir(legacy_dir):
            with open(_LEGACY_CONFIG_PATH, 'w') as f:
                json.dump(config_data, f, indent=2)
    except Exception:
        pass


def apply_llm_overrides(config, llm_url=None, llm_api_key=None, llm_api_key_file=None, insecure=False):
    if llm_url:
        config['llm_url'] = llm_url
    if llm_api_key:
        config['llm_api_key'] = llm_api_key
        config['llm_api_key_file'] = None
    if llm_api_key_file:
        config['llm_api_key_file'] = llm_api_key_file
        config['llm_api_key'] = None
    if insecure:
        config['llm_verify_tls'] = False
        config['llm_allow_insecure_remote'] = True
    return config


def apply_public_url_override(config, public_url, allow_insecure_remote=False):
    validated = validate_llm_endpoint(public_url, allow_insecure_remote=allow_insecure_remote)
    config['llm_url'] = public_url
    config['llm_verify_tls'] = validated['is_secure']
    if allow_insecure_remote:
        config['llm_allow_insecure_remote'] = True
    return config


def create_llm_client(config):
    try:
        return LLMClient(
            config.get('llm_url', DEFAULT_CONFIG['llm_url']),
            max_stream_chunks=config.get('max_stream_chunks', DEFAULT_CONFIG['max_stream_chunks']),
            api_key=config.get('llm_api_key'),
            api_key_file=config.get('llm_api_key_file'),
            verify_tls=config.get('llm_verify_tls', True),
            allow_insecure_remote=config.get('llm_allow_insecure_remote', False),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

def refresh_config_paths(project_dir=None):
    config = load_config()
    discovered = discover_tool_paths(project_dir)
    placeholder_values = {
        "nvcc",
        "nvidia-smi",
        "ncu",
        "nsys",
        "server.py",
    }

    changed = False
    for key, value in discovered.items():
        current = config.get(key)
        if value:
            if current != value:
                config[key] = value
                changed = True
            continue

        if key not in config:
            config[key] = None
            changed = True
            continue

        if isinstance(current, str) and current.strip().lower() in placeholder_values:
            config[key] = None
            changed = True

    if config.get("last_discovery_at") != datetime.now().strftime("%Y-%m-%d"):
        config["last_discovery_at"] = datetime.now().isoformat(timespec="seconds")
        changed = True

    if changed:
        save_config(config)

    return config

def save_report(report_data, filepath):
    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)



def get_resources_table():
    
    cpu_percent = psutil.cpu_percent(interval=None)
    

    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)
    

    gpu_info = ""
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            gpu_name = gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            vram_used_gb = mem.used / (1024 ** 3)
            vram_total_gb = mem.total / (1024 ** 3)
            vram_percent = (mem.used / mem.total) * 100
            
            gpu_info = f"[bold green]GPU {gpu_name}:[/bold green] [yellow]Core {util.gpu}%[/yellow] | [yellow]VRAM {vram_percent:.1f}% ({vram_used_gb:.2f}/{vram_total_gb:.2f} GB)[/yellow]"
        except Exception as e:
            gpu_info = f"[bold red][ERROR] GPU Error:[/bold red] {e}"
    else:
        gpu_info = "[bold red][WARNING] GPU:[/bold red] Offline"
        
    stats_str = (
        f"[bold cyan]CPU:[/bold cyan] [yellow]{cpu_percent}%[/yellow]  |  "
        f"[bold magenta]RAM:[/bold magenta] [yellow]{ram.percent}% ({ram_used_gb:.1f}/{ram_total_gb:.1f} GB)[/yellow]  |  "
        f"{gpu_info}"
    )
    
    return Panel(stats_str, title="Real-Time Hardware Resource Monitor", border_style="green")

class IterationDashboard:
    def __init__(self, iter_num, total_iters):
        self.iter_num = iter_num
        self.total_iters = total_iters
        self.status_msg = "Initializing..."
        self.spinner = Spinner("dots")
        
    def update_status(self, msg, spinner_name="aesthetic"):
        self.status_msg = msg.replace("[INFO] ", "").replace("[ERROR] ", "").strip()
        try:
            self.spinner = Spinner(spinner_name)
        except KeyError:
            self.spinner = Spinner("dots")
        
    def __rich__(self):
        table = get_resources_table()

        progress_width = 18
        total = max(self.total_iters, 1)
        filled = int(round(progress_width * self.iter_num / total))
        filled = max(0, min(progress_width, filled))
        progress_bar = f"[green]{'▰' * filled}[/green][dim]{'▱' * (progress_width - filled)}[/dim]"

        status_grid = Table.grid(expand=True)
        status_grid.add_column(ratio=1)
        status_grid.add_column(ratio=3)
        status_grid.add_row(
            "[bold cyan]Stage[/bold cyan]",
            Group(self.spinner, Text.from_markup(f" {self.status_msg}")),
        )
        status_grid.add_row(
            "[bold cyan]Progress[/bold cyan]",
            f"{progress_bar} [bold]{self.iter_num}/{self.total_iters}[/bold]",
        )

        status_panel = Panel(
            status_grid,
            border_style="cyan",
            title=f"Iteration {self.iter_num}/{self.total_iters}",
        )
        return Group(table, status_panel)


def print_optimization_summary(output, best_time, original_latency, total_tokens, compile_enabled, completed_iters, profiling_status):
    overview = Table(show_header=False, box=None, padding=(0, 1))
    overview.add_column("Metric", style="bold cyan", no_wrap=True)
    overview.add_column("Value", style="white")
    overview.add_row("Output File", output)
    overview.add_row("Generated Tokens", str(total_tokens))
    overview.add_row("Completed Iterations", str(completed_iters))

    if compile_enabled:
        performance = Table(show_header=False, box=None, padding=(0, 1))
        performance.add_column("Metric", style="bold cyan", no_wrap=True)
        performance.add_column("Value", style="white")
        performance.add_row("Best Latency", f"[bold yellow]{best_time:.4f} ms[/bold yellow]")
        performance.add_row("Original Latency", f"{original_latency} ms")
    else:
        performance = Table(show_header=False, box=None, padding=(0, 1))
        performance.add_column("Metric", style="bold cyan", no_wrap=True)
        performance.add_column("Value", style="white")
        performance.add_row("Verification", "[bold yellow]Skipped[/bold yellow]")
        performance.add_row("Latency", "[dim]unverified[/dim]")

    footer = Table.grid(expand=True)
    footer.add_column(ratio=3)
    footer.add_column(ratio=2, justify="right")
    footer.add_row(
        "[dim]Use --report for a JSON run log.[/dim]",
        f"[bold green]{profiling_status}[/bold green]",
    )

    return Panel(Group(overview, performance, footer), border_style="green", title="Optimization Summary")

def print_diff(old_code, new_code):
    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    diff_text = "\n".join(list(diff))
    if diff_text.strip():
        console.print(Panel(Syntax(diff_text, "diff", theme="monokai"), title="Code Modifications", border_style="blue"))

def print_dry_run_panel(title, lines, border_style="yellow"):
    console.print(Panel("\n".join(lines), title=title, border_style=border_style))

def render_environment_summary(title="Local CUDA Environment Status"):
    import socket
    from urllib.parse import urlparse

    with console.status("[bold green]Inspecting local hardware & environment...[/bold green]"):
        env_status = check_environment()
        config = refresh_config_paths(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


        llm_status = "Offline"
        try:
            parsed = urlparse(config.get('llm_url', ''))
            if parsed.hostname and parsed.port:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((parsed.hostname, parsed.port))
                llm_status = "Online (Port Accessible)"
                s.close()
            else:
                llm_status = "Unknown / Invalid URL"
        except Exception:
            llm_status = "Offline"

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    is_dev = os.path.exists(os.path.join(project_dir, '.git'))
    install_mode = "Development (editable)" if is_dev else "Global (pip install)"

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Platform", platform_info.platform_display_name())
    table.add_row("Install Mode", install_mode)
    table.add_row("Config Location", CONFIG_PATH)
    table.add_row("Cache Directory", str(platform_info.get_cache_dir()))
    table.add_row("-" * 25, "-" * 40)
    table.add_row("NVCC Toolchain", "Found" if env_status.get("nvcc_found") else "[ERROR] Not Found")
    table.add_row("NVIDIA SMI", "Found" if env_status.get("nvidia_smi_found") else "[ERROR] Not Found")
    table.add_row("Nsight Compute (ncu)", "Found" if env_status.get("ncu_found") else "[WARNING] Not Found (Fallback to timer)")
    table.add_row("Nsight Systems (nsys)", "Found" if env_status.get("nsys_found") else "[WARNING] Not Found")
    table.add_row("GPU Model", str(env_status.get("gpu_model", "N/A")))
    table.add_row("Compute Capability", str(env_status.get("compute_capability", "N/A")))
    table.add_row("VRAM Total", str(env_status.get("vram_total", "N/A")))
    table.add_row("CUDA Version", str(env_status.get("cuda_version", "N/A")))
    table.add_row("LLM URL", str(config.get("llm_url", "N/A")))
    table.add_row("LLM Connection", llm_status)
    table.add_row("NVCC Path", str(env_status.get("nvcc_path", "N/A")))
    table.add_row("NVIDIA SMI Path", str(env_status.get("nvidia_smi_path", "N/A")))
    table.add_row("Nsight Compute Path", str(env_status.get("ncu_path", "N/A")))
    table.add_row("Nsight Systems Path", str(env_status.get("nsys_path", "N/A")))

    console.print(table)


def print_cli_help():
        help_text = """cudallm CLI Help & Documentation
==================================================
Local CUDA performance engineering and AI optimization tools.

Commands
--------
cudallm init
    Inspect local GPU hardware, CUDA toolchain paths, Nsight tools, and config.

cudallm doctor
    Show the same environment summary as init.

cudallm check
    Alias for doctor.

cudallm setup-gpu
    Verify environment readiness for GPU kernel compiling.

cudallm serve [options]
    Check LM Studio connection status or display startup instructions.

cudallm optimize <file|folder> [options]
    Run the optimization loop: prompt the LLM, compile, profile, verify, and heal.

cudallm expert <exe> [options]
    Run NSYS -> NCU hotspot analysis, optional NVTX generation, and reruns.

cudallm audit <file|folder> [options]
    Perform static CUDA audits and export markdown reports.

cudallm ncu <exe> [options]
    Run Nsight Compute and export a CSV report.

cudallm nsys <exe> [options]
    Run Nsight Systems timeline capture.

cudallm profile <exe> [options]
    Auto-select ncu or nsys depending on what is available.

Useful flags
------------
optimize:
    --dry-run        Show the planned flow without compiling or profiling.
    --nvtx           Inject NVTX ranges into the generated harness.
    --apply-nvtx     Apply an NVTX suggestion file during compilation.
    --profile-mode   Choose none, auto, ncu, nsys, or code.

expert:
    --dry-run        Show the planned expert flow without running profilers.
    --auto-nvtx      Ask the LLM for NVTX insertion suggestions.
    --rerun          Re-run profiling after NVTX suggestions are generated.
    --code           Use cudaProfilerApi capture range for NSYS.

serve:
    --file           GGUF file name. Default: cudaLLM-8B.Q4_K_M.gguf
    --host           Bind host/interface for the server.
    --public-url     Reachable URL to store in config for clients.
    --api-key-file   Path to a file with API keys for server auth.
    --ssl-key-file   PEM private key for HTTPS.
    --ssl-cert-file  PEM certificate for HTTPS.
    --no-update      Deprecated (retained for backward compatibility).

Examples
--------
cudallm init
cudallm doctor
cudallm optimize path/to/kernel.cu --iters 3 --profile-mode auto --nvtx
cudallm optimize path/to/kernel.cu --dry-run
cudallm expert ./temp_cuda_kernel.exe --auto-nvtx --rerun
cudallm serve --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q4_K_M.gguf --ngl 24
"""
        console.print(help_text)

def optimize_single_file(input_file, output, iters, target, retries, fast_math, opt_level, report, llm, env_info, profile_mode='none', use_nvtx=False, ncu_metrics='', apply_nvtx=False):
    import uuid
    run_id = uuid.uuid4().hex[:8]
    with open(input_file, 'r') as f:
        original_code = f.read()

    flags = [f"-O{opt_level}"]
    if fast_math:
        flags.append("-use_fast_math")

    compile_enabled = os.path.splitext(input_file)[1].lower() == ".cu"
    if not compile_enabled:
        iters = 1
        console.print(Panel(
            "[WARNING] Non-.cu file detected. Compilation and profiling are skipped; output is not validated.",
            border_style="yellow",
            title="Header-Only Optimization"
        ))

    sandbox = CUDASandbox(input_file, flags=flags, profile_mode=profile_mode, use_nvtx=use_nvtx, profile_metrics=ncu_metrics, apply_nvtx_suggestion=apply_nvtx) if compile_enabled else None

    best_time = float('inf')
    best_code = original_code
    current_code = original_code
    history = []

    console.print(Panel(
        f"[INFO] [bold green]Starting Local CUDA Optimization Loop[/bold green]\n"
        f"  [bold]Input File:[/bold] {input_file}\n"
        f"  [bold]Target Metric:[/bold] {target}\n"
        f"  [bold]Iterations:[/bold] {iters}\n"
        f"  [bold]Optimization Level:[/bold] -O{opt_level} {'(with fast-math)' if fast_math else ''}\n"
        f"  [bold]Local LLM Backend:[/bold] {llm.url}",
        border_style="bold green",
        title="Local CUDA Optimization Agent"
    ))

    try:
        for i in range(iters):
            console.print(f"\n[bold cyan]Iteration {i+1}/{iters}[/bold cyan]")

            dashboard = IterationDashboard(i+1, iters)
            with Live(dashboard, refresh_per_second=4) as live:

                dashboard.update_status("Prompting local LLM for optimization...")
                prompt = llm.create_optimization_prompt(current_code, env_info, target, best_time, flags)

                new_code, gen_time = llm.generate_code(
                    prompt,
                    status_callback=dashboard.update_status,
                    prefill=True,
                )

                if not new_code:
                    dashboard.update_status("No CUDA code returned; skipping iteration.", "dots")
                    console.print(Panel(
                        "[bold red]No CUDA code was returned.[/bold red]\n"
                        "The model likely produced prose-only output or stopped before emitting a CUDA block.\n\n"
                        "[bold]What to check:[/bold]\n"
                        "- Tighten the prompt so the final answer must be a single ```cuda block.\n"
                        "- Keep code generation in prefill mode so the assistant starts inside a code fence.\n"
                        "- Verify the model is not truncating the response early.",
                        title="LLM Output Issue",
                        border_style="red",
                    ))
                    try:
                        t_file = f"temp_kernel_{run_id}.cu"
                        if os.path.exists(t_file):
                            os.remove(t_file)
                    except Exception:
                        pass
                    time.sleep(1.0)
                    continue

                if not compile_enabled:
                    best_code = new_code
                    current_code = new_code
                    history.append({
                        "iteration": i+1,
                        "latency": None,
                        "compile_success": False,
                        "gen_time": gen_time
                    })
                    dashboard.update_status("Skipping compilation for header-only file.", "dots")
                    time.sleep(0.5)
                    continue

                dashboard.update_status("Saving temporary kernel...")
                temp_file = f"temp_kernel_{run_id}.cu"
                with open(temp_file, 'w') as f:
                    f.write(new_code)

                sandbox.file_path = temp_file

                dashboard.update_status("Compiling generated CUDA code...")
                compile_res = sandbox.compile()

                verification_failed = False
                prof_res = {"latency": float('inf'), "raw_output": ""}

                if compile_res['success']:
                    dashboard.update_status("Profiling & verifying mathematical correctness...")
                    prof_res = sandbox.profile_latency()
                    if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                        verification_failed = True
                        compile_res['success'] = False
                        compile_res['error_log'] = prof_res["raw_output"]

                heal_attempts = 0
                while not compile_res['success'] and heal_attempts < retries:
                    live.stop()
                    if verification_failed:
                        console.print("[bold red][ERROR] Mathematical Verification Failed! Output does not match original baseline.[/bold red]")
                        console.print(Panel(compile_res['error_log'], title="[bold red][ERROR] Verification Error[/bold red]", border_style="red"))
                        repair_prompt = llm.create_healing_prompt(new_code, compile_res['error_log'])
                    else:
                        console.print(f"[bold yellow][WARNING] Compilation failed! Attempting self-healing ({heal_attempts+1}/{retries})...[/bold yellow]")
                        console.print(Panel(Syntax(compile_res['error_log'], "text", theme="monokai"), title="[bold red][ERROR] NVCC Error Log[/bold red]", border_style="red"))
                        repair_prompt = llm.create_compile_repair_prompt(
                            new_code,
                            compile_res['error_log'],
                            attempt_index=heal_attempts + 1,
                            max_attempts=retries,
                        )
                    live.start()

                    dashboard.update_status(f"Repairing CUDA code with LLM... ({heal_attempts+1}/{retries})")

                    new_code, gen_time = llm.generate_code(
                        repair_prompt,
                        status_callback=dashboard.update_status,
                        prefill=True,
                    )

                    if new_code:
                        with open(temp_file, 'w') as f:
                            f.write(new_code)
                        dashboard.update_status("Re-compiling healed CUDA code...")
                        compile_res = sandbox.compile()
                        if compile_res['success']:
                            dashboard.update_status("Re-profiling & verifying healed CUDA code...")
                            prof_res = sandbox.profile_latency()
                            if "VERIFICATION FAILURE" in prof_res.get("raw_output", ""):
                                verification_failed = True
                                compile_res['success'] = False
                                compile_res['error_log'] = prof_res["raw_output"]
                            else:
                                verification_failed = False
                    heal_attempts += 1

                if not compile_res['success']:
                    live.stop()
                    if verification_failed:
                        console.print("[bold red][ERROR] Failed to mathematically verify the code after max attempts. Skipping iteration.[/bold red]")
                    else:
                        console.print("[bold red][ERROR] Failed to heal the code after max attempts. Skipping iteration.[/bold red]")
                    console.print(Panel(compile_res['error_log'], title="[bold red][ERROR] Final Error Log[/bold red]", border_style="red"))
                    continue

                latency = prof_res["latency"]
                profile_note = prof_res.get("raw_output", "")
                profiling_ok = (
                    "NCU profiling failed" not in profile_note
                    and "nsys invocation failed" not in profile_note
                    and "VERIFICATION FAILURE" not in profile_note
                )
                if profiling_ok:
                    dashboard.update_status("Code compiled and verified mathematically!", "dots")
                else:
                    dashboard.update_status("Code compiled; profiler fallback used.", "dots")
                time.sleep(0.5)

            console.print("[bold green]Code compiled successfully![/bold green]")
            console.print(f"[bold]Kernel Latency:[/bold] [bold yellow]{latency:.4f} ms[/bold yellow] (Gen Time: {gen_time:.2f}s, Profiler: {prof_res.get('raw_output', 'Fallback')})")

            history.append({
                "iteration": i+1,
                "latency": latency,
                "compile_success": True,
                "gen_time": gen_time
            })

            if latency < best_time:
                best_time = latency
                console.print("[bold green]New Best Latency Achieved! Modifying code...[/bold green]")
                print_diff(best_code, new_code)
                best_code = new_code

            current_code = new_code

        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output, 'w') as f:
            f.write(best_code)

        if compile_enabled:
            original_latency = history[0]['latency'] if history else "unknown"
            profiling_status = "Profiled" if (history and history[0]['latency'] != 99999.0) else "Profiler Fallback"
            console.print(print_optimization_summary(
                output=output,
                best_time=best_time,
                original_latency=original_latency,
                total_tokens=llm.total_tokens,
                compile_enabled=compile_enabled,
                completed_iters=len(history),
                profiling_status=profiling_status,
            ))
        else:
            console.print(print_optimization_summary(
                output=output,
                best_time=float('inf'),
                original_latency="unverified",
                total_tokens=llm.total_tokens,
                compile_enabled=False,
                completed_iters=len(history),
                profiling_status="Unverified",
            ))

        if report:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "environment": env_info,
                "flags": flags,
                "best_latency": None if not compile_enabled else best_time,
                "history": history
            }
            report_dir = os.path.dirname(output) or "."
            report_file = os.path.join(report_dir, f"report_{os.path.basename(input_file)}.json")
            save_report(report_data, report_file)
            console.print(f"[bold blue][INFO] Detailed report saved to {report_file}[/bold blue]")

        return {
            "input": input_file,
            "output": output,
            "compile_enabled": compile_enabled,
            "best_latency": None if not compile_enabled else best_time
        }
    finally:
        temp_file = f"temp_kernel_{run_id}.cu"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        if sandbox:
            sandbox.cleanup()

@click.group()
def main():
    pass

@main.command()
def init():
    render_environment_summary()


@main.command()
def doctor():
    render_environment_summary()


@main.command(name='check')
def check():
    render_environment_summary()

@main.command()
@click.option('--instruction', default="Compile src/example.cu and profile it with nsys and ncu", help="Instruction for the agent")
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL')
@click.option('--llm-api-key', envvar='LLM_API_KEY', default=None, help='Override the LLM API key')
@click.option('--llm-api-key-file', envvar='LLM_API_KEY_FILE', default=None, type=click.Path(exists=True, dir_okay=False), help='Override the LLM API key file path')
@click.option('--insecure', is_flag=True, help='Bypass HTTPS/TLS verification and allow insecure remote HTTP connections')
def agent(instruction, llm_url, llm_api_key, llm_api_key_file, insecure):
    """Run LangChain-based autonomous performance engineering agent."""
    from .langchain_agent import create_agent_with_llmclient
    
    config = load_config()
    if llm_url:
        config["llm_url"] = llm_url
    if llm_api_key:
        config["llm_api_key"] = llm_api_key
    if llm_api_key_file:
        config["llm_api_key_file"] = llm_api_key_file
    if insecure:
        config["llm_allow_insecure_remote"] = True
        config["llm_verify_tls"] = False
        
    llm_client = create_llm_client(config)
    
    from .health_check import check_llm_health
    ok, msg = check_llm_health(config.get('llm_url'))
    if not ok:
        if "HTTPConnectionPool" in msg or "ConnectionRefusedError" in msg or "Max retries exceeded" in msg or "refused" in msg:
            from rich.panel import Panel
            guide_text = (
                "[bold red]Could not connect to the local LLM server (LM Studio).[/bold red]\n\n"
                "[bold white]Please start LM Studio manually following these steps:[/bold white]\n"
                "  1. [cyan]Open LM Studio[/cyan] on your machine.\n"
                "  2. [cyan]Click on the Developer Tab[/cyan] (the '< >' icon on the left sidebar).\n"
                "  3. [cyan]Select a model[/cyan] to load from the dropdown list at the top.\n"
                "  4. [cyan]Click the 'Start Server' button[/cyan] (default port is [yellow]1234[/yellow]).\n"
                "  5. Ensure your client configuration matches (run [yellow]cudallm init[/yellow] to check config).\n"
            )
            console.print(Panel(guide_text, title="[bold yellow]LM Studio Connection Error[/bold yellow]", border_style="yellow"))
        else:
            console.print(f"[bold red]LLM health-check failed:[/bold red] {msg}")
        sys.exit(1)
        
    console.print(f"[bold green]Starting LangChain Agent with instruction:[/bold green] '{instruction}'")
    agent_graph = create_agent_with_llmclient(llm_client)
    res = agent_graph.invoke({"messages": [("user", instruction)]})
    response = res["messages"][-1].content
    console.print("\n[bold green]Agent response:[/bold green]\n", response)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', default=None)
@click.option('-i', '--iters', default=3)
@click.option('--target', default='latency')
@click.option('--retries', default=3)
@click.option('--fast-math', is_flag=True)
@click.option('-O', '--opt-level', default='3')
@click.option('--report', is_flag=True)
@click.option('--recursive/--no-recursive', default=True, help='Recurse into subfolders when input is a directory')
@click.option('--profile-mode', type=click.Choice(['none','auto','ncu','nsys','code']), default='none', help='Profiling backend or mode to use')
@click.option('--nvtx', is_flag=True, help='Inject NVTX ranges into generated harness')
@click.option('--apply-nvtx', is_flag=True, help='If set, apply LLM-produced NVTX suggestion (nvtx_suggestion.cu) into the harness during compilation')
@click.option('--ncu-metrics', default='', help='Comma-separated list of ncu metrics to collect (e.g. sm__sass_thread_inst_executed_avg)')
@click.option('--dry-run', is_flag=True, help='Show the planned optimization flow without compiling or profiling')
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:1234/v1/completions)')
@click.option('--llm-api-key', envvar='LLM_API_KEY', default=None, help='Override the LLM API key per-run')
@click.option('--llm-api-key-file', envvar='LLM_API_KEY_FILE', default=None, type=click.Path(exists=True, dir_okay=False), help='Override the LLM API key file path per-run')
@click.option('--insecure', is_flag=True, help='Bypass HTTPS/TLS verification and allow insecure remote HTTP connections')
def optimize(input_file, output, iters, target, retries, fast_math, opt_level, report, recursive, profile_mode, nvtx, apply_nvtx, ncu_metrics, dry_run, llm_url, llm_api_key, llm_api_key_file, insecure):
    locate_and_setup_msvc()
    config = refresh_config_paths(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = apply_llm_overrides(config, llm_url, llm_api_key, llm_api_key_file, insecure)

    with console.status("[bold green][INFO] Checking Environment...[/bold green]"):
        env_info = check_environment()

    if os.path.isdir(input_file):
        base_name = os.path.basename(os.path.normpath(input_file))
        output_dir = output or os.path.join(os.path.dirname(input_file), f"optimized_{base_name}")

        exclude_dirs = set()
        abs_input = os.path.abspath(input_file)
        abs_output = os.path.abspath(output_dir)
        if os.path.commonpath([abs_input, abs_output]) == abs_input:
            exclude_dirs.add(os.path.basename(output_dir))

        files = collect_cuda_files(input_file, recursive=recursive, exclude_dirs=exclude_dirs)
        if not files:
            console.print("[bold red][ERROR] No CUDA source files found in the directory.[/bold red]")
            return

        if dry_run:
            print_dry_run_panel(
                "CUDA Folder Optimization Dry Run",
                [
                    f"Input Folder: {input_file}",
                    f"Output Folder: {output_dir}",
                    f"Files Detected: {len(files)}",
                    f"Iterations: {iters}",
                    f"Target Metric: {target}",
                    f"Retries: {retries}",
                    f"Optimization Level: -O{opt_level}{' (with fast-math)' if fast_math else ''}",
                    f"Profile Mode: {profile_mode}",
                    f"NVTX: {'enabled' if nvtx else 'disabled'} | Apply NVTX suggestion: {'enabled' if apply_nvtx else 'disabled'}",
                    f"NCU Metrics: {ncu_metrics or '(default)'}",
                    f"LLM Backend: {config.get('llm_url', 'N/A')}",
                    "Would prompt the LLM, compile each CUDA file, profile, and self-heal as needed.",
                ],
            )
            return

        llm = create_llm_client(config)

        console.print(Panel(
            f"[INFO] [bold green]Folder Mode Enabled[/bold green]\n"
            f"  [bold]Input Folder:[/bold] {input_file}\n"
            f"  [bold]Output Folder:[/bold] {output_dir}\n"
            f"  [bold]Files Detected:[/bold] {len(files)}\n"
            f"  [bold]Extensions:[/bold] {', '.join(sorted(CUDA_EXTENSIONS))}",
            border_style="bold green",
            title="CUDA Folder Optimization"
        ))

        results = []
        for file_path in files:
            rel_path = os.path.relpath(file_path, input_file)
            output_path = os.path.join(output_dir, rel_path)
            results.append(optimize_single_file(
                file_path,
                output_path,
                iters,
                target,
                retries,
                fast_math,
                opt_level,
                report,
                llm,
                env_info,
                        profile_mode,
                        nvtx,
                        ncu_metrics,
                        apply_nvtx
            ))

        compiled = sum(1 for r in results if r.get("compile_enabled"))
        console.print(Panel(
            f"[bold green]Folder Optimization Complete![/bold green]\n"
            f"  [bold]Files Processed:[/bold] {len(results)}\n"
            f"  [bold]Compiled Files:[/bold] {compiled}\n"
            f"  [bold]Output Folder:[/bold] {output_dir}",
            border_style="bold green",
            title="Folder Summary"
        ))
        return

    if not output:
        output = f"optimized_{os.path.basename(input_file)}"

    if dry_run:
        print_dry_run_panel(
            "CUDA Optimization Dry Run",
            [
                f"Input File: {input_file}",
                f"Output File: {output}",
                f"Iterations: {iters}",
                f"Target Metric: {target}",
                f"Retries: {retries}",
                f"Optimization Level: -O{opt_level}{' (with fast-math)' if fast_math else ''}",
                f"Profile Mode: {profile_mode}",
                f"NVTX: {'enabled' if nvtx else 'disabled'} | Apply NVTX suggestion: {'enabled' if apply_nvtx else 'disabled'}",
                f"NCU Metrics: {ncu_metrics or '(default)'}",
                f"LLM Backend: {config.get('llm_url', 'N/A')}",
                "Would prompt the LLM, compile the generated kernel, profile it, and run healing loops if needed.",
            ],
        )
        return

    llm = create_llm_client(config)

    optimize_single_file(
        input_file,
        output,
        iters,
        target,
        retries,
        fast_math,
        opt_level,
        report,
        llm,
        env_info,
        profile_mode,
        nvtx,
        ncu_metrics,
        apply_nvtx
    )

@main.command()
def help():
    """
    Display plain text help guide for cudallm CLI tool.
    """
    print_cli_help()


@main.command(name='sandbox-run')
@click.option('--image', required=True, help='Docker image to use for sandbox execution')
@click.option('--cmd', 'cmd_text', required=True, help='Command to run inside the container')
@click.option('--mount', multiple=True, help='Volume mount in host:container form (repeatable)')
@click.option('--workdir', default='/workspace', help='Working directory inside the container')
@click.option('--mount-cwd/--no-mount-cwd', default=True, help='Mount the current directory into the container at --workdir')
@click.option('--timeout', default=600, type=int, help='Timeout in seconds')
@click.option('--mem-limit-mb', default=None, type=int, help='Optional memory limit in MB')
def sandbox_run(image, cmd_text, mount, workdir, mount_cwd, timeout, mem_limit_mb):
    volumes = {}
    if mount_cwd:
        volumes[os.getcwd()] = workdir
    for item in mount:
        if ':' not in item:
            raise click.ClickException(f"Invalid mount '{item}'. Expected host:container")
        host, container = item.split(':', 1)
        volumes[host] = container

    result = run_in_docker(image=image, cmd=cmd_text, volumes=volumes or None, timeout=timeout, mem_limit_mb=mem_limit_mb)
    console.print(Panel(result.get('stdout', '') or '', title='Docker sandbox stdout'))
    if result.get('stderr'):
        console.print(Panel(result.get('stderr', ''), title='Docker sandbox stderr', border_style='yellow'))
    console.print(f"[bold]rc:[/bold] {result.get('rc')} | [bold]timed_out:[/bold] {result.get('timed_out')} | [bold]killed_by_limit:[/bold] {result.get('killed_by_limit')}")


@main.command()
@click.argument('exe', type=click.Path(exists=True))
@click.option('--metrics', default='', help='Comma-separated ncu metrics')
@click.option('-o', '--output', default=None, help='Output basename (for CSV or report)')
def ncu(exe, metrics, output):
    """Run Nsight Compute (ncu) against an executable and export CSV."""
    ncu_bin = find_ncu_path()
    if not ncu_bin:
        console.print('[bold red]ncu not found on this system.[/bold red]')
        return
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = output or f'ncu_report_{ts}'
    cmd = [ncu_bin]
    if metrics:
        cmd += ['--metrics', metrics]
    cmd += ['--csv', '-o', base, exe]
    try:
        console.print(f'[blue]Running:[/blue] {" ".join(cmd)}')
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        out = res.stdout + res.stderr
        console.print(Panel(out[:2000], title='ncu output'))
        csv_path = f"{base}.csv"
        if os.path.exists(csv_path):
            console.print(f'[green]NCU CSV:[/green] {os.path.abspath(csv_path)}')
        else:
            console.print('[yellow]NCU did not produce CSV; check output above.[/yellow]')
    except Exception as e:
        console.print(f'[red]Failed to run ncu: {e}[/red]')


@main.command()
@click.argument('exe', type=click.Path(exists=True))
@click.option('-o', '--output', default='nsys_report', help='Output basename')
@click.option('--code', is_flag=True, help='Use cudaProfilerApi capture range (for code-driven profiling)')
def nsys(exe, output, code):
    """Run Nsight Systems (nsys) timeline capture against an executable."""
    nsys_bin = find_nsys_path()
    if not nsys_bin:
        console.print('[bold red]nsys not found on this system.[/bold red]')
        return
    cmd = [nsys_bin, 'profile', '--output', output, '--trace', 'cuda,cudnn']
    if code:
        cmd += ['--capture-range=cudaProfilerApi']
    cmd += [exe]
    try:
        console.print(f'[blue]Running:[/blue] {" ".join(cmd)}')
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900)
        out = res.stdout + res.stderr
        console.print(Panel(out[:2000], title='nsys output'))
        console.print(f'[green]NSYS report basename:[/green] {os.path.abspath(output)}')
    except Exception as e:
        console.print(f'[red]Failed to run nsys: {e}[/red]')


@main.command()
@click.argument('exe', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['auto','ncu','nsys']), default='auto')
@click.option('--metrics', default='', help='ncu metrics, if mode=ncu or auto')
@click.option('--code', is_flag=True, help='Use code-driven capture for nsys')
def profile(exe, mode, metrics, code):
    """Profile an executable using ncu or nsys (auto-select allowed)."""
    ncu_bin = find_ncu_path()
    nsys_bin = find_nsys_path()
    selected = None
    if mode == 'ncu' and ncu_bin:
        selected = 'ncu'
    elif mode == 'nsys' and nsys_bin:
        selected = 'nsys'
    elif mode == 'auto':
        if ncu_bin:
            selected = 'ncu'
        elif nsys_bin:
            selected = 'nsys'
    if not selected:
        console.print('[red]No suitable profiler found for requested mode.[/red]')
        return
    if selected == 'ncu':
        ctx = click.get_current_context()
        ctx.invoke(ncu, exe=exe, metrics=metrics, output=None)
    else:
        ctx = click.get_current_context()
        ctx.invoke(nsys, exe=exe, output='nsys_report', code=code)


@main.command()
@click.argument('exe', type=click.Path(exists=True))
@click.option('--metrics', default='sm__cycles_elapsed.avg,dram__throughput.avg,sm__sass_thread_inst_executed_avg', help='Comma-separated ncu metrics for broad sweep')
@click.option('--run-deep', is_flag=True, help='If set, run a follow-up deep ncu collection for the identified hotspot')
@click.option('--code', is_flag=True, help='Use cudaProfilerApi capture range for nsys')
@click.option('--auto-nvtx', is_flag=True, help='Ask LLM to produce NVTX insertion suggestions and code snippets')
@click.option('--rerun', is_flag=True, help='Rerun profiling after generating NVTX suggestions and save separate outputs')
@click.option('--dry-run', is_flag=True, help='Show the planned expert workflow without running profilers or the LLM')
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:1234/v1/completions)')
@click.option('--llm-api-key', envvar='LLM_API_KEY', default=None, help='Override the LLM API key per-run')
@click.option('--llm-api-key-file', envvar='LLM_API_KEY_FILE', default=None, type=click.Path(exists=True, dir_okay=False), help='Override the LLM API key file path per-run')
@click.option('--insecure', is_flag=True, help='Bypass HTTPS/TLS verification and allow insecure remote HTTP connections')
def expert(exe, metrics, run_deep, code, auto_nvtx, rerun, dry_run, llm_url, llm_api_key, llm_api_key_file, insecure):
    """Run Expert Workflow: NSYS (system) -> NCU (kernel) to identify hotspots and suggest detailed NCU runs."""
    console.print(Panel(f"[bold cyan]Expert Profiling Workflow:[/bold cyan] NSYS -> NCU (broad sweep) -> identify hotspot"))

    config = load_config()
    config = apply_llm_overrides(config, llm_url, llm_api_key, llm_api_key_file, insecure)

    if dry_run:
        print_dry_run_panel(
            "Expert Workflow Dry Run",
            [
                f"Executable: {exe}",
                f"NSYS command: nsys profile --output nsys_expert --trace cuda,cudnn,nvtx{' --capture-range=cudaProfilerApi' if code else ''} {exe}",
                f"NCU command: ncu --csv --output ncu_expert_<timestamp>{' --metrics ' + metrics if metrics else ''} {exe}",
                f"Auto NVTX suggestions: {'enabled' if auto_nvtx else 'disabled'}",
                f"Rerun after NVTX: {'enabled' if rerun else 'disabled'}",
                f"LLM Backend: {config.get('llm_url', 'N/A')}",
                "Would not call the LLM, compile any code, or start profiler runs.",
            ],
        )
        return

    llm = create_llm_client(config)

    nsys_res = run_nsys(exe, output_base='nsys_expert', code=code)
    if nsys_res.get('error'):
        console.print(f"[yellow]NSYS: {nsys_res['error']} (skipping NSYS capture)[/yellow]")
    else:
        console.print(Panel(nsys_res.get('out','')[:2000], title='NSYS Output (truncated)'))
        console.print(f"[green]NSYS report basename:[/green] {nsys_res.get('basename')}")

    ncu_res = run_ncu_broad(exe, metrics=metrics)
    if ncu_res.get('error'):
        console.print(f"[red]NCU error: {ncu_res['error']}[/red]")
        return
    console.print(Panel(ncu_res.get('out','')[:2000], title='NCU Output (truncated)'))
    csv_path = ncu_res.get('csv')
    if csv_path and os.path.exists(csv_path):
        console.print(f"[green]NCU CSV:{os.path.abspath(csv_path)}[/green]")
        hotspot = parse_ncu_csv_for_hotspot(csv_path)
        if hotspot:
            console.print(f"[bold]Identified hotspot kernel:[/bold] [yellow]{hotspot['kernel']}[/yellow] (value={hotspot['value']})")
            suggested = f"ncu --metrics {metrics} --csv --output ncu_deep_{hotspot['kernel']} --kernel-name \"{hotspot['kernel']}\" {exe}"
            console.print(Panel(f"Suggested deep NCU command:\n{suggested}", title='Suggested Deep NCU Command'))

            profile_summary = summarize_profile_outputs(nsys_res.get('out',''), csv_path)
            console.print('[cyan]Requesting expert analysis from local LLM...[/cyan]')
            analysis = llm.analyze_profile(profile_summary)
            if analysis:
                console.print(Panel(analysis, title='LLM Expert Analysis'))
                try:
                    with open('profile_expert_analysis.txt', 'w', encoding='utf-8') as f:
                        f.write(analysis)
                    console.print('[green]Saved LLM analysis to profile_expert_analysis.txt[/green]')
                except Exception:
                    pass

            if auto_nvtx:
                console.print('[cyan]Requesting NVTX insertion suggestions from LLM...[/cyan]')
                nvtx_prompt = (
                    f"You are a CUDA performance engineer. Given the following profiling summary:\n\n{profile_summary}\n\n"
                    f"And the hotspot kernel name: {hotspot['kernel']}. Provide concise NVTX instrumentation code snippets and macro definitions that can be inserted into the CUDA harness to mark high-level regions and the hotspot kernel. Output only CUDA/C++ code inside a ```cuda or ```cpp block."
                )
                nvtx_code, _ = llm.generate_code(
                    nvtx_prompt,
                    status_callback=dashboard.update_status,
                    prefill=True,
                )
                if nvtx_code:
                    console.print(Panel(nvtx_code[:2000], title='NVTX Suggestion (truncated)'))
                    try:
                        with open('nvtx_suggestion.cu', 'w', encoding='utf-8') as f:
                            f.write(nvtx_code)
                        console.print('[green]Saved NVTX suggestion to nvtx_suggestion.cu[/green]')
                    except Exception:
                        pass
                    if rerun:
                        console.print('[cyan]Re-running profiling after NVTX suggestion...[/cyan]')
                        rerun_nsys = run_nsys(exe, output_base=f"nsys_expert_nvtx_{hotspot['kernel']}", code=code)
                        if rerun_nsys.get('error'):
                            console.print(f"[yellow]NVTX rerun NSYS: {rerun_nsys['error']}[/yellow]")
                        else:
                            console.print(Panel(rerun_nsys.get('out','')[:2000], title='NVTX Rerun NSYS Output (truncated)'))
                            console.print(f"[green]NVTX rerun NSYS report basename:[/green] {rerun_nsys.get('basename')}")

                        rerun_ncu = run_ncu_broad(exe, output_base=f"ncu_expert_nvtx_{hotspot['kernel']}", metrics=metrics)
                        if rerun_ncu.get('error'):
                            console.print(f"[yellow]NVTX rerun NCU: {rerun_ncu['error']}[/yellow]")
                        else:
                            console.print(Panel(rerun_ncu.get('out','')[:2000], title='NVTX Rerun NCU Output (truncated)'))
                            rerun_csv = rerun_ncu.get('csv')
                            if rerun_csv and os.path.exists(rerun_csv):
                                console.print(f"[green]NVTX rerun NCU CSV:[/green] {os.path.abspath(rerun_csv)}")
            if run_deep:
                console.print("[cyan]Running deep NCU collection (this may take time)...[/cyan]")
                try:
                    from .discover import find_ncu_path
                    ncu_bin = find_ncu_path() or 'ncu'
                    deep_cmd = [ncu_bin, '--metrics', metrics, '--csv', '--output', f"ncu_deep_{hotspot['kernel']}", exe]
                    res = subprocess.run(deep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
                    console.print(Panel((res.stdout + res.stderr)[:2000], title='Deep NCU Output (truncated)'))
                except Exception as e:
                    console.print(f"[red]Deep NCU run failed: {e}[/red]")
        else:
            console.print('[yellow]Could not auto-identify hotspot kernel from NCU CSV. Consider running detailed NCU manually.[/yellow]')
    else:
        console.print('[yellow]NCU did not produce a CSV; check the output above.[/yellow]')

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--markdown', is_flag=True)
@click.option('--recursive/--no-recursive', default=True, help='Recurse into subfolders when input is a directory')
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:1234/v1/completions)')
@click.option('--llm-api-key', envvar='LLM_API_KEY', default=None, help='Override the LLM API key per-run')
@click.option('--llm-api-key-file', envvar='LLM_API_KEY_FILE', default=None, type=click.Path(exists=True, dir_okay=False), help='Override the LLM API key file path per-run')
@click.option('--insecure', is_flag=True, help='Bypass HTTPS/TLS verification and allow insecure remote HTTP connections')
def audit(input_file, markdown, recursive, llm_url, llm_api_key, llm_api_key_file, insecure):
    config = load_config()
    config = apply_llm_overrides(config, llm_url, llm_api_key, llm_api_key_file, insecure)
    llm = create_llm_client(config)

    if os.path.isdir(input_file):
        base_name = os.path.basename(os.path.normpath(input_file))
        output_dir = os.path.join(os.path.dirname(input_file), f"audit_{base_name}")

        exclude_dirs = set()
        abs_input = os.path.abspath(input_file)
        abs_output = os.path.abspath(output_dir)
        if os.path.commonpath([abs_input, abs_output]) == abs_input:
            exclude_dirs.add(os.path.basename(output_dir))

        files = collect_cuda_files(input_file, recursive=recursive, exclude_dirs=exclude_dirs)
        if not files:
            console.print("[bold red][ERROR] No CUDA source files found in the directory.[/bold red]")
            return

        console.print(Panel(
            f"[INFO] [bold green]Folder Audit Mode Enabled[/bold green]\n"
            f"  [bold]Input Folder:[/bold] {input_file}\n"
            f"  [bold]Files Detected:[/bold] {len(files)}\n"
            f"  [bold]Extensions:[/bold] {', '.join(sorted(CUDA_EXTENSIONS))}\n"
            f"  [bold]Local LLM Backend:[/bold] {config['llm_url']}",
            border_style="bold magenta",
            title="Local CUDA Auditor"
        ))

        for file_path in files:
            with open(file_path, 'r') as f:
                code = f.read()
            prompt = llm.create_audit_prompt(code)

            console.print(Panel(
                f"[INFO] Auditing: {file_path}",
                border_style="bold magenta",
                title="File Audit"
            ))

            analysis, _ = llm.generate_code(prompt, early_terminate=False, prefill=False)

            if markdown:
                rel_path = os.path.relpath(file_path, input_file)
                out_path = os.path.join(output_dir, f"{rel_path}.md")
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(out_path, 'w') as f:
                    f.write(analysis)

        if markdown:
            console.print(f"\n[bold green][INFO] Audit reports saved to {output_dir}[/bold green]")
        else:
            console.print("\n[bold green][INFO] Architectural Audit complete![/bold green]")
        return

    with open(input_file, 'r') as f:
        code = f.read()
    prompt = llm.create_audit_prompt(code)

    console.print(Panel(
        f"[INFO] [bold green]Starting Local CUDA Architectural Audit[/bold green]\n"
        f"  [bold]Input File:[/bold] {input_file}\n"
        f"  [bold]Local LLM Backend:[/bold] {config['llm_url']}",
        border_style="bold magenta",
        title="Local CUDA Auditor"
    ))

    analysis, _ = llm.generate_code(prompt, early_terminate=False, prefill=False)

    if markdown:
        out_name = f"audit_{os.path.basename(input_file)}.md"
        with open(out_name, 'w') as f:
            f.write(analysis)
        console.print(f"\n[bold green][INFO] Audit report saved to {out_name}[/bold green]")
    else:
        console.print("\n[bold green][INFO] Architectural Audit complete![/bold green]")


@main.command(name='setup-gpu')
@click.option('--dry-run', is_flag=True, help='Show the commands without executing them')
@click.option('--force-reinstall', is_flag=True, default=True, help='Force reinstall (deprecated/ignored)')
def setup_gpu(dry_run, force_reinstall):
    """
    Verify environment readiness for GPU kernel compiling.
    """
    console.print(Panel("[bold green]Verifying environment readiness for GPU kernel compiling[/bold green]", border_style="green"))
    console.print(f"  [bold]Detected Terminal:[/bold] {TerminalManager.get_terminal_name()}")
    
    env_info = check_environment()
    
    if not env_info.get('nvcc_found'):
        console.print("[bold red][ERROR] CUDA Toolkit (nvcc) not found on this system.[/bold red]")
        console.print("Please install CUDA Toolkit and ensure 'nvcc' is in your PATH.")
        return
        
    cc = env_info.get('compute_capability', 'Unknown')
    cuda_ver = env_info.get('cuda_version', 'Unknown')
    gpu_model = env_info.get('gpu_model', 'Unknown')
    
    console.print(f"[green][SUCCESS] CUDA Environment check passed![/green]")
    console.print(f"  [bold]GPU Model:[/bold] {gpu_model}")
    console.print(f"  [bold]CUDA Version:[/bold] {cuda_ver}")
    console.print(f"  [bold]Compute Capability:[/bold] {cc}")
    
    console.print("\n[green][INFO] Note: llama-cpp-python installation is bypassed because cudallm-cli now interfaces directly with LM Studio.[/green]")

@main.command()
@click.option('--port', default=1234, help='Port LM Studio is running on')
@click.option('--host', default='127.0.0.1', help='Host/interface for LM Studio backend')
@click.option('--public-url', default=None, help='Reachable URL to store in config for clients on this network')
@click.option('--api-key', default=None, help='API key to require for server access (stored in config for client requests)')
@click.option('--api-key-file', default=None, type=click.Path(exists=True, dir_okay=False), help='Path to a file containing one or more API keys')
@click.option('--ssl-key-file', default=None, type=click.Path(exists=True, dir_okay=False), help='PEM-encoded SSL private key for HTTPS')
@click.option('--ssl-cert-file', default=None, type=click.Path(exists=True, dir_okay=False), help='PEM-encoded SSL certificate for HTTPS')
@click.option('--allow-unsafe-network', is_flag=True, help='Allow exposing the server without API key or TLS')
@click.option('--reuse-port', is_flag=True, help='Allow multiple sockets to bind to the same port')
@click.option('--repo', default='prithivMLmods/cudaLLM-8B-GGUF', help='HuggingFace repository name')
@click.option('--file', default='cudaLLM-8B.Q4_K_M.gguf', help='HuggingFace GGUF model file name')
@click.option('--local-model', default=None, help='Local model path (GGUF or HF local directory)')
@click.option('--use-cuda/--no-use-cuda', default=True, help='Enable CUDA for the Python backend')
@click.option('--ngl', default=33, help='Deprecated. Kept for compatibility; ignored by Python backend')
@click.option('--ctx', default=4096, help='Deprecated. Kept for compatibility; ignored by Python backend')
@click.option('--parallel', default=1, help='Deprecated. Kept for compatibility; ignored by Python backend')
@click.option('--no-update', is_flag=True, help='Deprecated. Kept for compatibility; ignored by Python backend')
def serve(port, host, public_url, api_key, api_key_file, ssl_key_file, ssl_cert_file, allow_unsafe_network, reuse_port, repo, file, local_model, use_cuda, ngl, ctx, parallel, no_update):
    """
    Check LM Studio connection status or display startup instructions.
    """
    console.print(Panel("[bold green]Checking LM Studio Server Connection[/bold green]", border_style="green"))
    console.print(f"  [bold]Detected Terminal:[/bold] {TerminalManager.get_terminal_name()}")

    scheme = "https" if (ssl_key_file and ssl_cert_file) else "http"
    

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = refresh_config_paths(project_dir)
    
    if public_url:
        config["llm_url"] = public_url
    else:
        config["llm_url"] = f"{scheme}://{host}:{port}/v1/completions"
        
    if api_key:
        config["llm_api_key"] = api_key
    if api_key_file:
        config["llm_api_key_file"] = api_key_file
        
    save_config(config)

   
    test_url = f"{scheme}://{host}:{port}/v1/models"
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif api_key_file and os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                key = f.read().strip()
                headers["Authorization"] = f"Bearer {key}"

        response = _probe_llm_server(test_url, headers, verify_tls=(scheme == "https"))
        if response.status_code == 200:
            models_data = response.json()
            models_list = models_data.get("data", [])
            
            table = Table(title="LM Studio Status: Online", show_header=True, header_style="bold green")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Connection URL", f"{scheme}://{host}:{port}")
            table.add_row("Configured endpoint", config["llm_url"])
            
            if models_list:
                loaded_models = ", ".join([m.get("id", "Unknown") for m in models_list])
                table.add_row("Loaded Model(s)", loaded_models)
            else:
                table.add_row("Loaded Model(s)", "[yellow]No models loaded in LM Studio[/yellow]")
                
            console.print(table)
            console.print("[bold green][SUCCESS] Successfully connected to LM Studio local server![/bold green]")
            return
    except Exception:
        pass


    guide_text = (
        "[bold red]Could not connect to LM Studio server.[/bold red]\n\n"
        "[bold white]Please start LM Studio manually following these steps:[/bold white]\n"
        "  1. [cyan]Open LM Studio[/cyan] on your machine.\n"
        "  2. [cyan]Click on the Developer Tab[/cyan] (the '< >' icon on the left sidebar).\n"
        "  3. [cyan]Select a model[/cyan] to load from the dropdown list at the top.\n"
        "  4. [cyan]Click the 'Start Server' button[/cyan] (default port is [yellow]1234[/yellow]).\n"
        "  5. Ensure your client configuration matches (e.g. run [yellow]cudallm init[/yellow] or check [yellow]config.json[/yellow]).\n"
    )
    console.print(Panel(guide_text, title="[bold yellow]LM Studio Integration Guide[/bold yellow]", border_style="yellow"))

if __name__ == '__main__':
    main()
