import click
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
from .discover import check_environment, discover_tool_paths, find_llm_backend_path, find_ncu_path, find_nsys_path
from .sandbox import CUDASandbox
from .llm_client import LLMClient
from .profiler_tools import run_nsys, run_ncu_broad, parse_ncu_csv_for_hotspot, summarize_profile_outputs
from .network_security import validate_llm_endpoint
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

console = Console()

CONFIG_PATH = str(platform_info.get_config_path())
_LEGACY_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

DEFAULT_CONFIG = {
    "llm_url": f"http://127.0.0.1:{platform_info.get_default_llm_port()}/completion",
    "llm_api_key": None,
    "llm_api_key_file": None,
    "llm_verify_tls": True,
    "llm_allow_insecure_remote": False,
    "max_stream_chunks": 2000,
}

CUDA_EXTENSIONS = {".cu", ".cuh"}
IGNORE_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "node_modules"}

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


def _parse_cuda_version(cuda_version_str):
    try:
        return float(str(cuda_version_str).split()[0])
    except (ValueError, IndexError, TypeError):
        return 12.4


def _get_latest_llama_tag():
    import urllib.request

    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            tag = data.get('tag_name')
            if tag and tag.startswith('b') and tag[1:].isdigit():
                return tag
    except Exception:
        pass
    return "b9222"


def _get_release_assets(tag):
    import urllib.request

    api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('assets', [])
    except Exception:
        return []


def _select_llama_asset(tag, cuda_version, is_windows=False):
    assets = _get_release_assets(tag)
    if not assets:
        return None

    version_tokens = ["cuda-13.1", "cu131", "cuda131"] if cuda_version >= 13.0 else ["cuda-12.4", "cu124", "cuda124"]
    platform_tokens = ["win"] if is_windows else ["ubuntu", "linux"]

    best_asset = None
    best_score = -1

    for asset in assets:
        name = asset.get('name', '')
        lowered = name.lower()
        score = 0

        if 'cuda' not in lowered and 'cu' not in lowered:
            continue

        if tag.lower() in lowered:
            score += 20

        if any(token in lowered for token in platform_tokens):
            score += 15
        else:
            continue

        if any(token in lowered for token in version_tokens):
            score += 25
        elif 'cuda' in lowered or 'cu' in lowered:
            score += 5

        if lowered.endswith('.zip') or lowered.endswith('.tar.gz') or lowered.endswith('.tgz'):
            score += 3

        if not is_windows and ('ubuntu' in lowered or 'linux' in lowered):
            score += 5

        if score > best_score:
            best_score = score
            best_asset = asset

    return best_asset


def _download_and_extract_asset(asset, dest_dir):
    import zipfile
    import tarfile

    url = asset.get('browser_download_url')
    name = asset.get('name', 'download')
    if not url:
        raise RuntimeError(f"Release asset {name} does not expose a download URL")

    archive_filepath = os.path.join(dest_dir, name)
    response = requests.get(url, stream=True, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP Status {response.status_code} for {name}")

    with open(archive_filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(archive_filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif name.lower().endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        raise RuntimeError(f"Unsupported archive format for {name}")

    if os.path.exists(archive_filepath):
        os.remove(archive_filepath)


def _find_executable(dest_dir, bin_name):
    direct_path = os.path.join(dest_dir, bin_name)
    if os.path.exists(direct_path):
        return direct_path

    for root, _, files_list in os.walk(dest_dir):
        if bin_name in files_list:
            return os.path.join(root, bin_name)
    return None


def _build_llama_server_from_source():
    if not os.path.exists("/content/llama.cpp"):
        code, out, err = run_command("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")
        if code != 0:
            raise RuntimeError(f"Failed to clone llama.cpp: {err}")

    os.makedirs("/content/llama.cpp/build", exist_ok=True)
    code, out, err = run_command("cmake -B /content/llama.cpp/build -S /content/llama.cpp -DGGML_CUDA=ON")
    if code != 0:
        raise RuntimeError(f"CMake configuration failed: {err}")

    code, out, err = run_command("cmake --build /content/llama.cpp/build --config Release --target llama-server -j$(nproc)")
    if code != 0:
        raise RuntimeError(f"Build failed: {err}")

    return "/content/llama.cpp/build/bin/llama-server"

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
        self.spinner = Spinner("aesthetic")
        
    def update_status(self, msg, spinner_name="aesthetic"):
        self.status_msg = msg
        try:
            self.spinner = Spinner(spinner_name)
        except KeyError:
            self.spinner = spinner_name
        
    def __rich__(self):
        table = get_resources_table()
      
        status_line = Group(
            self.spinner,
            f" {self.status_msg}"
        )
        status_panel = Panel(
            status_line,
            border_style="cyan",
            title=f"Iteration {self.iter_num}/{self.total_iters} Activity Log"
        )
        return Group(table, status_panel)

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
    table.add_row("─" * 25, "─" * 40)
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
    table.add_row("LLM Backend Script", str(config.get("llm_server_path", "N/A")))

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

cudallm serve [options]
    Launch Python LLM backend (`tools/server.py`) with CUDA support.

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

                live.stop()
                new_code, gen_time = llm.generate_code(prompt)
                live.start()

                if not new_code:
                    dashboard.update_status("[ERROR] No CUDA code returned; skipping iteration.", "dots")
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
                    else:
                        console.print(f"[bold yellow][WARNING] Compilation failed! Attempting self-healing ({heal_attempts+1}/{retries})...[/bold yellow]")
                        console.print(Panel(Syntax(compile_res['error_log'], "text", theme="monokai"), title="[bold red][ERROR] NVCC Error Log[/bold red]", border_style="red"))
                    live.start()

                    dashboard.update_status(f"Healing CUDA code with LLM... ({heal_attempts+1}/{retries})")
                    heal_prompt = llm.create_healing_prompt(new_code, compile_res['error_log'])

                    live.stop()
                    new_code, gen_time = llm.generate_code(heal_prompt)
                    live.start()

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
                dashboard.update_status("Code compiled and verified mathematically!", "dots")
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
            console.print(Panel(
                f"[bold green]Local Optimization Complete![/bold green]\n"
                f"  [bold]Saved Optimized Kernel to:[/bold] {output}\n"
                f"  [bold]Best Latency:[/bold] [bold yellow]{best_time:.4f} ms[/bold yellow] (Original was {original_latency} ms)\n"
                f"  [bold]Total Generated Tokens:[/bold] {llm.total_tokens}",
                border_style="bold green",
                title="Optimization Summary"
            ))
        else:
            console.print(Panel(
                f"[bold green]Optimization Complete (Unverified)![/bold green]\n"
                f"  [bold]Saved Optimized File to:[/bold] {output}\n"
                f"  [bold]Total Generated Tokens:[/bold] {llm.total_tokens}",
                border_style="bold yellow",
                title="Optimization Summary"
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
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:8080/completion)')
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
    cmd += ['--csv', '--output', base, exe]
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
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:8080/completion)')
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
                nvtx_code, _ = llm.generate_code(nvtx_prompt)
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
@click.option('--llm-url', envvar='LLM_URL', default=None, help='Override the LLM backend URL per-run (e.g. http://10.212.3.55:8080/completion)')
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

def check_and_prepare_python_server(project_dir, config):
    server_path = config.get("llm_server_path")
    if server_path and (not os.path.exists(server_path) or not str(server_path).lower().endswith(".py")):
        server_path = None
    if not server_path:
        server_path = find_llm_backend_path(project_dir)

    if not server_path:
        expected = os.path.join(project_dir, "tools", "server.py")
        raise click.ClickException(
            f"Python LLM backend not found. Expected `{expected}`. "
            "Create or restore `tools/server.py` and retry."
        )

    config["llm_server_path"] = server_path
    config.pop("llama_version", None)
    save_config(config)
    return server_path


def check_and_update_llama_server(project_dir, config, no_update=False):
    return check_and_prepare_python_server(project_dir, config)

@main.command()
@click.option('--port', default=8080, help='Port to run the LLM server on')
@click.option('--host', default='127.0.0.1', help='Host/interface for Python backend to bind to')
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
    Launch the local Python LLM backend (`tools/server.py`) with CUDA support.
    """
    console.print(Panel("[bold green]Launching Local Python LLM Backend with CUDA Support[/bold green]", border_style="green"))
    
   
    env = os.environ.copy()

 
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = refresh_config_paths(project_dir)
    server_path = check_and_prepare_python_server(project_dir, config)

    api_keys_enabled = bool(api_key or api_key_file)
    tls_enabled = bool(ssl_key_file and ssl_cert_file)
    exposed_bind = host not in {"127.0.0.1", "localhost", "::1"}
    if exposed_bind and not (api_keys_enabled or tls_enabled or allow_unsafe_network):
        console.print(
            "[bold red][ERROR] Refusing to expose the server on a network-facing host without API key or TLS.[/bold red]"
        )
        console.print("Use --api-key or --api-key-file, add --ssl-key-file and --ssl-cert-file, or pass --allow-unsafe-network to override.")
        return

    if public_url:
        try:
            apply_public_url_override(config, public_url, allow_insecure_remote=allow_unsafe_network)
        except ValueError as exc:
            console.print(f"[bold red][ERROR] Invalid public URL: {exc}[/bold red]")
            return
    else:
        scheme = "https" if tls_enabled else "http"
        if host in {"0.0.0.0", "::"}:
            console.print("[bold yellow][WARNING] Host is a wildcard bind address, so cudallm cannot derive a client URL automatically.[/bold yellow]")
            console.print("[yellow]Pass --public-url to store the reachable client URL in config.json.[/yellow]")
        else:
            config["llm_url"] = f"{scheme}://{host}:{port}/completion"

    if api_key_file:
        config["llm_api_key_file"] = api_key_file
    if tls_enabled:
        config["llm_verify_tls"] = True
    save_config(config)


    if any([no_update, ngl != 33, ctx != 4096, parallel != 1, reuse_port, api_key, api_key_file, ssl_key_file, ssl_cert_file]):
        console.print("[yellow][WARNING] Some options are deprecated or not enforced by the Python backend and will be ignored by the process launch.[/yellow]")

    cmd = [
        sys.executable,
        server_path,
        "--host", host,
        "--port", str(port),
    ]
    if local_model:
        cmd.extend(["--local-model", local_model])
    else:
        cmd.extend(["--hf-repo", repo, "--hf-file", file])
    if use_cuda:
        cmd.append("--use-cuda")
    
    console.print(f"[bold blue][INFO] Running command:[/bold blue] {' '.join(cmd)}")
    console.print("[bold yellow]Press Ctrl+C to terminate the server.[/bold yellow]\n")
    
    try:
       
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
    
        for line in process.stdout:
            line_str = line.strip()
            if "error" in line_str.lower() or "failed" in line_str.lower():
                console.print(f"[red]{line_str}[/red]")
            elif "cuda" in line_str.lower() or "gpu" in line_str.lower() or "device" in line_str.lower():
                console.print(f"[bold green]{line_str}[/bold green]")
            elif "listening" in line_str.lower() or "model loaded" in line_str.lower():
                console.print(f"[bold cyan]{line_str}[/bold cyan]")
            else:
                console.print(line_str)
                
        process.wait()
    except KeyboardInterrupt:
        console.print("\n[bold yellow][WARNING] Stopping LLM server...[/bold yellow]")
        if 'process' in locals():
            process.terminate()
            process.wait()
        console.print("[bold green][INFO] LLM server stopped cleanly.[/bold green]")
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to run Python LLM backend: {e}[/bold red]")

if __name__ == '__main__':
    main()
