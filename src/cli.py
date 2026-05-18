import click
import json
import os
import difflib
import sys
import subprocess
import glob
import time
from datetime import datetime
from .discover import check_environment
from .sandbox import CUDASandbox
from .llm_client import LLMClient
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

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

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
        

    if subprocess.run("where cl", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return True
        
   
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
        
    return False

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"llm_url": "http://127.0.0.1:8080/completion"}

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

def optimize_single_file(input_file, output, iters, target, retries, fast_math, opt_level, report, llm, env_info):
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

    sandbox = CUDASandbox(input_file, flags=flags) if compile_enabled else None

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
                    if os.path.exists("temp_kernel.cu"):
                        os.remove("temp_kernel.cu")
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
            temp_file = "temp_kernel.cu"
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

    for temp in ["temp_kernel.cu", "temp_cuda_kernel.exe", "temp_cuda_kernel.out", "temp_cuda_kernel.exp", "temp_cuda_kernel.lib"]:
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except Exception:
                pass

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

@click.group()
def main():
    pass

@main.command()
def init():
    with console.status("[bold green][INFO] Inspecting local hardware & environment...[/bold green]"):
        env_status = check_environment()
        config = load_config()
    
    table = Table(title="Local CUDA Environment Status", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Value", style="green")
    
    table.add_row("NVCC Toolchain", "Found" if env_status.get("nvcc_found") else "[ERROR] Not Found")
    table.add_row("NVIDIA SMI", "Found" if env_status.get("nvidia_smi_found") else "[ERROR] Not Found")
    table.add_row("Nsight Compute (ncu)", "Found" if env_status.get("ncu_found") else "[WARNING] Not Found (Fallback to timer)")
    table.add_row("GPU Model", str(env_status.get("gpu_model", "N/A")))
    table.add_row("Compute Capability", str(env_status.get("compute_capability", "N/A")))
    table.add_row("VRAM Total", str(env_status.get("vram_total", "N/A")))
    table.add_row("CUDA Version", str(env_status.get("cuda_version", "N/A")))
    table.add_row("LLM URL", str(config.get("llm_url", "N/A")))
    table.add_row("NVCC Path", str(config.get("nvcc_path", "N/A")))
    
    console.print(table)

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
def optimize(input_file, output, iters, target, retries, fast_math, opt_level, report, recursive):
    locate_and_setup_msvc()
    config = load_config()
    llm = LLMClient(config['llm_url'], max_stream_chunks=config.get("max_stream_chunks", 2000))

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
                env_info
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
        env_info
    )

@main.command()
def help():
    """
    Display plain text help guide for cudallm CLI tool.
    """
    help_text = """[INFO] Welcome to cudallm CLI Help & Documentation
==================================================
An elite Local CUDA Performance Engineering & AI Optimization suite.

Commands Summary:
-----------------
* cudallm init
  Inspect local GPU hardware environment, active VRAM details, compute capabilities, CUDA compiler paths, and configurations.

* cudallm serve [options]
  Launch the local llama-server backend with CUDA GPU offloading enabled, customized context windows, and automatic PyTorch DLL injection.

* cudallm optimize <file|folder> [options]
    Begin the dynamic, live-monitored AI performance loop. Auto-optimizes CUDA latency, runs hardware profilers, and utilizes self-healing loops.

* cudallm audit <file|folder> [options]
    Perform deep static architectural audits for Warp Divergence, Shared Memory Bank Conflicts, Coalescing issues, and export to markdown.

* cudallm help
  Open this CLI documentation center.

[INFO] Quick Start Guide & Standard Workflows:
------------------------------------------
1. Step 1: Check hardware status
   > cudallm init

2. Step 2: Spin up local AI LLM backend (RTX accelerated)
   > cudallm serve --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q2_K.gguf

3. Step 3: Run the Auto-tuning Live Engine
    > cudallm optimize src/my_kernel.cu --iters 3 -O3 --fast-math
    > cudallm optimize src/kernels --recursive

4. Step 4: Check for bottlenecks and export a report
    > cudallm audit src/my_kernel.cu --markdown
    > cudallm audit src/kernels --markdown --recursive

[INFO] Useful command line switches during optimization:
-----------------------------------------------------
  -i, --iters <num>    Number of optimization cycles to run (Default: 3)
  --target <metric>   Target metric to measure ('latency' or 'memory')
  --retries <num>    Max self-healing compilation correction attempts (Default: 3)
  -O, --opt-level <L>  NVCC optimization flags: O1, O2, O3 (Default: 3)
  --fast-math         Speed up float calculations with NVCC --use_fast_math flag
    --report            Save optimization history as a JSON configuration telemetry report
    --recursive/--no-recursive  Recurse into subfolders when input is a directory"""
    print(help_text)

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--markdown', is_flag=True)
@click.option('--recursive/--no-recursive', default=True, help='Recurse into subfolders when input is a directory')
def audit(input_file, markdown, recursive):
    config = load_config()
    llm = LLMClient(config['llm_url'], max_stream_chunks=config.get("max_stream_chunks", 2000))

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

            analysis, _ = llm.generate_code(prompt, early_terminate=False)

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

    analysis, _ = llm.generate_code(prompt, early_terminate=False)

    if markdown:
        out_name = f"audit_{os.path.basename(input_file)}.md"
        with open(out_name, 'w') as f:
            f.write(analysis)
        console.print(f"\n[bold green][INFO] Audit report saved to {out_name}[/bold green]")
    else:
        console.print("\n[bold green][INFO] Architectural Audit complete![/bold green]")

@main.command()
@click.option('--port', default=8080, help='Port to run the LLM server on')
@click.option('--repo', default='prithivMLmods/cudaLLM-8B-GGUF', help='HuggingFace repository name')
@click.option('--file', default=' cudaLLM-8B.Q2_K.gguf', help='HuggingFace GGUF model file name')
@click.option('--ngl', default=99, help='Number of layers to offload to GPU')
@click.option('--ctx', default=8192, help='Context size')
def serve(port, repo, file, ngl, ctx):
    """
    Launch the local llama-server with CUDA support and auto-dependency resolution.
    """
    console.print(Panel("[bold green][INFO] Launching Local LLM Server with CUDA Support[/bold green]", border_style="green"))
    
   
    env = os.environ.copy()
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    
    if os.path.exists(torch_lib):
        console.print(f"[bold green][INFO] Found PyTorch CUDA runtime libraries at:[/bold green] {torch_lib}")
        env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
    else:

        console.print("[yellow][WARNING] PyTorch CUDA runtime libraries not found in active virtual environment. Falling back to system PATH.[/yellow]")

 
    exe_path = None
   
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_pattern = os.path.join(project_dir, "llama-b*", "llama-server.exe")
    matching_paths = glob.glob(search_pattern)
    
    if matching_paths:
        exe_path = matching_paths[0]
        console.print(f"[bold green][INFO] Located llama-server at:[/bold green] {exe_path}")
    else:
     
        import shutil
        exe_in_path = shutil.which("llama-server")
        if exe_in_path:
            exe_path = exe_in_path
            console.print(f"[bold green][INFO] Located llama-server in system PATH:[/bold green] {exe_path}")
        else:
            console.print("[bold red][ERROR] Could not find llama-server.exe![/bold red]")
            console.print("Please place the 'llama-bXXXX-bin-win-cuda...' folder inside the cudallm-cli project directory.")
            return

    cmd = [
        exe_path,
        "--hf-repo", repo,
        "--hf-file", file,
        "-ngl", str(ngl),
        "-c", str(ctx),
        "--port", str(port)
    ]
    
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
            elif "cuda" in line_str.lower() or "offload" in line_str.lower() or "device" in line_str.lower():
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
        console.print(f"[bold red][ERROR] Failed to run llama-server: {e}[/bold red]")

if __name__ == '__main__':
    main()
