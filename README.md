# cudallm-cli

[![CI](https://github.com/ThemeHackers/cudallm-cli/actions/workflows/ncu-regression.yml/badge.svg)](https://github.com/ThemeHackers/cudallm-cli/actions/workflows/ncu-regression.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Local Autonomous CUDA Optimization Agent — A closed-loop tool that uses a local LLM to read, modify, compile, profile, and repair CUDA kernels automatically.

---

## Table of Contents
1. [What & Why](#what--why)
2. [Key Features](#key-features)
3. [Architecture & Workflow](#architecture--workflow)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Detailed Command Reference](#detailed-command-reference)
8. [Advanced Core Workflows](#advanced-core-workflows)
   - [1. Autonomous Optimization & Self-Healing](#1-autonomous-optimization--self-healing)
   - [2. Expert Profiling Workflow (nsys → ncu)](#2-expert-profiling-workflow-nsys--ncu)
9. [Networking & Secure Deployment](#networking--secure-deployment)
10. [CI/CD Regression Checks](#cicd-regression-checks)
11. [Configuration Schema](#configuration-schema)
12. [Troubleshooting](#troubleshooting)

---

## What & Why
* **What**: An automated local toolchain for CUDA kernel performance engineering using an LLM-backed edit/compile/profile loop.
* **Why**: Manual CUDA optimization is a tedious process of tweaking parameters (block sizes, tiling, memory layouts), compiling, running profilers (`ncu`, `nsys`), interpreting reports, and fixing compilation errors. `cudallm-cli` automates this entire loop.

---

## Key Features
- **Closed-Loop Self-Healing**: Automatically captures NVCC compiler error logs, compresses them into repair-friendly diagnostics, and sends them back to the LLM so syntax errors, missing includes, or CUDA API mismatches can be fixed without manual intervention.
- **Separate Repair Paths**: Compilation failures and mathematical verification failures are handled differently, so the LLM repairs the correct root cause instead of trying to solve syntax and correctness in the same pass.
- **Mathematical Correctness Validation**: Every compiled candidate is compared against the baseline output before it is treated as a valid performance result. A faster kernel that produces wrong values is rejected.
- **Profile-Mode Control**: The optimizer supports `none`, `auto`, `auto-strict`, `auto-relaxed`, `ncu`, `nsys`, and `code`, so you can choose between strict benchmarking, relaxed fallback ranking, or direct profiler control.
- **Deep Profiling Integration**: Connects directly with NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`) to collect detailed execution data, identify hotspots, and keep the kernel optimization loop grounded in real measurements.
- **Structured Console Dashboard**: Visualizes execution state, live hardware metrics (CPU, RAM, GPU, VRAM usage), progress, and code diffs in real time. Failure states are shown as clear status cards instead of ambiguous latency values.
- **Mode-Aware Reporting**: Report files and dashboard summaries are separated by profile mode, so runs do not overwrite each other and benchmark history can be compared across modes.
- **LM Studio Integration**: Interfaces directly with LM Studio's standard OpenAI-compatible API endpoint (`http://127.0.0.1:1234/v1/completions`) for fast, local, GPU-accelerated LLM reasoning without compiling local bindings.

---

## Feature Behavior Guide

This section explains what the main sub-features actually do at runtime.

### Optimization Loop
- Reads a CUDA source file or folder of files.
- Sends the current code and environment context to the local LLM.
- Compiles the generated kernel with NVCC.
- Runs the result to check correctness before measuring speed.
- Repeats the loop for the requested number of iterations.
- Saves the best accepted candidate to the requested output path.

### Self-Healing
- If compilation fails, the error output is summarized and sent back to the LLM.
- If the code compiles but produces incorrect results, the verification failure is routed to a correctness-focused repair prompt.
- This keeps syntax repairs and algorithmic repairs separate.

### Profile Modes

| Mode | What it does | When to use |
| :--- | :--- | :--- |
| `none` | Skips profiler-based timing and uses a basic executable timer path. | Quick sanity checks or environments where profiler tooling is unavailable. |
| `auto` | Chooses the best available profiler path automatically, but still treats invalid profiler output as unavailable. | General use when you want the tool to pick the profiler backend. |
| `auto-strict` | Uses profiler data only if it is valid. If profiling cannot produce a usable result, no fake latency is shown. | Benchmarking and result comparison where accuracy matters most. |
| `auto-relaxed` | Falls back to a timer-only value when profiler output is blocked or incomplete, but marks it as informational only. | Debugging or rough ranking when you still want a visible time estimate. |
| `ncu` | Forces Nsight Compute profiling. | Kernel-level performance analysis and hardware counter inspection. |
| `nsys` | Forces Nsight Systems profiling. | Timeline and system-trace analysis. |
| `code` | Uses `cudaProfilerStart()` / `cudaProfilerStop()` inside the harness. | When you want capture controlled from inside the generated code. |

### Reporting
- `--report` writes a JSON run log for the current optimization session.
- Report filenames are mode-aware, so `auto-strict`, `auto-relaxed`, `ncu`, and other modes do not overwrite one another.
- The report includes the selected profile mode, best latency, original latency, per-iteration history, and regression guard details.

---

## Prerequisites
- **OS**: Windows (with PowerShell/Cmd) or Linux (Git Bash supported).
- **GPU**: NVIDIA GPU with up-to-date drivers.
- **CUDA Toolkit**: Recommended v12.x or v13.x (with `nvcc` added to your system path).
- **NVIDIA Nsight Tools**: Nsight Compute (`ncu`) and Nsight Systems (`nsys`) for advanced profiling modes.

Verify your environment tools:
```powershell
nvcc --version
ncu --version
nsys --version
```

---

## Installation

### Windows Local Installation

1. **Prerequisites Check**:
   - Verify NVIDIA GPU drivers are installed:
   ```powershell
   nvidia-smi
   ```
   - Verify CUDA Toolkit (v12.x or v13.x):
   ```powershell
   nvcc --version
   ```
   - Verify Python 3.8+ is installed:
   ```powershell
   python --version
   ```

2. **Clone the repository**:
   ```powershell
   git clone https://github.com/ThemeHackers/cudallm-cli.git
   cd cudallm-cli
   ```

3. **Install and set up (Automated - Recommended)**:
   Simply run the automated installation script. It will automatically check or create the virtual environment, install all dependencies, and initialize the system:
   ```powershell
   # If you get an execution policy error, run this first:
   # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   .\install.ps1
   ```

4. **Install and set up (Manual)**:
   If you prefer manual setup:
   - Create and activate the virtual environment:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - Install dependencies and local packages:
     ```powershell
     pip install -r requirements.txt
     pip install -e .
     cudallm setup-gpu
     ```
   - Initialize and verify:
     ```powershell
     cudallm init
     cudallm doctor
     ```

> [!TIP]
> **LM Studio Setup**:
> Make sure to download and start **LM Studio**. Under the developer tab, load a GGUF model (e.g., `cudaLLM-8B` or similar) and start the local server on port `1234` before running the optimizer.

### Linux/macOS Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ThemeHackers/cudallm-cli.git
   cd cudallm-cli
   ```

2. **Install and set up (Automated - Recommended)**:
   Simply run the automated installation script. It will automatically check or create the virtual environment, install all dependencies, and initialize the system:
   ```bash
   chmod +x ./install.sh
   ./install.sh
   ```

3. **Install and set up (Manual)**:
   If you prefer manual setup:
   - Create and activate the virtual environment:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Install dependencies and local packages:
     ```bash
     pip install -r requirements.txt
     pip install -e .
     cudallm setup-gpu
     ```
   - Initialize and verify:
     ```bash
     cudallm init
     cudallm doctor
     ```
---

## Quick Start

### Step 1: Scan Environment
Run the discovery command to scan for compiler tools, profilers, and GPU capabilities. Discovered paths will be saved in `config/config.json`:
```powershell
cudallm init
```

### Step 2: Verify LM Studio Connection
Start **LM Studio** manually, load a model in the **Developer Tab**, start the server (default port `1234`), and verify the connection status:
```powershell
cudallm serve
```

### Step 3: Run Kernel Optimization
Optimize a target CUDA kernel using the default 3-iteration self-healing loop:
```powershell
cudallm optimize path/to/kernel.cu -o optimized.cu --iters 3 --profile-mode auto --nvtx
```

---

## Detailed Command Reference

| Command | Arguments | Options | Description |
| :--- | :--- | :--- | :--- |
| `init` | None | None | Scans system paths, locates compiler/profiler executables, and persists configuration. |
| `doctor` / `check` | None | None | Displays a clean summary table of discovered paths and GPU capability. |
| `setup-gpu` | None | `--dry-run`, `--force-reinstall` | Verifies environment readiness for GPU kernel compiling. Bypasses llama-cpp-python installation. |
| `serve` | None | `--host`, `--port`, `--public-url`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--ssl-cert-file`, `--allow-unsafe-network`, `--reuse-port` | Checks LM Studio connection status or displays startup instructions. |
| `agent` | None | `--instruction`, `--llm-url`, `--llm-api-key`, `--llm-api-key-file`, `--insecure` | Runs the LangChain-based autonomous performance engineering agent. |
| `dashboard` | None | `--port` | Start the CUDA LLM Optimizer Web Dashboard. |
| `sandbox-run` | None | `--image`, `--cmd`, `--mount`, `--workdir`, `--mount-cwd/--no-mount-cwd`, `--timeout`, `--mem-limit-mb` | Run a command inside a Docker sandbox container. |
| `optimize` | `<file_or_folder>` | `-o/--output`, `-i/--iters`, `--target`, `--retries`, `--fast-math`, `-O/--opt-level`, `--profile-mode`, `--nvtx`, `--apply-nvtx`, `--ncu-metrics`, `--dry-run`, `--llm-url`, `--insecure` | Runs the iterative optimization agent. Supports dry runs, folder batches, custom compilers flags, and NVTX injections. |
| `expert` | `<exe_path>` | `--metrics`, `--run-deep`, `--code`, `--auto-nvtx`, `--rerun`, `--dry-run`, `--llm-url` | Performs advanced profiling on a compiled binary, identifies hotspots, runs deep NCU sweeps, and outputs LLM analyses. |
| `audit` | `<file_or_folder>` | `--markdown`, `--recursive/--no-recursive`, `--llm-url` | Performs a static structural audit on CUDA kernels using LLM prompts. Can output reports in Markdown format. |
| `ncu` | `<exe_path>` | `-o/--output`, `--metrics`, `--raw`, `--timeout`, `--dry-run` | Direct wrapper to execute Nsight Compute, exporting performance metrics into a clean CSV file. Add `--raw` and put profiler options after `--` to pass them through verbatim. |
| `nsys` | `<exe_path>` | `--command`, `-o/--output`, `--trace`, `--capture-range`, `--code`, `--raw`, `--timeout`, `--dry-run` | Direct wrapper to run Nsight Systems commands. Use `--command` for `profile`, `launch`, `start`, `stats`, `analyze`, `export`, `sessions`, `status`, `stop`, or `shutdown`. Add `--raw` and put profiler options after `--` to pass them through verbatim. |
| `profile` | `<exe_path>` | `--mode [auto\|ncu\|nsys]`, `--metrics`, `--code` | Runs either NCU or NSYS based on availability. |
| `help` | None | None | Prints the plain text command reference page. |

---

## Direct Shell Integration

The `ncu` and `nsys` commands can now be used as thin shell wrappers around the NVIDIA tools. Use `--raw` when you want the underlying profiler options to be passed through unchanged.

Examples:
```powershell
cudallm ncu .\my_kernel.exe --raw -- --set full --section SpeedOfLight --page raw
cudallm nsys --command status
cudallm nsys .\my_kernel.exe --code --raw -- --trace cuda,nvtx --capture-range=cudaProfilerApi
```

For `nsys`, `--command` selects the NVIDIA subcommand directly. Use profile-style commands for timeline captures and non-profile commands for session/status/analyze/export workflows.

## Advanced Core Workflows

### 1. Autonomous Optimization & Self-Healing
When running `cudallm optimize`, the tool starts a closed feedback loop:
* **The Harness**: A temporary verification harness is generated wrapping your CUDA kernel.
* **NVCC Compilation**: The harness is compiled. If compilation fails, the compiler error output is automatically sent to the LLM alongside the generated source code with a prompt to "heal" the compile errors.
   The compiler diagnostics are condensed first, then the LLM is asked to repair the source and the tool recompiles immediately.
* **Correctness Check**: Once compiled, the binary runs and compares its mathematical output values against the baseline kernel. If verification fails, this is also sent to the LLM to fix logical discrepancies.
   Verification failures follow a separate repair path so the model can focus on correctness rather than syntax.
* **Profiling**: Only mathematically correct kernels are profiled to determine execution time, protecting against empty or dummy speedups.
   If Nsight profiling falls back or cannot export data cleanly, the terminal now reports that explicitly instead of labeling the result as verified.

#### Key Flags:
* `--fast-math`: Injects `-use_fast_math` flags into compilation.
* `--nvtx`: Instruments the generated harness using NVTX ranges.
* `--profile-mode code`: Uses `cudaProfilerStart()` / `cudaProfilerStop()` inside the harness for precise profiling.

---

### 2. Expert Profiling Workflow (nsys → ncu)
The `cudallm expert` command profiles a pre-compiled CUDA binary, identifies bottleneck kernels, and recommends optimizations:

1. **Capture System Timeline**: Runs `nsys` to capture CPU-GPU interactions and execution markers.
2. **Broad Sweep**: Runs `ncu` collecting standard metrics like cycles and DRAM throughput.
3. **Hotspot Analysis**: Parses the output CSV to locate the kernel with the highest execution bottleneck.
4. **Deep Collection Recommendations**: Recommends a targeted `ncu` command for the specific hotspot kernel.
5. **Auto NVTX Suggestions**: Query the LLM with the profiling logs to automatically generate an `nvtx_suggestion.cu` instrumented file.

Example command:
```powershell
cudallm expert ./my_cuda_binary.exe --auto-nvtx --rerun
```

---

### 3. LangChain Autonomous Performance Agent
The `cudallm agent` command launches an autonomous agent running on LangChain 1.x and LangGraph. This agent is equipped with low-level GPU tool bindings (`compile_cuda`, `profile_system`, `profile_kernel`, and `summarize_profile`) to solve complex performance engineering instructions:

* **Instruction Execution**: The agent iteratively decides which tools to invoke based on user prompts.
* **Tool Set**:
  * `compile_cuda`: Compiles code with `nvcc`.
  * `profile_system`: Runs system-wide profiling via `nsys`.
  * `profile_kernel`: Gathers kernel execution metrics using `ncu`.
  * `summarize_profile`: Provides detailed human-readable summary of profiling reports.

Example command:
```powershell
cudallm agent --instruction "Compile examples/vector_add.cu and profile it with nsys and ncu"
```

#### Hybrid Dual-Model Routing Setup

The system supports running a **Hybrid Dual-Model Setup** by loading both models concurrently in LM Studio on the exact same port (e.g. `1234`). The system will automatically route requests based on keyword matching:

* **Orchestrator/Agent Model (`cudallm agent`)**: Target model is **`Qwen2.5-3B-Instruct-GGUF`** (auto-selects any loaded model containing `qwen`, `llama`, or `instruct`).
* **Kernel Optimizer Model (`cudallm optimize`)**: Target model is **`cudaLLM-8B-GGUF`** (auto-selects any loaded model containing `cuda`, `llm`, or `coder`).

To explicitly define model names, configure them in `config/config.json`:
```json
{
  "llm_url": "http://127.0.0.1:1234/v1/completions",
  "llm_model_name": "cudaLLM-8B",
  "agent_llm_model_name": "qwen2.5-3b-instruct"
}
```

---

## Connection Configuration & Probing

The `cudallm serve` command is a helper command used to verify connectivity to your LM Studio server or configure client endpoints:

### Checking LM Studio Status
To probe the status of LM Studio and verify loaded models:
```powershell
cudallm serve
```

### Remote/Local Network Overrides
If LM Studio or your LLM server is hosted on a different machine or requires authentication, you can test and persist client endpoints:
* **Custom URL & Host**:
  ```powershell
  cudallm serve --host 192.168.1.100 --port 1234
  ```
* **API Key & TLS Verification**:
  Ensure secure connections when accessing remote endpoints:
  ```powershell
  cudallm serve --public-url https://my-remote-llm/v1/completions --api-key-file .\secrets\key.txt
  ```

---

## CI/CD Regression Checks

To prevent slow kernels from being merged into production repositories, `cudallm` includes scripts for automated CI regression testing.

### CSV Comparisons
`tools/compare_ncu.py` compares a baseline NCU CSV metrics file with a newly collected run:
```powershell
python tools/compare_ncu.py ci/baselines/baseline.csv ncu_output.csv -c dram__throughput.avg --threshold 0.02
```
* Returns exit code `3` if a regression exceeding the threshold (default 2%) is detected.

### Bash Regression Script
On Linux self-hosted runners, use the `ci/ncu_regression_check.sh` utility:
```bash
./ci/ncu_regression_check.sh ./my_compiled_kernel ci/baselines/kernel_baseline.csv "sm__cycles_elapsed.avg" "cycles"
```

---

## Configuration Schema

Config parameters are persisted inside `config/config.json`. Below is the schema structure:
```json
{
  "llm_url": "http://127.0.0.1:1234/v1/completions",
  "llm_api_key": null,
  "llm_api_key_file": null,
  "llm_verify_tls": true,
  "llm_allow_insecure_remote": false,
   "max_stream_chunks": 8000,
  "nvcc_path": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\nvcc.exe",
  "nvidia_smi_path": "C:\\Windows\\system32\\nvidia-smi.exe",
  "ncu_path": "C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2026.1.1\\ncu.bat",
  "nsys_path": "C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2026.1.1\\host\\target-windows-x64\\nsys.exe",
  "last_discovery_at": "2026-05-19T20:25:00"
}
```

---

## Troubleshooting

### Windows-Specific Issues

**PowerShell Execution Policy Error**
```
.venv\Scripts\activate : The term '.venv\Scripts\activate' is not recognized
```
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```

**NVCC Not Found in PATH**
```
'nvcc' is not recognized as an internal or external command
```
**Solution**:
- Add CUDA Toolkit to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
- Or run `cudallm init` to auto-detect NVCC path



**CUDA Out of Memory**
```
CUDA_ERROR_OUT_OF_MEMORY
```
**Solution**:
- Reduce batch size or model size
- Use smaller GGUF quantization (Q4_K_M instead of Q8_0) in LM Studio
- Close other GPU applications

### General Issues

**Profiler Not Found Error**
If `ncu` or `nsys` show `Not Found`, run `cudallm init` to force a workspace path refresh after installing NVIDIA Nsight.

**Empty NSYS Timelines**
If `nsys` reports do not display GPU kernel profiles, run with `--profile-mode code` to enable programmatic compiler activation or instrument kernels with NVTX.

**LLM Server Connection Refused**
```
Connection refused to http://localhost:1234/v1/completions
```
**Solution**:
- Verify LM Studio local server is running and listening by running: `cudallm serve`
- Check firewall settings
- Verify port is not in use: `netstat -ano | findstr :1234` (Windows) or `lsof -i :1234` (Linux)

**Compilation Errors Not Healing**
If the self-healing loop fails to fix compilation errors:
- Check that `nvcc` is working independently
- Verify CUDA version compatibility with your code
- Try manual compilation first to isolate the issue
- Check `config/config.json` for correct tool paths

**Python Version Incompatibility**
```
SyntaxError or ImportError after installation
```
**Solution**:
- Ensure Python 3.8 or later is installed
- Recreate virtual environment with correct Python version
- Update pip: `python -m pip install --upgrade pip`

### Environment Verification Commands

**Windows**:
```powershell
# Check all tools
nvcc --version
ncu --version
nsys --version
nvidia-smi
python --version

# Verify cudallm installation
cudallm doctor
cudallm init
```

**Linux**:
```bash
# Check all tools
nvcc --version
ncu --version
nsys --version
nvidia-smi
python3 --version

# Verify cudallm installation
cudallm doctor
cudallm init
```

