# cudallm-cli
**Local Autonomous CUDA Engineering Agent**

`cudallm-cli` is an autonomous CLI tool designed to act as your personal AI CUDA engineer. It runs entirely on your local machine, creating a closed-loop system where a local LLM can read your CUDA kernel, modify it, compile it, profile its latency on your actual GPU hardware, and automatically fix errors—all iteratively.

---

## 🚀 Key Premium Features

* **Closed-Loop Optimization**: Think ➔ Code ➔ Compile ➔ Profile ➔ Fix.
* **Live IDE Syntax Highlighting**: Includes a premium real-time terminal CUDA C++ syntax colorizer. Streams C++ code directly into the terminal with distinct color-coding for CUDA specifiers (Gold), types (Cyan), keywords (Magenta), functions (Bold Gold), comments (Gray), and numbers (Orange), providing a professional VS Code-like developer experience.
* **Instant Early Termination (Anti-Looping Engine)**: An intelligent state-driven stream parser that immediately aborts/terminates the LLM API request once the first closing code block (` ``` `) is received. This completely eliminates repetitive loop generation bugs in local LLMs, saving context windows, tokens, and VRAM.
* **Zero-Leak Stream Parser**: Cleanly intercepts, parses, and strips raw markdown tags (like ` ```cpp `, ` ```cuda ` or ` ``` `) from the live output, printing only the clean code block and separating the LLM reasoning process in cyan panels.
* **Auto-Detected Toolchain on Windows**: Automatically scans standard registry/folders (`Program Files`, `System32`, `NVIDIA Corporation`, etc.) to self-detect `nvcc`, `nvidia-smi`, and `Nsight Compute (ncu)` on Windows systems without requiring tedious manual environment PATH setups!
* **Compilation Self-Healing**: If the generated code fails to compile, the tool feeds the `nvcc` error log back to the LLM to fix its own mistakes automatically.
* **Hardware Injection**: Automatically detects your GPU model, VRAM, and Compute Capability via `nvidia-smi` and injects this into the LLM context.
* **Diffing UI**: See exactly what the LLM changed in your code with live colored terminal diffing.
* **Advanced Profiling**: Transparently uses NVIDIA Nsight Compute (`ncu`) if available for hardware analysis, gracefully falling back to high-precision GPU event timers.
* **Flexible Backend**: Fully compatible with both `llama.cpp` and `Ollama`.

---

## Prerequisites & System Setup

To unleash the full power of `cudallm-cli`, you need to set up the modern CUDA compiler and profiling toolchain on your system.

### 1. CUDA Toolkit (v11.6 - v13.0+)
* Required for the `nvcc` compiler and GPU runtimes.
* **Auto-Detected on Windows:** Scans the standard installation folder:
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
* Ensure `nvcc` is available in your terminal by running:
  ```powershell
  nvcc --version
  ```

### 2. NVIDIA Nsight Compute (ncu)
Required for deep GPU kernel profiling and metric gathering. If not present, the suite falls back to system GPU timers.

#### How to Install:
1. **Download Standalone Installer (Recommended):**
   * Download Nsight Compute directly from the official developer page: [NVIDIA Nsight Compute Standalone](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
2. **Alternative (CUDA Installer):**
   * Re-run your CUDA Toolkit installer, choose **Custom (Advanced)** installation, and ensure the **Nsight Compute** checkbox is ticked.

#### Windows Environment Setup (If not auto-detected):
By default, the installer does not add `ncu` to the system variables. The tool will auto-detect it under `C:\Program Files\NVIDIA Corporation`, but you can also add it manually:
1. Press the `Win` key, type **env**, and select **"Edit the system environment variables"**.
2. Click **"Environment Variables..."** at the bottom.
3. Under **"System variables"**, double-click **`Path`** to edit it.
4. Click **"New"** and add your Nsight Compute target directory, for example:
   ```text
   C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\
   ```
5. Click **"OK"** on all windows, restart your terminal, and verify the path by running:
   ```powershell
   ncu --version
   ```

---

## Installation

```bash
git clone https://github.com/ThemeHackers/cudallm-cli.git
cd cudallm-cli
pip install -e .
```

---

## Quick Start

1. **Start your local LLM server** (e.g. via `llama-server`):
   ```bash
   cudallm serve --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q2_K.gguf
   ```

2. **Check toolchain and environment status:**
   ```bash
   cudallm init
   ```

3. **Let the AI Agent optimize your kernel:**
   ```bash
   cudallm optimize path/to/kernel.cu --iters 5 --target latency --fast-math -O 3
   ```
   You can also pass a folder to optimize all detected CUDA files (currently `.cu` and `.cuh`):
   ```bash
   cudallm optimize path/to/kernels --recursive
   ```

4. **Audit code for Warp Divergence or Bank Conflicts:**
   ```bash
   cudallm audit path/to/kernel.cu
   ```
   Folder audits are supported too:
   ```bash
   cudallm audit path/to/kernels --markdown --recursive
   ```

---

## Core Architecture

* **`cli.py`**: Manages the CLI interface, live telemetry dashboard, and the self-healing loops.
* **`discover.py`**: Queries local hardware information, `nvidia-smi`, and active profiler paths.
* **`sandbox.py`**: Compiles temporary units, injects dynamic benchmark wrappers, and measures execution times.
* **`llm_client.py`**: Manages backend LLM requests, structures reasoning prompts, and monitors generation statistics.

---
*Created by ThemeHackers*
