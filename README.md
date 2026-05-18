# cudallm-cli 🚀

**Local Autonomous CUDA Engineering Agent**

`cudallm-cli` is an autonomous CLI tool designed to act as your personal AI CUDA engineer. It runs entirely on your local machine, creating a closed-loop system where a local LLM can read your CUDA kernel, modify it, compile it, profile its latency on your actual GPU hardware, and automatically fix errors—all iteratively.

## ✨ Features

- **Closed-Loop Optimization**: Think -> Code -> Compile -> Profile -> Fix.
- **Compilation Self-Healing**: If the generated code fails to compile, the tool feeds the `nvcc` error log back to the LLM to fix its own mistakes automatically.
- **Hardware Injection**: Automatically detects your GPU model, VRAM, and Compute Capability via `nvidia-smi` and injects this into the LLM context.
- **Diffing UI**: See exactly what the LLM changed in your code with live colored terminal diffing.
- **Advanced Profiling**: Transparently uses Nsight Compute (`ncu`) if available, gracefully falling back to `nvprof` or raw execution timers.
- **Flexible Backend**: Fully compatible with both `llama.cpp` and `Ollama`.

## 🛠️ Installation

```bash
git clone https://github.com/ThemeHackers/cudallm-cli.git
cd cudallm-cli
pip install -e .
```

## 🚀 Quick Start

1. Start your local LLM server (e.g., via `llama.cpp`):
```bash
llama-server -m cudaLLM-8B-Q8_0.gguf -ngl 99 -c 8192 --port 8080
```

2. Check environment and toolchains:
```bash
cudallm init
```

3. Let the Agent Optimize your Kernel:
```bash
cudallm optimize path/to/kernel.cu --iters 5 --target latency --fast-math -O 3
```

4. Audit your code for common bottlenecks (Warp Divergence, Bank Conflicts):
```bash
cudallm audit path/to/kernel.cu --markdown
```

## 🧠 Core Architecture

- **`cli.py`**: Manages the CLI interface, the optimization loop, generative diffs, and the self-healing retry mechanism.
- **`discover.py`**: Interacts with the system to discover CUDA toolkits (`nvcc`), `nvidia-smi`, and active profilers.
- **`sandbox.py`**: The execution environment that handles the risky business of compiling and safely profiling the output binary to extract real `ms` latencies.
- **`llm_client.py`**: Handles all AI interactions, supporting both `llama.cpp` and `Ollama` paradigms transparently while tracking tokens and generation times.

---
*Created by ThemeHackers*
