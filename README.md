# cudallm-cli

[![CI](https://github.com/ThemeHackers/cudallm-cli/actions/workflows/ncu-regression.yml/badge.svg)](https://github.com/ThemeHackers/cudallm-cli/actions/workflows/ncu-regression.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Local Autonomous CUDA Optimization Agent — closed-loop tool that uses a local LLM to read, modify, compile, profile and repair CUDA kernels.

Table of Contents
- [What & Why](#what--why)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CI / Regression Checks](#ci--regression-checks)

What & Why
-----------
What: automated local toolchain for CUDA kernel optimization using an LLM-backed edit/compile/profile loop.
Why: speeds iterative performance engineering by automating common edit/compile/profile/fix cycles and integrating `ncu`/`nsys` where available.

Key Features
------------
- Closed-loop optimization: generate → compile → profile → heal.
- NVTX + code-driven profiling support for precise `nsys` captures.
- `ncu` CSV export + simple regression comparator for CI.
- Auto-discovery of CUDA toolchain and local LLM server path (saved to `config/config.json`).

Prerequisites
-------------
- Windows machine with NVIDIA GPU and drivers.
- CUDA Toolkit v12.6 (install from NVIDIA). Verify:
```powershell
nvcc --version
```
- Nsight Compute (ncu). Verify:
```powershell
ncu --version
```
- Nsight Systems (nsys). Verify:
```powershell
nsys --version
```

Install
-------
1. Create and activate a Python virtual environment, then install:
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -e .
```

Quick Start
-----------
1) Scan environment and persist discovered paths:
```powershell
cudallm init
```

2) Start local LLM server (must have been unzipped into repo):
```powershell
cudallm serve --port 8080 --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q2_K.gguf
```

LAN / multi-network usage

- Bind to a LAN interface and require an API key:
```powershell
cudallm serve --host 0.0.0.0 --port 8080 --api-key-file .\secrets\llama-api.key --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q2_K.gguf
```
- Add TLS for cross-network deployments:
```powershell
cudallm serve --host 0.0.0.0 --port 8443 --ssl-key-file .\secrets\server.key --ssl-cert-file .\secrets\server.crt --api-key-file .\secrets\llama-api.key --repo prithivMLmods/cudaLLM-8B-GGUF --file cudaLLM-8B.Q2_K.gguf
```
- If you bind with a wildcard address such as `0.0.0.0`, pass `--public-url` so `config/config.json` stores the reachable client URL, for example `http://192.168.1.10:8080/completion` or `https://gateway.example.com:8443/completion`.

3) Optimize a kernel (single file):
```powershell
cudallm optimize path/to/kernel.cu -o optimized.cu --iters 3 --profile-mode auto --nvtx --ncu-metrics "sm__sass_thread_inst_executed_avg"
```

4) Audit source (markdown output):
```powershell
cudallm audit path/to/kernel.cu --markdown
```

Advanced Usage
--------------
Profiling modes

- `none`: no profiler — fallback to harness timers.
- `auto`: prefer `ncu` if available, fallback to `nsys`/timers.
- `ncu`: run Nsight Compute for detailed kernel metrics and CSV export.
- `nsys`: run Nsight Systems timeline capture (use with NVTX or code-driven profiling).
- `code`: in-process `cudaProfilerStart/Stop` + NVTX ranges (best for `nsys --capture-range=cudaProfilerApi`).

Environment and planning helpers

- `cudallm doctor` or `cudallm check` shows the discovered CUDA tools, LLM server, and config paths.
- `cudallm optimize ... --dry-run` and `cudallm expert ... --dry-run` print the planned flow without compiling or profiling.
- `cudallm expert ... --auto-nvtx --rerun` saves NVTX suggestions and runs a follow-up profiling pass.

NVTX & code-driven profiling

- To label regions, use `--nvtx` when calling `optimize` so the generated harness wraps kernels with NVTX push/pop.
- For precise `nsys` captures, run with `--profile-mode code` which uses `cudaProfilerStart()`/`cudaProfilerStop()` in the harness and lets `nsys` capture only the profiled region.

Example: run `nsys` manually against the harness produced by `optimize` (or the compiled exe)

```powershell
# produce a harness executable (optimize will compile a temp exe during its run)
cudallm optimize path/to/kernel.cu --iters 1 --profile-mode code --nvtx -o optimized.cu

# capture timeline (self-hosted / local machine)
nsys profile --output nsys_capture --capture-range=cudaProfilerApi --trace=cuda,cudnn ./temp_cuda_kernel.exe
```

Example: run `ncu` and export CSV

```powershell
ncu --target-processes all --csv --export-path ncu_output --metrics sm__sass_thread_inst_executed_avg,dram__throughput.avg ./temp_cuda_kernel.exe
```

Recommended `ncu` metric templates

- compute-focused: `sm__sass_thread_inst_executed_avg,sm__cycles_elapsed.avg`
- memory-focused: `dram__throughput.avg,lts__t_bytes` (or vendor-specific DRAM metrics)

Using `tools/compare_ncu.py`

- The script compares two `ncu` CSV exports. Supply baseline and current CSV paths and optionally `-c/--column` (column name or index) and `--threshold` (relative regression threshold, default 2%).

CI tips

- Keep CI runs short: capture a small, representative kernel with single iteration and limited input sizes.
- Store baseline CSVs under `ci/baselines/` and reference them in the workflow inputs.
- Use the provided `.github/workflows/ncu-regression.yml` as a template; run it on a self-hosted GPU runner that has `nvcc`, `ncu`, and `nsys` installed.

Troubleshooting

- If `ncu` or `nsys` are not found, run `cudallm init` after installing the tools to refresh `config/config.json`.
- If `nsys` captures empty timelines, ensure the harness uses `cudaProfilerStart()`/`cudaProfilerStop()` (use `--profile-mode code`) or add NVTX ranges.

Configuration
-------------
- `config/config.json` stores discovered tool paths, `llm_url`, `llm_api_key_file`, `llm_verify_tls`, `llm_allow_insecure_remote`, and `last_discovery_at`.
- Use `cudallm init` after installing toolchain to refresh and persist paths.
- `llama-server.exe` supports `--host`, `--api-key`, `--api-key-file`, `--ssl-key-file`, and `--ssl-cert-file`, so you can expose the server safely on a LAN or over routed networks.

### Per-run LLM Overrides
For `optimize`, `expert`, and `audit` commands, you can temporarily override the default LLM backend config on a per-run basis without modifying `config/config.json`:
- `--llm-url` (or environment variable `LLM_URL`): Custom endpoint URL (e.g. `--llm-url http://10.212.3.55:8080/completion`).
- `--llm-api-key` (or environment variable `LLM_API_KEY`): Custom API key.
- `--llm-api-key-file` (or environment variable `LLM_API_KEY_FILE`): Custom API key file path.
- `--insecure`: Bypass TLS/SSL verification and permit insecure remote HTTP connections (useful for self-signed VPNs/LANs).

CI / Regression Checks
----------------------
- `ci/ncu_regression_check.sh` runs `ncu` (CSV) and compares with `tools/compare_ncu.py`.
- Example workflow: `.github/workflows/ncu-regression.yml` (requires a self-hosted GPU runner with NVIDIA tools).

    
