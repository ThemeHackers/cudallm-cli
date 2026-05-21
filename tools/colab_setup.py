#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import socket
import shutil
import shlex
import re
import json
import urllib.request

import requests


def detect_cuda_version():
    """Detect CUDA version from nvidia-smi output."""
    print("[INFO] Detecting CUDA version...")
    code, smi_output, _ = run_command("nvidia-smi")
    if code != 0:
        print("[WARNING] Could not run nvidia-smi, assuming CUDA 12.1")
        return "121"

 
    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', smi_output)
    if match:
        major = match.group(1)
        minor = match.group(2)
        cuda_version = f"{major}{minor}"
        print(f"[INFO] Detected CUDA version: {major}.{minor} (using cu{cuda_version})")
        return cuda_version
    else:
        print("[WARNING] Could not parse CUDA version from nvidia-smi, assuming CUDA 12.1")
        return "121"


def verify_cuda_enabled():
    """Verify that llama-cpp-python has CUDA support enabled."""
    print("[INFO] Verifying CUDA support in llama-cpp-python...")
    check_cmd = f"{sys.executable} -c \"import llama_cpp; print('llama_cpp_version:', llama_cpp.__version__); print('CUDA supported:', hasattr(llama_cpp, 'llama_cpp_cuda')); print('CUDA module loaded successfully' if hasattr(llama_cpp, 'llama_cpp_cuda') else 'CUDA module not available')\""
    code, stdout, _ = run_command(check_cmd)
    if code != 0:
        print("[WARNING] CUDA module import failed, but this may be expected for CPU-only builds")
        print("[INFO] Output:", stdout)
        return False

    print("[INFO] CUDA verification output:")
    print(stdout)

    if "CUDA supported: True" in stdout:
        print("[SUCCESS] CUDA support is properly enabled!")
        return True
    else:
        print("[WARNING] CUDA support may not be properly enabled")
        return False


def ensure_python_dependencies(repo_root):
    req_file = os.path.join(repo_root, "requirements.txt")

    print("[INFO] Upgrading pip/setuptools/wheel...")
    code, _, err = run_command(f"{sys.executable} -m pip install -U pip setuptools wheel")
    if code != 0:
        print(f"[WARNING] Failed to upgrade packaging tools: {err}")

    if os.path.exists(req_file):
        print(f"[INFO] Installing Python dependencies from {req_file}...")
        base_deps = []
        with open(req_file, "r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                if item == "llama-cpp-python":
                    continue
                base_deps.append(item)

        if base_deps:
            deps_args = " ".join(shlex.quote(dep) for dep in base_deps)
            code, _, err = run_command(f"{sys.executable} -m pip install {deps_args}")
        else:
            code, _, err = 0, "", ""
        if code != 0:
            print(f"[ERROR] Failed to install requirements.txt: {err}")
            sys.exit(1)
    else:
        print("[WARNING] requirements.txt not found; skipping dependency install.")


    cuda_version = detect_cuda_version()

    if int(cuda_version) >= 130:
        wheel_candidates = ["131", "126", "121"]
    else:
        wheel_candidates = [cuda_version, "126", "121"]

    code = 1
    err = ""
    tried_candidates = []
    for candidate in wheel_candidates:
        if candidate in tried_candidates:
            continue
        tried_candidates.append(candidate)
        print(f"[INFO] Installing CUDA-enabled llama-cpp-python using pre-built wheels for CUDA {candidate}...")
        cuda_llama_cmd = (
            f'{sys.executable} -m pip install --no-cache-dir llama-cpp-python '
            f'--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu{candidate}'
        )
        code, _, err = run_command(cuda_llama_cmd)
        if code == 0:
            break

    if code != 0:
        print("[WARNING] All pre-built wheels failed, falling back to compile from source...")
        print("[INFO] This may take 10-20 minutes, please be patient...")
        cuda_llama_cmd = (
            f'CMAKE_ARGS="-DGGML_CUDA=on" '
            f'{sys.executable} -m pip install --no-cache-dir --force-reinstall --no-binary :all: llama-cpp-python'
        )
        code, _, err = run_command(cuda_llama_cmd)

    if code != 0:
        print(f"[ERROR] Failed to install CUDA-enabled llama-cpp-python: {err}")
        sys.exit(1)

   
    cuda_enabled = verify_cuda_enabled()
    if not cuda_enabled:
        print("[WARNING] CUDA support verification failed. The model may run on CPU, which will be very slow.")
        print("[INFO] If you want to force CPU-only mode, remove --use-cuda from server arguments.")
    else:
        print("[SUCCESS] CUDA support verified successfully!")

    check_cmd = (
        f"{sys.executable} -c \""
        "import fastapi,uvicorn,huggingface_hub; "
        "import llama_cpp; "
        "print('deps-ok')"
        "\""
    )
    code, _, err = run_command(check_cmd)
    if code != 0:
        print("[ERROR] Missing required Python packages for server backend (fastapi/uvicorn/huggingface_hub/llama_cpp).")
        print(f"[DETAIL] {err}")
        sys.exit(1)
def run_command(cmd, shell=True, timeout=None):
    print(f"[RUNNING] {cmd}")
    try:
        res = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if res.returncode != 0:
          
            out_tail = (res.stdout or "")[-2000:]
            err_tail = (res.stderr or "")[-2000:]
            print("[COMMAND FAILED] return code:", res.returncode)
            if out_tail:
                print("[STDOUT - tail]:\n" + out_tail)
            if err_tail:
                print("[STDERR - tail]:\n" + err_tail)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", e.stderr or "Timeout expired"


def get_latest_llama_tag():
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            tag = data.get("tag_name")
            if tag and tag.startswith("b") and tag[1:].isdigit():
                return tag
    except Exception:
        pass
    return "b9222"


def get_release_assets(tag):
    api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("assets", [])
    except Exception:
        return []


def parse_cuda_version(cuda_version_str):
    try:
        return float(str(cuda_version_str).split()[0])
    except (ValueError, IndexError, TypeError):
        return 12.4


def select_release_asset(tag, cuda_version, is_windows=False):
    assets = get_release_assets(tag)
    if not assets:
        return None

    version_token = "cuda-13.1" if cuda_version >= 13.0 else "cuda-12.4"
    platform_tokens = ["win"] if is_windows else ["ubuntu", "linux"]

    for asset in assets:
        name = asset.get("name", "")
        lowered = name.lower()
        if "cuda" not in lowered:
            continue
        if version_token not in lowered:
            continue
        if not any(token in lowered for token in platform_tokens):
            continue
        return asset

    return None


def download_and_extract_asset(asset, dest_dir):
    url = asset.get("browser_download_url")
    name = asset.get("name", "download")
    if not url:
        raise RuntimeError(f"Release asset {name} does not expose a download URL")

    archive_filepath = os.path.join(dest_dir, name)
    response = requests.get(url, stream=True, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP Status {response.status_code} for {name}")

    with open(archive_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if name.lower().endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive_filepath, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif name.lower().endswith((".tar.gz", ".tgz")):
        import tarfile

        with tarfile.open(archive_filepath, "r:gz") as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        raise RuntimeError(f"Unsupported archive format for {name}")

    if os.path.exists(archive_filepath):
        os.remove(archive_filepath)


def find_executable(dest_dir, bin_name):
    direct_path = os.path.join(dest_dir, bin_name)
    if os.path.exists(direct_path):
        return direct_path

    for root, _, files_list in os.walk(dest_dir):
        if bin_name in files_list:
            return os.path.join(root, bin_name)
    return None


def build_llama_server_from_source():
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

def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(('127.0.0.1', port))
        s.close()
        return True
    except Exception:
        return False

def main():
    print("="*60)
    print(" Google Colab CUDA LLM Server Setup & Diagnostics")
    print("="*60)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    ensure_python_dependencies(repo_root)

    print("[INFO] Installing cudallm in editable mode...")
    code, out, err = run_command(f"{sys.executable} -m pip install -e .")
    if code != 0:
        print(f"[WARNING] Failed to install cudallm in editable mode: {err}")
    else:
        print("[SUCCESS] Installed cudallm in editable mode.")

    if is_port_open(8081):
        print("[INFO] Port 8081 is in use. Attempting to terminate existing process...")

        if shutil.which("fuser"):
            run_command("fuser -k 8081/tcp")
            time.sleep(2)
        if is_port_open(8081) and shutil.which("lsof"):
            run_command("lsof -t -i:8081 | xargs kill -9")
            time.sleep(2)

    if is_port_open(8081):
        print("[ERROR] Could not free port 8081. Please check running processes manually.")
        sys.exit(1)
    else:
        print("[SUCCESS] Port 8081 is free.")

    server_py_path = os.path.join(repo_root, "tools", "server.py")
    model_repo = "prithivMLmods/cudaLLM-8B-GGUF"
    model_file = "cudaLLM-8B.Q4_K_M.gguf"

    if not os.path.exists(server_py_path):
        print(f"[ERROR] Python backend not found: {server_py_path}")
        sys.exit(1)

    print(f"[INFO] Found Python server at {server_py_path}. Launching with CUDA option...")
    py_cmd = shlex.join([
        sys.executable,
        server_py_path,
        "--hf-repo", model_repo,
        "--hf-file", model_file,
        "--host", "127.0.0.1",
        "--port", "8081",
        "--ctx", "4096",
        "--use-cuda",
    ])
    log_path = "/content/server_colab.log"
    with open(log_path, "a") as _:
        pass
    subprocess.Popen(f"nohup {py_cmd} > {log_path} 2>&1 &", shell=True)


    print("[INFO] Waiting for server initialization and model download/loading...")
    attempts = 0
    max_attempts = 120
    server_ready = False
    while attempts < max_attempts:
        time.sleep(5)
        attempts += 1
        
 
        if os.path.exists("/content/server_colab.log"):
            with open("/content/server_colab.log", "r") as f:
                log_content = f.read()
            if "HTTP server error" in log_content:
                print("[ERROR] Server encountered an HTTP server error.")
                print(log_content[-1000:])
                sys.exit(1)
            if "Traceback (most recent call last):" in log_content or "RuntimeError:" in log_content or "OSError:" in log_content:
                print("[ERROR] Server failed during startup. Log tail:")
                print(log_content[-2000:])
                sys.exit(1)
            if (
                "server is listening on" in log_content
                or "model loaded" in log_content
                or "Uvicorn running on" in log_content
                or "Application startup complete" in log_content
            ):
                server_ready = True
                break
        
        print(f"  - Waiting... (Attempt {attempts}/{max_attempts})")

    if not server_ready:
        print("[ERROR] Server startup timed out. Check /content/server_colab.log for details:")
        if os.path.exists("/content/server_colab.log"):
            with open("/content/server_colab.log", "r") as f:
                print(f.read()[-1000:])
        sys.exit(1)

    print("[SUCCESS] Server is online and listening on http://127.0.0.1:8081")

   
    _, smi_out, _ = run_command("nvidia-smi")
    print("\n" + "="*60)
    print(" Current GPU Status:")
    print("="*60)
    print(smi_out)
    
    if "python" in smi_out.lower() or "server.py" in smi_out.lower():
        print("\n[SUCCESS] Confirmed: Python LLM backend is running on the GPU!")
    else:
        print("\n[WARNING] Python backend process was not explicitly found in nvidia-smi processes table. Please check if VRAM usage is non-zero.")

    print("\nSetup finished successfully. You can now run:")
    print("!cudallm optimize examples/vector_add.cu --llm-url http://127.0.0.1:8081/completion --iters 1")
    print("="*60)

if __name__ == "__main__":
    main()
