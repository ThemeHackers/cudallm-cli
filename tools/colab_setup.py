#/usr/bin/env python3

import os
import sys
import subprocess
import time
import socket
import shutil
import shlex


def ensure_python_dependencies(repo_root):
    req_file = os.path.join(repo_root, "requirements.txt")

    print("[INFO] Upgrading pip/setuptools/wheel...")
    code, _, err = run_command(f"{sys.executable} -m pip install -U pip setuptools wheel")
    if code != 0:
        print(f"[WARNING] Failed to upgrade packaging tools: {err}")

    if os.path.exists(req_file):
        print(f"[INFO] Installing Python dependencies from {req_file}...")
        code, _, err = run_command(f"{sys.executable} -m pip install -r {shlex.quote(req_file)}")
        if code != 0:
            print(f"[ERROR] Failed to install requirements.txt: {err}")
            sys.exit(1)
    else:
        print("[WARNING] requirements.txt not found; skipping dependency install.")

    print("[INFO] Forcing CUDA-enabled llama-cpp-python install...")
    cuda_llama_cmd = (
        f'FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_CUDA=on" '
        f'{sys.executable} -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python'
    )
    code, _, err = run_command(cuda_llama_cmd)
    if code != 0:
        print(f"[ERROR] Failed to install CUDA-enabled llama-cpp-python: {err}")
        sys.exit(1)

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
