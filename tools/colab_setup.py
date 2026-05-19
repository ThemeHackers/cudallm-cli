import os
import sys
import subprocess
import time
import socket
import shutil

def run_command(cmd, shell=True, timeout=None):
    print(f"[RUNNING] {cmd}")
    try:
        res = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
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


    if is_port_open(8081):
        print("[INFO] Port 8081 is in use. Terminating existing process...")
        run_command("fuser -k 8081/tcp")
        time.sleep(2)
        if is_port_open(8081):
            run_command("lsof -t -i:8081 | xargs kill -9")
            time.sleep(2)

    if is_port_open(8081):
        print("[ERROR] Could not free port 8081. Please check running processes manually.")
        sys.exit(1)
    else:
        print("[SUCCESS] Port 8081 is free.")


    cuda_server_path = "/content/llama.cpp/build/bin/llama-server"
    if not os.path.exists(cuda_server_path):
        print("[INFO] Compiled llama-server with CUDA not found. Compiling now...")
        
       
        if not os.path.exists("/content/llama.cpp"):
            code, out, err = run_command("git clone https://github.com/ggml-org/llama.cpp /content/llama.cpp")
            if code != 0:
                print(f"[ERROR] Failed to clone llama.cpp: {err}")
                sys.exit(1)

      
        os.makedirs("/content/llama.cpp/build", exist_ok=True)
        code, out, err = run_command("cmake -B /content/llama.cpp/build -S /content/llama.cpp -DGGML_CUDA=ON")
        if code != 0:
            print(f"[ERROR] CMake configuration failed: {err}")
            sys.exit(1)


        code, out, err = run_command("cmake --build /content/llama.cpp/build --config Release -j$(nproc)")
        if code != 0:
            print(f"[ERROR] Build failed: {err}")
            sys.exit(1)

    if os.path.exists(cuda_server_path):
        print(f"[SUCCESS] CUDA-enabled llama-server found at: {cuda_server_path}")
    else:
        print("[ERROR] Failed to locate or compile CUDA-enabled llama-server.")
        sys.exit(1)

 
    model_repo = "prithivMLmods/cudaLLM-8B-GGUF"
    model_file = "cudaLLM-8B.Q4_K_M.gguf"
    
    server_cmd = (
        f"nohup {cuda_server_path} "
        f"--hf-repo {model_repo} "
        f"--hf-file {model_file} "
        f"-ngl 33 -c 4096 --host 127.0.0.1 --port 8081 --parallel 1 > /content/server_colab.log 2>&1 &"
    )
    print("[INFO] Launching llama-server with CUDA GPU acceleration in background...")
    subprocess.Popen(server_cmd, shell=True)

    # 4. Wait and verify server connection
    print("[INFO] Waiting for server initialization and model download/loading...")
    attempts = 0
    max_attempts = 30
    server_ready = False
    while attempts < max_attempts:
        time.sleep(5)
        attempts += 1
        
        # Check logs
        if os.path.exists("/content/server_colab.log"):
            with open("/content/server_colab.log", "r") as f:
                log_content = f.read()
            if "HTTP server error" in log_content:
                print("[ERROR] Server encountered an HTTP server error.")
                print(log_content[-1000:])
                sys.exit(1)
            if "server is listening on" in log_content or "model loaded" in log_content:
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

    # 5. Check nvidia-smi memory usage to confirm GPU allocation
    _, smi_out, _ = run_command("nvidia-smi")
    print("\n" + "="*60)
    print(" Current GPU Status:")
    print("="*60)
    print(smi_out)
    
    if "llama-server" in smi_out:
        print("\n[SUCCESS] Confirmed: llama-server is running on the GPU!")
    else:
        print("\n[WARNING] llama-server process was not explicitly found in nvidia-smi processes table. Please check if VRAM usage is non-zero.")

    print("\nSetup finished successfully. You can now run:")
    print("!cudallm optimize examples/vector_add.cu --llm-url http://127.0.0.1:8081/completion --iters 1")
    print("="*60)

if __name__ == "__main__":
    main()
