import subprocess
import os
import re
import shutil

class CUDASandbox:
    def __init__(self, file_path, flags=None):
        self.file_path = file_path
        self.exe_path = "./temp_cuda_kernel.exe" if os.name == 'nt' else "./temp_cuda_kernel.out"
        self.flags = flags or []

    def compile(self):
        cmd = ["nvcc", self.file_path, "-o", self.exe_path] + self.flags
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return {"success": result.returncode == 0, "error_log": result.stderr}
        except FileNotFoundError:
            return {"success": False, "error_log": "nvcc not found"}

    def profile_latency(self):
        if not os.path.exists(self.exe_path):
            return {"latency": float('inf'), "raw_output": ""}
            
        profiler = "ncu" if shutil.which("ncu") else ("nvprof" if shutil.which("nvprof") else None)
        if not profiler:
            try:
                import time
                start = time.time()
                subprocess.run([self.exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return {"latency": (time.time() - start) * 1000.0, "raw_output": "Fallback timer"}
            except Exception:
                return {"latency": 99999.0, "raw_output": ""}

        cmd = [profiler, self.exe_path]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout + result.stderr
            match = re.search(r'([\d\.]+)\s*(ms|us)', output)
            if match:
                val = float(match.group(1))
                latency = val / 1000.0 if match.group(2) == "us" else val
                return {"latency": latency, "raw_output": output[:500]}
            return {"latency": 99999.0, "raw_output": output[:500]}
        except Exception as e:
            return {"latency": 99999.0, "raw_output": str(e)}
