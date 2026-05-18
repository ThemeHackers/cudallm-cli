import subprocess
import shutil
import re
import os

def find_nvcc_path():

    sys_path = shutil.which('nvcc')
    if sys_path:
        return sys_path
        
 
    if os.name == 'nt':
        base_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base_dir):
            try:
                for folder in os.listdir(base_dir):
                    if folder.startswith("v"):
                           candidate = os.path.join(base_dir, folder, "bin", "nvcc.exe")
                           if os.path.exists(candidate):
                               return candidate
            except Exception:
                pass
    return None

def find_nvidia_smi_path():
  
    sys_path = shutil.which('nvidia-smi')
    if sys_path:
        return sys_path
        
   
    if os.name == 'nt':
        candidates = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return None

def find_ncu_path():
    sys_path = shutil.which('ncu')
    if sys_path:
        return sys_path
        
    if os.name == 'nt':
        base_dir = r"C:\Program Files\NVIDIA Corporation"
        if os.path.exists(base_dir):
            try:
                for folder in os.listdir(base_dir):
                    if folder.startswith("Nsight Compute"):
                        candidate = os.path.join(base_dir, folder, "target", "windows-desktop-win7-x64", "ncu.exe")
                        if os.path.exists(candidate):
                            return candidate
            except Exception:
                pass
    return None

def check_environment():
    nvcc_path = find_nvcc_path()
    nvidia_smi_path = find_nvidia_smi_path()
    ncu_path = find_ncu_path()
    
    env_info = {
        'nvcc_found': nvcc_path is not None,
        'nvcc_path': nvcc_path,
        'nvidia_smi_found': nvidia_smi_path is not None,
        'nvidia_smi_path': nvidia_smi_path,
        'ncu_found': ncu_path is not None,
        'ncu_path': ncu_path,
        'gpu_model': 'Unknown',
        'compute_capability': 'Unknown',
        'vram_total': 'Unknown',
        'cuda_version': 'Unknown'
    }
    
    if env_info['nvcc_found']:
        try:
            res = subprocess.run([env_info['nvcc_path'], '--version'], stdout=subprocess.PIPE, text=True)
            match = re.search(r'release (\d+\.\d+)', res.stdout)
            if match:
                env_info['cuda_version'] = match.group(1)
        except Exception:
            pass

    if env_info['nvidia_smi_found']:
        try:
            res = subprocess.run(
                [env_info['nvidia_smi_path'], '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader'],
                stdout=subprocess.PIPE, text=True
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split(', ')
                if len(parts) >= 3:
                    env_info['gpu_model'] = parts[0]
                    env_info['compute_capability'] = parts[1]
                    env_info['vram_total'] = parts[2]
        except Exception:
            pass
            
    return env_info
