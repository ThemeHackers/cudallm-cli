import subprocess
import shutil
import glob
import re
import os

def _pick_first_existing(candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

def _normalize_ext(path):
    if not path:
        return path
    base, ext = os.path.splitext(path)
    return base + ext.lower()

def find_nvcc_path():

    sys_path = shutil.which('nvcc')
    if sys_path:
        return _normalize_ext(sys_path)
        
 
    if os.name == 'nt':
        cuda_envs = []
        if os.environ.get("CUDA_PATH"):
            cuda_envs.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc.exe"))
        for key, value in os.environ.items():
            if key.startswith("CUDA_PATH_V") and value:
                cuda_envs.append(os.path.join(value, "bin", "nvcc.exe"))
        found_env = _pick_first_existing(cuda_envs)
        if found_env:
            return _normalize_ext(found_env)

        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        matches = []
        for root in roots:
            pattern = os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", "v*", "bin", "nvcc.exe")
            matches.extend(glob.glob(pattern))
        if matches:
            matches.sort(reverse=True)
            return _normalize_ext(matches[0])
    return None

def find_nvidia_smi_path():
  
    sys_path = shutil.which('nvidia-smi')
    if sys_path:
        return _normalize_ext(sys_path)
        
   
    if os.name == 'nt':
        candidates = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        ]
        found = _pick_first_existing(candidates)
        if found:
            return _normalize_ext(found)
    return None

def find_ncu_path():
    sys_path = shutil.which('ncu')
    if sys_path:
        return _normalize_ext(sys_path)
        
    if os.name == 'nt':
        roots = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "NVIDIA Corporation"),
        ]
        for root in roots:
            if not os.path.exists(root):
                continue
            candidates = []
            candidates.extend(glob.glob(os.path.join(root, "Nsight Compute*", "target", "**", "ncu.exe"), recursive=True))
            candidates.extend(glob.glob(os.path.join(root, "**", "ncu.exe"), recursive=True))
            if candidates:
                return _normalize_ext(candidates[0])
    return None

def find_nsys_path():
    sys_path = shutil.which('nsys')
    if sys_path:
        return _normalize_ext(sys_path)

    if os.name == 'nt':
        roots = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "NVIDIA Corporation"),
        ]
        for root in roots:
            if not os.path.exists(root):
                continue
            candidates = []
            candidates.extend(glob.glob(os.path.join(root, "Nsight Systems*", "target*", "**", "nsys.exe"), recursive=True))
            candidates.extend(glob.glob(os.path.join(root, "**", "nsys.exe"), recursive=True))
            if candidates:
                return _normalize_ext(candidates[0])
    return None

def find_llm_server_path(project_dir=None):
    for tool_name in ('llm-server', 'llama-server'):
        sys_path = shutil.which(tool_name)
        if sys_path:
            return _normalize_ext(sys_path)

    if project_dir and os.path.exists(project_dir):
        ext = ".exe" if os.name == 'nt' else ""
        patterns = [
            os.path.join(project_dir, "llm-b*", f"llm-server{ext}"),
            os.path.join(project_dir, "llama-b*", f"llama-server{ext}"),
            os.path.join(project_dir, "**", f"llm-server{ext}"),
            os.path.join(project_dir, "**", f"llama-server{ext}"),
        ]
        for pattern in patterns:
            for candidate in glob.glob(pattern, recursive=True):
                if os.path.exists(candidate):
                    return _normalize_ext(candidate)

    return None

def discover_tool_paths(project_dir=None):
    return {
        'nvcc_path': find_nvcc_path(),
        'nvidia_smi_path': find_nvidia_smi_path(),
        'ncu_path': find_ncu_path(),
        'nsys_path': find_nsys_path(),
        'llm_server_path': find_llm_server_path(project_dir),
    }

def check_environment():
    nvcc_path = find_nvcc_path()
    nvidia_smi_path = find_nvidia_smi_path()
    ncu_path = find_ncu_path()
    nsys_path = find_nsys_path()
    
    env_info = {
        'nvcc_found': nvcc_path is not None,
        'nvcc_path': nvcc_path,
        'nvidia_smi_found': nvidia_smi_path is not None,
        'nvidia_smi_path': nvidia_smi_path,
        'ncu_found': ncu_path is not None,
        'ncu_path': ncu_path,
        'nsys_found': nsys_path is not None,
        'nsys_path': nsys_path,
        'gpu_model': 'Unknown',
        'compute_capability': 'Unknown',
        'vram_total': 'Unknown',
        'cuda_version': 'Unknown'
    }
    
    if env_info['nvcc_found']:
        try:
            res = subprocess.run([env_info['nvcc_path'], '--version'], stdout=subprocess.PIPE, text=True, timeout=10)
            match = re.search(r'release (\d+\.\d+)', res.stdout)
            if match:
                env_info['cuda_version'] = match.group(1)
        except Exception:
            pass

    if env_info['nvidia_smi_found']:
        try:
            res = subprocess.run(
                [env_info['nvidia_smi_path'], '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader'],
                stdout=subprocess.PIPE, text=True, timeout=10
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
