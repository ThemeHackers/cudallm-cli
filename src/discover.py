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


def _version_sort_key(path):
    parts = re.findall(r"\d+", path)
    if not parts:
        return [0]
    return [int(part) for part in parts]

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
            def _cuda_version_key(path):
                m = re.search(r'[\\/]CUDA[\\/]v([\d\.]+)', path, re.IGNORECASE)
                if m:
                    try:
                        return [int(x) for x in m.group(1).split('.')]
                    except ValueError:
                        pass
                return [0]
            matches.sort(key=_cuda_version_key, reverse=True)
            return _normalize_ext(matches[0])
    else:
        candidates = [
            "/usr/local/cuda/bin/nvcc",
            "/usr/bin/nvcc",
        ]
        candidates.extend(glob.glob("/usr/local/cuda-*/bin/nvcc"))
        found = _pick_first_existing(candidates)
        if found:
            return _normalize_ext(found)
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
    else:
        candidates = [
            "/usr/bin/nvidia-smi",
            "/usr/local/cuda/bin/nvidia-smi",
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
            nsight_dirs = glob.glob(os.path.join(root, "Nsight Compute *"))
            if nsight_dirs:
                nsight_dirs.sort(reverse=True)
                for nsight_dir in nsight_dirs:
                    target_dirs = glob.glob(os.path.join(nsight_dir, "target", "*"))
                    for target_dir in target_dirs:
                        candidate = os.path.join(target_dir, "ncu.exe")
                        if os.path.exists(candidate):
                            return _normalize_ext(candidate)
            candidates = []
            candidates.extend(glob.glob(os.path.join(root, "Nsight Compute*", "target", "**", "ncu.exe"), recursive=True))
            candidates.extend(glob.glob(os.path.join(root, "**", "ncu.exe"), recursive=True))
            if candidates:
                candidates.sort(key=_version_sort_key, reverse=True)
                return _normalize_ext(candidates[0])
    else:
        roots = ["/usr/local", "/opt/nvidia"]
        candidates = []
        for root in roots:
            if os.path.exists(root):
                candidates.extend(glob.glob(os.path.join(root, "cuda*", "NsightCompute*", "target", "*", "ncu")))
                candidates.extend(glob.glob(os.path.join(root, "NVIDIA-Nsight-Compute*", "target", "*", "ncu")))
                candidates.extend(glob.glob(os.path.join(root, "nsight-compute*", "target", "*", "ncu")))
        candidates.extend([
            "/usr/local/cuda/bin/ncu",
            "/usr/bin/ncu"
        ])
        found = _pick_first_existing(candidates)
        if found:
            return _normalize_ext(found)
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
            nsight_dirs = glob.glob(os.path.join(root, "Nsight Systems *"))
            if nsight_dirs:
                nsight_dirs.sort(reverse=True)
                for nsight_dir in nsight_dirs:
                    target_dirs = glob.glob(os.path.join(nsight_dir, "target", "*"))
                    for target_dir in target_dirs:
                        candidate = os.path.join(target_dir, "nsys.exe")
                        if os.path.exists(candidate):
                            return _normalize_ext(candidate)
            candidates = []
            candidates.extend(glob.glob(os.path.join(root, "Nsight Systems*", "target*", "**", "nsys.exe"), recursive=True))
            candidates.extend(glob.glob(os.path.join(root, "**", "nsys.exe"), recursive=True))
            if candidates:
                candidates.sort(key=_version_sort_key, reverse=True)
                return _normalize_ext(candidates[0])
    else:
        roots = ["/usr/local", "/opt/nvidia"]
        candidates = []
        for root in roots:
            if os.path.exists(root):
                candidates.extend(glob.glob(os.path.join(root, "cuda*", "nsight-systems*", "bin", "nsys")))
                candidates.extend(glob.glob(os.path.join(root, "nsight-systems*", "bin", "nsys")))
                candidates.extend(glob.glob(os.path.join(root, "nsys", "bin", "nsys")))
        candidates.extend([
            "/usr/local/cuda/bin/nsys",
            "/usr/bin/nsys"
        ])
        found = _pick_first_existing(candidates)
        if found:
            return _normalize_ext(found)
    return None

def discover_tool_paths(project_dir=None):
    return {
        'nvcc_path': find_nvcc_path(),
        'nvidia_smi_path': find_nvidia_smi_path(),
        'ncu_path': find_ncu_path(),
        'nsys_path': find_nsys_path(),
    }

def find_cmake_path():
    """Locate cmake binary for source builds."""
    sys_path = shutil.which('cmake')
    if sys_path:
        return sys_path
    candidates = [
        "/usr/bin/cmake",
        "/usr/local/bin/cmake",
        "/snap/bin/cmake",
    ]
    return _pick_first_existing(candidates)


def detect_cuda_version_from_toolkit():
    """
    Parse CUDA version from the toolkit's version.txt or version.json
    without requiring nvidia-smi.
    """
    search_roots = []
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        search_roots.append(cuda_path)
    search_roots.extend(glob.glob("/usr/local/cuda-*"))
    search_roots.append("/usr/local/cuda")

    for root in search_roots:
        version_file = os.path.join(root, "version.txt")
        if os.path.isfile(version_file):
            try:
                with open(version_file) as f:
                    match = re.search(r'(\d+\.\d+)', f.read())
                    if match:
                        return match.group(1)
            except Exception:
                pass
        version_json = os.path.join(root, "version.json")
        if os.path.isfile(version_json):
            try:
                import json
                with open(version_json) as f:
                    data = json.load(f)
                    cuda_info = data.get("cuda", {})
                    ver = cuda_info.get("version") or cuda_info.get("name", "")
                    match = re.search(r'(\d+\.\d+)', str(ver))
                    if match:
                        return match.group(1)
            except Exception:
                pass
    return None


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

    if env_info['cuda_version'] == 'Unknown':
        toolkit_ver = detect_cuda_version_from_toolkit()
        if toolkit_ver:
            env_info['cuda_version'] = toolkit_ver

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
