import subprocess
import shutil
import re

def check_environment():
    env_info = {
        'nvcc_found': shutil.which('nvcc') is not None,
        'nvidia_smi_found': shutil.which('nvidia-smi') is not None,
        'ncu_found': shutil.which('ncu') is not None,
        'nvprof_found': shutil.which('nvprof') is not None,
        'gpu_model': 'Unknown',
        'compute_capability': 'Unknown',
        'vram_total': 'Unknown',
        'cuda_version': 'Unknown'
    }
    
    if env_info['nvcc_found']:
        try:
            res = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, text=True)
            match = re.search(r'release (\d+\.\d+)', res.stdout)
            if match:
                env_info['cuda_version'] = match.group(1)
        except Exception:
            pass

    if env_info['nvidia_smi_found']:
        try:
            res = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader'],
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
