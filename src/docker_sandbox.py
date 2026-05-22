import subprocess
from typing import List, Dict, Optional
from .sandbox_security import run_sandboxed


def run_in_docker(image: str, cmd: List[str] or str, volumes: Optional[Dict[str, str]] = None, timeout: int = 600, mem_limit_mb: Optional[int] = None) -> Dict:
    base_gpu = ["docker", "run", "--rm", "--gpus", "all"]
    if volumes:
        for host, cont in volumes.items():
            base_gpu += ["-v", f"{host}:{cont}"]
    base_gpu += ["--network", "none"]
    base_gpu += [image]
    if isinstance(cmd, str):
        full_gpu = base_gpu + ["/bin/sh", "-c", cmd]
    else:
        full_gpu = base_gpu + cmd
        
    result = run_sandboxed(full_gpu, timeout=timeout, mem_limit_mb=mem_limit_mb)
    
    gpu_err_keywords = [
        "could not select device driver",
        "unknown flag: --gpus",
        "unknown or invalid runtime",
        "nvidia-container-cli"
    ]
    has_gpu_err = any(kw in result.get("stderr", "").lower() for kw in gpu_err_keywords)
    
    if result.get("rc") != 0 and has_gpu_err:
        base_no_gpu = ["docker", "run", "--rm"]
        if volumes:
            for host, cont in volumes.items():
                base_no_gpu += ["-v", f"{host}:{cont}"]
        base_no_gpu += ["--network", "none"]
        base_no_gpu += [image]
        if isinstance(cmd, str):
            full_no_gpu = base_no_gpu + ["/bin/sh", "-c", cmd]
        else:
            full_no_gpu = base_no_gpu + cmd
        return run_sandboxed(full_no_gpu, timeout=timeout, mem_limit_mb=mem_limit_mb)
        
    return result
