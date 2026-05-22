import subprocess
from typing import List, Dict, Optional
from .sandbox_security import run_sandboxed


def run_in_docker(image: str, cmd: List[str] or str, volumes: Optional[Dict[str, str]] = None, timeout: int = 600, mem_limit_mb: Optional[int] = None) -> Dict:
    base = ["docker", "run", "--rm"]
    if volumes:
        for host, cont in volumes.items():
            base += ["-v", f"{host}:{cont}"]
    base += ["--network", "none"]
    base += [image]
    if isinstance(cmd, str):
        full = base + ["/bin/sh", "-c", cmd]
    else:
        full = base + cmd
    return run_sandboxed(full, timeout=timeout, mem_limit_mb=mem_limit_mb)
