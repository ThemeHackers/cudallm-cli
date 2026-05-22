import subprocess
import threading
import time
import shlex
from typing import List, Optional, Dict, Any

try:
    import psutil
except Exception:
    psutil = None


def _monitor_process(proc: subprocess.Popen, mem_limit_mb: Optional[int], cpu_affinity: Optional[List[int]], stop_event: threading.Event) -> Dict[str, Any]:
    info = {"killed": False, "killed_reason": None}
    if psutil is None:
        return info

    try:
        p = psutil.Process(proc.pid)
        if cpu_affinity and hasattr(p, 'cpu_affinity'):
            try:
                p.cpu_affinity(cpu_affinity)
            except Exception:
                pass

        while not stop_event.is_set():
            try:
                mem = p.memory_info().rss / (1024 * 1024)
                if mem_limit_mb and mem > mem_limit_mb:
                    p.kill()
                    info['killed'] = True
                    info['killed_reason'] = 'mem_limit_exceeded'
                    break
            except psutil.NoSuchProcess:
                break
            except Exception:
                pass
            time.sleep(0.2)
    except Exception:
        pass
    return info


def run_sandboxed(cmd: List[str] or str, cwd: Optional[str] = None, timeout: int = 300, mem_limit_mb: Optional[int] = None, cpu_affinity: Optional[List[int]] = None, shell: bool = False) -> Dict[str, Any]:
    """Run a command with basic sandboxing: timeout, memory limit, cpu affinity.

    - `cmd` can be a list or a string. If string and shell=False, it will be split.
    - Returns dict with keys: rc, stdout, stderr, timed_out, killed_by_limit
    """
    if isinstance(cmd, str) and not shell:
        args = shlex.split(cmd)
    else:
        args = cmd

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, shell=shell)
    stop_event = threading.Event()
    monitor_info = {}

    monitor_thread = threading.Thread(target=lambda: None)
    if psutil is not None and (mem_limit_mb or cpu_affinity):
        monitor_thread = threading.Thread(target=lambda: _monitor_process(proc, mem_limit_mb, cpu_affinity, stop_event))
        monitor_thread.daemon = True
        monitor_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        stdout, stderr = proc.communicate()
        rc = -1
        timed_out = True

    stop_event.set()
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=1)

   
    killed_by_limit = False
    killed_reason = None
    if psutil is not None:
        try:
       
            if rc == -1 and timed_out is False:
                killed_by_limit = True
        except Exception:
            pass

    return {"rc": rc, "stdout": stdout.decode('utf-8', errors='ignore') if isinstance(stdout, bytes) else stdout, "stderr": stderr.decode('utf-8', errors='ignore') if isinstance(stderr, bytes) else stderr, "timed_out": timed_out, "killed_by_limit": killed_by_limit, "killed_reason": killed_reason}
