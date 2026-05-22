import os
import json
import subprocess
import tempfile
from typing import Callable, List, Dict, Optional

from .discover import find_nvcc_path, find_nsys_path, find_ncu_path
from .profiler_tools import parse_ncu_csv_for_hotspot, summarize_profile_outputs
from .sandbox_security import run_sandboxed
import shlex


def _run_cmd(cmd: List[str], timeout: int = 300) -> Dict:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return {"rc": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "stdout": "", "stderr": "timeout"}


def compile_cuda(source_path: str, out_path: Optional[str] = None, extra_args: Optional[List[str]] = None, timeout: int = 120, mem_limit_mb: int = 1024) -> Dict:
    nvcc = find_nvcc_path()
    if not nvcc:
        return {"error": "nvcc not found"}
    if not out_path:
        base = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.abspath(f"{base}.so")

    cmd = [nvcc, '-O3', '-shared', '-Xcompiler', '-fPIC', '-o', out_path, source_path]
    if extra_args:
        cmd[1:1] = extra_args
    return run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)


def profile_with_nsys(cmdline: str, output_base: Optional[str] = None, timeout: int = 900, mem_limit_mb: int = 4096) -> Dict:
    nsys = find_nsys_path()
    if not nsys:
        return {"error": "nsys not found"}
    output_base = output_base or "nsys_agent_out"
    parts = shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
    cmd = [nsys, 'profile', '--output', output_base, '--trace', 'cuda,cudnn,nvtx'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    return {"out": out, "basename": output_base}


def profile_with_ncu(cmdline: str, metrics: Optional[str] = None, timeout: int = 600, mem_limit_mb: int = 4096) -> Dict:
    ncu = find_ncu_path()
    if not ncu:
        return {"error": "ncu not found"}
    parts = shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
    cmd = [ncu, '--csv']
    if metrics:
        cmd += ['--metrics', metrics]
    cmd += ['-o', 'ncu_agent_out'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    csv_path = 'ncu_agent_out.csv'
    return {"out": out, "csv": csv_path, "basename": 'ncu_agent_out'}


class LangChainTool:
    def __init__(self, name: str, description: str, func: Callable[..., Dict]):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args, **kwargs) -> str:
        try:
            res = self.func(*args, **kwargs)
            return json.dumps(res, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})


def get_tools() -> List[LangChainTool]:
    return [
        LangChainTool(
            "compile_cuda",
            "Compile a CUDA source file using nvcc. Args: source_path, out_path (optional). Returns JSON with rc/stdout/stderr.",
            compile_cuda,
        ),
        LangChainTool(
            "profile_system",
            "Run system-wide profiling with nsys. Args: command_line, output_base (optional). Returns parser output.",
            profile_with_nsys,
        ),
        LangChainTool(
            "profile_kernel",
            "Run kernel-level profiling with ncu. Args: command_line, metrics (optional). Returns CSV path and raw output.",
            profile_with_ncu,
        ),
        LangChainTool(
            "summarize_profile",
            "Summarize nsys/ncu outputs. Args: nsys_out (string), ncu_csv_path (path). Returns human-readable summary.",
            lambda nsys_out, ncu_csv: summarize_profile_outputs(nsys_out, ncu_csv),
        ),
    ]


def create_langchain_agent_example(llm=None):
    try:
        import langchain
        from langchain.agents import initialize_agent
        from langchain.tools import Tool
        from langchain.agents import AgentType
    except Exception:
        raise RuntimeError("langchain is not installed or not importable")

    tools = []
    for t in get_tools():
        def _make_callable(func):
            return lambda arg: func(*json.loads(arg) if arg else [])

        tools.append(Tool(name=t.name, func=_make_callable(t.func), description=t.description))

    if llm is None:
        try:
            from langchain.llms import OpenAI
            llm = OpenAI()
        except Exception:
            raise RuntimeError("No LLM provided and OpenAI LLM could not be instantiated")

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent


def create_agent_with_llmclient(llm_client):
    try:
        from langchain.agents import initialize_agent
        from langchain.tools import Tool
        from langchain.agents import AgentType
        from langchain.llms.base import LLM
    except Exception:
        raise RuntimeError("langchain is not installed or not importable")

    class LocalLLM(LLM):
        def _call(self, prompt: str, stop=None) -> str:
            code, _ = llm_client.generate_code(prompt)
            return code or ""

        async def _acall(self, prompt: str, stop=None) -> str:
            return self._call(prompt, stop=stop)

        @property
        def _identifying_params(self):
            return {"implementation": "LLMClientLocal"}

    tools = []
    for t in get_tools():
        def _make_callable(func):
            return lambda arg: func(*json.loads(arg) if arg else [])
        tools.append(Tool(name=t.name, func=_make_callable(t.func), description=t.description))

    local_llm = LocalLLM()
    agent = initialize_agent(tools, local_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent
