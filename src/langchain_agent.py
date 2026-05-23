import os
import sys
import json
from typing import List, Optional, Any
from pydantic import Field
from rich.console import Console

console = Console()
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain.agents import create_agent

from .discover import find_nvcc_path, find_nsys_path, find_ncu_path, find_ncu_sections_path
from .profiler_tools import summarize_profile_outputs
from .sandbox_security import run_sandboxed
from .cli import load_config, locate_and_setup_msvc
import shlex

@tool
def compile_cuda(source_path: str, out_path: Optional[str] = None, extra_args: Optional[List[str]] = None, timeout: int = 120, mem_limit_mb: int = 1024) -> str:
    """Compile a CUDA source file using nvcc. Args: source_path (path to .cu file), out_path (optional output path, must end with .exe for Windows executables), extra_args (optional compiler flags). Returns JSON string with rc/stdout/stderr and output_path.
    IMPORTANT: For Windows executables, out_path MUST end with '.exe'. Check the 'output_path' field in the result to know where the file was created. Use find_file if you can't locate it."""
    locate_and_setup_msvc()
    nvcc = find_nvcc_path()
    if not nvcc:
        return json.dumps({"error": "nvcc not found"})

    is_shared = True
    if out_path and out_path.endswith((".exe", ".bin")):
        is_shared = False
    if extra_args and any(arg in extra_args for arg in ("-shared", "--shared")):
        is_shared = True

    if not out_path:
        base = os.path.splitext(os.path.basename(source_path))[0]
        ext = ".so" if sys.platform != "win32" else ".dll"
        out_path = os.path.abspath(f"{base}{ext}")
    else:
        out_path = os.path.abspath(out_path)

    cmd = [nvcc, '-O3', '-o', out_path, source_path]
    if is_shared:
        if sys.platform == "win32":
            cmd[2:2] = ['-shared']
        else:
            cmd[2:2] = ['-shared', '-Xcompiler', '-fPIC']

    if extra_args:
        cmd[1:1] = extra_args

    if sys.platform == "win32" and "-allow-unsupported-compiler" not in cmd:
        cmd.append("-allow-unsupported-compiler")

    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    res["output_path"] = out_path
    return json.dumps(res)

@tool
def profile_system(cmdline: str, output_base: Optional[str] = None, timeout: int = 900, mem_limit_mb: int = 4096) -> str:
    """Run system-wide profiling with nsys. Args: cmdline (executable name and arguments, e.g., 'program.exe arg1 arg2'), output_base (optional). Returns JSON string with out and basename.
    IMPORTANT: Do NOT include nsys flags in cmdline. The tool handles nsys arguments automatically. Only provide the executable and its arguments."""
    nsys = find_nsys_path()
    if not nsys:
        return json.dumps({"error": "nsys not found"})
    output_base = output_base or "nsys_agent_out"
    parts = shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
    
    # Filter out nsys flags from user input to avoid conflicts
    _skip_flags = {'--trace', '--output', '-o', '--duration', '-t', '--delay', '-d', 
                   '--force-overwrite', '-f', '--capture-range', '--capture-range-end',
                   '--capture-range-type', '--cudabacktrace', '--env', '--inherit-env',
                   '--kill', '--profile-while-disconnected', '--stop', '--target-pids',
                   '--target-processes', '--wait', '--wd'}
    _skip_next = False
    clean_parts = []
    for p in parts:
        if _skip_next:
            _skip_next = False
            continue
        if p in _skip_flags:
            if p in ('--output', '-o', '--duration', '-t', '--delay', '-d', '--capture-range',
                     '--capture-range-end', '--capture-range-type', '--env', '--inherit-env',
                     '--target-pids', '--target-processes', '--wd'):
                _skip_next = True
            continue
        clean_parts.append(p)
    parts = clean_parts
    
    cmd = [nsys, 'profile', '--output', output_base, '--trace', 'cuda,nvtx'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    return json.dumps({"out": out, "basename": output_base})

@tool
def profile_kernel(cmdline: str, metrics: Optional[str] = None, timeout: int = 600, mem_limit_mb: int = 4096) -> str:
    """Run kernel-level profiling with ncu. Args: cmdline (executable name and arguments, e.g., 'program.exe arg1 arg2'), metrics (optional comma-separated metric names, leave empty for defaults). Returns JSON string with out, csv path, and basename.
    IMPORTANT: If the executable is not found, use find_file to locate it first, then use the absolute path. Do NOT include ncu flags in cmdline."""
    ncu = find_ncu_path()
    if not ncu:
        return json.dumps({"error": "ncu not found"})

  
    if sys.platform == "win32" and isinstance(cmdline, str):
    
        import re
        parts = re.findall(r'[^"\s]+|"[^"]+"', cmdline)
        parts = [p.strip('"') for p in parts]
    else:
        parts = shlex.split(cmdline) if isinstance(cmdline, str) else list(cmdline)

    _skip_flags = {'-f', '--force-overwrite', '--csv', '-o', '--output',
                   '--section-folder', '--set', '--metrics'}
    _skip_next = False
    clean_parts = []
    for p in parts:
        if _skip_next:
            _skip_next = False
            continue
        if p in _skip_flags:
            if p in ('-o', '--output', '--section-folder', '--set', '--metrics'):
                _skip_next = True
            continue
        clean_parts.append(p)
    parts = clean_parts
   
    if parts:
       
        parts[0] = parts[0].lstrip('./').lstrip('.\\')
     
        if parts[0] and not os.path.isabs(parts[0]):
            if os.path.exists(parts[0]):
                parts[0] = os.path.abspath(parts[0])
        
       
        if parts[0] and not os.path.exists(parts[0]):
            return json.dumps({
                "error": f"Executable '{parts[0]}' not found. Please compile it first using compile_cuda, or use find_file to locate it.",
                "suggestion": "Run compile_cuda with out_path ending in '.exe' before profiling."
            })

    BASELINE_METRICS = (
        "gpu__time_duration.sum,"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
        "sm__warps_active.avg.pct_of_peak_sustained_active,"
        "l1tex__t_bytes.sum,"
        "l2__t_bytes.sum"
    )

    if metrics:
        valid_list = [m.strip() for m in metrics.split(',') if '__' in m]
        effective_metrics = ','.join(valid_list) if valid_list else BASELINE_METRICS
    else:
        effective_metrics = BASELINE_METRICS

    cmd = [ncu, '--csv', '--metrics', effective_metrics, '-f', '-o', 'ncu_agent_out'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    csv_path = 'ncu_agent_out.csv'
    return json.dumps({"out": out, "csv": csv_path, "basename": 'ncu_agent_out'})

@tool
def summarize_profile(nsys_out: str, ncu_csv: Optional[str] = None) -> str:
    """Summarize nsys/ncu outputs. Args: nsys_out (string content of nsys output, can be empty), ncu_csv (optional path to CSV file). Returns human-readable summary string.
    IMPORTANT: ncu_csv is optional. If only nsys data is available, summary will be based on nsys only. If both are empty, returns a message indicating no data available."""
    if not nsys_out and not ncu_csv:
        return json.dumps({"error": "No profiling data available. Please run profile_system or profile_kernel first."})
    try:
        return summarize_profile_outputs(nsys_out, ncu_csv)
    except Exception as e:
        return json.dumps({"error": f"Failed to summarize profile data: {str(e)}"})

@tool
def audit_cuda_code(source_path: str) -> str:
    """Statically audit a CUDA source file for performance issues and safety vulnerabilities. Args: source_path. Returns an audit report string."""
    import re
    if not os.path.exists(source_path):
        return json.dumps({"error": f"File {source_path} does not exist"})
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return json.dumps({"error": str(e)})

    issues = []
    if "cudaDeviceSynchronize" not in code and ("cudaMemcpy" not in code or "cudaMemcpyDeviceToHost" not in code):
        issues.append("- WARNING: Missing device synchronization or memory copies to host. Kernel results might not be flushed or validated.")
    if "__shared__" in code and "__syncthreads()" not in code:
        issues.append("- WARNING: Shared memory is declared but __syncthreads() is missing. This can cause data hazards and race conditions.")
    if re.search(r'if\s*\(\s*threadIdx\.x\s*[<>!=]=?', code):
        issues.append("- INFO: Thread conditional branching found. Review for potential warp divergence.")
    if "for " in code and "#pragma unroll" not in code:
        issues.append("- INFO: For loop found without '#pragma unroll'. Consider loop unrolling if loop boundaries are known.")
    if "__shared__" in code and "[threadIdx.x]" in code and "stride" not in code:
        issues.append("- INFO: Direct threadIdx.x indexing into shared memory. Review to prevent bank conflicts.")

    if not issues:
        return json.dumps({"status": "Clean", "message": "No obvious static code optimization or safety issues found."})
    return json.dumps({"status": "Issues Found", "report": "\n".join(issues)})

@tool
def get_cwd() -> str:
    """Get current working directory. Returns JSON string with current directory path."""
    return json.dumps({"cwd": os.getcwd()})

@tool
def resolve_path(path: str) -> str:
    """Convert a relative path to absolute path. Args: path (relative or absolute path). Returns JSON string with absolute path."""
    try:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        return json.dumps({"original": path, "absolute": abs_path, "exists": exists})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory. Args: directory (path to list, default '.'), pattern (glob pattern, default '*'). Returns JSON string with file list.
    Use this to see what files are available after compilation."""
    import glob
    try:
        if not os.path.isdir(directory):
            return json.dumps({"error": f"Directory {directory} does not exist"})
        files = glob.glob(os.path.join(directory, pattern))
        rel_files = [os.path.relpath(f, directory) for f in files]
        return json.dumps({"directory": directory, "pattern": pattern, "files": rel_files})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def find_file(name: str, directory: str = ".") -> str:
    """Find a file by name recursively. Args: name (filename to find, e.g., 'vector_add.exe'), directory (search root, default '.'). Returns JSON string with absolute paths if found.
    Use this when profile_kernel reports 'file not found' or after compilation to locate the executable."""
    import glob
    try:
        matches = glob.glob(os.path.join(directory, "**", name), recursive=True)
        if matches:
            abs_matches = [os.path.abspath(m) for m in matches]
            return json.dumps({"found": True, "paths": abs_matches})
        return json.dumps({"found": False, "error": f"File {name} not found in {directory}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def optimize_cuda(source_path: str, out_path: Optional[str] = None, iters: int = 3, target: str = "latency") -> str:
    """Optimize a CUDA kernel using the iterative self-healing cudallm optimizer (powered by the specialized cudaLLM model).
    Args: source_path (path to .cu file), out_path (optional output path), iters (number of optimization loops, default 3), target (default 'latency').
    Returns a JSON string with the results of the optimization run."""
    import sys
    
    config = load_config()
    optimizer_url = config.get("llm_url")
    
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    cmd = [sys.executable, "-m", "src.cli", "optimize", source_path, "--iters", str(iters), "--target", target]
    if out_path:
        cmd += ["-o", out_path]
    if optimizer_url:
        cmd += ["--llm-url", optimizer_url]
        
    res = run_sandboxed(cmd, cwd=project_root, timeout=iters * 300, mem_limit_mb=2048)
    return json.dumps(res)

def get_tools() -> List[Any]:
    return [compile_cuda, profile_system, profile_kernel, summarize_profile, audit_cuda_code, optimize_cuda, list_files, find_file, get_cwd, resolve_path]

class LocalLMStudioChat(BaseChatModel):
    base_url: str
    api_key: str = "lm-studio"
    openai_tools: List[Any] = Field(default_factory=list)
    model_name: Optional[str] = Field(default=None)

    @property
    def _llm_type(self) -> str:
        return "local_lm_studio_chat"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        self.openai_tools = []
        for t in tools:
            schema = t.args_schema.schema() if hasattr(t, "args_schema") and t.args_schema else {"type": "object", "properties": {}}
            self.openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema
                }
            })
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from openai import OpenAI
        openai_messages = []
        for m in messages:
            role = "user"
            if m.type == "human":
                role = "user"
            elif m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            elif m.type == "tool":
                role = "tool"

            msg_dict = {"role": role, "content": m.content}
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for tc in m.tool_calls
                ]
            if role == "tool" and hasattr(m, "tool_call_id"):
                msg_dict["tool_call_id"] = m.tool_call_id

            openai_messages.append(msg_dict)

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        api_kwargs = {
            "model": self.model_name or "local-model",
            "messages": openai_messages,
        }
        if self.openai_tools:
            api_kwargs["tools"] = self.openai_tools

        response = client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        message = choice.message
        lc_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                lc_tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "id": tc.id,
                    "type": "tool_call"
                })
        usage_metadata = None
        if hasattr(response, "usage") and response.usage:
            usage_metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        ai_msg = AIMessage(
            content=message.content or "", 
            tool_calls=lc_tool_calls,
            usage_metadata=usage_metadata
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

def create_langchain_agent_example(llm=None):
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o")
        except Exception:
            raise RuntimeError("langchain_openai is not installed, and no custom LLM model was provided.")

    tools = get_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful CUDA performance engineer.")
    return agent

def create_agent_with_llmclient(llm_client):
    base_url = llm_client.url
    if "/v1/completions" in base_url:
        base_url = base_url.replace("/v1/completions", "/v1")
    elif "/v1/chat/completions" in base_url:
        base_url = base_url.replace("/v1/chat/completions", "/v1")
    elif base_url.endswith("/v1"):
        pass
    else:
        base_url = base_url.rstrip("/") + "/v1"

    api_key = "lm-studio"
    auth = llm_client.request_headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        api_key = auth.split(" ")[1]

    config = load_config()
    agent_model = config.get("agent_llm_model_name")

    if not agent_model:
        try:
            import requests
            models_url = base_url.rstrip("/") + "/models"
            headers = {}
            if api_key and api_key != "lm-studio":
                headers["Authorization"] = f"Bearer {api_key}"
            r = requests.get(models_url, headers=headers, timeout=3)
            if r.status_code == 200:
                models = r.json().get("data", [])
                non_embed_models = [m.get("id") for m in models if "embed" not in m.get("id", "").lower()]
                if non_embed_models:
                    agent_models = [m for m in non_embed_models if any(k in m.lower() for k in ("qwen", "llama", "instruct"))]
                    if agent_models:
                        agent_model = agent_models[0]
                    else:
                        agent_model = non_embed_models[0]
        except Exception:
            pass

    local_model = LocalLMStudioChat(base_url=base_url, api_key=api_key, model_name=agent_model)
    console.print(f"[bold blue][INFO] Agent LLM Endpoint:[/bold blue] {base_url} | [bold]Model:[/bold] {agent_model or 'local-model'}")
    tools = get_tools()
    system_prompt = (
        "CUDA engineer on Windows. Rules:\n"
        "1. Exe must end with '.exe', use bare name (no './').\n"
        "2. compile_cuda needs out_path='name.exe' for runnable exe. Check output_path in result.\n"
        "3. profile_kernel: only exe name + args, no ncu flags. Use find_file if exe not found.\n"
        "4. Don't invent metrics, leave empty for defaults.\n"
        "5. Use get_cwd to know current directory, resolve_path to convert relative to absolute.\n"
        "6. Use list_files/find_file to locate executables after compilation.\n"
        "7. CRITICAL: Call tools IMMEDIATELY when needed. Don't say 'I will call X' - just call it.\n"
        "8. Don't re-profile if profiling already done. Check existing files first.\n"
        "9. For analysis: use summarize_profile with nsys_out/ncu_csv data. Don't just describe what you'll do.\n"
        "10. profile_system: only exe name + args, no nsys flags. Tool handles flags automatically."
    )
    agent = create_agent(model=local_model, tools=tools, system_prompt=system_prompt)
    return agent
