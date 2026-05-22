import os
import json
from typing import List, Optional, Any
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain.agents import create_agent

from .discover import find_nvcc_path, find_nsys_path, find_ncu_path
from .profiler_tools import summarize_profile_outputs
from .sandbox_security import run_sandboxed
import shlex

@tool
def compile_cuda(source_path: str, out_path: Optional[str] = None, extra_args: Optional[List[str]] = None, timeout: int = 120, mem_limit_mb: int = 1024) -> str:
    """Compile a CUDA source file using nvcc. Args: source_path, out_path (optional), extra_args (optional). Returns JSON string with rc/stdout/stderr."""
    nvcc = find_nvcc_path()
    if not nvcc:
        return json.dumps({"error": "nvcc not found"})
    if not out_path:
        base = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.abspath(f"{base}.so")

    cmd = [nvcc, '-O3', '-shared', '-Xcompiler', '-fPIC', '-o', out_path, source_path]
    if extra_args:
        cmd[1:1] = extra_args
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    return json.dumps(res)

@tool
def profile_system(cmdline: str, output_base: Optional[str] = None, timeout: int = 900, mem_limit_mb: int = 4096) -> str:
    """Run system-wide profiling with nsys. Args: cmdline (command line string), output_base (optional). Returns JSON string with out and basename."""
    nsys = find_nsys_path()
    if not nsys:
        return json.dumps({"error": "nsys not found"})
    output_base = output_base or "nsys_agent_out"
    parts = shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
    cmd = [nsys, 'profile', '--output', output_base, '--trace', 'cuda,cudnn,nvtx'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    return json.dumps({"out": out, "basename": output_base})

@tool
def profile_kernel(cmdline: str, metrics: Optional[str] = None, timeout: int = 600, mem_limit_mb: int = 4096) -> str:
    """Run kernel-level profiling with ncu. Args: cmdline (command line string), metrics (optional comma-separated string). Returns JSON string with out, csv path, and basename."""
    ncu = find_ncu_path()
    if not ncu:
        return json.dumps({"error": "ncu not found"})
    parts = shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
    cmd = [ncu, '--csv']
    if metrics:
        cmd += ['--metrics', metrics]
    cmd += ['-o', 'ncu_agent_out'] + parts
    res = run_sandboxed(cmd, timeout=timeout, mem_limit_mb=mem_limit_mb)
    out = (res.get('stdout') or '') + (res.get('stderr') or '')
    csv_path = 'ncu_agent_out.csv'
    return json.dumps({"out": out, "csv": csv_path, "basename": 'ncu_agent_out'})

@tool
def summarize_profile(nsys_out: str, ncu_csv: str) -> str:
    """Summarize nsys/ncu outputs. Args: nsys_out (string), ncu_csv (path to CSV file). Returns human-readable summary string."""
    return summarize_profile_outputs(nsys_out, ncu_csv)

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

def get_tools() -> List[Any]:
    return [compile_cuda, profile_system, profile_kernel, summarize_profile, audit_cuda_code]

class LocalLMStudioChat(BaseChatModel):
    base_url: str
    api_key: str = "lm-studio"
    openai_tools: List[Any] = Field(default_factory=list)

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
            "model": "local-model",
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
        ai_msg = AIMessage(content=message.content or "", tool_calls=lc_tool_calls)
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

    local_model = LocalLMStudioChat(base_url=base_url, api_key=api_key)
    tools = get_tools()
    agent = create_agent(model=local_model, tools=tools, system_prompt="You are a helpful CUDA performance engineer.")
    return agent
