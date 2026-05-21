import requests
import json
import time
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.live import Live
from .terminal_manager import TerminalManager

from .network_security import build_auth_headers, validate_llm_endpoint

console = Console()

def colorize_cuda_line(line):
    
    if line.strip().startswith("//") or line.strip().startswith("/*"):
        return f"\033[90m{line}\033[0m"
        
    comment_part = ""
    if "//" in line:
        line, comment_part = line.split("//", 1)
        comment_part = f"\033[90m//{comment_part}\033[0m"
        
    if line.strip().startswith("#"):
        line = re.sub(r"(#\w+)\s+(<[^>]+>|\"[^\"]+\")", "\x1b[34m\\1\x1b[37m \x1b[32m\\2\x1b[37m", line)
        line = re.sub(r"(#\w+)", "\x1b[34m\\1\x1b[37m", line)
        return f"\x1b[37m{line}\x1b[0m" + comment_part
        
    cuda_specifiers = [
        "__global__", "__device__", "__host__", "__shared__", "__constant__", 
        "__launch_bounds__", "__forceinline__", "__restrict__"
    ]
    cpp_keywords = [
        "return", "if", "else", "for", "while", "do", "break", "continue", "extern"
    ]
    types = [
        "void", "float", "int", "char", "double", "unsigned", "bool", "size_t",
        "uchar4", "float4", "float2", "int2", "short"
    ]

  
    line = re.sub(r"\b(\w+)\s*\(", "\x00FUNC\\1\x00ENDFUNC(", line)
    

    for spec in cuda_specifiers:
        line = re.sub(rf"\b{re.escape(spec)}\b", f"\033[93m{spec}\033[37m", line)
        
   
    for kw in cpp_keywords:
        line = re.sub(rf"\b{kw}\b", f"\033[35m{kw}\033[37m", line)
        
   
    for t in types:
        line = re.sub(rf"\b{t}\b", f"\033[36m{t}\033[37m", line)
        
   
    line = re.sub(r"\b(\d+\.?\d*f?)\b", "\x1b[38;5;208m\\1\x1b[37m", line)
    
   
    line = line.replace("\x00FUNC", "\033[1;33m").replace("\x00ENDFUNC", "\033[37m")
    
    return f"\033[37m{line}\033[0m" + comment_part

class LLMClient:
    def __init__(
        self,
        url,
        max_stream_chunks=2000,
        api_key=None,
        api_key_file=None,
        verify_tls=True,
        allow_insecure_remote=False,
    ):
        self.url = url
        from urllib.parse import urlparse
        try:
            parsed = urlparse(self.url)
            if not parsed.path or parsed.path == '/':
                if "ollama" in self.url.lower() or "11434" in self.url:
                    self.url = self.url.rstrip('/') + '/api/generate'
                else:
                    self.url = self.url.rstrip('/') + '/v1/completions'
        except Exception:
            pass

        self.total_tokens = 0
        self.max_stream_chunks = max_stream_chunks
        self.request_headers = build_auth_headers(api_key=api_key, api_key_file=api_key_file)
        self.verify_tls = verify_tls
        self.endpoint_info = validate_llm_endpoint(self.url, allow_insecure_remote=allow_insecure_remote)
        self._validate_url()
        
    def _is_ollama(self):
        return "ollama" in self.url.lower() or "11434" in self.url

    def _validate_url(self):
        try:
            parsed = re.match(r'^(https?)://([^/:]+)(:\d+)?(/.*)?$', self.url)
            if not parsed:
                console.print(f"[yellow]Warning: LLM URL does not look like HTTP(S): {self.url}[/yellow]")
            elif self.endpoint_info.get("is_private"):
                console.print(f"[green]LLM endpoint classified as private network: {self.endpoint_info.get('host')}[/green]")
            elif self.endpoint_info.get("is_secure"):
                console.print(f"[green]LLM endpoint uses HTTPS: {self.url}[/green]")
        except Exception:
            console.print(f"[yellow]Warning: unable to validate LLM URL: {self.url}[/yellow]")

    def _looks_like_cuda_code(self, text):
        snippet = text.strip()
        if not snippet:
            return False


        clean_text = re.sub(r'//.*$', '', snippet, flags=re.MULTILINE)
        clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)
       
        clean_text = re.sub(r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'', '', clean_text)
        
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        if not lines:
            return False


        code_line_count = 0
        valid_endings = (';', '{', '}', ',', '\\')
        for line in lines:
            if line.startswith('#'):
                code_line_count += 1
            elif line.endswith(valid_endings):
                code_line_count += 1
           
            elif re.match(r'^(if|for|while|else|do)\b', line):
                code_line_count += 1
            
            elif line.endswith(':'):
                code_line_count += 1

        code_ratio = code_line_count / len(lines)

      
        stop_words = [
            "the", "is", "are", "was", "were", "have", "has", "had", 
            "would", "could", "should", "they", "them", "their", "you", "your", 
            "we", "our", "about", "because", "here", "there", "why", "how", 
            "what", "who", "which", "please", "okay", "sorry", "thank", "thanks"
        ]
        stop_word_pattern = r'\b(' + '|'.join(stop_words) + r')\b'
        prose_word_matches = re.findall(stop_word_pattern, clean_text, re.IGNORECASE)

        cuda_markers = [
            "__global__", "__device__", "__host__", "#include",
            "cudaMalloc", "cudaMemcpy", "dim3", "<<<", ">>>"
        ]
        has_cuda_marker = any(marker in snippet for marker in cuda_markers)
        has_function_signature = re.search(r"\b(?:void|int|float|double|__global__|__device__)\s+\w+\s*\([^)]*\)\s*{?", clean_text) is not None

      
        if code_ratio < 0.4:
            return False
        if len(prose_word_matches) > 3:
            return False

        return has_cuda_marker or has_function_signature
        
    def _extract_kernels(self, code):
        kernels = []
        for line in code.splitlines():
            if "__global__" in line:
                match = re.search(r'\b(?:void|\w+)\s+(\w+)\s*\(', line)
                if match:
                    kernels.append(match.group(1))
        seen = set()
        unique_kernels = []
        for k in kernels:
            if k not in seen:
                seen.add(k)
                unique_kernels.append(k)
        return unique_kernels

    def _summarize_diagnostics(self, diagnostics_text, max_significant_lines=10, tail_lines=4):
        lines = [line.rstrip() for line in diagnostics_text.splitlines() if line.strip()]
        if not lines:
            return "(no diagnostics available)"

        significant_patterns = re.compile(
            r"\b(error|warning|fatal|undefined|cannot|expected|not found|mismatch|failed|invalid|no such)\b",
            re.IGNORECASE,
        )

        summary_lines = []
        seen = set()
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            if significant_patterns.search(normalized) or normalized.startswith("==ERROR=="):
                if normalized not in seen:
                    seen.add(normalized)
                    summary_lines.append(normalized)
            if len(summary_lines) >= max_significant_lines:
                break

        if not summary_lines:
            summary_lines = lines[:max_significant_lines]

        if len(lines) > len(summary_lines):
            tail = lines[-tail_lines:]
            tail = [re.sub(r"\s+", " ", line).strip() for line in tail]
            condensed = summary_lines + ["...", *tail]
        else:
            condensed = summary_lines

        return "\n".join(condensed)

    def generate_code(self, prompt, max_tokens=4096, status_callback=None, early_terminate=True, prefill=False):
        start_time = time.time()

        prefilled = False
        if prefill and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "```cuda\n"
            prefilled = True

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "repeat_penalty": 1.25,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "stream": True
        }
        if self._is_ollama():
            payload = {
                "model": "codellama",
                "prompt": prompt,
                "options": {
                    "temperature": 0.2,
                    "repeat_penalty": 1.25,
                    "num_predict": max_tokens
                },
                "stream": True
            }


        attempt = 0
        max_attempts = 3
        backoff = 1.0
        resp = None
        while attempt < max_attempts:
            try:
                resp = requests.post(
                    self.url,
                    json=payload,
                    headers=self.request_headers,
                    stream=True,
                    timeout=30,
                    verify=self.verify_tls,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    console.print(f"[bold red]LLM request failed after {attempt} attempts: {e}[/bold red]")
                    return None, 0
                time.sleep(backoff)
                backoff *= 2

        if resp is None:
            console.print("[bold red]Failed to get response from LLM.[/bold red]")
            return None, 0


        result_text = "```cuda\n" if prefilled else ""
        tokens_generated = 0
        if status_callback:
            status_callback("AI Thinking & Code Generation started...")
        else:
            console.print("[bold yellow]AI Thinking & Code Generation started...[/bold yellow]")

        in_think = False
        has_printed_think_header = False
        has_printed_think_footer = False
        in_code_block = False
        has_passed_opening_line = False
        has_printed_code_header = False
        stream_buffer = ""
        think_buffer = ""
        think_live = None

        last_line = None
        repeat_count = 0
        MAX_REPEATS = 5
        last_chunk_text = None
        chunk_repeat_count = 0
        MAX_CHUNK_REPEATS = 8

        try:
            for chunk in resp.iter_lines():
                if not chunk:
                    continue
                chunk_str = chunk.decode('utf-8').strip()
                if chunk_str.startswith('data:'):
                    chunk_str = chunk_str[5:].strip()
                if not chunk_str or chunk_str == '[DONE]':
                    continue
                try:
                    res_json = json.loads(chunk_str)
                    if self._is_ollama():
                        content = res_json.get('response', '')
                        done = res_json.get('done', False)
                    else:
                        choices = res_json.get('choices', [])
                        if choices:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', choices[0].get('text', ''))
                        else:
                            content = res_json.get('content', '')
                        done = res_json.get('stop', False) or res_json.get('stop', '') == 'done' or (choices and choices[0].get('finish_reason') is not None)

                    normalized_chunk = content.strip()
                    if normalized_chunk:
                        if normalized_chunk == last_chunk_text:
                            chunk_repeat_count += 1
                        else:
                            last_chunk_text = normalized_chunk
                            chunk_repeat_count = 1
                        if chunk_repeat_count >= MAX_CHUNK_REPEATS:
                            console.print(f"\n[bold yellow][WARNING] Repetition loop detected at chunk level ({chunk_repeat_count}x). Force-stopping generation.[/bold yellow]")
                            done = True

                    result_text += content
                    tokens_generated += 1

                    if tokens_generated > self.max_stream_chunks:
                        console.print(f"\n[bold yellow][WARNING] Token limit reached ({self.max_stream_chunks}). Force-stopping generation.[/bold yellow]")
                        done = True

                    if tokens_generated % 5 == 0 and tokens_generated > 50:
                        recent_lines = [l.strip() for l in result_text.split('\n')[-12:] if l.strip()]
                        if len(recent_lines) >= 6:
                            from collections import Counter
                            counts = Counter(recent_lines)
                            most_common_line, most_common_count = counts.most_common(1)[0]
                            if most_common_count >= 4:
                                console.print(f"\n[bold yellow][WARNING] Degenerate repetition detected: '{most_common_line[:60]}' repeated {most_common_count}x. Force-stopping.[/bold yellow]")
                                done = True

                    if status_callback:
                        if "<think>" in result_text and "</think>" not in result_text:
                            current_think = result_text.split("<think>")[1].strip()
                            lines = [line.strip() for line in current_think.splitlines() if line.strip()]
                            display_think = lines[-1] if lines else "Analyzing performance..."
                            if len(display_think) > 80:
                                display_think = display_think[:77] + "..."
                            status_callback(f"AI Reasoning: [dim cyan]{display_think}[/dim cyan]")
                        elif "</think>" in result_text:
                            status_callback(f"Writing CUDA Code... (Generated {tokens_generated} tokens)")
                        else:
                            status_callback(f"AI Reasoning: [dim cyan]Analyzing kernel constraints... ({tokens_generated} tokens)[/dim cyan]")
                    else:
                        if "<think>" in result_text and "</think>" not in result_text:
                            in_think = True
                            if not has_printed_think_header:
                                think_live = Live(
                                    Panel("", title="[bold cyan]AI Reasoning Process[/bold cyan]", border_style="dim", box=TerminalManager.get_box_style(), style="dim cyan"),
                                    console=console,
                                    refresh_per_second=10,
                                    transient=False
                                )
                                think_live.start()
                                has_printed_think_header = True
                        elif "</think>" in result_text:
                            in_think = False
                            if not has_printed_think_footer:
                                if think_live:
                                    think_live.stop()
                                    think_live = None
                                has_printed_think_footer = True

                        text_no_think = result_text
                        if "</think>" in text_no_think:
                            text_no_think = text_no_think.split("</think>")[-1]
                        elif "<think>" in text_no_think:
                            text_no_think = text_no_think.split("<think>")[0]

                        num_backticks = text_no_think.count("```")

                        if num_backticks % 2 == 1:
                            if not in_code_block:
                                in_code_block = True
                                if not has_printed_think_footer:
                                    if think_live:
                                        think_live.stop()
                                        think_live = None
                                    has_printed_think_footer = True
                                if not has_printed_code_header:
                                    console.print("[bold green]Writing Optimized CUDA Code...[/bold green]\n")
                                    has_printed_code_header = True
                                has_passed_opening_line = False
                                stream_buffer = ""
                        else:
                            if in_code_block:
                                in_code_block = False
                                if early_terminate:
                                    done = True

                        if in_think:
                            clean_content = content
                            for tag in ["<think>", "</think>"]:
                                if tag in clean_content:
                                    clean_content = clean_content.replace(tag, "")
                            if clean_content:
                                think_buffer += clean_content
                                if think_live:
                                    think_live.update(
                                        Panel(think_buffer, title="[bold cyan]AI Reasoning Process[/bold cyan]", border_style="dim", box=TerminalManager.get_box_style(), style="dim cyan")
                                    )
                        elif in_code_block:
                            if not has_passed_opening_line:
                                if "\n" in content:
                                    has_passed_opening_line = True
                                    parts = content.split("\n", 1)
                                    if len(parts) > 1:
                                        stream_buffer += parts[1]
                            else:
                                clean_code_chunk = content
                                if "```" in clean_code_chunk:
                                    clean_code_chunk = clean_code_chunk.split("```")[0]
                                stream_buffer += clean_code_chunk

                                if has_passed_opening_line and stream_buffer:
                                    if "\n" in stream_buffer:
                                        lines = stream_buffer.split("\n")
                                        for line in lines[:-1]:
                                            stripped = line.strip()
                                            if stripped == last_line and stripped:
                                                repeat_count += 1
                                                if repeat_count >= MAX_REPEATS:
                                                    console.print(f"\n[bold yellow][WARNING] Repetition loop detected ({repeat_count}x). Force-stopping generation.[/bold yellow]")
                                                    done = True
                                                    break
                                                continue
                                            else:
                                                last_line = stripped
                                                repeat_count = 0
                                            sys.stdout.write(colorize_cuda_line(line) + "\n")
                                            sys.stdout.flush()
                                        stream_buffer = lines[-1]
                        else:
                            clean_content = content
                            if "```" in clean_content:
                                clean_content = clean_content.replace("```", "")
                            if clean_content:
                                sys.stdout.write(f"\033[37m{clean_content}\033[0m")
                                sys.stdout.flush()

                    if done:
                        break
                except Exception as e:
                    with open("client_err.log", "a") as f:
                        f.write(f"Err: {e}\n")

            self.total_tokens += tokens_generated
            if stream_buffer:
                sys.stdout.write(colorize_cuda_line(stream_buffer) + "\n")
                sys.stdout.flush()

            if not status_callback:
                if not has_printed_think_header and result_text:
                    if "```" in result_text:
                        console.print("\n[bold green]Writing Optimized CUDA Code...[/bold green]\n")
                    else:
                        console.print(Panel(result_text, title="[bold cyan]AI Reasoning Process[/bold cyan]", border_style="dim", box=TerminalManager.get_box_style(), style="dim cyan"))
                elif in_think and not has_printed_think_footer:
                    if think_live:
                        think_live.stop()
                        think_live = None

                console.print(f"\n[bold green]Generation complete! Total tokens: {tokens_generated} (In {time.time() - start_time:.2f}s)[/bold green]")

           
            think_match = re.search(r"<think>(.*?)</think>", result_text, re.DOTALL)
            if think_match:
                result_text = result_text.replace(think_match.group(0), "").strip()
            else:
                if "<think>" in result_text:
                    parts = result_text.split("<think>")
                    result_text = parts[0].strip()

       
            code_match = re.search(r"```(?:cuda|cpp|c)?\n(.*?)\n```", result_text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip(), time.time() - start_time

        
            code_match_open = re.search(r"```(?:cuda|cpp|c)?\n(.*)", result_text, re.DOTALL)
            if code_match_open:
                return code_match_open.group(1).strip(), time.time() - start_time

         
            if self._looks_like_cuda_code(result_text):
                return result_text.strip(), time.time() - start_time

            if not status_callback:
                console.print("[bold red][ERROR] LLM did not return CUDA code. Skipping output.[/bold red]")
            return "", time.time() - start_time
        except Exception as e:
            console.print(Panel(
                f"[bold red][ERROR] Connection Error[/bold red]\n"
                f"[yellow]Could not communicate with the local LLM server.[/yellow]\n\n"
                f"[dim]Details:[/dim] {escape(str(e))}\n\n"
                f"[italic]Please ensure that the LLM server is actively running by typing [bold cyan]cudallm serve[/bold cyan] in another terminal.[/italic]",
                title="[bold red]LLM Server Offline[/bold red]",
                border_style="red"
            ))
            return "", 0.0
        finally:
            if think_live:
                try:
                    think_live.stop()
                except Exception:
                    pass

    def create_optimization_prompt(self, code, env_info, target, best_time, flags):
        kernels = self._extract_kernels(code)
        if kernels:
            kernel_list_str = ", ".join(kernels)
            kernel_instruction = f"all kernels ({kernel_list_str})"
        else:
            kernel_instruction = "all kernels"

        system_prompt = (
            f"You are a world-class CUDA optimization expert. Optimize the user's CUDA code for high performance on {env_info['gpu_model']} "
            f"(Compute {env_info['compute_capability']}, CUDA {env_info['cuda_version']}).\n"
        )
        if best_time != float('inf'):
            system_prompt += f"Current latency: {best_time}ms. Target: {target}.\n"
        if flags:
            system_prompt += f"Compiler flags used: {' '.join(flags)}.\n"
            
        system_prompt += (
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do not provide reasoning or commentary.\n"
            "2. The final answer must be exactly one complete ```cuda ... ``` markdown block with the full optimized source.\n"
            "3. Do NOT write any introduction, explanation, apology, summary, or text outside the CUDA code block.\n"
            "4. Start the final answer by continuing the CUDA code block immediately. Never narrate before the code fence.\n"
            "5. CRITICAL: DO NOT USE ANY PLACEHOLDERS OR ABBREVIATIONS like '...', 'rest of the code', 'TODO', 'code remains unchanged', etc. You MUST output every single function in its entirety, with every single variable declaration, loop, conditional check, and math statement written out completely. If a function is not modified, you MUST copy its code verbatim from the original input. Failure to output the complete code will break compilation.\n"
            f"6. CRITICAL: You MUST preserve the exact function names, argument counts, parameter types, and overall function signatures of {kernel_instruction} exactly as provided in the input code. Do NOT alter parameter types, as they are strictly benchmarked by an external host wrapper."
        )
        
        user_prompt = f"Optimize the following CUDA code:\n\n```cuda\n{code}\n```"
        
        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def create_healing_prompt(self, code, error_log):
        kernels = self._extract_kernels(code)
        if kernels:
            kernel_list_str = ", ".join(kernels)
            kernel_instruction = f"all kernels ({kernel_list_str})"
        else:
            kernel_instruction = "all kernels"

        system_prompt = (
            "You are a world-class CUDA compiler and debugging assistant. Your task is to fix compilation errors or correctness failures "
            "in the user's CUDA code.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do not provide reasoning or commentary.\n"
            "2. The final answer must be exactly one complete ```cuda ... ``` markdown block with the full fixed source.\n"
            "3. Do NOT write any introduction, explanation, apology, summary, or text outside the CUDA code block.\n"
            "4. Start the final answer by continuing the CUDA code block immediately. Never narrate before the code fence.\n"
            "5. CRITICAL: DO NOT USE ANY PLACEHOLDERS OR ABBREVIATIONS like '...', 'rest of the code', 'TODO', 'code remains unchanged', etc. You MUST output every single function in its entirety, with every single variable declaration, loop, conditional check, and math statement written out completely. If a function is not modified, you MUST copy its code verbatim from the original input. Failure to output the complete code will break compilation.\n"
            f"6. CRITICAL: You MUST preserve the exact function names, argument counts, parameter types, and overall function signatures of {kernel_instruction} exactly as provided. Do NOT alter parameter types, change argument lists, or change variable types of the parameters, as it will break the compilation with the benchmark wrapper."
        )
        
        user_prompt = (
            f"This CUDA code failed to compile/verify:\n```cuda\n{code}\n```\n\n"
            f"Error log summary:\n{self._summarize_diagnostics(error_log)}\n\n"
            f"Raw error log:\n{error_log}\n\nPlease fix it so it compiles and is mathematically correct."
        )
        
        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def create_compile_repair_prompt(self, code, error_log, attempt_index=1, max_attempts=1):
        kernels = self._extract_kernels(code)
        if kernels:
            kernel_list_str = ", ".join(kernels)
            kernel_instruction = f"all kernels ({kernel_list_str})"
        else:
            kernel_instruction = "all kernels"

        system_prompt = (
            "You are a world-class CUDA compiler engineer. Repair the user's CUDA source so it compiles cleanly.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do not provide reasoning or commentary.\n"
            "2. The final answer must be exactly one complete ```cuda ... ``` markdown block with the full corrected source.\n"
            "3. Do NOT write any introduction, explanation, apology, summary, or text outside the CUDA code block.\n"
            "4. Prioritize the compiler diagnostics over the previous code. Fix the reported errors directly and preserve all unaffected logic.\n"
            f"5. This is repair attempt {attempt_index}/{max_attempts}. If a previous repair failed, assume the last emitted code was still invalid and re-check the exact compiler diagnostics before returning code.\n"
            "6. CRITICAL: DO NOT USE ANY PLACEHOLDERS OR ABBREVIATIONS like '...', 'rest of the code', 'TODO', 'code remains unchanged', etc. You MUST output every single function in its entirety, with every single variable declaration, loop, conditional check, and math statement written out completely. If a function is not modified, You MUST copy its code verbatim from the original input. Failure to output the complete code will break compilation.\n"
            f"7. CRITICAL: You MUST preserve the exact function names, argument counts, parameter types, and overall function signatures of {kernel_instruction} exactly as provided. Do NOT alter parameter types, change argument lists, or change variable types of the parameters, as it will break the compilation with the benchmark wrapper."
        )

        diagnostic_summary = self._summarize_diagnostics(error_log)

        user_prompt = (
            f"This CUDA code failed to compile:\n```cuda\n{code}\n```\n\n"
            f"Compiler diagnostics summary:\n{diagnostic_summary}\n\n"
            f"Raw compiler diagnostics:\n{error_log}\n\n"
            "Return a corrected full CUDA file that resolves those compile errors."
        )

        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
    def create_audit_prompt(self, code):
        system_prompt = (
            "You are a world-class CUDA architect. Audit the user's CUDA code for performance bottlenecks, Warp Divergence, "
            "Bank Conflicts, and Memory Access issues.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. First, write your detailed architectural analysis inside <think>...</think> tags. "
            "Keep this reasoning block extremely concise (maximum 150 words).\n"
            "2. Second, write your final recommendations as standard markdown text, followed by specific audited and optimized code snippets inside standard ```cuda ... ``` blocks.\n"
            "Do NOT output the entire provided code; only show the optimized code snippets for the specific sections you recommended improving.\n"
            "Keep the report focused, readable, and highly technical."
        )
        
        user_prompt = f"Audit the following CUDA code:\n\n```cuda\n{code}\n```"
        
        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def analyze_profile(self, summary_text, max_tokens=2000):
        """Send profile summary text to LLM and request a structured expert analysis and action plan."""
        system_prompt = (
            "You are a senior GPU performance engineer. Analyze the following profiling summary (NSYS timeline excerpt and NCU CSV preview).\n"
            "Provide: 1) Short diagnostic summary (one paragraph), 2) Roofline-based verdict (memory-bound or compute-bound), 3) Concrete optimization recommendations (code-level hints, NVTX marker suggestions), and 4) Suggested NCU metrics and commands for deeper analysis.\n"
            "Respond in clear sections with headings: DIAGNOSIS, ROOFLINE_VERDICT, RECOMMENDATIONS, DEEP_NCU_COMMANDS, NVTX_SUGGESTIONS. Keep it concise and technical."
        )
        user_prompt = f"Profiling summary:\n\n{summary_text}\n\nPlease analyze and provide the sections requested."

        payload = {
            "prompt": system_prompt + "\n" + user_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        if self._is_ollama():
            payload = {
                "model": "codellama",
                "prompt": system_prompt + "\n" + user_prompt,
                "options": {
                    "temperature": 0.0,
                    "num_predict": max_tokens
                },
                "stream": False
            }

        attempt = 0
        max_attempts = 3
        backoff = 1.0
        while attempt < max_attempts:
            try:
                r = requests.post(
                    self.url,
                    json=payload,
                    headers=self.request_headers,
                    timeout=60,
                    verify=self.verify_tls,
                )
                r.raise_for_status()

                try:
                    j = r.json()
                    if isinstance(j, dict):
                        if 'response' in j:
                            return j['response']
                        if 'content' in j:
                            return j['content']
                except Exception:
                    pass
                return r.text
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    console.print(f"[bold red]LLM analysis failed after {attempt} attempts: {e}[/bold red]")
                    return None
                time.sleep(backoff)
                backoff *= 2
