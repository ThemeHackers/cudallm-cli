import requests
import json
import time
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

console = Console()

def colorize_cuda_line(line):
    # Base color is light gray/white: \033[37m
    # CUDA Specifiers: Yellow/Gold \033[93m
    # C++ Keywords: Magenta \033[35m
    # Types: Cyan \033[36m
    # Function names: Orange/Gold \033[1;33m
    # Comments: Dim gray \033[90m
    # Numbers: Light orange \033[38;5;208m
    
    if line.strip().startswith("//") or line.strip().startswith("/*"):
        return f"\033[90m{line}\033[0m"
        
    comment_part = ""
    if "//" in line:
        line, comment_part = line.split("//", 1)
        comment_part = f"\033[90m//{comment_part}\033[0m"
        
    if line.strip().startswith("#"):
      
        line = re.sub(r"(#\w+)\s+(<[^>]+>|\"[^\"]+\")", r"\033[34m\1\033[37m \033[32m\2\033[37m", line)
        line = re.sub(r"(#\w+)", r"\033[34m\1\033[37m", line)
        return f"\033[37m{line}\033[0m" + comment_part
        
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

  
    line = re.sub(r"\b(\w+)\s*\(", r"\x00FUNC\1\x00ENDFUNC(", line)
    

    for spec in cuda_specifiers:
        line = re.sub(rf"\b{re.escape(spec)}\b", f"\033[93m{spec}\033[37m", line)
        
   
    for kw in cpp_keywords:
        line = re.sub(rf"\b{kw}\b", f"\033[35m{kw}\033[37m", line)
        
   
    for t in types:
        line = re.sub(rf"\b{t}\b", f"\033[36m{t}\033[37m", line)
        
   
    line = re.sub(r"\b(\d+\.?\d*f?)\b", r"\033[38;5;208m\1\033[37m", line)
    
   
    line = line.replace("\x00FUNC", "\033[1;33m").replace("\x00ENDFUNC", "\033[37m")
    
    return f"\033[37m{line}\033[0m" + comment_part

class LLMClient:
    def __init__(self, url, max_stream_chunks=2000):
        self.url = url
        self.total_tokens = 0
        self.max_stream_chunks = max_stream_chunks
        
    def _is_ollama(self):
        return "ollama" in self.url.lower() or "11434" in self.url

    def _looks_like_cuda_code(self, text):
        snippet = text.strip()
        if not snippet:
            return False

        markers = [
            "__global__", "__device__", "__host__", "#include",
            "cudaMalloc", "cudaMemcpy", "dim3", "<<<", ">>>"
        ]
        if any(marker in snippet for marker in markers):
            return True

        return re.search(r"\b(?:void|int|float|double|__global__|__device__)\s+\w+\s*\([^)]*\)\s*{", snippet) is not None
        
    def generate_code(self, prompt, max_tokens=4096, status_callback=None, early_terminate=True):
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
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
            
        try:
            response = requests.post(self.url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            result_text = ""
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
           
            last_line = None
            repeat_count = 0
            MAX_REPEATS = 5
            last_chunk_text = None
            chunk_repeat_count = 0
            MAX_CHUNK_REPEATS = 8
            
            for chunk in response.iter_lines():
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
                                console.print("\n" + "=" * 45 + " AI REASONING PROCESS " + "=" * 45, style="bold cyan")
                                has_printed_think_header = True
                        elif "</think>" in result_text:
                            in_think = False
                            if not has_printed_think_footer:
                                console.print("\n" + "=" * 114 + "\n", style="bold cyan")
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
                                    console.print("\n" + "=" * 114 + "\n", style="bold cyan")
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
                                sys.stdout.write(f"\033[96m{clean_content}\033[0m")
                                sys.stdout.flush()
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
                        console.print("\n" + "=" * 45 + " AI REASONING PROCESS " + "=" * 45, style="bold cyan")
                        sys.stdout.write(f"\033[96m{result_text}\033[0m")
                        sys.stdout.flush()
                        console.print("\n" + "=" * 114 + "\n", style="bold cyan")
                elif in_think and not has_printed_think_footer:
                    console.print("\n" + "=" * 114 + "\n", style="bold cyan")
                    
                console.print(f"\n[bold green]Generation complete! Total tokens: {tokens_generated} (In {time.time() - start_time:.2f}s)[/bold green]")
             
            think_match = re.search(r"<think>(.*?)</think>", result_text, re.DOTALL)
            if think_match:
                result_text = result_text.replace(think_match.group(0), "").strip()
            else:
                if "<think>" in result_text:
                    parts = result_text.split("<think>")
                    result_text = parts[0].strip()
            
            code_match = re.search(r"```(?:cuda|cpp|c)?\n(.*?)```", result_text, re.DOTALL)
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

    def create_optimization_prompt(self, code, env_info, target, best_time, flags):
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
            "1. First, write your detailed step-by-step reasoning, optimization strategy, and analytical thoughts inside <think>...</think> tags. "
            "Keep your reasoning inside <think> extremely concise, clear, and direct (maximum 150-200 words) so you have enough token budget for the code.\n"
            "2. Second, write the final complete optimized CUDA code inside a ```cuda ... ``` markdown block.\n"
            "Do not output anything else outside these blocks.\n"
            "3. CRITICAL: NEVER use lazy placeholders like '...' or leave parts of the code as 'TODO'. You MUST output the entire, fully-functional, and mathematically complete CUDA kernels!\n"
            "4. CRITICAL: You MUST preserve the exact function names, argument counts, parameter types, and overall function signatures of all kernels (dft_kernel, fft_shared_kernel, vision_filter_kernel, pattern_match_kernel) exactly as provided in the input code. Do NOT alter parameter types (e.g., do not change uchar4* to unsigned char*, or remove arguments), as they are strictly benchmarked by an external host wrapper."
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
        system_prompt = (
            "You are a world-class CUDA compiler and debugging assistant. Your task is to fix compilation errors or correctness failures "
            "in the user's CUDA code.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. First, write your compilation error analysis and correction strategy inside <think>...</think> tags. "
            "Keep this reasoning block extremely concise (maximum 100 words).\n"
            "2. Second, write the fixed, complete CUDA code inside a ```cuda ... ``` markdown block.\n"
            "Do not output anything else outside these blocks.\n"
            "3. CRITICAL: NEVER use lazy placeholders like '...' or leave parts of the code as 'TODO'. You MUST output the entire, fully-functional, and mathematically complete CUDA kernels!\n"
            "4. CRITICAL: You MUST preserve the exact function names, argument counts, parameter types, and overall function signatures of all kernels (dft_kernel, fft_shared_kernel, vision_filter_kernel, pattern_match_kernel) exactly as provided. Do NOT alter parameter types, change argument lists, or change variable types of the parameters, as it will break the compilation with the benchmark wrapper."
        )
        
        user_prompt = (
            f"This CUDA code failed to compile/verify:\n```cuda\n{code}\n```\n\n"
            f"Error log:\n{error_log}\n\nPlease fix it so it compiles and is mathematically correct."
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
