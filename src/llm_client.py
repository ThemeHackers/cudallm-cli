import requests
import json
import re
import time

class LLMClient:
    def __init__(self, url):
        self.url = url
        self.total_tokens = 0
        
    def _is_ollama(self):
        return "11434" in self.url

    def generate_code(self, prompt, max_tokens=2048):
        start_time = time.time()
        if self._is_ollama():
            payload = {"model": "llama3", "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": max_tokens}}
        else:
            payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.2, "stop": ["```\n\n", "<|eot_id|>"]}
            
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            
            if self._is_ollama():
                res_json = response.json()
                result_text = res_json.get('response', '')
                self.total_tokens += res_json.get('eval_count', 0)
            else:
                res_json = response.json()
                result_text = res_json.get('content', '')
                self.total_tokens += res_json.get('tokens_predicted', 0)
            
            code_match = re.search(r"```(?:cuda|cpp|c)?\n(.*?)```", result_text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip(), time.time() - start_time
            return result_text.strip(), time.time() - start_time
        except Exception:
            return "", 0.0

    def create_optimization_prompt(self, code, env_info, target, best_time, flags):
        sys_prompt = f"Optimize CUDA code for {env_info['gpu_model']} (Compute {env_info['compute_capability']}, CUDA {env_info['cuda_version']})."
        if best_time != float('inf'):
            sys_prompt += f" Current latency: {best_time}ms. Target: {target}."
        if flags:
            sys_prompt += f" Compiler flags used: {' '.join(flags)}."
        return f"{sys_prompt}\nOnly return CUDA code inside markdown.\n```cuda\n{code}\n```"

    def create_healing_prompt(self, code, error_log):
        return f"This CUDA code failed to compile:\n```cuda\n{code}\n```\nError:\n{error_log}\nFix it and return only CUDA code."
        
    def create_audit_prompt(self, code):
        return f"Audit this CUDA code for Warp Divergence, Bank Conflicts, Memory Access. Explain fixes.\n```cuda\n{code}\n```"
