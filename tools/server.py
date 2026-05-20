"""
Simple GPU-backed LLM HTTP server (FastAPI).

Behavior:
- Prefer llama-cpp-python (if installed) to load GGUF/llama.cpp models (uses CUDA if llama.cpp is compiled with CUDA)
- Otherwise fall back to Hugging Face `transformers` (works with CUDA via `device_map="auto"` and `bitsandbytes`/`accelerate`)

Usage (Colab):
- Install dependencies (choose one path):
  # For llama-cpp-python (GGUF / llama.cpp)
  pip install "llama-cpp-python"

  # For transformers + bitsandbytes (HF checkpoints)
  pip install transformers accelerate bitsandbytes safetensors[torch]

Run:
    python tools/server.py --hf-repo prithivMLmods/cudaLLM-8B-GGUF --hf-file cudaLLM-8B.Q2_K.gguf --host 127.0.0.1 --port 8081 --use-cuda

Endpoints:
- GET /health
- POST /completion  {"prompt": "...", "max_tokens": 128}

Note: This server is a minimal example. Production deployments should add auth, batching, rate limits, and careful memory management.
"""

import argparse
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = None
ENGINE = None


@asynccontextmanager
async def lifespan(app):
   
    yield


app = FastAPI(lifespan=lifespan)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128

class CompletionResponse(BaseModel):
    text: str


def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


class LLMWrapper:
    def __init__(self, model_path=None, hf_repo=None, hf_file=None, use_cuda=False):
        self.model_path = model_path
        self.hf_repo = hf_repo
        self.hf_file = hf_file.strip() if isinstance(hf_file, str) else hf_file
        self.use_cuda = use_cuda
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._init_backend()

    def _init_backend(self):
        wants_gguf = bool((self.hf_file and self.hf_file.lower().endswith('.gguf')) or (self.model_path and str(self.model_path).lower().endswith('.gguf')))

        llama_cpp = try_import('llama_cpp')
        if llama_cpp is not None:
            try:
                from llama_cpp import Llama
                print('[INFO] Using llama-cpp-python backend')
                self.backend = 'llama_cpp'

                if self.model_path:
                    llm_kwargs = {'model_path': self.model_path}
                elif self.hf_repo and self.hf_file:
                    print(f'[INFO] Downloading GGUF from HF via llama-cpp-python: repo={self.hf_repo} file={self.hf_file}')
                    llm_kwargs = {
                        'repo_id': self.hf_repo,
                        'filename': self.hf_file,
                    }
                elif self.hf_file and os.path.exists(self.hf_file):
                    llm_kwargs = {'model_path': self.hf_file}
                else:
                    raise RuntimeError('No model path supplied for llama-cpp')

                if self.use_cuda:
                    llm_kwargs['n_gpu_layers'] = -1

                self.model = Llama.from_pretrained(**llm_kwargs)
                return
            except Exception as e:
                print('[WARN] llama-cpp import or init failed:', e)

        if wants_gguf:
            raise RuntimeError(
                'GGUF model requested, but llama-cpp-python is unavailable or failed to initialize. '
                'Install llama-cpp-python (and huggingface_hub for --hf-repo/--hf-file).'
            )

      
        transformers = try_import('transformers')
        torch = try_import('torch')
        if transformers is None or torch is None:
            raise RuntimeError('Neither llama-cpp-python nor transformers+torch are available. Install one of them.')

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            print('[INFO] Using transformers backend')
            self.backend = 'transformers'
            model_id = self.hf_repo or self.model_path
            if not model_id:
                raise RuntimeError('No model id/path provided for transformers backend')

          
            device_map = 'auto' if self.use_cuda else None
            kwargs = {}
            if self.use_cuda:
            
                kwargs['dtype'] = getattr(torch, 'float16', None)

            print(f'[INFO] Loading model {model_id} (this may take a while)')
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, **kwargs)
            gen_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if self.use_cuda else -1)
            self.tokenizer = tokenizer
            self.model = model
            self.pipeline = gen_pipeline
            return
        except Exception as e:
            raise RuntimeError('Failed to initialize transformers backend: ' + str(e))

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        if self.backend == 'llama_cpp':
            
            try:
                out = self.model(prompt, max_tokens=max_tokens)

                if isinstance(out, dict) and 'choices' in out:
                    return out['choices'][0]['text']
                if isinstance(out, str):
                    return out
    
                return str(out)
            except Exception as e:
                raise RuntimeError('llama-cpp generation error: ' + str(e))
        elif self.backend == 'transformers':
            try:
                res = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)
                if isinstance(res, list) and 'generated_text' in res[0]:
                    return res[0]['generated_text']
              
                return str(res)
            except Exception as e:
                raise RuntimeError('transformers generation error: ' + str(e))
        else:
            raise RuntimeError('No backend available')


@app.get('/health')
def health():
    return {'status': 'ok', 'backend': MODEL.backend if MODEL else None}


@app.post('/completion', response_model=CompletionResponse)
def completion(req: CompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        text = MODEL.generate(req.prompt, max_tokens=req.max_tokens or 128)
        return CompletionResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(args):
  
    global MODEL
    MODEL = LLMWrapper(model_path=args.local_model, hf_repo=args.hf_repo, hf_file=args.hf_file, use_cuda=args.use_cuda)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-model', type=str, help='Path to local model file (GGUF or HF id)')
    parser.add_argument('--hf-repo', type=str, help='HF repo id or path')
    parser.add_argument('--hf-file', type=str, help='HF file or GGUF filename for llama-cpp')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--use-cuda', action='store_true', help='Attempt to use CUDA/GPU')
    parsed = parser.parse_args()

    print('[INFO] Starting server with args:', parsed)
    run_server(parsed)
