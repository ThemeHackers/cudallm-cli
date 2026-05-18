import subprocess
import os
import re
import shutil
import time
from src.discover import find_ncu_path

class CUDASandbox:
    def __init__(self, file_path, flags=None):
        self.file_path = file_path
        self.exe_path = "./temp_cuda_kernel.exe" if os.name == 'nt' else "./temp_cuda_kernel.out"
        self.flags = flags or []
        self.reference_checksums = {}
        self.has_custom_harness = False

    def _generate_harness(self, code):
        has_dft = "dft_kernel" in code
        has_fft = "fft_shared_kernel" in code
        has_vision = "vision_filter_kernel" in code
        has_pattern = "pattern_match_kernel" in code

        defines = []
        if has_dft: defines.append("#define HAS_DFT")
        if has_fft: defines.append("#define HAS_FFT")
        if has_vision: defines.append("#define HAS_VISION")
        if has_pattern: defines.append("#define HAS_PATTERN")

        define_block = "\n".join(defines)

        harness_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

{define_block}

// Helper to check CUDA errors
#define CUDA_CHECK(val) {{ \
    cudaError_t err = val; \
    if (err != cudaSuccess) {{ \
        fprintf(stderr, "CUDA Error: %s at line %d\\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }} \
}}

// Function to generate random float data
void init_rand_float(float* arr, int n) {{
    for (int i = 0; i < n; i++) {{
        arr[i] = (float)rand() / (float)RAND_MAX;
    }}
}}

int main(int argc, char** argv) {{
    // Check if we are running in verification mode
    bool verify = false;
    float ref_dft = 0.0f;
    float ref_fft = 0.0f;
    float ref_vision = 0.0f;
    int ref_pattern = 0;

    if (argc >= 2) {{
        verify = true;
        // Parse expected checksums passed as arguments
        for (int i = 1; i < argc; i++) {{
            sscanf(argv[i], "--ref_dft=%f", &ref_dft);
            sscanf(argv[i], "--ref_fft=%f", &ref_fft);
            sscanf(argv[i], "--ref_vision=%f", &ref_vision);
            sscanf(argv[i], "--ref_pattern=%d", &ref_pattern);
        }}
    }}

    srand(42); // Pin seed for reproducibility
    float total_latency_ms = 0.0f;

    // DFT Kernel Test
    #ifdef HAS_DFT
    {{
        int n = 256;
        float *h_real = (float*)malloc(n * sizeof(float));
        float *h_imag = (float*)malloc(n * sizeof(float));
        float *h_mag = (float*)malloc(n * sizeof(float));
        init_rand_float(h_real, n);
        init_rand_float(h_imag, n);

        float *d_real, *d_imag, *d_mag;
        CUDA_CHECK(cudaMalloc(&d_real, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_imag, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mag, n * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_real, h_real, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_imag, h_imag, n * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        dft_kernel<<<1, 256>>>(d_real, d_imag, d_mag, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_latency_ms += ms;

        CUDA_CHECK(cudaMemcpy(h_mag, d_mag, n * sizeof(float), cudaMemcpyDeviceToHost));

        float dft_sum = 0.0f;
        for (int i = 0; i < n; i++) dft_sum += h_mag[i];

        printf("DFT_LATENCY: %f ms\\n", ms);
        printf("DFT_CHECKSUM: %f\\n", dft_sum);

        if (verify && fabs(dft_sum - ref_dft) > 1e-2f) {{
            fprintf(stderr, "VERIFICATION FAILURE: DFT output checksum mismatch! Expected %f, got %f\\n", ref_dft, dft_sum);
            exit(2);
        }}

        cudaFree(d_real); cudaFree(d_imag); cudaFree(d_mag);
        free(h_real); free(h_imag); free(h_mag);
    }}
    #endif

    // FFT Shared Kernel Test
    #ifdef HAS_FFT
    {{
        int n = 256;
        float *h_real = (float*)malloc(n * sizeof(float));
        float *h_imag = (float*)malloc(n * sizeof(float));
        init_rand_float(h_real, n);
        init_rand_float(h_imag, n);

        float *d_real, *d_imag;
        CUDA_CHECK(cudaMalloc(&d_real, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_imag, n * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_real, h_real, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_imag, h_imag, n * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        fft_shared_kernel<<<1, 512, 2 * 512 * sizeof(float)>>>(d_real, d_imag, n, 8);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_latency_ms += ms;

        CUDA_CHECK(cudaMemcpy(h_real, d_real, n * sizeof(float), cudaMemcpyDeviceToHost));

        float fft_sum = 0.0f;
        for (int i = 0; i < n; i++) fft_sum += fabs(h_real[i]);

        printf("FFT_LATENCY: %f ms\\n", ms);
        printf("FFT_CHECKSUM: %f\\n", fft_sum);

        if (verify && fabs(fft_sum - ref_fft) > 1e-2f) {{
            fprintf(stderr, "VERIFICATION FAILURE: FFT output checksum mismatch! Expected %f, got %f\\n", ref_fft, fft_sum);
            exit(2);
        }}

        cudaFree(d_real); cudaFree(d_imag);
        free(h_real); free(h_imag);
    }}
    #endif

    // Vision Filter Kernel Test
    #ifdef HAS_VISION
    {{
        int w = 640, h = 480;
        uchar4 *h_pixels = (uchar4*)malloc(w * h * sizeof(uchar4));
        for (int i = 0; i < w * h; i++) {{
            h_pixels[i] = make_uchar4(rand() % 256, rand() % 256, rand() % 256, 255);
        }}

        uchar4 *d_pixels;
        CUDA_CHECK(cudaMalloc(&d_pixels, w * h * sizeof(uchar4)));
        CUDA_CHECK(cudaMemcpy(d_pixels, h_pixels, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

        dim3 blocks((w + 15) / 16, (h + 15) / 16);
        dim3 threads(16, 16);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        vision_filter_kernel<<<blocks, threads>>>(d_pixels, w, h);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_latency_ms += ms;

        CUDA_CHECK(cudaMemcpy(h_pixels, d_pixels, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

        float vis_sum = 0.0f;
        for (int i = 0; i < w * h; i += 100) vis_sum += h_pixels[i].x;

        printf("VISION_LATENCY: %f ms\\n", ms);
        printf("VISION_CHECKSUM: %f\\n", vis_sum);

        if (verify && fabs(vis_sum - ref_vision) > 1.0f) {{
            fprintf(stderr, "VERIFICATION FAILURE: Vision filter output checksum mismatch! Expected %f, got %f\\n", ref_vision, vis_sum);
            exit(2);
        }}

        cudaFree(d_pixels);
        free(h_pixels);
    }}
    #endif

    // Pattern Match Kernel Test
    #ifdef HAS_PATTERN
    {{
        int data_len = 1024;
        int pat_len = 16;
        unsigned char *h_data = (unsigned char*)malloc(data_len);
        for (int i = 0; i < data_len; i++) h_data[i] = rand() % 256;
        
        unsigned char *d_data;
        int *d_found;
        int h_found = -1;
        CUDA_CHECK(cudaMalloc(&d_data, data_len));
        CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_data, h_data, data_len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        pattern_match_kernel<<<4, 256, pat_len>>>(d_data, data_len, pat_len, d_found);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_latency_ms += ms;

        CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));

        printf("PATTERN_LATENCY: %f ms\\n", ms);
        printf("PATTERN_CHECKSUM: %d\\n", h_found);

        if (verify && h_found != ref_pattern) {{
            fprintf(stderr, "VERIFICATION FAILURE: Pattern match output mismatch! Expected %d, got %d\\n", ref_pattern, h_found);
            exit(2);
        }}

        cudaFree(d_data); cudaFree(d_found);
        free(h_data);
    }}
    #endif

    printf("TOTAL_LATENCY: %f ms\\n", total_latency_ms);
    return 0;
}}
"""
        return harness_code

    def compile(self):
        try:
            with open(self.file_path, "r") as f:
                code_content = f.read()
        except Exception as e:
            return {"success": False, "error_log": f"Could not read file: {e}"}

        self.has_custom_harness = "main(" not in code_content

        target_file = self.file_path
        if self.has_custom_harness:
            harness_code = self._generate_harness(code_content)
            temp_compile_file = "./temp_consolidated_unit.cu"
            
            with open(temp_compile_file, "w") as f:
                f.write(code_content)
                f.write("\n")
                f.write(harness_code)
            
            target_file = temp_compile_file

        from src.discover import find_nvcc_path
        nvcc_bin = find_nvcc_path() or "nvcc"
        cmd = [nvcc_bin, target_file, "-o", self.exe_path] + self.flags
        if os.name == 'nt' and "-allow-unsupported-compiler" not in cmd:
            cmd.append("-allow-unsupported-compiler")
            
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            error_log = ""
            if result.stderr:
                error_log += result.stderr.strip()
            if result.stdout:
                if error_log:
                    error_log += "\n"
                error_log += result.stdout.strip()
            
            if self.has_custom_harness and os.path.exists("./temp_consolidated_unit.cu"):
                try:
                    os.remove("./temp_consolidated_unit.cu")
                except Exception:
                    pass

            success = result.returncode == 0
            
            if success and self.has_custom_harness and not self.reference_checksums:
                try:
                    run_res = subprocess.run([self.exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if run_res.returncode == 0:
                        stdout = run_res.stdout
                        for key in ["dft", "fft", "vision", "pattern"]:
                            match = re.search(f"{key.upper()}_CHECKSUM:\\s*([\\d\\.-]+)", stdout)
                            if match:
                                self.reference_checksums[key] = match.group(1)
                except Exception:
                    pass

            return {"success": success, "error_log": error_log}
        except FileNotFoundError:
            return {"success": False, "error_log": "nvcc not found"}

    def profile_latency(self):
        if not os.path.exists(self.exe_path):
            return {"latency": float('inf'), "raw_output": ""}

        args = []
        if self.has_custom_harness and self.reference_checksums:
            for k, v in self.reference_checksums.items():
                args.append(f"--ref_{k}={v}")

        profiler = find_ncu_path()
        
        if not profiler:
            try:
                cmd = [self.exe_path] + args
                start = time.time()
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if result.returncode != 0:
                    err_msg = result.stderr if result.stderr else "Verification mismatch!"
                    return {"latency": 99999.0, "raw_output": f"VERIFICATION FAILURE: {err_msg.strip()}"}
                
                stdout = result.stdout
                match = re.search(r"TOTAL_LATENCY:\s*([\d\.]+)\s*ms", stdout)
                if match:
                    latency = float(match.group(1))
                else:
                    latency = (time.time() - start) * 1000.0
                    
                return {"latency": latency, "raw_output": stdout[:500]}
            except Exception as e:
                return {"latency": 99999.0, "raw_output": str(e)}

        cmd = [profiler, self.exe_path] + args
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout + result.stderr
            
            if result.returncode != 0:
                return {"latency": 99999.0, "raw_output": f"VERIFICATION FAILURE:\n{output[:500]}"}

            match = re.search(r'([\d\.]+)\s*(ms|us)', output)
            if match:
                val = float(match.group(1))
                latency = val / 1000.0 if match.group(2) == "us" else val
                return {"latency": latency, "raw_output": output[:500]}
                
            return {"latency": 99999.0, "raw_output": output[:500]}
        except Exception as e:
            return {"latency": 99999.0, "raw_output": str(e)}
