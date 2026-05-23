import subprocess
import os
import re
import shutil
import time
from datetime import datetime
from .discover import find_ncu_path, find_nsys_path, find_cl_exe_dir

class CUDASandbox:
    def __init__(self, file_path, flags=None, profile_mode='auto', use_nvtx=False, profile_metrics='', apply_nvtx_suggestion=False):
        import uuid
        self.run_id = uuid.uuid4().hex[:8]
        self.file_path = file_path
        self.original_file_path = file_path
        self.exe_path = f"./temp_cuda_kernel_{self.run_id}.exe" if os.name == 'nt' else f"./temp_cuda_kernel_{self.run_id}.out"
        self.flags = flags or []
        self.profile_mode = profile_mode
        self.use_nvtx = use_nvtx
        self.profile_metrics = profile_metrics
        self.apply_nvtx_suggestion = apply_nvtx_suggestion
        self.reference_checksums = {}
        self.has_custom_harness = False

    def cleanup(self):
        temp_files = [
            self.exe_path,
            self.exe_path.replace(".exe", ".exp") if os.name == 'nt' else "",
            self.exe_path.replace(".exe", ".lib") if os.name == 'nt' else "",
            f"./temp_consolidated_unit_{self.run_id}.cu",
        ]
        for pattern in [f"ncu_report_{self.run_id}*", f"nsys_report_{self.run_id}*"]:
            import glob
            temp_files.extend(glob.glob(pattern))

        for tf in temp_files:
            if tf and os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass

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

        profiler_includes = ""
        if self.use_nvtx:
            profiler_includes += "#include <nvToolsExt.h>\n"
        profiler_includes += "#include <cuda_profiler_api.h>\n"

        harness_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

    {profiler_includes}
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
    bool enable_profiler = false;
    bool have_reference = false;
    float ref_dft = 0.0f;
    float ref_fft = 0.0f;
    float ref_vision = 0.0f;
    int ref_pattern = 0;

    if (argc >= 2) {{
        // Parse expected checksums passed as arguments
        for (int i = 1; i < argc; i++) {{
            if (strcmp(argv[i], "--profiler=on") == 0) {{
                enable_profiler = true;
            }}
            if (sscanf(argv[i], "--ref_dft=%f", &ref_dft) == 1) {{
                have_reference = true;
            }}
            if (sscanf(argv[i], "--ref_fft=%f", &ref_fft) == 1) {{
                have_reference = true;
            }}
            if (sscanf(argv[i], "--ref_vision=%f", &ref_vision) == 1) {{
                have_reference = true;
            }}
            if (sscanf(argv[i], "--ref_pattern=%d", &ref_pattern) == 1) {{
                have_reference = true;
            }}
        }}
    }}

    verify = have_reference;

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

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStart());
        }}

        #ifdef USE_NVTX
        nvtxRangePushA("DFT_KERNEL");
        #endif

        CUDA_CHECK(cudaEventRecord(start));
        dft_kernel<<<1, 256>>>(d_real, d_imag, d_mag, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        #ifdef USE_NVTX
        nvtxRangePop();
        #endif

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStop());
        }}

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

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStart());
        }}

        #ifdef USE_NVTX
        nvtxRangePushA("FFT_KERNEL");
        #endif

        CUDA_CHECK(cudaEventRecord(start));
        fft_shared_kernel<<<1, 512, 2 * 512 * sizeof(float)>>>(d_real, d_imag, n, 8);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        #ifdef USE_NVTX
        nvtxRangePop();
        #endif

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStop());
        }}

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

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStart());
        }}

        #ifdef USE_NVTX
        nvtxRangePushA("VISION_KERNEL");
        #endif

        CUDA_CHECK(cudaEventRecord(start));
        vision_filter_kernel<<<blocks, threads>>>(d_pixels, w, h);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        #ifdef USE_NVTX
        nvtxRangePop();
        #endif

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStop());
        }}

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

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStart());
        }}

        #ifdef USE_NVTX
        nvtxRangePushA("PATTERN_KERNEL");
        #endif

        CUDA_CHECK(cudaEventRecord(start));
        pattern_match_kernel<<<4, 256, pat_len>>>(d_data, data_len, pat_len, d_found);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        #ifdef USE_NVTX
        nvtxRangePop();
        #endif

        if (enable_profiler) {{
            CUDA_CHECK(cudaProfilerStop());
        }}

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

    def _collect_reference_checksums(self):
        try:
            run_res = subprocess.run([self.exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            return f"Could not collect reference checksums: {exc}"

        if run_res.returncode != 0:
            stderr = run_res.stderr.strip() if run_res.stderr else ""
            stdout = run_res.stdout.strip() if run_res.stdout else ""
            details = stderr or stdout or f"exit code {run_res.returncode}"
            return f"Could not collect reference checksums: harness execution failed ({details})"

        stdout = run_res.stdout
        for key in ["dft", "fft", "vision", "pattern"]:
            match = re.search(f"{key.upper()}_CHECKSUM:\\s*([\\d\\.-]+)", stdout)
            if match:
                self.reference_checksums[key] = match.group(1)
        return None

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
            temp_compile_file = f"./temp_consolidated_unit_{self.run_id}.cu"
            
            with open(temp_compile_file, "w") as f:
            
                if self.apply_nvtx_suggestion and os.path.exists("nvtx_suggestion.cu"):
                    try:
                        with open("nvtx_suggestion.cu", "r", encoding='utf-8') as sf:
                            f.write(sf.read())
                            f.write("\n")
                    except Exception:
                        pass
                f.write(code_content)
                f.write("\n")
                f.write(harness_code)

            target_file = temp_compile_file

        from .discover import find_nvcc_path
        nvcc_bin = find_nvcc_path() or "nvcc"
        src_dir = os.path.dirname(os.path.abspath(self.original_file_path or self.file_path))
        
      
        include_dirs = [src_dir]
        curr = src_dir
        for _ in range(4):
            if not curr or curr == os.path.dirname(curr):
                break
            for name in ["include", "includes", "headers", "src", "kernels"]:
                candidate = os.path.join(curr, name)
                if os.path.isdir(candidate):
                    include_dirs.append(candidate)
            try:
                for entry in os.scandir(curr):
                    if entry.is_dir() and entry.name not in [".git", ".venv", "__pycache__", "build", "dist"]:
                        has_headers = False
                        try:
                            for sub_entry in os.scandir(entry.path):
                                if sub_entry.is_file() and sub_entry.name.lower().endswith(('.h', '.cuh')):
                                    has_headers = True
                                    break
                        except Exception:
                            pass
                        if has_headers or entry.name.lower() in ["include", "includes", "headers"]:
                            include_dirs.append(entry.path)
            except Exception:
                pass
            curr = os.path.dirname(curr)


        cmd = [nvcc_bin, target_file]
        unique_dirs = []
        for d in include_dirs:
            d_abs = os.path.abspath(d)
            if d_abs not in unique_dirs and os.path.isdir(d_abs):
                unique_dirs.append(d_abs)
                cmd += ["-I", d_abs]

        cmd += ["-o", self.exe_path] + self.flags
        if os.name == 'nt' and "-allow-unsupported-compiler" not in cmd:
            cmd.append("-allow-unsupported-compiler")

        # On Windows, inject cl.exe directory into PATH so nvcc can find the host compiler
        compile_env = None
        if os.name == 'nt':
            cl_dir = find_cl_exe_dir()
            if cl_dir:
                compile_env = os.environ.copy()
                current_path = compile_env.get('PATH', '')
                if cl_dir.lower() not in current_path.lower():
                    compile_env['PATH'] = cl_dir + ';' + current_path

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=compile_env)
            error_log = ""
            if result.stderr:
                error_log += result.stderr.strip()
            if result.stdout:
                if error_log:
                    error_log += "\n"
                error_log += result.stdout.strip()
            
            temp_compile_file = f"./temp_consolidated_unit_{self.run_id}.cu"
            if self.has_custom_harness and os.path.exists(temp_compile_file):
                try:
                    os.remove(temp_compile_file)
                except Exception:
                    pass

            success = result.returncode == 0
            
            if success and self.has_custom_harness and not self.reference_checksums:
                warning = self._collect_reference_checksums()
                if warning:
                    if error_log:
                        error_log += "\n"
                    error_log += f"WARNING: {warning}"
            return {"success": success, "error_log": error_log}
        except FileNotFoundError:
            return {"success": False, "error_log": "nvcc not found"}

    def _format_ncu_failure_message(self, output):
        if "ERR_NVGPUCTRPERM" in output:
            return (
                "System issue: ERR_NVGPUCTRPERM (Profiler Fallback).\n"
                "Cause: cudallm attempted to run NVIDIA Nsight Compute (NCU) for accurate performance profiling, "
                "but the current user is blocked from accessing NVIDIA hardware performance counters. "
                "The run therefore used fallback timer mode, and benchmark quality is reduced (Best Latency may be unavailable). "
                "This run is not a valid benchmark.\n\n"
                "Windows quick fix:\n"
                "1) Open NVIDIA Control Panel as Administrator.\n"
                "2) In the Desktop menu, enable Developer Settings (if not already enabled).\n"
                "3) Go to Developer > Manage GPU Performance Counters.\n"
                "4) Set \"Allow access to the GPU performance counters to all users\", then click Apply.\n"
                "5) Close Terminal/VS Code and reopen as Administrator, then rerun cudallm optimize.\n\n"
                f"Output logs:\n{output[:500]}"
            )

        return f"NCU profiling failed to generate CSV report. Output logs:\n{output[:500]}"

    def _format_nsys_failure_message(self, output):
        if "ERR_NVGPUCTRPERM" in output:
            return (
                "System issue: ERR_NVGPUCTRPERM (Profiler Fallback).\n"
                "Cause: cudallm attempted to run NVIDIA Nsight Systems (NSYS) for timeline profiling, "
                "but the current user is blocked from accessing NVIDIA hardware performance counters. "
                "The run therefore used fallback timer mode, and benchmark quality is reduced (Best Latency may be unavailable). "
                "This run is not a valid benchmark.\n\n"
                "Windows quick fix:\n"
                "1) Open NVIDIA Control Panel as Administrator.\n"
                "2) In the Desktop menu, enable Developer Settings (if not already enabled).\n"
                "3) Go to Developer > Manage GPU Performance Counters.\n"
                "4) Set \"Allow access to the GPU performance counters to all users\", then click Apply.\n"
                "5) Close Terminal/VS Code and reopen as Administrator, then rerun cudallm optimize.\n\n"
                f"Output logs:\n{output[:500]}"
            )

        return f"NSYS profiling failed to generate timeline report. Output logs:\n{output[:500]}"

    def _extract_latency_ms(self, output, wallclock_start=None):
        match = re.search(r"TOTAL_LATENCY:\s*([\d\.]+)\s*(ms|us)?", output, re.IGNORECASE)
        if match:
            latency = float(match.group(1))
            if match.group(2) == 'us':
                latency /= 1000.0
            return latency

        match = re.search(r"([\d\.]+)\s*(ms|us)", output)
        if match:
            latency = float(match.group(1))
            if match.group(2) == 'us':
                latency /= 1000.0
            return latency

        if wallclock_start is not None:
            return (time.time() - wallclock_start) * 1000.0

        return None

    def _run_timer_fallback(self, args):
        cmd = [self.exe_path] + args
        started = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            err_text = (result.stderr or result.stdout or "fallback execution failed").strip()
            return {"latency": None, "error": err_text, "raw_output": (result.stdout + result.stderr)[:500]}

        stdout = result.stdout or ""
        latency = self._extract_latency_ms(stdout, wallclock_start=started)
        return {"latency": latency, "raw_output": stdout[:500]}

    def profile_latency(self, target_metric=None):
        if not os.path.exists(self.exe_path):
            return {"latency": float('inf'), "raw_output": ""}

        try:
            args = []
            if self.has_custom_harness and self.reference_checksums:
                for key, value in self.reference_checksums.items():
                    args.append(f"--ref_{key}={value}")

            if self.profile_mode == 'code':
                args.append("--profiler=on")

            ncu_bin = find_ncu_path()
            nsys_bin = find_nsys_path()

            profiler = None
            if self.profile_mode == 'nsys' and nsys_bin:
                profiler = ('nsys', nsys_bin)
            elif self.profile_mode == 'ncu' and ncu_bin:
                profiler = ('ncu', ncu_bin)
            elif self.profile_mode in ('auto', 'auto-strict', 'auto-relaxed'):
                if ncu_bin:
                    profiler = ('ncu', ncu_bin)
                elif nsys_bin:
                    profiler = ('nsys', nsys_bin)
            elif self.profile_mode == 'code' and nsys_bin:
                profiler = ('nsys', nsys_bin)

            if not profiler:
                cmd = [self.exe_path] + args
                start = time.time()
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode != 0:
                    err_msg = result.stderr if result.stderr else "Verification mismatch!"
                    return {"latency": 99999.0, "raw_output": f"VERIFICATION FAILURE: {err_msg.strip()}"}

                stdout = result.stdout
                latency = self._extract_latency_ms(stdout, wallclock_start=start)
                if latency is None:
                    latency = (time.time() - start) * 1000.0

                return {"latency": latency, "raw_output": stdout[:500]}

            kind, binpath = profiler
            if kind == 'ncu':
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                base = f"ncu_report_{self.run_id}_{ts}"
                csv_file = f"{base}.csv"
                cmd = [binpath]
                from .discover import find_ncu_sections_path
                sections_path = find_ncu_sections_path(binpath)
                if sections_path:
                    cmd.extend(['--section-folder', sections_path])
                # Add kernel profiling section to ensure kernels are captured
                cmd.extend(['--set', 'full', '--section', 'SpeedOfLight'])
                if self.profile_metrics:
                    cmd.extend(['--metrics', self.profile_metrics])
                cmd.extend(['--csv', '-o', base, self.exe_path])
                cmd.extend(args)
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    output = result.stdout + result.stderr
                except Exception:
                    cmd2 = [binpath]
                    if sections_path:
                        cmd2.extend(['--section-folder', sections_path])
                    cmd2.extend(['--set', 'full', '--section', 'SpeedOfLight'])
                    cmd2.extend([self.exe_path] + args)
                    result = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    output = result.stdout + result.stderr

                rep_path = f"{base}.ncu-rep"
                if os.path.exists(rep_path):
                    export_cmd = [binpath]
                    if sections_path:
                        export_cmd.extend(['--section-folder', sections_path])
                    export_cmd.extend(['--import', rep_path, '--csv'])
                    try:
                        exp_res = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                        if exp_res.returncode == 0 and exp_res.stdout.strip():
                            with open(csv_file, 'w', encoding='utf-8') as f:
                                f.write(exp_res.stdout)
                    except Exception:
                        pass

                if os.path.exists(csv_file):
                    try:
                        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                            out_text = f.read()[:500]
                    except Exception:
                        out_text = ''
                    latency = 99999.0
                    try:
                        from .profiler_tools import parse_ncu_csv_for_hotspot
                        hotspot = parse_ncu_csv_for_hotspot(csv_file, target_metric)
                        if hotspot and hotspot.get("value") is not None:
                            latency = float(hotspot["value"])
                    except Exception:
                        pass
                    return {"latency": latency, "raw_output": out_text, "ncu_csv": os.path.abspath(csv_file)}
                else:
                    failure_message = self._format_ncu_failure_message(output)
                    if "ERR_NVGPUCTRPERM" in output:
                        fallback = self._run_timer_fallback(args)
                        fallback_latency = fallback.get("latency")
                        if fallback_latency is not None:
                            failure_message += (
                                f"\n\nFallback executable timing: {fallback_latency:.4f} ms "
                                "(non-NCU; informational only)."
                            )
                        elif fallback.get("error"):
                            failure_message += f"\n\nFallback executable timing failed: {fallback['error']}"
                        return {
                            "latency": 99999.0,
                            "raw_output": failure_message,
                            "fallback_latency": fallback_latency,
                            "fallback_raw_output": fallback.get("raw_output", ""),
                            "profiling_blocked_reason": "ncu_permission",
                        }
                    return {"latency": 99999.0, "raw_output": failure_message}

            cmd = [binpath, 'profile', '--output', f'nsys_report_{self.run_id}', '--trace', 'cuda']
            if self.profile_mode == 'code':
                cmd.extend(['--capture-range=cudaProfilerApi'])
            cmd.append(self.exe_path)
            cmd.extend(args)

            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
                output = result.stdout + result.stderr
            except Exception:
                try:
                    cmd2 = [binpath, self.exe_path] + args
                    result = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
                    output = result.stdout + result.stderr
                except Exception as e:
                    output = f"nsys invocation failed: {e}"

            latency = self._extract_latency_ms(output)
            if latency is not None:
                return {"latency": latency, "raw_output": output[:500]}

            return {"latency": 99999.0, "raw_output": self._format_nsys_failure_message(output)}
        except Exception as e:
            return {"latency": 99999.0, "raw_output": self._format_nsys_failure_message(str(e))}
