import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import csv

from src.discover import find_nvcc_path
from src.llm_client import LLMClient
from src.profiler_tools import parse_ncu_csv_for_hotspot
from tools.compare_ncu import numeric_columns
from src.profiler_tools import run_ncu_broad


class TestHarnessFixes(unittest.TestCase):
    @patch("src.discover.os.name", "nt")
    @patch("src.discover.glob.glob")
    @patch("src.discover.os.environ.get")
    @patch("src.discover.shutil.which")
    @patch("src.discover._pick_first_existing")
    def test_cuda_version_sorting(self, mock_pick, mock_which, mock_env_get, mock_glob):
        mock_which.return_value = None
        mock_pick.return_value = None
        mock_env_get.side_effect = lambda key, default=None: default

        mock_glob.return_value = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe",
        ]

        nvcc_path = find_nvcc_path()
        self.assertIn("v12.0", nvcc_path)

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_llm_url_endpoint_normalization(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client_ollama = LLMClient(url="http://127.0.0.1:11434")
        self.assertEqual(client_ollama.url, "http://127.0.0.1:11434/api/generate")

        client_llama = LLMClient(url="http://127.0.0.1:8080")
        self.assertEqual(client_llama.url, "http://127.0.0.1:8080/v1/completions")

        client_specified = LLMClient(url="http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(client_specified.url, "http://127.0.0.1:8080/v1/chat/completions")

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    @patch("src.llm_client.requests.post")
    def test_generate_code_prefills_cuda_block(self, mock_post, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [b"data: [DONE]"]
        mock_post.return_value = mock_response

        client = LLMClient(url="http://127.0.0.1:8080")
        client.generate_code("<|im_start|>system\nTest<|im_end|>\n<|im_start|>assistant\n", prefill=True)

        payload = mock_post.call_args.kwargs["json"]
        self.assertTrue(payload["prompt"].endswith("```cuda\n"))

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_optimization_prompt_requires_single_cuda_block(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")
        prompt = client.create_optimization_prompt(
            code="__global__ void vectorAdd(float* A, float* B, float* C, int n) {}",
            env_info={"gpu_model": "RTX 2060", "compute_capability": "7.5", "cuda_version": "13.0"},
            target="latency",
            best_time=float("inf"),
            flags=["-O3"],
        )

        self.assertIn("Do not provide reasoning or commentary.", prompt)
        self.assertIn("final answer must be exactly one complete ```cuda", prompt)
        self.assertIn("Do NOT write any introduction, explanation, apology, summary, or text outside the CUDA code block", prompt)

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_healing_prompt_requires_single_cuda_block(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")
        prompt = client.create_healing_prompt(
            code="__global__ void vectorAdd(float* A, float* B, float* C, int n) {}",
            error_log="nvcc error",
        )

        self.assertIn("Do not provide reasoning or commentary.", prompt)
        self.assertIn("final answer must be exactly one complete ```cuda", prompt)
        self.assertIn("Do NOT write any introduction, explanation, apology, summary, or text outside the CUDA code block", prompt)

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_compile_repair_prompt_prioritizes_diagnostics(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")
        prompt = client.create_compile_repair_prompt(
            code="__global__ void vectorAdd(float* A, float* B, float* C, int n) {}",
            error_log="error: expected ';' before '}' token",
        )

        self.assertIn("Repair the user's CUDA source so it compiles cleanly", prompt)
        self.assertIn("Prioritize the compiler diagnostics over the previous code", prompt)
        self.assertIn("This is repair attempt 1/1", prompt)
        self.assertIn("Compiler diagnostics summary:", prompt)
        self.assertIn("Raw compiler diagnostics:", prompt)

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_compile_repair_prompt_carries_retry_context(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")
        prompt = client.create_compile_repair_prompt(
            code="__global__ void vectorAdd(float* A, float* B, float* C, int n) {}",
            error_log="error: expected ';' before '}' token",
            attempt_index=2,
            max_attempts=3,
        )

        self.assertIn("This is repair attempt 2/3", prompt)
        self.assertIn("If a previous repair failed", prompt)

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_diagnostics_summarizer_keeps_key_errors(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")
        summary = client._summarize_diagnostics(
            """
            temp_kernel.cu(16): error: a value of type \"void *\" cannot be used to initialize an entity of type \"float *\"
            temp_kernel.cu(17): error: a value of type \"void *\" cannot be used to initialize an entity of type \"float *\"
            note: some unrelated note
            """
        )

        self.assertIn("cannot be used to initialize", summary)
        self.assertIn("temp_kernel.cu(16)", summary)
        self.assertIn("temp_kernel.cu(17)", summary)

    def test_generated_harness_includes_profiler_toggle(self):
        from src.sandbox import CUDASandbox

        sandbox = CUDASandbox("dummy.cu")
        harness = sandbox._generate_harness("__global__ void dft_kernel() {}")

        self.assertIn("bool enable_profiler = false;", harness)
        self.assertIn("bool have_reference = false;", harness)
        self.assertIn("--profiler=on", harness)
        self.assertIn("strcmp(argv[i], \"--profiler=on\") == 0", harness)
        self.assertIn("verify = have_reference;", harness)

    @patch("src.sandbox.subprocess.run")
    def test_collect_reference_checksums_records_values(self, mock_run):
        from src.sandbox import CUDASandbox

        sandbox = CUDASandbox("dummy.cu")
        sandbox.exe_path = "dummy.exe"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "DFT_CHECKSUM: 12.5\nFFT_CHECKSUM: 9.0\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        warning = sandbox._collect_reference_checksums()

        self.assertIsNone(warning)
        self.assertEqual(sandbox.reference_checksums["dft"], "12.5")
        self.assertEqual(sandbox.reference_checksums["fft"], "9.0")

    @patch("src.sandbox.subprocess.run")
    def test_collect_reference_checksums_surfaces_failure(self, mock_run):
        from src.sandbox import CUDASandbox

        sandbox = CUDASandbox("dummy.cu")
        sandbox.exe_path = "dummy.exe"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "runtime failure"
        mock_run.return_value = mock_result

        warning = sandbox._collect_reference_checksums()

        self.assertIsNotNone(warning)
        self.assertIn("harness execution failed", warning)
        self.assertIn("runtime failure", warning)

    @patch("src.discover.find_nvcc_path", return_value="nvcc")
    @patch("src.sandbox.subprocess.run")
    def test_compile_surfaces_reference_collection_warning(self, mock_run, mock_find_nvcc):
        from src.sandbox import CUDASandbox

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "kernel.cu")
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("__global__ void dft_kernel() {}\n")

            sandbox = CUDASandbox(source_path)

            mock_compile = MagicMock()
            mock_compile.returncode = 0
            mock_compile.stdout = ""
            mock_compile.stderr = ""

            mock_run.side_effect = [mock_compile, MagicMock(returncode=1, stdout="", stderr="runtime failure")]

            result = sandbox.compile()

        self.assertTrue(result["success"])
        self.assertIn("WARNING: Could not collect reference checksums", result["error_log"])

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    def test_llm_code_detection_accepts_preprocessor_directives(self, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        client = LLMClient(url="http://127.0.0.1:8080")

        self.assertTrue(client._looks_like_cuda_code("#include <cuda_runtime.h>\nint main() { return 0; }"))
        self.assertFalse(client._looks_like_cuda_code("This is just plain prose."))
        
     
        prose_with_keywords = (
            "Okay, let's see. The user's code failed because they tried to allocate memory "
            "using cudaMalloc but didn't cast the result properly on the host side. "
            "We should probably write float* d_A = (float*)malloc(size); or call "
            "cudaMemcpy. Let me analyze what went wrong."
        )
        self.assertFalse(client._looks_like_cuda_code(prose_with_keywords))

    @patch("src.llm_client.validate_llm_endpoint")
    @patch("src.llm_client.build_auth_headers")
    @patch("src.llm_client.requests.post")
    def test_llm_analyze_profile(self, mock_post, mock_headers, mock_validate):
        mock_validate.return_value = {"is_private": True, "is_secure": False}
        mock_headers.return_value = {}

        mock_response_llama = MagicMock()
        mock_response_llama.status_code = 200
        mock_response_llama.json.return_value = {"content": "Llama analysis results"}

        mock_response_ollama = MagicMock()
        mock_response_ollama.status_code = 200
        mock_response_ollama.json.return_value = {"response": "Ollama analysis results"}

        mock_post.return_value = mock_response_llama
        client_llama = LLMClient(url="http://127.0.0.1:8080")
        res_llama = client_llama.analyze_profile("dummy summary")
        self.assertEqual(res_llama, "Llama analysis results")

        mock_post.return_value = mock_response_ollama
        client_ollama = LLMClient(url="http://127.0.0.1:11434")
        res_ollama = client_ollama.analyze_profile("dummy summary")
        self.assertEqual(res_ollama, "Ollama analysis results")

    def test_ncu_csv_metadata_row_skipping(self):
        dummy_csv_content = [
            ["## Nsight Compute CSV Export"],
            ["## Version: 2024.1.0"],
            [],
            ["ID", "Process ID", "Kernel Name", "sm__cycles_elapsed.avg"],
            ["0", "1234", "vector_add_kernel", "4500.5"],
            ["1", "1234", "matrix_mul_kernel", "9500.2"],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_report.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(dummy_csv_content)

            hotspot = parse_ncu_csv_for_hotspot(csv_path)
            self.assertIsNotNone(hotspot)
            self.assertEqual(hotspot["kernel"], "matrix_mul_kernel")
            self.assertEqual(hotspot["value"], 9500.2)

            col_match = numeric_columns(csv_path, "sm__cycles_elapsed.avg")
            self.assertIsNotNone(col_match)
            idx, values = col_match
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0], 4500.5)
            self.assertEqual(values[1], 9500.2)

    @patch("src.discover.shutil.which")
    @patch("src.discover.os.name", "nt")
    @patch("src.discover.glob.glob")
    @patch("src.discover.os.path.exists")
    def test_nsight_fast_path_detection(self, mock_exists, mock_glob, mock_which):
        from src.discover import find_ncu_path, find_nsys_path

        mock_which.return_value = None
        mock_exists.return_value = True

        mock_glob.side_effect = lambda pattern, recursive=False: (
            ["C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2024.1"] if "Nsight Compute *" in pattern
            else ["C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2024.1"] if "Nsight Systems *" in pattern
            else ["C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2024.1\\target\\windows-desktop-win7-x64"] if "Compute" in pattern and "target" in pattern
            else ["C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2024.1\\target\\windows-desktop-win7-x64"] if "Systems" in pattern and "target" in pattern
            else []
        )

        ncu_path = find_ncu_path()
        self.assertIsNotNone(ncu_path)
        self.assertIn("Nsight Compute 2024.1", ncu_path)
        self.assertIn("ncu.exe", ncu_path.lower())

        nsys_path = find_nsys_path()
        self.assertIsNotNone(nsys_path)
        self.assertIn("Nsight Systems 2024.1", nsys_path)
        self.assertIn("nsys.exe", nsys_path.lower())

    @patch("src.discover.os.name", "nt")
    @patch("src.discover.os.path.exists")
    @patch("src.discover.glob.glob")
    @patch("src.discover.shutil.which")
    def test_nsight_recursive_fallback_prefers_newer_version(self, mock_which, mock_glob, mock_exists):
        from src.discover import find_ncu_path, find_nsys_path

        mock_which.return_value = None
        mock_exists.return_value = True

        def glob_side_effect(pattern, recursive=False):
            if "Nsight Compute*" in pattern or "Nsight Systems*" in pattern:
                return []
            if pattern.endswith("**\\ncu.exe") or pattern.endswith("**/ncu.exe"):
                return [
                    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3\target\windows-desktop\ncu.exe",
                    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1\target\windows-desktop\ncu.exe",
                ]
            if pattern.endswith("**\\nsys.exe") or pattern.endswith("**/nsys.exe"):
                return [
                    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.3\target\windows-desktop\nsys.exe",
                    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1\target\windows-desktop\nsys.exe",
                ]
            return []

        mock_glob.side_effect = glob_side_effect

        ncu_path = find_ncu_path()
        self.assertIn("2025.1", ncu_path)

        nsys_path = find_nsys_path()
        self.assertIn("2025.1", nsys_path)

    @patch("src.discover.find_ncu_path", return_value="ncu")
    @patch("src.profiler_tools.subprocess.run")
    def test_run_ncu_broad_uses_export_flag(self, mock_run, mock_find_ncu):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_ncu_broad("dummy.exe", output_base="sample_report", metrics="sm__cycles_elapsed.avg")

        cmd = mock_run.call_args.args[0]
        self.assertIn("-o", cmd)
        self.assertIn("sample_report", cmd)
        self.assertNotIn("--output", cmd)

    @patch("src.sandbox.find_ncu_path", return_value="ncu")
    @patch("src.sandbox.find_nsys_path", return_value=None)
    @patch("src.sandbox.subprocess.run")
    def test_sandbox_profile_latency_uses_export_flag(self, mock_run, mock_nsys, mock_ncu):
        from src.sandbox import CUDASandbox

        with tempfile.TemporaryDirectory() as tmpdir:
            exe_path = os.path.join(tmpdir, "kernel.exe")
            with open(exe_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")

            sandbox = CUDASandbox(exe_path, profile_mode="ncu")
            sandbox.exe_path = exe_path

            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("src.sandbox.os.path.exists", side_effect=lambda path: path == exe_path):
                sandbox.profile_latency()

        cmd = mock_run.call_args.args[0]
        self.assertIn("-o", cmd)
        self.assertNotIn("--output", cmd)

    @patch("src.discover.os.path.exists")
    @patch("src.discover.glob.glob")
    @patch("src.discover.shutil.which")
    def test_python_backend_discovery_prefers_tools_server(self, mock_which, mock_glob, mock_exists):
        from src.discover import find_llm_server_path

        mock_which.return_value = None
        def exists_side_effect(path):
            p = str(path).replace("\\", "/")
            return p.endswith("/project") or p.endswith("/project/tools/server.py")
        mock_exists.side_effect = exists_side_effect

        def glob_side_effect(pattern, recursive=False):
            return []

        mock_glob.side_effect = glob_side_effect

        result = find_llm_server_path(r"C:\project")
        self.assertTrue(result.replace("\\", "/").endswith("tools/server.py"))

    @patch("src.discover.find_ncu_path")
    @patch("src.profiler_tools.subprocess.run")
    @patch("src.profiler_tools.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_run_ncu_broad_exports_csv(self, mock_file, mock_exists, mock_run, mock_find_ncu):
        mock_find_ncu.return_value = "ncu"
        
        mock_exists.side_effect = lambda path: path.endswith(".ncu-rep")

        mock_profile_res = MagicMock()
        mock_profile_res.stdout = "profiling..."
        mock_profile_res.stderr = ""
        mock_profile_res.returncode = 0
        
        mock_export_res = MagicMock()
        mock_export_res.stdout = "CSV,headers,data\n1,2,3"
        mock_export_res.stderr = ""
        mock_export_res.returncode = 0
        
        mock_run.side_effect = [mock_profile_res, mock_export_res]
        
        from src.profiler_tools import run_ncu_broad
        res = run_ncu_broad("dummy.exe", output_base="test_report")
        
        self.assertEqual(res["csv"], "test_report.csv")
        self.assertEqual(res["basename"], "test_report")
        
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(mock_run.call_args_list[0][0][0], ["ncu", "--csv", "-o", "test_report", "dummy.exe"])
        self.assertEqual(mock_run.call_args_list[1][0][0], ["ncu", "--import", "test_report.ncu-rep", "--csv"])

        mock_file.assert_called_once_with("test_report.csv", "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("CSV,headers,data\n1,2,3")

    @patch("src.sandbox.find_ncu_path")
    @patch("src.sandbox.subprocess.run")
    @patch("src.sandbox.os.path.exists")
    @patch("src.profiler_tools.parse_ncu_csv_for_hotspot")
    @patch("builtins.open", new_callable=mock_open)
    def test_sandbox_ncu_profile_exports_csv(self, mock_file, mock_parse_hotspot, mock_exists, mock_run, mock_find_ncu):
        mock_find_ncu.return_value = "ncu"
        mock_parse_hotspot.return_value = {"kernel": "kernel_name", "value": 100.0}
        
        mock_exists.return_value = True
        
        mock_profile_res = MagicMock()
        mock_profile_res.stdout = "profiling..."
        mock_profile_res.stderr = ""
        mock_profile_res.returncode = 0
        
        mock_export_res = MagicMock()
        mock_export_res.stdout = "ID,Process ID,Kernel Name,sm__cycles_elapsed.avg\n0,123,kernel_name,100"
        mock_export_res.stderr = ""
        mock_export_res.returncode = 0
        
        mock_run.side_effect = [mock_profile_res, mock_export_res]
        
        from src.sandbox import CUDASandbox
        sandbox = CUDASandbox("dummy.cu", profile_mode="ncu")
        sandbox.exe_path = "dummy.exe"
        
        res = sandbox.profile_latency()
        self.assertEqual(res["latency"], 100.0)
        self.assertEqual(mock_run.call_count, 2)
        mock_parse_hotspot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
