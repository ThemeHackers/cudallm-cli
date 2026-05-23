import unittest
import io
import tempfile
import os
import zipfile
import tarfile
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import main


def get_option(command_name, option_name):
    command = main.commands[command_name]
    return next(param for param in command.params if param.name == option_name)


class CLIDefaultsTest(unittest.TestCase):
    def test_serve_default_filename_is_trimmed(self):
        file_option = get_option("serve", "file")

        self.assertEqual(file_option.default, "cudaLLM-8B.Q4_K_M.gguf")
        self.assertFalse(str(file_option.default).startswith(" "))

    def test_max_stream_chunks_default_is_raised(self):
        from src.cli import DEFAULT_CONFIG

        self.assertEqual(DEFAULT_CONFIG["max_stream_chunks"], 8000)

    def test_profiling_failure_latency_sentinel(self):
        from src.cli import PROFILING_FAILURE_LATENCY, is_profiling_failure

        self.assertEqual(PROFILING_FAILURE_LATENCY, 99999.0)
        self.assertTrue(is_profiling_failure(99999.0))
        self.assertFalse(is_profiling_failure(1.0))

    def test_help_uses_click_help_and_lists_new_commands(self):
        result = CliRunner().invoke(main, ["help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertNotIn("[INFO]", result.output)
        self.assertIn("doctor", result.output)
        self.assertIn("check", result.output)
        self.assertIn("setup-gpu", result.output)
        self.assertIn("sandbox-run", result.output)
        self.assertIn("profile", result.output)
        self.assertIn("--allow-unsafe-network", result.output)
        self.assertIn("--llm-api-key-file", result.output)

    def test_optimize_and_expert_flags_exist(self):
        optimize_options = {param.name for param in main.commands["optimize"].params}
        expert_options = {param.name for param in main.commands["expert"].params}
        serve_options = {param.name for param in main.commands["serve"].params}
        audit_options = {param.name for param in main.commands["audit"].params}
        agent_options = {param.name for param in main.commands["agent"].params}
        dashboard_options = {param.name for param in main.commands["dashboard"].params}

        self.assertIn("dry_run", optimize_options)
        self.assertIn("ncu_metrics", optimize_options)
        self.assertIn("dry_run", expert_options)
        self.assertIn("port", dashboard_options)

        self.assertIn("rerun", expert_options)
        self.assertIn("host", serve_options)
        self.assertIn("public_url", serve_options)
        self.assertIn("api_key_file", serve_options)
        self.assertIn("ssl_key_file", serve_options)
        self.assertIn("ssl_cert_file", serve_options)
        self.assertIn("allow_unsafe_network", serve_options)
        self.assertIn("no_update", serve_options)
        self.assertIn("instruction", agent_options)

       
        for opt in ["llm_url", "llm_api_key", "llm_api_key_file", "insecure"]:
            self.assertIn(opt, optimize_options)
            self.assertIn(opt, expert_options)
            self.assertIn(opt, audit_options)
            self.assertIn(opt, agent_options)

    def test_optimize_profile_mode_supports_strict_and_relaxed_auto(self):
        profile_mode_option = get_option("optimize", "profile_mode")
        self.assertIn("auto-strict", profile_mode_option.type.choices)
        self.assertIn("auto-relaxed", profile_mode_option.type.choices)

    def test_effective_latency_for_selection_uses_fallback_in_relaxed_mode(self):
        from src.cli import effective_latency_for_selection, PROFILING_FAILURE_LATENCY

        strict_latency = effective_latency_for_selection(
            PROFILING_FAILURE_LATENCY,
            {"fallback_latency": 1.5},
            "auto-strict",
        )
        self.assertIsNone(strict_latency)

        relaxed_latency = effective_latency_for_selection(
            PROFILING_FAILURE_LATENCY,
            {"fallback_latency": 1.5},
            "auto-relaxed",
        )
        self.assertEqual(relaxed_latency, 1.5)

    def test_profile_mode_label_is_readable(self):
        from src.cli import profile_mode_label

        self.assertEqual(profile_mode_label("auto-strict"), "auto-strict (benchmark-only)")
        self.assertEqual(profile_mode_label("auto-relaxed"), "auto-relaxed (fallback ranking enabled)")
        self.assertEqual(profile_mode_label("ncu"), "ncu (Nsight Compute)")

    def test_build_report_filename_is_mode_specific(self):
        from src.cli import build_report_filename

        self.assertEqual(build_report_filename("examples/vector_add.cu", "auto-strict"), "report_vector_add_auto-strict.cu.json")
        self.assertEqual(build_report_filename("examples/vector_add.cu", "auto-relaxed"), "report_vector_add_auto-relaxed.cu.json")
        self.assertEqual(build_report_filename("examples/vector_add.cu", "ncu"), "report_vector_add_ncu.cu.json")

    def test_profiler_failure_display_suppresses_result_in_strict_mode(self):
        from src.cli import build_profiler_failure_panel

        panel = build_profiler_failure_panel(
            99999.0,
            {"raw_output": "NCU profiling failed to generate CSV report."},
            1.23,
            "auto-strict",
        )

        self.assertEqual(panel.title, "Profiler Status")
        body_text = panel.renderable.plain
        self.assertIn("No usable profiler result", body_text)
        self.assertIn("Latency: unavailable", body_text)

    def test_profiler_failure_display_shows_fallback_only_in_relaxed_mode(self):
        from src.cli import build_profiler_failure_panel

        panel = build_profiler_failure_panel(
            99999.0,
            {"raw_output": "fallback", "fallback_latency": 2.5},
            1.23,
            "auto-relaxed",
        )

        self.assertEqual(panel.title, "Profiler Status")
        body_text = panel.renderable.plain
        self.assertIn("Fallback timing available", body_text)
        self.assertIn("2.5000 ms", body_text)

    def test_apply_llm_overrides(self):
        from src.cli import apply_llm_overrides
        config = {
            "llm_url": "http://127.0.0.1:8080/completion",
            "llm_api_key": "old-key",
            "llm_api_key_file": "old-file",
            "llm_verify_tls": True,
            "llm_allow_insecure_remote": False,
        }

        c = apply_llm_overrides(config.copy(), llm_url="http://new-url:8080/completion")
        self.assertEqual(c["llm_url"], "http://new-url:8080/completion")

    
        c = apply_llm_overrides(config.copy(), llm_api_key="new-key")
        self.assertEqual(c["llm_api_key"], "new-key")
        self.assertIsNone(c["llm_api_key_file"])

     
        c = apply_llm_overrides(config.copy(), llm_api_key_file="new-file")
        self.assertEqual(c["llm_api_key_file"], "new-file")
        self.assertIsNone(c["llm_api_key"])

       
        c = apply_llm_overrides(config.copy(), insecure=True)
        self.assertFalse(c["llm_verify_tls"])
        self.assertTrue(c["llm_allow_insecure_remote"])

    def test_apply_public_url_override_validates_endpoint(self):
        from src.cli import apply_public_url_override

        config = {
            "llm_url": "http://127.0.0.1:8080/completion",
            "llm_verify_tls": True,
            "llm_allow_insecure_remote": False,
        }

        updated = apply_public_url_override(config.copy(), "http://192.168.1.10:8080/completion")
        self.assertEqual(updated["llm_url"], "http://192.168.1.10:8080/completion")
        self.assertFalse(updated["llm_verify_tls"])

        with self.assertRaises(ValueError):
            apply_public_url_override(config.copy(), "http://example.com:8080/completion")


    def test_probe_llm_server_uses_tls_verification_for_https(self):
        from unittest.mock import patch
        from src.cli import _probe_llm_server

        with patch("src.cli.requests.get") as mock_get:
            _probe_llm_server("https://example.com/v1/models", {"Authorization": "Bearer x"}, verify_tls=True)

        self.assertEqual(mock_get.call_args.kwargs["verify"], True)

    def test_probe_llm_server_can_skip_verification_for_http(self):
        from unittest.mock import patch
        from src.cli import _probe_llm_server

        with patch("src.cli.requests.get") as mock_get:
            _probe_llm_server("http://127.0.0.1:1234/v1/models", {}, verify_tls=False)

        self.assertEqual(mock_get.call_args.kwargs["verify"], False)

    def test_setup_gpu_command_flags(self):
        setup_options = {param.name for param in main.commands["setup-gpu"].params}
        self.assertIn("dry_run", setup_options)
        self.assertIn("force_reinstall", setup_options)

    def test_setup_gpu_dry_run_execution(self):
        from unittest.mock import patch
        
        mock_env = {
            'nvcc_found': True,
            'compute_capability': '7.5',
            'cuda_version': '13.0'
        }
        
        with patch("src.cli.check_environment", return_value=mock_env):
            result = CliRunner().invoke(main, ["setup-gpu", "--dry-run"])
            
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("CUDA Environment check passed!", result.output)
        self.assertIn("bypassed", result.output)

    @patch("src.cli.console.print")
    @patch("src.cli.console.status")
    @patch("src.cli.Live")
    @patch("src.cli.IterationDashboard")
    @patch("src.cli.CUDASandbox")
    @patch("src.cli.create_llm_client")
    @patch("src.cli.refresh_config_paths")
    @patch("src.cli.check_environment")
    @patch("src.cli.locate_and_setup_msvc")
    def test_optimize_initial_generation_uses_prefill(
        self,
        mock_msvc,
        mock_check_environment,
        mock_refresh_config_paths,
        mock_create_llm_client,
        mock_sandbox_class,
        mock_dashboard_class,
        mock_live_class,
        mock_status,
        mock_print,
    ):
        mock_msvc.return_value = True
        mock_check_environment.return_value = {
            "nvcc_found": True,
            "compute_capability": "7.5",
            "cuda_version": "13.0",
            "gpu_model": "RTX 2060",
        }
        mock_refresh_config_paths.return_value = {
            "llm_url": "http://127.0.0.1:1234/v1/completions",
            "llm_api_key": None,
            "llm_api_key_file": None,
            "llm_verify_tls": True,
            "llm_allow_insecure_remote": False,
            "max_stream_chunks": 8000,
        }

        mock_llm = MagicMock()
        mock_llm.url = "http://127.0.0.1:1234/v1/completions"
        mock_llm.total_tokens = 5
        mock_llm.create_optimization_prompt.return_value = "<|im_start|>system\nTest\n<|im_start|>assistant\n"
        mock_llm.generate_code.return_value = ("```cuda\n__global__ void vectorAdd() {}\n```", 0.1)
        mock_create_llm_client.return_value = mock_llm

        sandbox = MagicMock()
        sandbox.compile.return_value = {"success": True, "error_log": ""}
        sandbox.profile_latency.return_value = {"latency": 1.23, "raw_output": ""}
        mock_sandbox_class.return_value = sandbox

        dashboard = MagicMock()
        mock_dashboard_class.return_value = dashboard

        live_cm = MagicMock()
        live_cm.__enter__.return_value = MagicMock(stop=MagicMock(), start=MagicMock())
        live_cm.__exit__.return_value = False
        mock_live_class.return_value = live_cm

        mock_status_cm = MagicMock()
        mock_status_cm.__enter__.return_value = None
        mock_status_cm.__exit__.return_value = False
        mock_status.return_value = mock_status_cm

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "kernel.cu")
            output_path = os.path.join(tmp_dir, "out.cu")
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write("__global__ void vectorAdd() {}\n")

            from src.cli import optimize_single_file

            optimize_single_file(
                input_path,
                output_path,
                iters=1,
                target="latency",
                retries=1,
                fast_math=False,
                opt_level="3",
                report=False,
                llm=mock_llm,
                env_info={"gpu_model": "RTX 2060", "compute_capability": "7.5", "cuda_version": "13.0"},
            )

        self.assertTrue(mock_llm.generate_code.called)
        self.assertTrue(mock_llm.generate_code.call_args.kwargs.get("prefill"))


if __name__ == "__main__":
    unittest.main()