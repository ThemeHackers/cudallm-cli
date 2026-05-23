import unittest
import threading
import time
import requests
import socket
import json
import os
import tempfile

from src.dashboard import DASHBOARD_API_TOKEN, active_run, active_run_lock, start_dashboard_server

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class DashboardServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = find_free_port()
        cls.server_thread = threading.Thread(
            target=start_dashboard_server,
            args=(cls.port,)
        )
        cls.server_thread.daemon = True
        cls.server_thread.start()
       
        time.sleep(1.0)

    def api_headers(self, extra=None):
        headers = {"X-Cudallm-Token": DASHBOARD_API_TOKEN}
        if extra:
            headers.update(extra)
        return headers

    def test_serve_index_html(self):
        url = f"http://localhost:{self.port}/"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("CUDA LLM Optimizer Dashboard", resp.text)

    def test_serve_favicon(self):
        url = f"http://localhost:{self.port}/favicon.ico"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "image/x-icon")

        url_png = f"http://localhost:{self.port}/favicon.png"
        resp_png = requests.get(url_png, timeout=5)
        self.assertEqual(resp_png.status_code, 200)
        self.assertEqual(resp_png.headers.get("Content-Type"), "image/png")

        url_svg = f"http://localhost:{self.port}/favicon.svg"
        resp_svg = requests.get(url_svg, timeout=5)
        self.assertEqual(resp_svg.status_code, 200)
        self.assertEqual(resp_svg.headers.get("Content-Type"), "image/svg+xml")

    def test_api_status(self):
        url = f"http://localhost:{self.port}/api/status"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cpu_usage", data)
        self.assertIn("ram_total", data)
        self.assertIn("ram_used", data)
        self.assertIn("gpu_model", data)
        self.assertIn("tools", data)

    def test_api_status_includes_profile_mode_label(self):
        with active_run_lock:
            active_run["profile_mode"] = "auto-relaxed"

        try:
            url = f"http://localhost:{self.port}/api/status"
            resp = requests.get(url, headers=self.api_headers(), timeout=5)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["current_profile_mode"], "auto-relaxed")
            self.assertIn("auto-relaxed", data["current_profile_mode_label"])
        finally:
            with active_run_lock:
                active_run["profile_mode"] = "auto"

    def test_api_files(self):
        url = f"http://localhost:{self.port}/api/files"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("files", data)
        self.assertIn("sources", data)
        self.assertTrue(any(source["key"] == "examples" for source in data["sources"]))
        self.assertTrue(any(source["key"] == "optimized" for source in data["sources"]))
     
        self.assertTrue(any("vector_add.cu" in f for f in data["files"]))

    def test_api_files_includes_custom_source_directory(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            source_file = os.path.join(temp_dir, "custom_kernel.cu")
            with open(source_file, "w", encoding="utf-8") as handle:
                handle.write("__global__ void custom_kernel() {}\n")

            previous_value = os.environ.get("CUDALLM_DASHBOARD_SOURCE_DIRS")
            relative_root = os.path.relpath(temp_dir, os.getcwd()).replace("\\", "/")
            os.environ["CUDALLM_DASHBOARD_SOURCE_DIRS"] = relative_root
            try:
                url = f"http://localhost:{self.port}/api/files"
                resp = requests.get(url, headers=self.api_headers(), timeout=5)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(any(source["key"] == relative_root for source in data["sources"]))
                self.assertTrue(any(source["root"] == relative_root for source in data["sources"]))
                self.assertTrue(any("custom_kernel.cu" in f for f in data["files"]))
            finally:
                if previous_value is None:
                    os.environ.pop("CUDALLM_DASHBOARD_SOURCE_DIRS", None)
                else:
                    os.environ["CUDALLM_DASHBOARD_SOURCE_DIRS"] = previous_value

    def test_api_optimize_status(self):
        url = f"http://localhost:{self.port}/api/optimize/status"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["stage"], "Idle")
        self.assertIsNone(data["best_time"])

    def test_api_optimize_status_includes_profiling_fields(self):
        url = f"http://localhost:{self.port}/api/optimize/status"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotIn("original_code", data)
        self.assertNotIn("best_code", data)
        self.assertNotIn("current_code", data)
        self.assertNotIn("original_latency_raw_output", data)

    def test_cross_origin_post_rejected_without_token(self):
        url = f"http://localhost:{self.port}/api/file"
        resp = requests.post(
            url,
            headers={"Origin": "https://attacker.example"},
            json={"path": "examples/a.cu", "content": "x"},
            timeout=5,
        )
        self.assertIn(resp.status_code, (401, 403))

    def test_post_rejected_without_token(self):
        url = f"http://localhost:{self.port}/api/file"
        resp = requests.post(
            url,
            json={"path": "examples/a.cu", "content": "x"},
            timeout=5,
        )
        self.assertEqual(resp.status_code, 401)

    def test_rejects_path_escape_with_sibling_prefix(self):
        url = f"http://localhost:{self.port}/api/file"
        resp = requests.post(
            url,
            headers=self.api_headers(),
            json={"path": "examples/../../cudallm-cli-poc/owned.txt", "content": "x"},
            timeout=5,
        )
        self.assertEqual(resp.status_code, 403)

    def test_status_redacts_code_fields(self):
        url = f"http://localhost:{self.port}/api/optimize/status"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertNotIn("original_code", body)
        self.assertNotIn("best_code", body)
        self.assertNotIn("current_code", body)

    def test_history_redacts_code_snapshots(self):
        url = f"http://localhost:{self.port}/api/benchmark/history"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("code_snapshot", resp.text)
        self.assertNotIn("parent_code_snapshot", resp.text)

    def test_api_benchmark_history(self):
        url = f"http://localhost:{self.port}/api/benchmark/history"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("items", data)
        self.assertIsInstance(data["items"], list)

    def test_benchmark_summary_includes_profile_mode_label(self):
        from src.dashboard import build_benchmark_summary

        run_state = {
            "run_id": "test-run",
            "status": "completed",
            "stage": "Done",
            "input_file": "examples/vector_add.cu",
            "target": "latency",
            "preset": "balanced",
            "profile_mode": "auto-strict",
            "profile_metrics": "",
            "flags": [],
            "total_iterations": 1,
            "best_time": 1.23,
            "original_latency": 2.34,
            "regression_guard": {"max_regression_pct": 5.0},
            "history": [
                {
                    "iteration": 1,
                    "latency": 1.23,
                    "compile_success": True,
                    "profiling_failed": False,
                    "status": "success",
                }
            ],
        }

        summary = build_benchmark_summary(run_state)
        self.assertEqual(summary["profile_mode_label"], "auto-strict (benchmark-only)")

    def test_api_optimizer_presets(self):
        url = f"http://localhost:{self.port}/api/optimizer/presets"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)
        self.assertTrue(any(item["name"] == "balanced" for item in data["presets"]))

    def test_api_benchmark_replay_requires_fields(self):
        url = f"http://localhost:{self.port}/api/benchmark/replay"
        resp = requests.post(url, headers=self.api_headers(), json={}, timeout=5)
        self.assertEqual(resp.status_code, 400)

    def test_profiling_failure_sentinel_remains_serializable(self):
        active_run["original_latency"] = 99999.0
        active_run["original_latency_profiling_failed"] = True
        active_run["original_latency_raw_output"] = "NCU profiling failed to generate CSV report. Output logs:\nexample"

        try:
            url = f"http://localhost:{self.port}/api/optimize/status"
            resp = requests.get(url, headers=self.api_headers(), timeout=5)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["original_latency"], 99999.0)
            self.assertTrue(data["original_latency_profiling_failed"])
            self.assertNotIn("original_latency_raw_output", data)
        finally:
            active_run["original_latency"] = None
            active_run["original_latency_profiling_failed"] = False
            active_run["original_latency_raw_output"] = ""

    def test_api_roofline_fails_on_non_existent(self):
        url = f"http://localhost:{self.port}/api/roofline?path=examples/does_not_exist.cu"
        resp = requests.get(url, headers=self.api_headers(), timeout=5)
        self.assertEqual(resp.status_code, 404)

if __name__ == "__main__":
    unittest.main()
