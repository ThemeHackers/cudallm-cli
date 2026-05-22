import unittest
import threading
import time
import requests
import socket
import json
import os

from src.dashboard import start_dashboard_server, active_run

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
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
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cpu_usage", data)
        self.assertIn("ram_total", data)
        self.assertIn("ram_used", data)
        self.assertIn("gpu_model", data)
        self.assertIn("tools", data)

    def test_api_files(self):
        url = f"http://localhost:{self.port}/api/files"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("files", data)
     
        self.assertTrue(any("vector_add.cu" in f for f in data["files"]))

    def test_api_optimize_status(self):
        url = f"http://localhost:{self.port}/api/optimize/status"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["stage"], "Idle")
        self.assertIsNone(data["best_time"])

    def test_api_optimize_status_includes_profiling_fields(self):
        url = f"http://localhost:{self.port}/api/optimize/status"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("original_latency_profiling_failed", data)
        self.assertIn("original_latency_raw_output", data)

    def test_api_benchmark_history(self):
        url = f"http://localhost:{self.port}/api/benchmark/history"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("items", data)
        self.assertIsInstance(data["items"], list)

    def test_api_optimizer_presets(self):
        url = f"http://localhost:{self.port}/api/optimizer/presets"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)
        self.assertTrue(any(item["name"] == "balanced" for item in data["presets"]))

    def test_api_benchmark_replay_requires_fields(self):
        url = f"http://localhost:{self.port}/api/benchmark/replay"
        resp = requests.post(url, json={}, timeout=5)
        self.assertEqual(resp.status_code, 400)

    def test_profiling_failure_sentinel_remains_serializable(self):
        active_run["original_latency"] = 99999.0
        active_run["original_latency_profiling_failed"] = True
        active_run["original_latency_raw_output"] = "NCU profiling failed to generate CSV report. Output logs:\nexample"

        try:
            url = f"http://localhost:{self.port}/api/optimize/status"
            resp = requests.get(url, timeout=5)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["original_latency"], 99999.0)
            self.assertTrue(data["original_latency_profiling_failed"])
            self.assertIn("NCU profiling failed", data["original_latency_raw_output"])
        finally:
            active_run["original_latency"] = None
            active_run["original_latency_profiling_failed"] = False
            active_run["original_latency_raw_output"] = ""

    def test_api_roofline_fails_on_non_existent(self):
        url = f"http://localhost:{self.port}/api/roofline?path=does_not_exist.cu"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 404)

if __name__ == "__main__":
    unittest.main()
