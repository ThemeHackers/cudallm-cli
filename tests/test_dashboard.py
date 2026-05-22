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

    def test_api_roofline_fails_on_non_existent(self):
        url = f"http://localhost:{self.port}/api/roofline?path=does_not_exist.cu"
        resp = requests.get(url, timeout=5)
        self.assertEqual(resp.status_code, 404)

if __name__ == "__main__":
    unittest.main()
