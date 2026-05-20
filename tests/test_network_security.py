import tempfile
import unittest
from pathlib import Path

from src.network_security import build_auth_headers, is_private_network_host, validate_llm_endpoint


class NetworkSecurityTest(unittest.TestCase):
    def test_private_hosts_are_allowed(self):
        self.assertTrue(is_private_network_host("127.0.0.1"))
        self.assertTrue(is_private_network_host("192.168.1.10"))
        self.assertTrue(is_private_network_host("llama-gateway.local"))
        self.assertFalse(is_private_network_host("example.com"))
        self.assertFalse(is_private_network_host("llama-gateway"))

    def test_validate_llm_endpoint_blocks_public_http(self):
        with self.assertRaises(ValueError):
            validate_llm_endpoint("http://example.com:8080/completion")

    def test_validate_llm_endpoint_allows_private_http(self):
        info = validate_llm_endpoint("http://192.168.1.10:8080/completion")

        self.assertTrue(info["is_private"])
        self.assertFalse(info["is_secure"])

    def test_validate_llm_endpoint_allows_explicit_override(self):
        info = validate_llm_endpoint("http://example.com:8080/completion", allow_insecure_remote=True)

        self.assertFalse(info["is_private"])
        self.assertFalse(info["is_secure"])

    def test_build_auth_headers_from_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            key_file = Path(tmp_dir) / "api.key"
            key_file.write_text("sk-demo-secret\n", encoding="utf-8")

            headers = build_auth_headers(api_key_file=str(key_file))

        self.assertEqual(headers["Authorization"], "Bearer sk-demo-secret")

    def test_build_auth_headers_uses_first_non_empty_key_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            key_file = Path(tmp_dir) / "api.key"
            key_file.write_text("\n  sk-first-secret  \n\nsk-second-secret\n", encoding="utf-8")

            headers = build_auth_headers(api_key_file=str(key_file))

        self.assertEqual(headers["Authorization"], "Bearer sk-first-secret")


if __name__ == "__main__":
    unittest.main()