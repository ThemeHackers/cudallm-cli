import tempfile
import unittest
import os
from pathlib import Path

from src.network_security import build_auth_headers, generate_secure_dashboard_token, generate_secure_dashboard_token_openssl, get_secure_dashboard_token, is_private_network_host, load_dotenv_file, upsert_env_value, validate_llm_endpoint


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

    def test_load_dotenv_file_reads_values_without_overriding_existing_env(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("CUDALLM_DASHBOARD_TOKEN=from_file\nSAMPLE_VALUE=hello\n", encoding="utf-8")

            original = os.environ.get("CUDALLM_DASHBOARD_TOKEN")
            try:
                os.environ["CUDALLM_DASHBOARD_TOKEN"] = "from_env"
                cwd = os.getcwd()
                os.chdir(tmp_dir)
                try:
                    loaded = load_dotenv_file()
                finally:
                    os.chdir(cwd)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["CUDALLM_DASHBOARD_TOKEN"], "from_env")
                self.assertEqual(os.environ["SAMPLE_VALUE"], "hello")
            finally:
                if original is None:
                    os.environ.pop("CUDALLM_DASHBOARD_TOKEN", None)
                else:
                    os.environ["CUDALLM_DASHBOARD_TOKEN"] = original
                os.environ.pop("SAMPLE_VALUE", None)

    def test_upsert_env_value_writes_token_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            upsert_env_value(env_path, "CUDALLM_DASHBOARD_TOKEN", "token-value")

            content = env_path.read_text(encoding="utf-8")

        self.assertIn("CUDALLM_DASHBOARD_TOKEN=token-value", content)

    def test_get_secure_dashboard_token_returns_urlsafe_value(self):
        token = get_secure_dashboard_token()

        self.assertGreaterEqual(len(token), 32)
        self.assertNotIn(" ", token)
        self.assertNotIn("/", token)

    def test_generate_secure_dashboard_token_openssl_returns_urlsafe_value(self):
        token = generate_secure_dashboard_token_openssl()

        self.assertGreaterEqual(len(token), 32)
        self.assertNotIn(" ", token)
        self.assertNotIn("+", token)
        self.assertNotIn("/", token)

    def test_generate_secure_dashboard_token_prefers_openssl(self):
        token = generate_secure_dashboard_token(prefer_openssl=True)

        self.assertGreaterEqual(len(token), 32)
        self.assertNotIn(" ", token)


if __name__ == "__main__":
    unittest.main()