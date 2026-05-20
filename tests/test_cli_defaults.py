import unittest

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

    def test_help_uses_click_help_and_lists_new_commands(self):
        result = CliRunner().invoke(main, ["help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertNotIn("[INFO]", result.output)
        self.assertIn("doctor", result.output)
        self.assertIn("check", result.output)

    def test_optimize_and_expert_flags_exist(self):
        optimize_options = {param.name for param in main.commands["optimize"].params}
        expert_options = {param.name for param in main.commands["expert"].params}
        serve_options = {param.name for param in main.commands["serve"].params}
        audit_options = {param.name for param in main.commands["audit"].params}

        self.assertIn("dry_run", optimize_options)
        self.assertIn("ncu_metrics", optimize_options)
        self.assertIn("dry_run", expert_options)
        self.assertIn("rerun", expert_options)
        self.assertIn("host", serve_options)
        self.assertIn("public_url", serve_options)
        self.assertIn("api_key_file", serve_options)
        self.assertIn("ssl_key_file", serve_options)
        self.assertIn("ssl_cert_file", serve_options)
        self.assertIn("allow_unsafe_network", serve_options)
        self.assertIn("no_update", serve_options)

       
        for opt in ["llm_url", "llm_api_key", "llm_api_key_file", "insecure"]:
            self.assertIn(opt, optimize_options)
            self.assertIn(opt, expert_options)
            self.assertIn(opt, audit_options)

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

    def test_check_and_prepare_python_server_uses_existing_script(self):
        from src.cli import check_and_prepare_python_server
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            server_bin = os.path.join(tmpdir, "tools", "server.py")
            os.makedirs(os.path.dirname(server_bin), exist_ok=True)
            with open(server_bin, "w", encoding="utf-8") as f:
                f.write("dummy")

            config = {
                "llm_server_path": server_bin
            }

            path = check_and_prepare_python_server(tmpdir, config)
            self.assertEqual(path, server_bin)
            self.assertNotIn("llama_version", config)


if __name__ == "__main__":
    unittest.main()