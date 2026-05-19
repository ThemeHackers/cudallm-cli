import unittest

from click.testing import CliRunner

from src.cli import main


def get_option(command_name, option_name):
    command = main.commands[command_name]
    return next(param for param in command.params if param.name == option_name)


class CLIDefaultsTest(unittest.TestCase):
    def test_serve_default_filename_is_trimmed(self):
        file_option = get_option("serve", "file")

        self.assertEqual(file_option.default, "cudaLLM-8B.Q2_K.gguf")
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

        # Assert LLM overrides exist
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

        # 1. URL override
        c = apply_llm_overrides(config.copy(), llm_url="http://new-url:8080/completion")
        self.assertEqual(c["llm_url"], "http://new-url:8080/completion")

        # 2. API Key override (should clear out key file)
        c = apply_llm_overrides(config.copy(), llm_api_key="new-key")
        self.assertEqual(c["llm_api_key"], "new-key")
        self.assertIsNone(c["llm_api_key_file"])

        # 3. API Key File override (should clear out key)
        c = apply_llm_overrides(config.copy(), llm_api_key_file="new-file")
        self.assertEqual(c["llm_api_key_file"], "new-file")
        self.assertIsNone(c["llm_api_key"])

        # 4. Insecure override
        c = apply_llm_overrides(config.copy(), insecure=True)
        self.assertFalse(c["llm_verify_tls"])
        self.assertTrue(c["llm_allow_insecure_remote"])


if __name__ == "__main__":
    unittest.main()