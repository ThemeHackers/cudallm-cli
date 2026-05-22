import unittest
from unittest.mock import patch, MagicMock
import json
from langchain_core.messages import AIMessage, HumanMessage
from src.langchain_agent import (
    compile_cuda,
    profile_system,
    profile_kernel,
    summarize_profile,
    get_tools,
    LocalLMStudioChat,
    create_agent_with_llmclient,
)

class LangChainAgentTest(unittest.TestCase):
    @patch("src.langchain_agent.find_nvcc_path")
    @patch("src.langchain_agent.run_sandboxed")
    def test_compile_cuda_tool(self, mock_run, mock_find_nvcc):
        mock_find_nvcc.return_value = "nvcc"
        mock_run.return_value = {"rc": 0, "stdout": "ok", "stderr": ""}

        res_str = compile_cuda.invoke({"source_path": "kernel.cu", "out_path": "kernel.so"})
        res = json.loads(res_str)
        self.assertEqual(res["rc"], 0)
        self.assertEqual(res["stdout"], "ok")

    @patch("src.langchain_agent.find_nsys_path")
    @patch("src.langchain_agent.run_sandboxed")
    def test_profile_system_tool(self, mock_run, mock_find_nsys):
        mock_find_nsys.return_value = "nsys"
        mock_run.return_value = {"rc": 0, "stdout": "nsys-done", "stderr": ""}

        res_str = profile_system.invoke({"cmdline": "./binary", "output_base": "out"})
        res = json.loads(res_str)
        self.assertEqual(res["basename"], "out")
        self.assertIn("nsys-done", res["out"])

    @patch("src.langchain_agent.find_ncu_path")
    @patch("src.langchain_agent.run_sandboxed")
    def test_profile_kernel_tool(self, mock_run, mock_find_ncu):
        mock_find_ncu.return_value = "ncu"
        mock_run.return_value = {"rc": 0, "stdout": "ncu-done", "stderr": ""}

        res_str = profile_kernel.invoke({"cmdline": "./binary", "metrics": "sm__throughput"})
        res = json.loads(res_str)
        self.assertEqual(res["csv"], "ncu_agent_out.csv")
        self.assertIn("ncu-done", res["out"])

    def test_get_tools(self):
        tools = get_tools()
        names = [t.name for t in tools]
        self.assertIn("compile_cuda", names)
        self.assertIn("profile_system", names)
        self.assertIn("profile_kernel", names)
        self.assertIn("summarize_profile", names)

    @patch("openai.resources.chat.completions.Completions.create")
    def test_local_lm_studio_chat_generate(self, mock_create):
      
        mock_message = MagicMock()
        mock_message.content = "Response from model"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_create.return_value = mock_response

        chat = LocalLMStudioChat(base_url="http://localhost:1234/v1")
        messages = [HumanMessage(content="Hello")]
        result = chat._generate(messages)
        
        self.assertEqual(len(result.generations), 1)
        self.assertEqual(result.generations[0].message.content, "Response from model")

    def test_create_agent_with_llmclient(self):
        mock_llm_client = MagicMock()
        mock_llm_client.url = "http://localhost:1234/v1/completions"
        mock_llm_client.request_headers = {"Authorization": "Bearer test-api-key"}

        agent = create_agent_with_llmclient(mock_llm_client)
        self.assertIsNotNone(agent)
    
        self.assertTrue(hasattr(agent, "invoke"))

if __name__ == "__main__":
    unittest.main()
