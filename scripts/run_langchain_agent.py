#!/usr/bin/env python3
import argparse
import json
from src.langchain_agent import create_langchain_agent_example, create_agent_with_llmclient, get_tools
from src.cli import load_config, create_llm_client
from src.health_check import check_llm_health


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="Compile src/example.cu and profile it with nsys and ncu")
    args = p.parse_args()

    try:
        config = load_config()
        llm_client = create_llm_client(config)
        ok, msg = check_llm_health(config.get('llm_url'))
        if not ok:
            raise RuntimeError(f"LLM health-check failed: {msg}")
        agent = create_agent_with_llmclient(llm_client)
    except Exception as e:
        print("Failed to create agent with local LLMClient:", e)
        print("Falling back to generic LangChain LLM (OpenAI) if available...")
        try:
            agent = create_langchain_agent_example()
        except Exception as e2:
            print("Failed to create agent:", e2)
            print("Available lightweight tools (no langchain required):")
            for t in get_tools():
                print(f"- {t.name}: {t.description}")
            return

    response = agent.run(args.instruction)
    print("Agent response:\n", response)


if __name__ == '__main__':
    main()
