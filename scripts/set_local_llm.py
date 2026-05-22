#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from src.platform_info import get_config_path
from src.health_check import check_llm_health


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=1234)
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}"
    cfg = {
        "llm_url": url,
        "llm_api_key": None,
        "llm_verify_tls": False,
        "llm_allow_insecure_remote": True
    }

    cfg_path = Path(get_config_path())
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    ok, msg = check_llm_health(url)
    if ok:
        print(f"Wrote config to {cfg_path} and health-check OK: {msg}")
        print("You can now run the agent:")
        print("  python scripts/run_langchain_agent.py --instruction \"Compile example_kernel.cu and profile\"")
    else:
        print(f"Wrote config to {cfg_path} but health-check failed: {msg}")
        print("Please ensure LM Studio is running at the provided host/port and try again.")


if __name__ == '__main__':
    main()
