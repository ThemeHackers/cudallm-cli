#!/usr/bin/env python3
import argparse
import json
from src.feedback_pipeline import run_feedback_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="CUDA source file to compile (e.g., kernel.cu)")
    p.add_argument("--verify-cmd", default="python verify.py", help="Command to run that executes the compiled kernel")
    p.add_argument("--metrics", default=None, help="Optional metrics string to pass to ncu")
    args = p.parse_args()

    report = run_feedback_pipeline(args.source, verify_cmd=args.verify_cmd, metrics=args.metrics)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
