# AGENTS.md — Guidance for AI coding agents

Purpose
- Provide concise, actionable guidance so AI coding agents can be immediately productive in this repository.

Quick setup
- Create and activate a virtualenv, install deps, install editable package:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
pip install -e .
```

Run tests
- Run the full test suite locally with:

```bash
python -m pytest -q
```

- Run a single test file:

```bash
python -m pytest tests/test_platform_info.py -q
```

Build
- Build source and wheel:

```bash
python setup.py sdist bdist_wheel
```

Where to look (quick links)
- Project overview: [README.md](README.md)
- Packaging/config: [pyproject.toml](pyproject.toml), [setup.py](setup.py), [requirements.txt](requirements.txt)
- Source code: [src/](src/)
- Important modules:
  - [src/cli.py](src/cli.py)
  - [src/llm_client.py](src/llm_client.py)
  - [src/platform_info.py](src/platform_info.py)
  - [src/profiler_tools.py](src/profiler_tools.py)
- Tests: [tests/](tests/)
- CI and developer scripts: [ci/](ci/), [scripts/](scripts/), [tools/](tools/)

Conventions and recommendations for agents
- Preserve and link to existing documentation rather than copy-pasting it.
- Run the test(s) relevant to any file you change before creating suggestions or patches.
- Prefer small, focused edits and include or update tests where behavior changes.
- If changing packaging or dependencies, update both `requirements.txt` and `pyproject.toml`/`setup.py` as appropriate and run the test suite.
- Do not modify `ci/baselines/` files or `ci` scripts without running the corresponding CI checks locally and documenting the reason.

Common developer tasks
- Run linters or formatters if present (none are enforced in repo). If you add tooling, include commands here.
- Use `scripts/check_env.ps1` on Windows to verify environment assumptions.

When in doubt
- Link to the relevant file(s) in a PR description and include the minimal reproducer for any bugfix.

Suggested next customizations
- Create a short `.github/copilot-instructions.md` or expand `skills/` with targeted skills for test-running and packaging checks.

---
Generated: AGENTS.md — added to help AI agents understand repository structure and workflow.
