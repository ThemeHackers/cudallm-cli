
# run-tests SKILL

TL;DR
- This skill documents how AI agents should run and triage tests for local development and PR validation.

Purpose
- Help AI agents run tests quickly and reliably and follow a minimal diagnostic workflow when tests fail.

Environment setup (Windows)
- Create and activate a virtual environment, then install dependencies and the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Cross-platform (bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the full test suite

```bash
python -m pytest -q
```

Run a single test file

```bash
python -m pytest tests/test_platform_info.py -q
```

Run a subset by keyword

```bash
python -m pytest -k "keyword" -q
```

Recommended agent workflow for changed files
1) Identify changed file(s).
2) Find tests that import or exercise those modules (search the `tests/` folder).
3) Run the specific test file(s) or use `-k` to narrow the run.

CI and baseline notes
- CI may run additional checks not present locally; do not change files under `ci/baselines/` without CI verification.

Agent assertions and expectations
- Run the targeted tests locally before proposing code changes.
- When a test fails, include the exact command used and a short excerpt of failing output in the PR or patch description.
- Add or update tests for any behavioral change.

Quick links
- Tests folder: [tests/](tests/)
- Environment check script (Windows): [scripts/check_env.ps1](scripts/check_env.ps1)

Example agent checklist
- Create or activate venv and install deps.
- Run targeted tests for the changed module(s).
- If failures occur, capture minimal failing output, write a focused fix, and add/update tests.

Generated: English `run-tests` skill to help agents run and triage tests in this repository.
