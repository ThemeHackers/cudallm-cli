# packaging-checks SKILL

TL;DR
- This skill documents how AI agents should validate packaging, dependencies, and distribution builds for this project.

Purpose
- Help AI agents ensure packaging changes are correct, dependencies are synchronized, and builds succeed before proposing changes.

Key packaging files
- `setup.py` - Build configuration and metadata
- `pyproject.toml` - Modern Python packaging configuration
- `requirements.txt` - Runtime dependencies
- `README.md` - Project documentation (may contain install instructions)

Build source and wheel distributions

```bash
python setup.py sdist bdist_wheel
```

Verify build artifacts

```bash
# Check that dist/ directory contains expected files
ls dist/
```

Packaging validation checklist
1. **Dependency synchronization**: Ensure `requirements.txt`, `setup.py` (install_requires), and `pyproject.toml` (dependencies) are consistent
2. **Version consistency**: Verify version numbers match across `setup.py`, `pyproject.toml`, and `__version__` in source code
3. **Build success**: Run `python setup.py sdist bdist_wheel` and confirm no errors
4. **Metadata completeness**: Check that author, description, license, and classifiers are present in `setup.py`
5. **README validation**: Ensure installation instructions in README.md match actual package structure

Common packaging tasks
- **Add a new dependency**: Update all three files (requirements.txt, setup.py, pyproject.toml)
- **Bump version**: Update version in setup.py, pyproject.toml, and source code __version__
- **Change package structure**: Update packages/ or py_modules in setup.py if needed
- **Update classifiers**: Add or remove Trove classifiers in setup.py

Agent workflow for packaging changes
1. Identify which packaging files need modification
2. Make synchronized changes across all relevant files
3. Run build command to verify: `python setup.py sdist bdist_wheel`
4. Run test suite to ensure no regressions: `python -m pytest -q`
5. If tests pass, propose the changes with a note about which files were updated

Cross-platform notes
- On Windows, use the same commands (Python handles platform differences)
- Build artifacts will be in `dist/` directory regardless of OS

Troubleshooting
- **Build fails with missing module**: Check that `packages` or `py_modules` in setup.py correctly references the source
- **Dependency conflict**: Use `pip check` to identify conflicts between installed packages
- **Version mismatch**: Search for `__version__` assignments in source code and ensure consistency

Quick links
- Packaging files: [setup.py](setup.py), [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt)
- Build guide: [AGENTS.md](AGENTS.md) (Build section)

Example agent checklist
- [ ] Updated requirements.txt with new dependency
- [ ] Updated setup.py install_requires with same dependency
- [ ] Updated pyproject.toml dependencies with same dependency
- [ ] Ran `python setup.py sdist bdist_wheel` successfully
- [ ] Ran `python -m pytest -q` to verify no test regressions
- [ ] Documented the change in commit message or PR description

Generated: packaging-checks skill to help agents validate packaging and dependency changes in this repository.
