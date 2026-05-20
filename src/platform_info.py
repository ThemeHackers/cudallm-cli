"""
Platform detection and cross-platform path utilities for cudallm-cli.

Centralizes all platform-specific logic so the rest of the codebase
can call simple helpers instead of scattering ``os.name == 'nt'`` checks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def detect_platform() -> str:
    """Return a short platform tag: ``"windows"``, ``"colab"``, or ``"linux"``."""
    if is_colab():
        return "colab"
    if os.name == "nt":
        return "windows"
    return "linux"


def is_colab() -> bool:
    """Detect whether we are running inside a Google Colab notebook."""
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    if os.path.isdir("/content") and os.path.isfile("/etc/os-release"):
        try:
            with open("/etc/os-release") as f:
                if "colab" in f.read().lower():
                    return True
        except Exception:
            pass
    return False


def is_wsl() -> bool:
    """Detect whether we are running inside Windows Subsystem for Linux."""
    if os.name == "nt":
        return False
    try:
        with open("/proc/version") as f:
            version_text = f.read().lower()
            return "microsoft" in version_text or "wsl" in version_text
    except Exception:
        return False


def is_windows() -> bool:
    """Return ``True`` on native Windows (not WSL)."""
    return os.name == "nt"


def is_linux() -> bool:
    """Return ``True`` on any Linux variant (including Colab and WSL)."""
    return os.name != "nt"


_APP_DIR_NAME = ".cudallm"


def get_home_dir() -> Path:
    """Return the user's home directory."""
    return Path.home()


def get_config_dir() -> Path:
    """
    Return the directory used for persistent configuration.

    Layout::

        ~/.cudallm/
            config.json
            cache/          (downloaded binaries, models)
    """
    return get_home_dir() / _APP_DIR_NAME


def get_config_path() -> Path:
    """Return the path to the main JSON configuration file."""
    return get_config_dir() / "config.json"


def get_cache_dir() -> Path:
    """Return the directory used for cached downloads (models, backend assets, etc.)."""
    return get_config_dir() / "cache"


def get_legacy_config_path() -> Path | None:
    """
    Return the legacy project-relative config path if it exists,
    otherwise ``None``.
    """
    project_dir = Path(__file__).resolve().parent.parent
    legacy = project_dir / "config" / "config.json"
    if legacy.is_file():
        return legacy
    return None


def get_exe_extension() -> str:
    """Return ``".exe"`` on Windows, ``""`` on Linux/Colab."""
    return ".exe" if is_windows() else ""


def get_compiled_binary_name(run_id: str) -> str:
    """Return the temp binary filename with the correct platform extension."""
    ext = ".exe" if is_windows() else ".out"
    return f"./temp_cuda_kernel_{run_id}{ext}"


def get_default_llm_port() -> int:
    """
    Return the conventional default port for the local LLM server.

    On Colab we use 8081 to avoid potential clashes with Jupyter's
    built-in services; on Windows/Linux we default to 8080.
    """
    return 8081 if is_colab() else 8080


def platform_display_name() -> str:
    """Return a human-readable platform string for CLI output."""
    plat = detect_platform()
    labels = {
        "windows": "Windows (Native)",
        "colab": "Google Colab (Linux)",
        "linux": "Linux",
    }
    label = labels.get(plat, plat)
    if plat == "linux" and is_wsl():
        label = "WSL (Windows Subsystem for Linux)"
    return label
