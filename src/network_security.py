from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlparse


_LOCAL_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
}

_INTERNAL_HOST_SUFFIXES = (".local", ".lan", ".internal", ".home.arpa")


def _normalize_host(host: str | None) -> str:
    if not host:
        return ""
    return host.strip().strip("[]").lower()


def is_private_network_host(host: str | None) -> bool:
    normalized = _normalize_host(host)
    if not normalized:
        return False

    if normalized in _LOCAL_HOSTNAMES:
        return True

    if normalized.endswith(_INTERNAL_HOST_SUFFIXES):
        return True

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False

    return address.is_private or address.is_loopback or address.is_link_local


def validate_llm_endpoint(url: str, allow_insecure_remote: bool = False) -> dict[str, str | bool]:
    parsed = urlparse(url)
    host = _normalize_host(parsed.hostname)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported LLM URL scheme: {parsed.scheme or '(missing)'}")
    if not host:
        raise ValueError("LLM URL must include a host")

    is_private = is_private_network_host(host)
    if parsed.scheme == "http" and not is_private and not allow_insecure_remote:
        raise ValueError(
            "Refusing to use plain HTTP for a non-private LLM endpoint. "
            "Use HTTPS or pass allow_insecure_remote=True."
        )

    return {
        "scheme": parsed.scheme,
        "host": host,
        "is_private": is_private,
        "is_secure": parsed.scheme == "https",
    }


def load_api_key(api_key: str | None = None, api_key_file: str | None = None) -> str | None:
    if api_key and api_key.strip():
        return api_key.strip()

    if api_key_file:
        key_path = Path(api_key_file).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")
        for line in key_path.read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if key:
                return key
        return None

    return None


def build_auth_headers(api_key: str | None = None, api_key_file: str | None = None) -> dict[str, str]:
    key = load_api_key(api_key=api_key, api_key_file=api_key_file)
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}