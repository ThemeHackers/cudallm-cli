from __future__ import annotations

import ipaddress
import os
import subprocess
import secrets
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


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if value and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_dotenv_file(env_path: str | os.PathLike[str] | None = None, *, override: bool = False) -> bool:
    path = Path(env_path or Path.cwd() / ".env")
    if not path.exists() or not path.is_file():
        return False

    loaded = False
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True
    return loaded


def get_secure_dashboard_token(length: int = 32) -> str:
    if length < 16:
        raise ValueError("Token length must be at least 16 bytes")
    return generate_secure_dashboard_token(length=length)


def generate_secure_dashboard_token_openssl(length: int = 32) -> str:
    if length < 16:
        raise ValueError("Token length must be at least 16 bytes")

    result = subprocess.run(
        ["openssl", "rand", "-base64", str(length)],
        check=True,
        capture_output=True,
        text=True,
    )
    token = result.stdout.strip().replace("+", "-").replace("/", "_").replace("=", "")
    if not token:
        raise RuntimeError("OpenSSL did not produce a token")
    return token


def generate_secure_dashboard_token(length: int = 32, prefer_openssl: bool = True) -> str:
    if prefer_openssl:
        try:
            return generate_secure_dashboard_token_openssl(length=length)
        except Exception:
            pass
    return secrets.token_urlsafe(length)


def upsert_env_value(env_path: str | os.PathLike[str], key: str, value: str) -> None:
    path = Path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    found = False
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed and parsed[0] == key:
                if not found:
                    lines.append(f"{key}={value}")
                    found = True
                continue
            lines.append(raw_line)

    if not found:
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")