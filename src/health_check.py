import requests
from typing import Tuple


def check_llm_health(url: str, timeout: int = 5) -> Tuple[bool, str]:
    if not url:
        return False, "no url provided"
    try:
        payload = {"prompt": "ping", "max_tokens": 1}
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code >= 200 and r.status_code < 300:
            return True, f"ok ({r.status_code})"
        r2 = requests.get(url, timeout=timeout)
        if r2.status_code >= 200 and r2.status_code < 300:
            return True, f"ok-get ({r2.status_code})"
        return False, f"unexpected status {r.status_code}"
    except Exception as e:
        return False, str(e)
