import requests
from typing import Tuple


def check_llm_health(url: str, timeout: int = 5) -> Tuple[bool, str]:
    if not url:
        return False, "no url provided"
    try:
        model_name = None
        if "/v1/completions" in url:
            try:
                models_url = url.replace("/v1/completions", "/v1/models")
                r_models = requests.get(models_url, timeout=timeout)
                if r_models.status_code == 200:
                    data = r_models.json()
                    models = data.get("data", [])
                    if models:
                        non_embed_models = [m.get("id") for m in models if "embed" not in m.get("id", "").lower()]
                        if non_embed_models:
                            model_name = non_embed_models[0]
                        else:
                            model_name = models[0].get("id")
            except Exception:
                pass

        payload = {"prompt": "ping", "max_tokens": 1}
        if model_name:
            payload["model"] = model_name
        elif "ollama" not in url.lower() and "11434" not in url:
            payload["model"] = "local-model"

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
