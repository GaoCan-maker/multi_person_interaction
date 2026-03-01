import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def _join_url(base_url: str, path: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    path = "/" + path.lstrip("/")
    return base_url + path


def openai_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = 60,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call an OpenAI-compatible Chat Completions endpoint and return the assistant message content.
    Works with OpenAI / Azure OpenAI compatible gateways / many self-hosted "OpenAI API" servers.
    """
    url = _join_url(base_url, "/chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTPError {e.code}: {err}") from e
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    try:
        data = json.loads(payload)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM response parse failed: {e}; raw={payload[:4000]}") from e


def get_api_key_from_env(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key: environment variable '{env_name}' is empty. "
            f"Please export it before running inference."
        )
    return key


