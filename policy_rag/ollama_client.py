"""Local Ollama client for lightweight language generation."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"


def generate_local(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_ctx: int = 2048,
    num_predict: int = 256,
    timeout: int = 180,
    response_format: str | dict | None = None,
) -> dict:
    """Generate text through the local Ollama API."""

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    
    if response_format is not None:
        payload["format"] = response_format

    request = Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not connect to local Ollama. "
            "Ensure that 'ollama serve' is running."
        ) from exc

    if "prompt_eval_duration" not in result:
        prompt_eval_duration_ns = None
    else:
        prompt_eval_duration_ns = result[
            "prompt_eval_duration"
        ]

        if (
            type(prompt_eval_duration_ns) is not int
            or prompt_eval_duration_ns < 0
        ):
            raise RuntimeError(
                "Ollama returned an invalid "
                "prompt_eval_duration"
            )

    generated_text = result.get("response", "").strip()

    if not generated_text:
        raise RuntimeError(
            "Ollama returned an empty response"
        )

    return {
        "text": generated_text,
        "model": result.get("model", model),
        "actual_model": result.get("model"),
        "total_duration_ns": result.get("total_duration"),
        "load_duration_ns": result.get("load_duration"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "eval_count": result.get("eval_count"),
        "eval_duration_ns": result.get("eval_duration"),
    }
