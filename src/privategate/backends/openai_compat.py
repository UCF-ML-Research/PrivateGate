"""OpenAI-compatible HTTP backend.

Covers both options for D2 with one implementation:
  (a) local Llama-3-8B-Instruct via **vLLM** — expose `base_url=http://host:8000/v1`
  (b) **OpenAI** `gpt-4o-mini` — `base_url=https://api.openai.com/v1`

Most open-source inference servers (vLLM, TGI, llama.cpp server, LMDeploy)
expose the same `/v1/chat/completions` SSE streaming surface, so one
adapter handles the lot.

`httpx` is imported lazily so the rest of the package (types, policy,
transformers) stays dependency-free for unit tests that don't touch
the wire.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

from ..types import Mode, ResponseChunk, TransformedRequest


class OpenAICompatBackend:
    """Streaming OpenAI-compat Chat Completions backend."""

    mode = Mode.PLAINTEXT

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    async def dispatch(
        self, request: TransformedRequest
    ) -> AsyncIterator[ResponseChunk]:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("httpx is required for OpenAICompatBackend") from e

        if not isinstance(request.payload, str):
            raise TypeError(
                f"plaintext backend expects str payload, got {type(request.payload).__name__}"
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.payload}],
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield ResponseChunk(data="", is_final=True)
                        return
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content", "")
                        yield ResponseChunk(data=delta, is_final=False)
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue


def openai_cloud(model: str = "gpt-4o-mini") -> OpenAICompatBackend:
    """Preset for D2(b): OpenAI cloud (API key read from env)."""
    return OpenAICompatBackend(
        base_url="https://api.openai.com/v1",
        model=model,
    )


def vllm_local(
    base_url: str = "http://localhost:8000/v1",
    model: str = "meta-llama/Llama-3-8B-Instruct",
) -> OpenAICompatBackend:
    """Preset for D2(a): local Llama-3-8B via vLLM."""
    return OpenAICompatBackend(base_url=base_url, model=model, api_key="")
