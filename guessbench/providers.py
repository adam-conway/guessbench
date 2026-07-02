"""Model clients: a minimal protocol plus Ollama and Anthropic implementations."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class ModelClient(Protocol):
    """Interface every provider implements; tests use stubs of this."""

    def complete(self, model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None = None) -> str:
        """Return one completion for a single-turn user prompt."""
        ...

    def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding vector for the text."""
        ...


class OllamaClient:
    """Client for a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def complete(self, model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None = None) -> str:
        options: dict = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        resp = self._http.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed(self, model: str, text: str) -> list[float]:
        resp = self._http.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


class AnthropicClient:
    """Client for the Anthropic Messages API (completions only; Anthropic has no
    embeddings endpoint, so pair with Ollama or another provider for Strategy B)."""

    def __init__(self, api_key: str | None = None, timeout: float = 300.0) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._http = httpx.Client(timeout=timeout)

    def complete(self, model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None = None) -> str:
        # The Messages API has no seed parameter; independence comes from sampling.
        resp = self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return "".join(block["text"] for block in resp.json()["content"] if block["type"] == "text")

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not provide an embeddings API")


def make_client(provider: str, ollama_base_url: str = "http://localhost:11434") -> ModelClient:
    if provider == "ollama":
        return OllamaClient(base_url=ollama_base_url)
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown provider: {provider!r}")
