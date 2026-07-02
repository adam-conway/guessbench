from pathlib import Path

import pytest


class StubClient:
    """ModelClient stub: returns queued completions and fixed embeddings."""

    def __init__(self, completions=None, embeddings=None):
        self.completions = list(completions or [])
        self.embeddings = dict(embeddings or {})
        self.complete_calls = []
        self.embed_calls = []

    def complete(self, model, prompt, temperature, max_tokens, seed=None):
        self.complete_calls.append(
            {"model": model, "prompt": prompt, "temperature": temperature, "seed": seed}
        )
        if not self.completions:
            raise AssertionError("StubClient ran out of queued completions")
        return self.completions.pop(0)

    def embed(self, model, text):
        self.embed_calls.append({"model": model, "text": text})
        return self.embeddings[text]


@pytest.fixture
def stub_client_factory():
    return StubClient


@pytest.fixture
def tmp_cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"
