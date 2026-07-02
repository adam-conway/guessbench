"""Content-addressed cache for model calls (SPEC 3.1).

Every model call is cached under a SHA-256 key derived from the full set of
inputs that determine the response. Raw responses are stored as JSON files.
The cache is only bypassed via an explicit flag (--no-cache).
"""

from __future__ import annotations

import json
from pathlib import Path

from guessbench.config import stable_hash


class Cache:
    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, kind: str, key: str) -> Path:
        return self.cache_dir / kind / f"{key}.json"

    def get(self, kind: str, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(kind, key)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self.misses += 1
        return None

    def put(self, kind: str, key: str, value: dict) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "enabled": self.enabled}


def sample_key(
    reference_model: str, temperature: float, max_tokens: int, artifact_text: str, sample_index: int
) -> str:
    return stable_hash("sample", reference_model, temperature, max_tokens, artifact_text, sample_index)


def judgment_key(
    judge_model: str,
    judge_prompt_version: str,
    artifact_text: str,
    output_a: str,
    output_b: str,
    presentation_order: str,
) -> str:
    return stable_hash(
        "judgment",
        judge_model,
        judge_prompt_version,
        artifact_text,
        output_a,
        output_b,
        presentation_order,
    )


def embedding_key(embedding_model: str, text: str) -> str:
    return stable_hash("embedding", embedding_model, text)
