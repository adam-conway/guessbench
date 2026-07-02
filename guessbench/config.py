"""Run configuration and content hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Spec-pinned sampling defaults (SPEC 2.1).
DEFAULT_N = 20
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 800

# Model defaults target local Ollama per user request (spec default is
# claude-sonnet-4-6 via Anthropic; see DECISIONS.md). Spec pins judge != reference.
DEFAULT_PROVIDER = "ollama"
DEFAULT_REFERENCE_MODEL = "llama3.1:8b"
DEFAULT_JUDGE_MODEL = "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

JUDGE_PROMPT_VERSION = "v1"


def stable_hash(*parts: object) -> str:
    """SHA-256 over a JSON-serialized tuple of parts, hex digest."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RunConfig:
    """Full configuration for a scoring run. Every field is recorded in reports."""

    reference_model: str = DEFAULT_REFERENCE_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    provider: str = DEFAULT_PROVIDER
    n_samples: int = DEFAULT_N
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    strategy: str = "llm_judge"
    embedding_threshold: float = 0.35
    seed: int = 0
    judge_prompt_version: str = JUDGE_PROMPT_VERSION
    bootstrap_iterations: int = 1000
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    use_cache: bool = True
    ollama_base_url: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        if self.judge_model == self.reference_model:
            raise ValueError(
                "judge_model must differ from reference_model (SPEC 4.2: "
                "self-judging invites self-preference artifacts)"
            )
        self.cache_dir = Path(self.cache_dir)

    def config_hash(self) -> str:
        """Reproducibility hash over the score-affecting configuration."""
        d = asdict(self)
        # Cache location/usage and endpoint do not affect the score itself.
        for key in ("cache_dir", "use_cache", "ollama_base_url"):
            d.pop(key)
        return stable_hash(d)[:16]

    def to_record(self) -> dict:
        """JSON-safe dict of the full config for report embedding."""
        d = asdict(self)
        d["cache_dir"] = str(d["cache_dir"])
        d["config_hash"] = self.config_hash()
        return d
