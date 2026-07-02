"""Sampling stage: draw N independent completions, cached, with refusal flagging."""

from __future__ import annotations

import re
from dataclasses import dataclass

from guessbench.cache import Cache, sample_key
from guessbench.config import RunConfig
from guessbench.providers import ModelClient

# Heuristic refusal / meta-response detection (SPEC 9). Refusals still cluster
# like any other sample; this only feeds the report's refusal fraction.
_REFUSAL_PATTERNS = [
    r"\bI (?:can'?not|can'?t|won'?t|am (?:not able|unable)) (?:to )?(?:help|assist|comply|do|write|provide|fulfill)",
    r"\bI'?m sorry,? (?:but )?I (?:can'?not|can'?t)",
    r"\bI must (?:decline|refuse)\b",
    r"\bas an AI\b.{0,60}\b(?:can'?not|can'?t|unable)\b",
]
_CLARIFICATION_PATTERNS = [
    r"\b(?:could|can|would) you (?:please )?(?:clarify|specify|provide more|tell me more)\b",
    r"\bbefore I (?:can )?(?:help|proceed|answer|start)\b.{0,80}\?",
    r"\b(?:what|which) (?:kind|type|sort) of\b.{0,80}\?\s*$",
    r"\bI need (?:a bit )?more (?:information|details|context)\b",
]


def is_refusal(text: str) -> bool:
    """True if the sample declines the task or asks a clarifying question instead
    of completing it."""
    head = text.strip()[:500]
    for pattern in _REFUSAL_PATTERNS + _CLARIFICATION_PATTERNS:
        if re.search(pattern, head, flags=re.IGNORECASE):
            return True
    return False


@dataclass
class SampleSet:
    """N completions of one artifact plus refusal accounting."""

    artifact_text: str
    samples: list[str]
    refusal_flags: list[bool]

    @property
    def refusal_fraction(self) -> float:
        return sum(self.refusal_flags) / len(self.samples) if self.samples else 0.0

    @property
    def low_confidence(self) -> bool:
        return self.refusal_fraction > 0.20


def draw_samples(artifact_text: str, config: RunConfig, client: ModelClient, cache: Cache) -> SampleSet:
    """Draw config.n_samples independent completions (SPEC 2.1), content-addressed
    cached per (model, T, max_tokens, artifact, sample_index)."""
    samples: list[str] = []
    for i in range(config.n_samples):
        # The run seed offsets the sample index so runs with disjoint seeds (T3)
        # draw disjoint samples while keeping the spec-pinned cache key formula.
        sample_index = config.seed * config.n_samples + i
        key = sample_key(
            config.reference_model, config.temperature, config.max_tokens, artifact_text, sample_index
        )
        cached = cache.get("samples", key)
        if cached is not None:
            text = cached["text"]
        else:
            text = client.complete(
                model=config.reference_model,
                prompt=artifact_text,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                seed=sample_index,
            )
            cache.put(
                "samples",
                key,
                {
                    "text": text,
                    "reference_model": config.reference_model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "sample_index": sample_index,
                },
            )
        samples.append(text)
    return SampleSet(
        artifact_text=artifact_text,
        samples=samples,
        refusal_flags=[is_refusal(s) for s in samples],
    )
