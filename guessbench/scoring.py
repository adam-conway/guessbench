"""Clusters -> entropy -> Effective Interpretations, with bootstrap CI (SPEC 2.2-2.3)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


def entropy_nats(assignments: list[int]) -> float:
    """Shannon entropy in nats over cluster proportions."""
    n = len(assignments)
    if n == 0:
        return 0.0
    counts = np.array(list(Counter(assignments).values()), dtype=float)
    p = counts / n
    return float(-np.sum(p * np.log(p)))


def effective_interpretations(assignments: list[int]) -> float:
    """EI = e^H. Range [1, N]: 1.0 when all samples share a cluster, N when all distinct."""
    return float(np.exp(entropy_nats(assignments)))


@dataclass
class Score:
    ei: float
    ei_ci_low: float
    ei_ci_high: float
    entropy: float
    k: int
    n: int


def bootstrap_ci(assignments: list[int], iterations: int = 1000, seed: int = 0) -> tuple[float, float]:
    """90% bootstrap CI on EI: resample cluster assignments with replacement,
    recompute EI, take 5th/95th percentiles (SPEC 2.3)."""
    rng = np.random.default_rng(seed)
    arr = np.array(assignments)
    n = len(arr)
    eis = np.empty(iterations)
    for b in range(iterations):
        resampled = arr[rng.integers(0, n, size=n)]
        eis[b] = effective_interpretations(resampled.tolist())
    low, high = np.percentile(eis, [5, 95])
    return float(low), float(high)


def score_assignments(assignments: list[int], bootstrap_iterations: int = 1000, seed: int = 0) -> Score:
    h = entropy_nats(assignments)
    ci_low, ci_high = bootstrap_ci(assignments, iterations=bootstrap_iterations, seed=seed)
    return Score(
        ei=float(np.exp(h)),
        ei_ci_low=ci_low,
        ei_ci_high=ci_high,
        entropy=h,
        k=len(set(assignments)),
        n=len(assignments),
    )
