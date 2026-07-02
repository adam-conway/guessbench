"""Common interface for equivalence strategies (SPEC 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.providers import ModelClient


@dataclass
class ClusteringResult:
    """Cluster assignment per sample, plus strategy-specific diagnostics."""

    assignments: list[int]
    # Difference-dimension labels emitted by DIFFERENT verdicts (Strategy A).
    difference_labels: list[str] = field(default_factory=list)
    # Observed non-transitive judgment triples, logged for the report (SPEC 4.1).
    non_transitivity_log: list[dict] = field(default_factory=list)
    # Raw pairwise judgments or distances, for the judge audit (T5).
    pair_records: list[dict] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(set(self.assignments))


class EquivalenceStrategy(ABC):
    """Decides which samples count as the same interpretation."""

    strategy_id: str
    version: str = "v1"

    @abstractmethod
    def cluster(
        self,
        artifact_text: str,
        samples: list[str],
        config: RunConfig,
        client: ModelClient,
        cache: Cache,
    ) -> ClusteringResult:
        """Partition samples into interpretation-level clusters."""
        ...
