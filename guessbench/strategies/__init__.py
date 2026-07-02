from guessbench.strategies.base import ClusteringResult, EquivalenceStrategy
from guessbench.strategies.embedding import EmbeddingStrategy
from guessbench.strategies.llm_judge import LLMJudgeStrategy

__all__ = ["ClusteringResult", "EquivalenceStrategy", "EmbeddingStrategy", "LLMJudgeStrategy"]


def make_strategy(name: str, **kwargs) -> EquivalenceStrategy:
    if name == "llm_judge":
        return LLMJudgeStrategy(**kwargs)
    if name == "embedding":
        return EmbeddingStrategy(**kwargs)
    raise ValueError(f"Unknown strategy: {name!r}")
