"""Strategy B: embeddings + agglomerative clustering with a distance threshold (SPEC 4.3)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from guessbench.cache import Cache, embedding_key
from guessbench.config import RunConfig
from guessbench.providers import ModelClient
from guessbench.strategies.base import ClusteringResult, EquivalenceStrategy


def get_embeddings(
    texts: list[str], config: RunConfig, client: ModelClient, cache: Cache
) -> np.ndarray:
    """Embed each text, content-addressed cached per (embedding_model, text)."""
    vectors = []
    for text in texts:
        key = embedding_key(config.embedding_model, text)
        cached = cache.get("embeddings", key)
        if cached is not None:
            vec = cached["vector"]
        else:
            vec = client.embed(model=config.embedding_model, text=text)
            cache.put("embeddings", key, {"vector": vec, "embedding_model": config.embedding_model})
        vectors.append(vec)
    return np.array(vectors, dtype=float)


def cluster_vectors(vectors: np.ndarray, threshold: float) -> list[int]:
    """Agglomerative clustering (average linkage, cosine distance) cut at the threshold."""
    if len(vectors) == 1:
        return [0]
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(vectors).tolist()


class EmbeddingStrategy(EquivalenceStrategy):
    strategy_id = "embedding"
    version = "v1"

    def __init__(self, threshold: float | None = None) -> None:
        # Falls back to config.embedding_threshold when not set explicitly.
        self.threshold = threshold

    def cluster(
        self,
        artifact_text: str,
        samples: list[str],
        config: RunConfig,
        client: ModelClient,
        cache: Cache,
    ) -> ClusteringResult:
        threshold = self.threshold if self.threshold is not None else config.embedding_threshold
        vectors = get_embeddings(samples, config, client, cache)
        assignments = cluster_vectors(vectors, threshold)

        # Record pairwise cosine distances for the judge audit's near-boundary
        # pair selection (T5).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.where(norms == 0, 1, norms)
        distances = 1 - normalized @ normalized.T
        pair_records = [
            {
                "sample_i": i,
                "sample_j": j,
                "distance": float(distances[i, j]),
                "same_cluster": assignments[i] == assignments[j],
            }
            for i in range(len(samples))
            for j in range(i + 1, len(samples))
        ]
        return ClusteringResult(assignments=assignments, pair_records=pair_records)


def sweep_thresholds(
    samples_by_artifact: dict[str, list[str]],
    thresholds: list[float],
    config: RunConfig,
    client: ModelClient,
    cache: Cache,
) -> dict[float, dict[str, list[int]]]:
    """Cluster every artifact's samples at each candidate threshold.

    Embeddings are computed once (cached); only the cut varies. The acceptance
    suite selects the winning threshold (SPEC 4.3).
    """
    vectors_by_artifact = {
        artifact: get_embeddings(samples, config, client, cache)
        for artifact, samples in samples_by_artifact.items()
    }
    return {
        threshold: {
            artifact: cluster_vectors(vectors, threshold)
            for artifact, vectors in vectors_by_artifact.items()
        }
        for threshold in thresholds
    }
