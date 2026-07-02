import numpy as np

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.strategies.embedding import (
    EmbeddingStrategy,
    cluster_vectors,
    sweep_thresholds,
)


def make_config(**overrides):
    defaults = dict(cache_dir="unused")
    defaults.update(overrides)
    return RunConfig(**defaults)


class EmbedClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embed_calls = 0

    def embed(self, model, text):
        self.embed_calls += 1
        return self.embeddings[text]

    def complete(self, *args, **kwargs):
        raise NotImplementedError


# Two tight groups far apart in cosine distance.
GROUP_A = {"a1": [1.0, 0.0, 0.01], "a2": [1.0, 0.01, 0.0], "a3": [0.99, 0.0, 0.0]}
GROUP_B = {"b1": [0.0, 1.0, 0.01], "b2": [0.01, 1.0, 0.0]}
EMBEDDINGS = {**GROUP_A, **GROUP_B}


class TestClusterVectors:
    def test_two_well_separated_groups(self):
        vectors = np.array(list(EMBEDDINGS.values()))
        assignments = cluster_vectors(vectors, threshold=0.5)
        a_ids = set(assignments[:3])
        b_ids = set(assignments[3:])
        assert len(a_ids) == 1
        assert len(b_ids) == 1
        assert a_ids != b_ids

    def test_tiny_threshold_splits_everything(self):
        vectors = np.array(list(EMBEDDINGS.values()))
        assert len(set(cluster_vectors(vectors, threshold=1e-9))) == 5

    def test_huge_threshold_merges_everything(self):
        vectors = np.array(list(EMBEDDINGS.values()))
        assert len(set(cluster_vectors(vectors, threshold=2.0))) == 1

    def test_single_sample(self):
        assert cluster_vectors(np.array([[1.0, 0.0]]), threshold=0.5) == [0]


class TestEmbeddingStrategy:
    def test_clusters_and_caches(self, tmp_cache_dir):
        cache = Cache(tmp_cache_dir)
        client = EmbedClient(EMBEDDINGS)
        samples = list(EMBEDDINGS.keys())
        result = EmbeddingStrategy(threshold=0.5).cluster(
            "artifact", samples, make_config(), client, cache
        )
        assert result.k == 2
        assert len(result.pair_records) == 10  # C(5,2) pairwise distances

        client2 = EmbedClient(EMBEDDINGS)
        EmbeddingStrategy(threshold=0.5).cluster("artifact", samples, make_config(), client2, cache)
        assert client2.embed_calls == 0

    def test_threshold_falls_back_to_config(self, tmp_cache_dir):
        client = EmbedClient(EMBEDDINGS)
        samples = list(EMBEDDINGS.keys())
        result = EmbeddingStrategy().cluster(
            "artifact", samples, make_config(embedding_threshold=2.0), client, Cache(tmp_cache_dir)
        )
        assert result.k == 1


class TestSweep:
    def test_sweep_embeds_once_per_text(self, tmp_cache_dir):
        cache = Cache(tmp_cache_dir)
        client = EmbedClient(EMBEDDINGS)
        samples = list(EMBEDDINGS.keys())
        results = sweep_thresholds(
            {"artifact": samples}, [1e-9, 0.5, 2.0], make_config(), client, cache
        )
        assert client.embed_calls == 5
        assert len(set(results[1e-9]["artifact"])) == 5
        assert len(set(results[0.5]["artifact"])) == 2
        assert len(set(results[2.0]["artifact"])) == 1
