import pytest

from guessbench.cache import Cache, embedding_key, judgment_key, sample_key
from guessbench.config import RunConfig
from guessbench.sampling import draw_samples, is_refusal


def make_config(**overrides):
    defaults = dict(n_samples=3, cache_dir="unused")
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestCache:
    def test_roundtrip(self, tmp_cache_dir):
        cache = Cache(tmp_cache_dir)
        assert cache.get("samples", "abc") is None
        cache.put("samples", "abc", {"text": "hello"})
        assert cache.get("samples", "abc") == {"text": "hello"}
        assert cache.stats() == {"hits": 1, "misses": 1, "enabled": True}

    def test_disabled_cache_never_hits(self, tmp_cache_dir):
        cache = Cache(tmp_cache_dir, enabled=False)
        cache.put("samples", "abc", {"text": "hello"})
        assert cache.get("samples", "abc") is None

    def test_sample_key_sensitivity(self):
        base = sample_key("m", 1.0, 800, "prompt", 0)
        assert sample_key("m", 1.0, 800, "prompt", 1) != base
        assert sample_key("m2", 1.0, 800, "prompt", 0) != base
        assert sample_key("m", 0.5, 800, "prompt", 0) != base

    def test_judgment_key_includes_presentation_order(self):
        a = judgment_key("j", "v1", "art", "x", "y", "AB")
        b = judgment_key("j", "v1", "art", "x", "y", "BA")
        assert a != b

    def test_embedding_key(self):
        assert embedding_key("e", "x") != embedding_key("e", "y")


class TestConfig:
    def test_judge_must_differ_from_reference(self):
        with pytest.raises(ValueError):
            RunConfig(reference_model="same-model", judge_model="same-model")

    def test_config_hash_stable_and_sensitive(self):
        assert make_config().config_hash() == make_config().config_hash()
        assert make_config().config_hash() != make_config(temperature=0.7).config_hash()
        # Cache toggles do not affect the score hash.
        assert make_config().config_hash() == make_config(use_cache=False).config_hash()


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "I'm sorry, but I can't help with that request.",
            "I cannot provide that information.",
            "Could you clarify what kind of sorting you need?",
            "I need more information before I can help.",
        ],
    )
    def test_detects_refusals_and_clarifications(self, text):
        assert is_refusal(text)

    @pytest.mark.parametrize(
        "text",
        [
            "def sort_desc(nums):\n    return sorted(nums, reverse=True)",
            "The sky appears blue because of Rayleigh scattering.",
            "HELLO",
        ],
    )
    def test_normal_outputs_not_flagged(self, text):
        assert not is_refusal(text)


class TestDrawSamples:
    def test_draws_n_and_caches(self, stub_client_factory, tmp_cache_dir):
        config = make_config(cache_dir=tmp_cache_dir)
        cache = Cache(tmp_cache_dir)
        client = stub_client_factory(completions=["out A", "out B", "out A"])
        result = draw_samples("test prompt", config, client, cache)
        assert result.samples == ["out A", "out B", "out A"]
        assert len(client.complete_calls) == 3

        # Second run is fully served from cache.
        client2 = stub_client_factory(completions=[])
        result2 = draw_samples("test prompt", config, client2, cache)
        assert result2.samples == result.samples
        assert client2.complete_calls == []

    def test_disjoint_seeds_use_disjoint_cache_keys(self, stub_client_factory, tmp_cache_dir):
        cache = Cache(tmp_cache_dir)
        client = stub_client_factory(completions=["a", "b", "c", "d", "e", "f"])
        draw_samples("p", make_config(cache_dir=tmp_cache_dir, seed=0), client, cache)
        draw_samples("p", make_config(cache_dir=tmp_cache_dir, seed=1), client, cache)
        assert len(client.complete_calls) == 6

    def test_refusal_fraction_and_low_confidence(self, stub_client_factory, tmp_cache_dir):
        config = make_config(cache_dir=tmp_cache_dir, n_samples=4)
        cache = Cache(tmp_cache_dir)
        client = stub_client_factory(
            completions=[
                "I'm sorry, but I can't help with that.",
                "Fine answer.",
                "Another fine answer.",
                "Third fine answer.",
            ]
        )
        result = draw_samples("p", config, client, cache)
        assert result.refusal_fraction == 0.25
        assert result.low_confidence
