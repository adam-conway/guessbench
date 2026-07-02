import pytest

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.strategies.llm_judge import LLMJudgeStrategy, parse_judge_response


def make_config(**overrides):
    defaults = dict(n_samples=4, cache_dir="unused")
    defaults.update(overrides)
    return RunConfig(**defaults)


class VerdictClient:
    """Judge stub with a fixed verdict function over (text_a, text_b) pairs."""

    def __init__(self, verdict_fn):
        self.verdict_fn = verdict_fn
        self.calls = []

    def complete(self, model, prompt, temperature, max_tokens, seed=None):
        self.calls.append(prompt)
        a = prompt.split("<output_A>\n")[1].split("\n</output_A>")[0]
        b = prompt.split("<output_B>\n")[1].split("\n</output_B>")[0]
        verdict, label = self.verdict_fn(a, b)
        if verdict == "INTERCHANGEABLE":
            return "VERDICT: INTERCHANGEABLE"
        return f"VERDICT: DIFFERENT\nDIMENSION: {label}"

    def embed(self, model, text):
        raise NotImplementedError


class TestParseJudgeResponse:
    def test_interchangeable(self):
        assert parse_judge_response("VERDICT: INTERCHANGEABLE") == ("INTERCHANGEABLE", None)

    def test_different_with_label(self):
        verdict, label = parse_judge_response("VERDICT: DIFFERENT\nDIMENSION: sort direction.")
        assert verdict == "DIFFERENT"
        assert label == "sort direction"

    def test_different_without_label_gets_placeholder(self):
        assert parse_judge_response("VERDICT: DIFFERENT") == ("DIFFERENT", "unlabeled")

    def test_strips_think_blocks(self):
        raw = "<think>VERDICT: DIFFERENT hmm no wait</think>VERDICT: INTERCHANGEABLE"
        assert parse_judge_response(raw) == ("INTERCHANGEABLE", None)

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            parse_judge_response("I think they are similar")


class TestGreedyClustering:
    def test_all_interchangeable_one_cluster(self, tmp_cache_dir):
        client = VerdictClient(lambda a, b: ("INTERCHANGEABLE", None))
        result = LLMJudgeStrategy().cluster(
            "artifact", ["s0", "s1", "s2", "s3"], make_config(), client, Cache(tmp_cache_dir)
        )
        assert result.k == 1
        assert len(result.assignments) == 4

    def test_all_different_n_clusters(self, tmp_cache_dir):
        client = VerdictClient(lambda a, b: ("DIFFERENT", "content"))
        result = LLMJudgeStrategy().cluster(
            "artifact", ["s0", "s1", "s2", "s3"], make_config(), client, Cache(tmp_cache_dir)
        )
        assert result.k == 4
        assert "content" in result.difference_labels

    def test_two_groups(self, tmp_cache_dir):
        def verdict_fn(a, b):
            # Samples prefixed x are one interpretation, y the other.
            if a[0] == b[0]:
                return ("INTERCHANGEABLE", None)
            return ("DIFFERENT", "group")

        samples = ["x one", "y one", "x two", "y two"]
        client = VerdictClient(verdict_fn)
        result = LLMJudgeStrategy().cluster(
            "artifact", samples, make_config(), client, Cache(tmp_cache_dir)
        )
        assert result.k == 2
        assert result.assignments[0] == result.assignments[2]
        assert result.assignments[1] == result.assignments[3]
        assert result.assignments[0] != result.assignments[1]

    def test_judge_call_count_is_n_times_k_bounded(self, tmp_cache_dir):
        client = VerdictClient(lambda a, b: ("DIFFERENT", "content"))
        LLMJudgeStrategy().cluster(
            "artifact", ["s0", "s1", "s2", "s3"], make_config(), client, Cache(tmp_cache_dir)
        )
        # All-different worst case: 0 + 1 + 2 + 3 comparisons.
        assert len(client.calls) == 6

    def test_judgments_cached_across_runs(self, tmp_cache_dir):
        cache = Cache(tmp_cache_dir)
        client = VerdictClient(lambda a, b: ("INTERCHANGEABLE", None))
        LLMJudgeStrategy().cluster("artifact", ["s0", "s1"], make_config(), client, cache)
        client2 = VerdictClient(lambda a, b: ("INTERCHANGEABLE", None))
        LLMJudgeStrategy().cluster("artifact", ["s0", "s1"], make_config(), client2, cache)
        assert client2.calls == []

    def test_seed_changes_processing_order(self, tmp_cache_dir):
        # Non-transitive verdicts make cluster shapes order-dependent; greedy
        # clustering absorbs them without crashing (SPEC 4.1).
        def verdict_fn(a, b):
            pair = {a, b}
            if pair in ({"s0", "s1"}, {"s1", "s2"}):
                return ("INTERCHANGEABLE", None)
            return ("DIFFERENT", "content")

        for seed in (0, 1, 2):
            client = VerdictClient(verdict_fn)
            result = LLMJudgeStrategy().cluster(
                "artifact", ["s0", "s1", "s2"], make_config(seed=seed), client, Cache(tmp_cache_dir / str(seed))
            )
            # If s1 is processed first it represents a cluster both others join
            # (k=1); otherwise the non-transitive edge forces a split (k=2).
            assert result.k in (1, 2)

    def test_presentation_order_is_deterministic_per_seed(self, tmp_cache_dir):
        client = VerdictClient(lambda a, b: ("DIFFERENT", "content"))
        result1 = LLMJudgeStrategy().cluster(
            "artifact", ["s0", "s1", "s2"], make_config(seed=5), client, Cache(tmp_cache_dir / "a")
        )
        client2 = VerdictClient(lambda a, b: ("DIFFERENT", "content"))
        result2 = LLMJudgeStrategy().cluster(
            "artifact", ["s0", "s1", "s2"], make_config(seed=5), client2, Cache(tmp_cache_dir / "b")
        )
        orders1 = [r["presentation_order"] for r in result1.pair_records]
        orders2 = [r["presentation_order"] for r in result2.pair_records]
        assert orders1 == orders2
