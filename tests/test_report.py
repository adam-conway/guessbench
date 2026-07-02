from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.report import build_report, render_report, score_artifact
from guessbench.sampling import SampleSet
from guessbench.strategies.base import ClusteringResult


def make_config(**overrides):
    defaults = dict(n_samples=4, bootstrap_iterations=100, cache_dir="unused")
    defaults.update(overrides)
    return RunConfig(**defaults)


def make_report(tmp_cache_dir, **config_overrides):
    sample_set = SampleSet(
        artifact_text="artifact",
        samples=["alpha output", "beta output", "alpha again", "gamma output"],
        refusal_flags=[False, False, False, True],
    )
    clustering = ClusteringResult(
        assignments=[0, 1, 0, 2],
        difference_labels=["tone", "tone", "content"],
        non_transitivity_log=[{"triple": [0, 1, 2], "verdicts": ["I", "I", "D"]}],
    )
    config = make_config(**config_overrides)
    return build_report(
        "artifact", sample_set, clustering, config, Cache(tmp_cache_dir), strategy_version="v1"
    )


class TestBuildReport:
    def test_core_fields(self, tmp_cache_dir):
        report = make_report(tmp_cache_dir)
        assert report["n"] == 4
        assert report["k"] == 3
        assert report["ei_ci_90"][0] <= report["ei"] <= report["ei_ci_90"][1]
        assert "llama3.1:8b" in report["stamp"]
        assert "T=1.0" in report["stamp"]
        assert report["difference_labels"] == {"tone": 2, "content": 1}

    def test_refusal_flagging(self, tmp_cache_dir):
        report = make_report(tmp_cache_dir)
        assert report["refusal_fraction"] == 0.25
        assert report["low_confidence"] is True

    def test_cluster_table_sorted_by_size(self, tmp_cache_dir):
        report = make_report(tmp_cache_dir)
        sizes = [row["size"] for row in report["cluster_table"]]
        assert sizes == sorted(sizes, reverse=True)
        assert report["cluster_table"][0]["exemplar"].startswith("alpha")

    def test_render_contains_stamp_and_warning(self, tmp_cache_dir):
        rendered = render_report(make_report(tmp_cache_dir))
        assert "EI@{model=llama3.1:8b" in rendered
        assert "LOW_CONFIDENCE" in rendered
        assert "2x tone" in rendered


class TestScoreArtifactEndToEnd:
    def test_pipeline_with_stub_judge(self, stub_client_factory, tmp_cache_dir):
        class PipelineStub:
            """Reference completions from a queue; judge verdicts by first token."""

            def complete(self, model, prompt, temperature, max_tokens, seed=None):
                if "VERDICT" in prompt:
                    a = prompt.split("<output_A>\n")[1].split("\n</output_A>")[0]
                    b = prompt.split("<output_B>\n")[1].split("\n</output_B>")[0]
                    if a.split()[0] == b.split()[0]:
                        return "VERDICT: INTERCHANGEABLE"
                    return "VERDICT: DIFFERENT\nDIMENSION: content"
                return self.samples.pop(0)

            def embed(self, model, text):
                raise NotImplementedError

            samples = ["x one", "x two", "y one", "x three"]

        config = make_config(cache_dir=tmp_cache_dir)
        report = score_artifact("prompt", config, PipelineStub(), Cache(tmp_cache_dir))
        assert report["n"] == 4
        assert report["k"] == 2
        assert report["config"]["strategy"] == "llm_judge"
