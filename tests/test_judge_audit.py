import json

import pytest

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.judge_audit import (
    cohens_kappa,
    compute_kappa_from_files,
    select_audit_pairs,
    write_audit_export,
)


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["INTERCHANGEABLE", "DIFFERENT"] * 5
        assert cohens_kappa(labels, list(labels)) == 1.0

    def test_hand_computed(self):
        # 2x2 table: both-I=4, both-D=4, disagreements 1+1.
        # po=0.8, pe=0.5, kappa=0.6.
        rater_a = ["I"] * 5 + ["D"] * 5
        rater_b = ["I"] * 4 + ["D"] + ["I"] + ["D"] * 4
        assert cohens_kappa(rater_a, rater_b) == pytest.approx(0.6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([], [])


class FixedEmbedClient:
    """Embeddings depend on the sample's leading token, so same-prefix samples
    are close and cross-prefix samples are far."""

    VECTORS = {"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [0.7, 0.7]}

    def embed(self, model, text):
        return self.VECTORS[text.split()[0]]

    def complete(self, *args, **kwargs):
        raise NotImplementedError


def make_audit_inputs():
    # Two artifacts; assignments follow prefixes; z-samples sit near the boundary.
    return {
        "art1": {
            "text": "prompt one",
            "samples": ["x a", "x b", "y a", "y b", "z a"],
            "assignments": [0, 0, 1, 1, 0],
        },
        "art2": {
            "text": "prompt two",
            "samples": ["x c", "y c", "z c"],
            "assignments": [0, 1, 2],
        },
    }


class TestSelectAuditPairs:
    def test_strata_and_blindness(self, tmp_cache_dir, tmp_path):
        config = RunConfig(cache_dir=tmp_cache_dir)
        pairs = select_audit_pairs(
            make_audit_inputs(), config, FixedEmbedClient(), Cache(tmp_cache_dir), per_stratum=3
        )
        strata = {p["stratum"] for p in pairs}
        assert strata == {"same_cluster", "cross_cluster", "near_boundary"}
        # Near-boundary pairs are cross-cluster by definition.
        for p in pairs:
            if p["stratum"] == "near_boundary":
                assert not p["same_cluster"]

        sheet_path, key_path = write_audit_export(pairs, tmp_path / "audit")
        sheet = json.loads(sheet_path.read_text())
        # The labeling sheet must not leak strategy verdicts (SPEC 8).
        for row in sheet:
            assert set(row) == {"pair_id", "artifact_text", "output_a", "output_b", "human_label"}
        key = json.loads(key_path.read_text())
        assert {r["pair_id"] for r in key} == {r["pair_id"] for r in sheet}

    def test_deterministic_for_seed(self, tmp_cache_dir):
        config = RunConfig(cache_dir=tmp_cache_dir)
        args = (make_audit_inputs(), config, FixedEmbedClient(), Cache(tmp_cache_dir))
        ids1 = [p["pair_id"] for p in select_audit_pairs(*args, per_stratum=3, seed=42)]
        ids2 = [p["pair_id"] for p in select_audit_pairs(*args, per_stratum=3, seed=42)]
        assert ids1 == ids2


class TestKappaFromFiles:
    def test_round_trip(self, tmp_cache_dir, tmp_path):
        config = RunConfig(cache_dir=tmp_cache_dir)
        pairs = select_audit_pairs(
            make_audit_inputs(), config, FixedEmbedClient(), Cache(tmp_cache_dir), per_stratum=3
        )
        sheet_path, key_path = write_audit_export(pairs, tmp_path / "audit")

        # Human agrees with the strategy on every pair except one, leaves one blank.
        sheet = json.loads(sheet_path.read_text())
        key = {r["pair_id"]: r["strategy_verdict"] for r in json.loads(key_path.read_text())}
        for i, row in enumerate(sheet):
            row["human_label"] = key[row["pair_id"]]
        sheet[0]["human_label"] = (
            "DIFFERENT" if key[sheet[0]["pair_id"]] == "INTERCHANGEABLE" else "INTERCHANGEABLE"
        )
        sheet[1]["human_label"] = ""
        sheet_path.write_text(json.dumps(sheet))

        result = compute_kappa_from_files(sheet_path, key_path)
        assert result["n_skipped"] == 1
        assert result["n_labeled"] == len(sheet) - 1
        assert 0 < result["kappa"] < 1
