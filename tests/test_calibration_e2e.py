"""End-to-end calibration test against a simulated model with controlled ambiguity.

The simulated reference model emits samples tagged with a meaning id; the number
of distinct meanings per artifact is chosen so a correct pipeline passes T1-T4.
The simulated judge and embedder both key off the meaning tag.
"""

import json
import re

import numpy as np
import pytest

from guessbench import cli
from guessbench.acceptance import calibration_artifacts, run_calibration
from guessbench.cache import Cache
from guessbench.config import RunConfig

MEANING_RE = re.compile(r"meaning-(\d+)")

N_MEANINGS_BY_ARTIFACT = {
    "A1": 1,
    "A2": 1,
    "A3": 2,
    "A4": 10,
    "L1": 6,
    "L2": 4,
    "L3": 3,
    "L4": 2,
    "L5": 1,
}


def n_meanings_for(artifact_id: str) -> int:
    if artifact_id in N_MEANINGS_BY_ARTIFACT:
        return N_MEANINGS_BY_ARTIFACT[artifact_id]
    return N_MEANINGS_BY_ARTIFACT[artifact_id.split(":")[1]]


class SimulatedModel:
    """Plays reference model, judge, and embedder for calibration artifacts."""

    def __init__(self):
        self.text_to_id = {text: aid for aid, text in calibration_artifacts().items()}

    def complete(self, model, prompt, temperature, max_tokens, seed=None):
        if "VERDICT" in prompt:
            a = MEANING_RE.search(prompt.split("<output_A>")[1].split("</output_A>")[0])
            b = MEANING_RE.search(prompt.split("<output_B>")[1].split("</output_B>")[0])
            if a.group(1) == b.group(1):
                return "VERDICT: INTERCHANGEABLE"
            return "VERDICT: DIFFERENT\nDIMENSION: simulated meaning"
        artifact_id = self.text_to_id[prompt]
        meaning = seed % n_meanings_for(artifact_id)
        return f"meaning-{meaning} wording-{seed}"

    def embed(self, model, text):
        match = re.match(r"meaning-(\d+) wording-(\d+)", text)
        meaning, wording = int(match.group(1)), int(match.group(2))
        vec = np.zeros(12)
        vec[meaning] = 1.0
        vec[11] = 0.02 * (wording % 5)  # wording jitter within a meaning
        return vec.tolist()


@pytest.fixture(scope="module")
def calibration_result(tmp_path_factory):
    cache = Cache(tmp_path_factory.mktemp("cache"))
    config = RunConfig(cache_dir="unused", bootstrap_iterations=200)
    return run_calibration(config, SimulatedModel(), cache)


class TestCalibrationEndToEnd:
    def test_both_strategies_pass_t1_to_t4(self, calibration_result):
        for strategy, results in calibration_result["results_by_strategy"].items():
            for result in results:
                assert result.passed, f"{strategy} {result.test_id}: {result.detail}"

    def test_threshold_selected_by_sweep(self, calibration_result):
        assert calibration_result["selected_embedding_threshold"] is not None
        sweep = calibration_result["threshold_sweep"]
        assert sweep[calibration_result["selected_embedding_threshold"]]["passes"] == 3

    def test_reports_cover_all_artifacts(self, calibration_result):
        for reports in calibration_result["reports_by_strategy"].values():
            assert len(reports) == 29

    def test_table_renders(self, calibration_result):
        table = calibration_result["table"]
        assert "llm_judge" in table and "embedding" in table
        assert "FAIL" not in table.splitlines()[1]


class TestCliScore(object):
    def test_score_command(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "make_client", lambda *a, **k: SimulatedModel())
        artifact = tmp_path / "artifact.txt"
        # A real calibration prompt so the simulated model recognizes it.
        artifact.write_text("Choose either 'heads' or 'tails'. Reply with just your choice.")

        rc = cli.main(
            [
                "score",
                str(artifact),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out-dir",
                str(tmp_path / "runs"),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "EI@{model=llama3.1:8b" in out
        report_files = list((tmp_path / "runs").glob("score_*.json"))
        assert len(report_files) == 1
        report = json.loads(report_files[0].read_text())
        assert report["k"] == 2

    def test_calibrate_command(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "make_client", lambda *a, **k: SimulatedModel())
        rc = cli.main(
            [
                "calibrate",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out-dir",
                str(tmp_path / "runs"),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "PENDING_REVIEW" in out
        assert (tmp_path / "runs" / "calibration.json").exists()

    def test_judge_audit_export_and_kappa(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "make_client", lambda *a, **k: SimulatedModel())
        out_dir = tmp_path / "audit"
        rc = cli.main(
            [
                "judge-audit",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        sheet_path = out_dir / "t5_labeling_sheet.json"
        key_path = out_dir / "t5_answer_key.json"
        assert sheet_path.exists() and key_path.exists()

        # A perfectly agreeing human labeler -> kappa 1.0, exit 0.
        sheet = json.loads(sheet_path.read_text())
        key = {r["pair_id"]: r["strategy_verdict"] for r in json.loads(key_path.read_text())}
        for row in sheet:
            row["human_label"] = key[row["pair_id"]]
        sheet_path.write_text(json.dumps(sheet))
        rc = cli.main(
            ["judge-audit", "--labeled-file", str(sheet_path), "--answer-key", str(key_path)]
        )
        assert rc == 0
        assert "PASS" in capsys.readouterr().out
