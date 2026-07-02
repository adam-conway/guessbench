"""Acceptance suite T1-T4 (SPEC 6), calibration data loading, and the calibration runner."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml
from scipy.stats import spearmanr

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.providers import ModelClient
from guessbench.report import build_report
from guessbench.sampling import draw_samples
from guessbench.scoring import effective_interpretations
from guessbench.strategies import make_strategy
from guessbench.strategies.embedding import sweep_thresholds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A4's absolute floor is a tunable (SPEC 5.3); the ordinal constraint is pinned.
A4_MIN_EI = 4.0
A4_WIDE_MARGIN_RATIO = 2.0  # "wide margin": EI(A4) at least 2x EI(A2)


def load_anchors(path: Path | None = None) -> list[dict]:
    text = (path or DATA_DIR / "anchors.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["anchors"]


def load_ladders(directory: Path | None = None) -> list[dict]:
    directory = directory or DATA_DIR / "ladders"
    ladders = [
        yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.yaml"))
    ]
    for ladder in ladders:
        ladder["levels"] = sorted(ladder["levels"], key=lambda lv: lv["level"])
    return ladders


@dataclass
class TestResult:
    test_id: str
    passed: bool
    detail: str


def check_t1_anchors(anchor_reports: dict[str, dict]) -> TestResult:
    """T1: all four anchor assertions hold (SPEC 5.3)."""
    failures = []
    a1 = anchor_reports["A1"]
    if not (a1["k"] == 1 and a1["ei"] == 1.0):
        failures.append(f"A1: expected k=1, EI=1.0; got k={a1['k']}, EI={a1['ei']}")
    a2 = anchor_reports["A2"]
    if not a2["ei"] <= 1.3:
        failures.append(f"A2: expected EI<=1.3; got {a2['ei']}")
    a3 = anchor_reports["A3"]
    if not (a3["k"] <= 2 and 1.0 <= a3["ei"] <= 2.0):
        failures.append(f"A3: expected k<=2, EI in [1,2]; got k={a3['k']}, EI={a3['ei']}")
    a4 = anchor_reports["A4"]
    if not (a4["ei"] >= A4_MIN_EI and a4["ei"] >= A4_WIDE_MARGIN_RATIO * a2["ei"]):
        failures.append(
            f"A4: expected EI>={A4_MIN_EI} and >={A4_WIDE_MARGIN_RATIO}x EI(A2)={a2['ei']}; "
            f"got {a4['ei']}"
        )
    return TestResult("T1", not failures, "; ".join(failures) or "all anchor assertions hold")


def check_t2_monotonicity(ladder_eis: dict[str, dict[int, float]]) -> TestResult:
    """T2: per ladder, Spearman rho(level, EI) <= -0.9 and EI(L5) <= 1.5."""
    failures = []
    details = []
    for family, eis_by_level in ladder_eis.items():
        levels = sorted(eis_by_level)
        eis = [eis_by_level[lv] for lv in levels]
        rho = float(spearmanr(levels, eis).statistic)
        l5 = eis_by_level[max(levels)]
        details.append(f"{family}: rho={rho:.3f}, EI(L5)={l5:.2f}")
        if not (rho <= -0.9 and l5 <= 1.5):
            failures.append(f"{family} (rho={rho:.3f}, EI(L5)={l5:.2f})")
    detail = "; ".join(details) + ("; FAILED: " + ", ".join(failures) if failures else "")
    return TestResult("T2", not failures, detail)


def check_t3_stability(run_pairs: dict[str, tuple[dict, dict]]) -> TestResult:
    """T3: for each artifact, two disjoint-seed runs have overlapping 90% CIs
    AND |dEI| <= max(0.5, 15% relative)."""
    failures = []
    details = []
    for artifact_id, (run_a, run_b) in run_pairs.items():
        ci_a, ci_b = run_a["ei_ci_90"], run_b["ei_ci_90"]
        overlap = ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]
        delta = abs(run_a["ei"] - run_b["ei"])
        allowed = max(0.5, 0.15 * max(run_a["ei"], run_b["ei"]))
        details.append(f"{artifact_id}: dEI={delta:.2f} (allowed {allowed:.2f}), overlap={overlap}")
        if not (overlap and delta <= allowed):
            failures.append(artifact_id)
    detail = "; ".join(details) + ("; FAILED: " + ", ".join(failures) if failures else "")
    return TestResult("T3", not failures, detail)


def check_t4_surface_invariance(a2_report: dict) -> TestResult:
    """T4: A2 specifically — EI(A2) <= 1.3. A strategy that fails is counting
    wording, not interpretation."""
    passed = a2_report["ei"] <= 1.3
    return TestResult("T4", passed, f"EI(A2)={a2_report['ei']}")


def evaluate_strategy(
    anchor_reports: dict[str, dict],
    ladder_eis: dict[str, dict[int, float]],
    stability_pairs: dict[str, tuple[dict, dict]],
) -> list[TestResult]:
    """Run T1-T4 over one strategy's calibration outputs."""
    return [
        check_t1_anchors(anchor_reports),
        check_t2_monotonicity(ladder_eis),
        check_t3_stability(stability_pairs),
        check_t4_surface_invariance(anchor_reports["A2"]),
    ]


def acceptance_table(results_by_strategy: dict[str, list[TestResult]]) -> str:
    """Render the strategy x test pass/fail comparison table (SPEC 4.5)."""
    test_ids = ["T1", "T2", "T3", "T4"]
    lines = ["strategy        " + "".join(f"{tid:<8}" for tid in test_ids)]
    for strategy, results in results_by_strategy.items():
        by_id = {r.test_id: r for r in results}
        cells = "".join(
            f"{('PASS' if by_id[tid].passed else 'FAIL') if tid in by_id else '-':<8}"
            for tid in test_ids
        )
        lines.append(f"{strategy:<16}{cells}")
    lines.append("")
    for strategy, results in results_by_strategy.items():
        for r in results:
            lines.append(f"[{strategy}] {r.test_id}: {r.detail}")
    return "\n".join(lines)


# --- Calibration runner ---------------------------------------------------

# T3 artifacts: one anchor, one L1, one L3 (SPEC 6).
STABILITY_ARTIFACTS = ["A2", "code:L1", "code:L3"]
SWEEP_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]


def calibration_artifacts() -> dict[str, str]:
    """All calibration artifacts keyed by id: 4 anchors + 5 ladders x 5 levels."""
    artifacts: dict[str, str] = {}
    for anchor in load_anchors():
        artifacts[anchor["id"]] = anchor["prompt"]
    for ladder in load_ladders():
        slug = ladder["family"].split("/")[0].split()[0]
        for level in ladder["levels"]:
            artifacts[f"{slug}:L{level['level']}"] = level["prompt"]
    return artifacts


def _strategy_reports(
    strategy_name: str,
    artifacts: dict[str, str],
    samples_by_artifact: dict[str, object],
    config: RunConfig,
    client: ModelClient,
    cache: Cache,
    threshold: float | None = None,
) -> dict[str, dict]:
    kwargs = {"threshold": threshold} if strategy_name == "embedding" else {}
    strategy = make_strategy(strategy_name, **kwargs)
    reports = {}
    for artifact_id, text in artifacts.items():
        sample_set = samples_by_artifact[artifact_id]
        clustering = strategy.cluster(text, sample_set.samples, config, client, cache)
        run_config = dataclasses.replace(config, strategy=strategy_name)
        reports[artifact_id] = build_report(
            text, sample_set, clustering, run_config, cache, strategy.version
        )
    return reports


def _split_reports(reports: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict[int, float]]]:
    anchor_reports = {aid: r for aid, r in reports.items() if aid.startswith("A")}
    ladder_eis: dict[str, dict[int, float]] = {}
    for artifact_id, report in reports.items():
        if ":" not in artifact_id:
            continue
        family, level = artifact_id.split(":L")
        ladder_eis.setdefault(family, {})[int(level)] = report["ei"]
    return anchor_reports, ladder_eis


def select_embedding_threshold(
    samples_by_artifact: dict[str, object],
    config: RunConfig,
    client: ModelClient,
    cache: Cache,
) -> tuple[float, dict[float, dict]]:
    """Sweep thresholds; the acceptance suite picks the winner (SPEC 4.3).

    A threshold passes if T1, T2, T4 hold on seed-0 data (T3 is checked after
    selection). Among passing thresholds the median is returned; if none pass,
    the one failing fewest tests.
    """
    texts_by_artifact = {aid: ss.samples for aid, ss in samples_by_artifact.items()}
    assignments_by_threshold = sweep_thresholds(
        texts_by_artifact, SWEEP_THRESHOLDS, config, client, cache
    )
    sweep_results: dict[float, dict] = {}
    for threshold, per_artifact in assignments_by_threshold.items():
        eis = {aid: effective_interpretations(assign) for aid, assign in per_artifact.items()}
        anchor_eis = {aid: {"ei": eis[aid], "k": len(set(per_artifact[aid]))} for aid in eis if aid.startswith("A")}
        _, ladder_eis = _split_reports({aid: {"ei": ei} for aid, ei in eis.items()})
        t1 = check_t1_anchors(anchor_eis)
        t2 = check_t2_monotonicity(ladder_eis)
        t4 = check_t4_surface_invariance(anchor_eis["A2"])
        sweep_results[threshold] = {
            "passes": sum(t.passed for t in (t1, t2, t4)),
            "results": [t1, t2, t4],
        }
    passing = [t for t, res in sweep_results.items() if res["passes"] == 3]
    if passing:
        winner = sorted(passing)[len(passing) // 2]
    else:
        winner = max(sweep_results, key=lambda t: sweep_results[t]["passes"])
    return winner, sweep_results


def run_calibration(
    config: RunConfig,
    client: ModelClient,
    cache: Cache,
    strategies: list[str] | None = None,
) -> dict:
    """Run anchors + ladders across strategies; emit acceptance results (SPEC 7)."""
    strategies = strategies or ["llm_judge", "embedding"]
    artifacts = calibration_artifacts()

    samples_seed0 = {
        aid: draw_samples(text, config, client, cache) for aid, text in artifacts.items()
    }
    config_seed1 = dataclasses.replace(config, seed=config.seed + 1)
    stability_artifacts = {aid: artifacts[aid] for aid in STABILITY_ARTIFACTS}
    samples_seed1 = {
        aid: draw_samples(text, config_seed1, client, cache)
        for aid, text in stability_artifacts.items()
    }

    results_by_strategy: dict[str, list[TestResult]] = {}
    reports_by_strategy: dict[str, dict[str, dict]] = {}
    selected_threshold: float | None = None
    sweep_summary = None

    for strategy_name in strategies:
        threshold = None
        if strategy_name == "embedding":
            selected_threshold, sweep = select_embedding_threshold(
                samples_seed0, config, client, cache
            )
            threshold = selected_threshold
            sweep_summary = {
                t: {"passes": res["passes"]} for t, res in sorted(sweep.items())
            }

        reports = _strategy_reports(
            strategy_name, artifacts, samples_seed0, config, client, cache, threshold
        )
        anchor_reports, ladder_eis = _split_reports(reports)

        rerun_reports = _strategy_reports(
            strategy_name, stability_artifacts, samples_seed1, config_seed1, client, cache, threshold
        )
        stability_pairs = {
            aid: (reports[aid], rerun_reports[aid]) for aid in STABILITY_ARTIFACTS
        }

        results_by_strategy[strategy_name] = evaluate_strategy(
            anchor_reports, ladder_eis, stability_pairs
        )
        reports_by_strategy[strategy_name] = reports

    return {
        "results_by_strategy": results_by_strategy,
        "reports_by_strategy": reports_by_strategy,
        "selected_embedding_threshold": selected_threshold,
        "threshold_sweep": sweep_summary,
        "table": acceptance_table(results_by_strategy),
    }
