"""Score reports: machine-readable JSON plus a human-readable rendering (SPEC 2.5).

Every score is stamped EI@{model, T, N, strategy} because EI is a property of
(text, model, temperature), not of the text alone (SPEC 2.4).
"""

from __future__ import annotations

import hashlib
from collections import Counter

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.providers import ModelClient
from guessbench.sampling import SampleSet, draw_samples
from guessbench.scoring import score_assignments
from guessbench.strategies import make_strategy
from guessbench.strategies.base import ClusteringResult


def prompt_hash(artifact_text: str) -> str:
    return hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()[:16]


def _snippet(text: str, limit: int = 160) -> str:
    flat = " ".join(text.strip().split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def build_report(
    artifact_text: str,
    sample_set: SampleSet,
    clustering: ClusteringResult,
    config: RunConfig,
    cache: Cache,
    strategy_version: str,
) -> dict:
    score = score_assignments(
        clustering.assignments,
        bootstrap_iterations=config.bootstrap_iterations,
        seed=config.seed,
    )

    cluster_table = []
    members_by_cluster: dict[int, list[int]] = {}
    for sample_idx, cluster_id in enumerate(clustering.assignments):
        members_by_cluster.setdefault(cluster_id, []).append(sample_idx)
    for cluster_id, members in sorted(
        members_by_cluster.items(), key=lambda item: -len(item[1])
    ):
        cluster_table.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "fraction": len(members) / score.n,
                "exemplar": _snippet(sample_set.samples[members[0]]),
                "member_indices": members,
            }
        )

    return {
        "stamp": (
            f"EI@{{model={config.reference_model}, T={config.temperature}, "
            f"N={config.n_samples}, strategy={config.strategy}:{strategy_version}}}"
        ),
        "ei": round(score.ei, 3),
        "ei_ci_90": [round(score.ei_ci_low, 3), round(score.ei_ci_high, 3)],
        "entropy_nats": round(score.entropy, 4),
        "k": score.k,
        "n": score.n,
        "config": config.to_record(),
        "prompt_hash": prompt_hash(artifact_text),
        "strategy_version": strategy_version,
        "cluster_table": cluster_table,
        "difference_labels": dict(Counter(clustering.difference_labels)),
        "non_transitivity_log": clustering.non_transitivity_log,
        "refusal_fraction": round(sample_set.refusal_fraction, 3),
        "low_confidence": sample_set.low_confidence,
        "cache_stats": cache.stats(),
    }


def render_report(report: dict) -> str:
    """Human-readable rendering of a report dict."""
    lines = [
        report["stamp"],
        "",
        f"  EI = {report['ei']}   90% CI [{report['ei_ci_90'][0]}, {report['ei_ci_90'][1]}]",
        f"  H  = {report['entropy_nats']} nats   k = {report['k']} clusters   N = {report['n']} samples",
        f"  prompt_hash = {report['prompt_hash']}   config_hash = {report['config']['config_hash']}",
        f"  refusal fraction = {report['refusal_fraction']:.0%}"
        + ("   ** LOW_CONFIDENCE (refusals > 20%) **" if report["low_confidence"] else ""),
        "",
        "  Clusters:",
    ]
    for row in report["cluster_table"]:
        lines.append(
            f"    [{row['size']:>2}/{report['n']}] ({row['fraction']:.0%})  {row['exemplar']}"
        )
    if report["difference_labels"]:
        lines.append("")
        lines.append("  Difference dimensions (from DIFFERENT verdicts):")
        for label, count in sorted(report["difference_labels"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {count}x {label}")
    if report["non_transitivity_log"]:
        lines.append("")
        lines.append(
            f"  Non-transitive judgment triples observed: {len(report['non_transitivity_log'])}"
        )
    stats = report["cache_stats"]
    lines.append("")
    lines.append(f"  Cache: {stats['hits']} hits / {stats['misses']} misses")
    return "\n".join(lines)


def score_artifact(artifact_text: str, config: RunConfig, client: ModelClient, cache: Cache) -> dict:
    """Full pipeline for one artifact: sample -> cluster -> score -> report."""
    sample_set = draw_samples(artifact_text, config, client, cache)
    strategy = make_strategy(config.strategy)
    clustering = strategy.cluster(artifact_text, sample_set.samples, config, client, cache)
    return build_report(artifact_text, sample_set, clustering, config, cache, strategy.version)
