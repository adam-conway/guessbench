"""T5 judge audit: export sample pairs for blind human labeling; compute Cohen's kappa.

The export is labelable in under an hour (SPEC 8): pairs in randomized order, no
strategy verdicts visible. Verdicts live in a separate answer-key file that is
joined back by pair_id when computing kappa.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.providers import ModelClient
from guessbench.strategies.embedding import get_embeddings


def _pair_id(artifact_id: str, i: int, j: int) -> str:
    return f"{artifact_id}:{min(i, j)}-{max(i, j)}"


def select_audit_pairs(
    artifacts: dict[str, dict],
    config: RunConfig,
    client: ModelClient,
    cache: Cache,
    per_stratum: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Stratified pair selection (SPEC 6 T5): ~10 same-cluster, ~10 cross-cluster,
    ~10 near-boundary (closest cross-cluster pairs by embedding distance).

    `artifacts` maps artifact_id -> {"text", "samples", "assignments"}.
    """
    rng = random.Random(seed)
    same_cluster: list[dict] = []
    cross_cluster: list[dict] = []

    for artifact_id, art in artifacts.items():
        assignments = art["assignments"]
        vectors = get_embeddings(art["samples"], config, client, cache)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.where(norms == 0, 1, norms)
        distances = 1 - normalized @ normalized.T
        n = len(assignments)
        for i in range(n):
            for j in range(i + 1, n):
                pair = {
                    "pair_id": _pair_id(artifact_id, i, j),
                    "artifact_id": artifact_id,
                    "artifact_text": art["text"],
                    "output_a": art["samples"][i],
                    "output_b": art["samples"][j],
                    "same_cluster": assignments[i] == assignments[j],
                    "distance": float(distances[i, j]),
                }
                (same_cluster if pair["same_cluster"] else cross_cluster).append(pair)

    rng.shuffle(same_cluster)
    rng.shuffle(cross_cluster)
    # Near-boundary: closest cross-cluster pairs by embedding distance.
    near_boundary = sorted(cross_cluster, key=lambda p: p["distance"])[:per_stratum]
    boundary_ids = {p["pair_id"] for p in near_boundary}
    plain_cross = [p for p in cross_cluster if p["pair_id"] not in boundary_ids][:per_stratum]

    selected = same_cluster[:per_stratum] + plain_cross + near_boundary
    for pair, stratum in zip(
        selected,
        ["same_cluster"] * len(same_cluster[:per_stratum])
        + ["cross_cluster"] * len(plain_cross)
        + ["near_boundary"] * len(near_boundary),
    ):
        pair["stratum"] = stratum
    rng.shuffle(selected)
    return selected


def write_audit_export(pairs: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Write the blind labeling sheet and the separate answer key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labeling_sheet = [
        {
            "pair_id": p["pair_id"],
            "artifact_text": p["artifact_text"],
            "output_a": p["output_a"],
            "output_b": p["output_b"],
            "human_label": "",  # fill with INTERCHANGEABLE or DIFFERENT
        }
        for p in pairs
    ]
    answer_key = [
        {
            "pair_id": p["pair_id"],
            "strategy_verdict": "INTERCHANGEABLE" if p["same_cluster"] else "DIFFERENT",
            "stratum": p["stratum"],
            "distance": p["distance"],
        }
        for p in pairs
    ]
    sheet_path = out_dir / "t5_labeling_sheet.json"
    key_path = out_dir / "t5_answer_key.json"
    sheet_path.write_text(json.dumps(labeling_sheet, indent=2, ensure_ascii=False), encoding="utf-8")
    key_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    return sheet_path, key_path


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two binary raters."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label lists must be equal-length and non-empty")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    categories = set(labels_a) | set(labels_b)
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def compute_kappa_from_files(labeled_sheet_path: Path, answer_key_path: Path) -> dict:
    """Join human labels to strategy verdicts by pair_id and compute kappa (T5:
    pass at kappa >= 0.7)."""
    sheet = json.loads(labeled_sheet_path.read_text(encoding="utf-8"))
    key = {row["pair_id"]: row for row in json.loads(answer_key_path.read_text(encoding="utf-8"))}

    human, strategy = [], []
    skipped = []
    for row in sheet:
        label = row.get("human_label", "").strip().upper()
        if label not in ("INTERCHANGEABLE", "DIFFERENT"):
            skipped.append(row["pair_id"])
            continue
        human.append(label)
        strategy.append(key[row["pair_id"]]["strategy_verdict"])

    kappa = cohens_kappa(human, strategy)
    return {
        "kappa": round(kappa, 4),
        "n_labeled": len(human),
        "n_skipped": len(skipped),
        "skipped_pair_ids": skipped,
        "passed": kappa >= 0.7,
    }
