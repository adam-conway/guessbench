"""CLI: score, calibrate, judge-audit (SPEC 7)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from guessbench.acceptance import calibration_artifacts, load_ladders, run_calibration
from guessbench.cache import Cache
from guessbench.config import RunConfig
from guessbench.judge_audit import compute_kappa_from_files, select_audit_pairs, write_audit_export
from guessbench.providers import make_client
from guessbench.report import render_report, score_artifact
from guessbench.sampling import draw_samples
from guessbench.strategies import make_strategy


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--provider", default=None, choices=["ollama", "anthropic"])
    parser.add_argument("--strategy", default=None, choices=["llm_judge", "embedding"])
    parser.add_argument("-n", "--n-samples", type=int, default=None)
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--embedding-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    parser.add_argument("--out-dir", default="runs", help="directory for output files")


def build_config(args: argparse.Namespace) -> RunConfig:
    overrides = {
        name: value
        for name, value in {
            "reference_model": args.reference_model,
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model,
            "provider": args.provider,
            "strategy": args.strategy,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "embedding_threshold": args.embedding_threshold,
            "seed": args.seed,
            "cache_dir": args.cache_dir,
        }.items()
        if value is not None
    }
    if args.no_cache:
        overrides["use_cache"] = False
    return RunConfig(**overrides)


def make_runtime(config: RunConfig):
    client = make_client(config.provider, ollama_base_url=config.ollama_base_url)
    cache = Cache(config.cache_dir, enabled=config.use_cache)
    return client, cache


def cmd_score(args: argparse.Namespace) -> int:
    config = build_config(args)
    client, cache = make_runtime(config)
    artifact_text = Path(args.artifact_file).read_text(encoding="utf-8").strip()

    report = score_artifact(artifact_text, config, client, cache)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"score_{report['prompt_hash']}_{report['config']['config_hash']}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(render_report(report))
    print(f"\nJSON report: {out_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    config = build_config(args)
    client, cache = make_runtime(config)
    strategies = args.strategies.split(",") if args.strategies else None

    result = run_calibration(config, client, cache, strategies=strategies)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reports_by_strategy": result["reports_by_strategy"],
        "selected_embedding_threshold": result["selected_embedding_threshold"],
        "threshold_sweep": result["threshold_sweep"],
        "results_by_strategy": {
            strategy: [dataclasses.asdict(r) for r in results]
            for strategy, results in result["results_by_strategy"].items()
        },
    }
    out_path = out_dir / "calibration.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ladders_pending = [lad["family"] for lad in load_ladders() if lad.get("status") == "PENDING_REVIEW"]
    print(result["table"])
    if result["selected_embedding_threshold"] is not None:
        print(f"\nSelected embedding threshold: {result['selected_embedding_threshold']}")
    if ladders_pending:
        print(
            "\nNOTE: ladders are PENDING_REVIEW — T2 results are not real until a "
            f"human approves the ladder content ({', '.join(ladders_pending)})."
        )
    print("NOTE: T5 (judge agreement) is human-gated; run `guessbench judge-audit export`.")
    print(f"\nJSON results: {out_path}")
    return 0


AUDIT_ARTIFACT_IDS = ["A2", "A3", "code:L1", "code:L3", "business:L1", "how-to:L2"]


def cmd_judge_audit(args: argparse.Namespace) -> int:
    config = build_config(args)

    if args.labeled_file:
        result = compute_kappa_from_files(Path(args.labeled_file), Path(args.answer_key))
        print(json.dumps(result, indent=2))
        print("\nT5:", "PASS (kappa >= 0.7)" if result["passed"] else "FAIL (kappa < 0.7)")
        return 0 if result["passed"] else 1

    client, cache = make_runtime(config)
    all_artifacts = calibration_artifacts()
    strategy = make_strategy(
        config.strategy, **({"threshold": config.embedding_threshold} if config.strategy == "embedding" else {})
    )
    audit_inputs = {}
    for artifact_id in AUDIT_ARTIFACT_IDS:
        text = all_artifacts[artifact_id]
        sample_set = draw_samples(text, config, client, cache)
        clustering = strategy.cluster(text, sample_set.samples, config, client, cache)
        audit_inputs[artifact_id] = {
            "text": text,
            "samples": sample_set.samples,
            "assignments": clustering.assignments,
        }

    pairs = select_audit_pairs(audit_inputs, config, client, cache, seed=config.seed)
    sheet_path, key_path = write_audit_export(pairs, Path(args.out_dir))
    print(f"Labeling sheet ({len(pairs)} pairs, blind, randomized): {sheet_path}")
    print(f"Answer key (do not open while labeling):              {key_path}")
    print(
        "\nFill each pair's human_label with INTERCHANGEABLE or DIFFERENT, then run:\n"
        f"  guessbench judge-audit --labeled-file {sheet_path} --answer-key {key_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guessbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="score one artifact file (SPEC 2.5 report)")
    p_score.add_argument("artifact_file")
    add_common_args(p_score)
    p_score.set_defaults(func=cmd_score)

    p_cal = sub.add_parser("calibrate", help="run anchors + ladders across strategies (T1-T4)")
    p_cal.add_argument("--strategies", default=None, help="comma-separated, default: llm_judge,embedding")
    add_common_args(p_cal)
    p_cal.set_defaults(func=cmd_calibrate)

    p_audit = sub.add_parser("judge-audit", help="export T5 pairs or compute kappa from labels")
    p_audit.add_argument("--labeled-file", default=None, help="human-labeled sheet (computes kappa)")
    p_audit.add_argument("--answer-key", default=None, help="answer key path (with --labeled-file)")
    add_common_args(p_audit)
    p_audit.set_defaults(func=cmd_judge_audit)

    args = parser.parse_args(argv)
    if getattr(args, "labeled_file", None) and not args.answer_key:
        parser.error("--labeled-file requires --answer-key")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
