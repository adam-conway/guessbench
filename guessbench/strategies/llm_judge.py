"""Strategy A: LLM pairwise judge with greedy clustering against representatives (SPEC 4.1-4.2)."""

from __future__ import annotations

import random
import re

from guessbench.cache import Cache, judgment_key
from guessbench.config import RunConfig, stable_hash
from guessbench.providers import ModelClient
from guessbench.strategies.base import ClusteringResult, EquivalenceStrategy

JUDGE_PROMPT_TEMPLATE = """You are judging whether two responses to the same request count as the same interpretation of that request.

<request>
{artifact}
</request>

<output_A>
{output_a}
</output_A>

<output_B>
{output_b}
</output_B>

Question: would the author of this request consider these two outputs interchangeable — differing only in ways the request left open to chance and the author would not care to distinguish — or meaningfully different?

Ignore differences in wording, phrasing, formatting, or style that do not change what the output substantively is or does. Focus on decisions the request left unresolved: if the two outputs resolved such a decision differently (e.g., different programming language, different tone, different sort direction, different content choices the author would care about), they are DIFFERENT.

Answer in exactly this format:
VERDICT: INTERCHANGEABLE
or
VERDICT: DIFFERENT
DIMENSION: <2-5 word label naming the dimension of difference>"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(INTERCHANGEABLE|DIFFERENT)", re.IGNORECASE)
_DIMENSION_RE = re.compile(r"DIMENSION:\s*(.+)", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_judge_response(text: str) -> tuple[str, str | None]:
    """Extract (verdict, dimension_label) from a judge response.

    Reasoning-model think blocks are stripped first; the last VERDICT found wins.
    Unparseable responses raise, surfacing judge failures instead of guessing.
    """
    cleaned = _THINK_RE.sub("", text)
    verdicts = _VERDICT_RE.findall(cleaned)
    if not verdicts:
        raise ValueError(f"Judge response contains no parseable verdict: {text[:200]!r}")
    verdict = verdicts[-1].upper()
    label = None
    if verdict == "DIFFERENT":
        match = _DIMENSION_RE.search(cleaned)
        label = match.group(1).strip().strip(".").lower() if match else "unlabeled"
    return verdict, label


class LLMJudgeStrategy(EquivalenceStrategy):
    strategy_id = "llm_judge"
    version = "v1"

    def _judge_pair(
        self,
        artifact_text: str,
        text_i: str,
        text_j: str,
        order: str,
        config: RunConfig,
        client: ModelClient,
        cache: Cache,
    ) -> tuple[str, str | None]:
        """One independent judge call (no accumulated context, SPEC 9), cached with
        presentation order in the key."""
        output_a, output_b = (text_i, text_j) if order == "AB" else (text_j, text_i)
        key = judgment_key(
            config.judge_model,
            config.judge_prompt_version,
            artifact_text,
            output_a,
            output_b,
            order,
        )
        cached = cache.get("judgments", key)
        if cached is not None:
            return cached["verdict"], cached.get("label")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            artifact=artifact_text, output_a=output_a, output_b=output_b
        )
        raw = client.complete(
            model=config.judge_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=2000,
        )
        verdict, label = parse_judge_response(raw)
        cache.put(
            "judgments",
            key,
            {
                "verdict": verdict,
                "label": label,
                "raw": raw,
                "presentation_order": order,
                "judge_model": config.judge_model,
                "judge_prompt_version": config.judge_prompt_version,
            },
        )
        return verdict, label

    def cluster(
        self,
        artifact_text: str,
        samples: list[str],
        config: RunConfig,
        client: ModelClient,
        cache: Cache,
    ) -> ClusteringResult:
        n = len(samples)
        order_indices = list(range(n))
        rng = random.Random(config.seed)
        rng.shuffle(order_indices)

        # clusters: list of member sample-indices; first member is the representative.
        clusters: list[list[int]] = []
        difference_labels: list[str] = []
        pair_records: list[dict] = []
        # Judgment graph over sample indices, for non-transitivity scanning.
        judged: dict[tuple[int, int], str] = {}

        for idx in order_indices:
            joined = False
            for cluster in clusters:
                rep = cluster[0]
                # Seeded per-pair presentation order (washes out position bias, SPEC 4.2).
                pair_seed = stable_hash(config.seed, artifact_text, rep, idx)
                order = "AB" if random.Random(pair_seed).random() < 0.5 else "BA"
                verdict, label = self._judge_pair(
                    artifact_text, samples[idx], samples[rep], order, config, client, cache
                )
                judged[(min(idx, rep), max(idx, rep))] = verdict
                pair_records.append(
                    {
                        "sample_i": idx,
                        "sample_j": rep,
                        "verdict": verdict,
                        "label": label,
                        "presentation_order": order,
                    }
                )
                if verdict == "INTERCHANGEABLE":
                    cluster.append(idx)
                    joined = True
                    break
                if label:
                    difference_labels.append(label)
            if not joined:
                clusters.append([idx])

        assignments = [0] * n
        for cluster_id, members in enumerate(clusters):
            for member in members:
                assignments[member] = cluster_id

        return ClusteringResult(
            assignments=assignments,
            difference_labels=difference_labels,
            non_transitivity_log=_find_non_transitive_triples(judged),
            pair_records=pair_records,
        )


def _find_non_transitive_triples(judged: dict[tuple[int, int], str]) -> list[dict]:
    """Scan recorded judgments for triples where A~B, B~C, but A!~C (SPEC 4.1).

    The pinned greedy algorithm stops at the first INTERCHANGEABLE match, so fully
    judged inconsistent triangles are rare; this logs any that do appear.
    """
    nodes = sorted({i for pair in judged for i in pair})
    inconsistent = []
    for a_pos, a in enumerate(nodes):
        for b in nodes[a_pos + 1 :]:
            for c in nodes:
                if c <= b:
                    continue
                ab = judged.get((a, b))
                bc = judged.get((b, c))
                ac = judged.get((a, c))
                if None in (ab, bc, ac):
                    continue
                same = [ab, bc, ac].count("INTERCHANGEABLE")
                if same == 2:
                    inconsistent.append({"triple": [a, b, c], "verdicts": [ab, bc, ac]})
    return inconsistent
