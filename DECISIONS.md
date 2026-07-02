# DECISIONS.md

Every [AGENT'S CHOICE] exercised and every assumption made under genuine ambiguity, per SPEC 0.1.

## Models

- **Local models via Ollama instead of the spec's `claude-sonnet-4-6` default** — explicit user request; the spec makes models configurable and always recorded, so defaults are `llama3.1:8b` (reference), `qwen3:8b` (judge), `nomic-embed-text` (embeddings). Anthropic remains available via `--provider anthropic`.
- **Judge model `qwen3:8b`** — satisfies the pinned judge ≠ reference constraint within the local stack; different model family reduces shared-bias risk.
- **Embedding model `nomic-embed-text`** — served by the same Ollama instance, so Strategy B needs no extra provider or API key.
- **Judge parse strips `<think>...</think>` blocks** — qwen3 is a reasoning model; the last VERDICT outside think blocks wins, and unparseable judge responses raise instead of guessing.

## Stack and layout

- **Python 3.11+, argparse CLI, pytest** — the spec's suggested obvious path.
- **httpx / numpy / scipy / scikit-learn / pyyaml** — minimal set covering API calls, entropy math, Spearman, agglomerative clustering, and data files.
- **Cache: JSON files on disk keyed by SHA-256** — content-addressed per SPEC 3.1, human-inspectable, no DB dependency.
- **Ladder file format: YAML** — more readable than JSON for multi-line prompts.

## Algorithmic choices

- **Judge prompt wording** (`guessbench/strategies/llm_judge.py`) — asks the pinned author-interchangeability question, instructs the judge to ignore wording/formatting, and demands the constrained `VERDICT:` / `DIMENSION:` format.
- **Judge calls at temperature 0, max_tokens 2000** — determinism for equivalence decisions; headroom for reasoning-model think tokens.
- **Presentation order (A/B) derived from a per-pair seeded hash** — deterministic per (seed, artifact, pair) yet ~50/50 across pairs, and recorded in the judgment cache key as pinned.
- **Sample cache key uses `seed * N + i` as `sample_index`** — keeps the spec-pinned key formula while giving T3's disjoint-seed runs disjoint samples.
- **Embedding clustering: average linkage, cosine distance** — standard for text embeddings; the distance threshold is selected by the acceptance sweep (grid 0.05–0.7), not hand-picked.
- **When several thresholds pass the sweep, the median passing value wins** — most robust to boundary effects; if none pass, the least-failing is reported (and the build is declared not done, per SPEC 4.5).
- **"Wide margin" for the A4 > A2 ordinal assertion: EI(A4) ≥ 2× EI(A2)** — the spec pins the ordinal constraint but not the margin; 2x is unambiguous and generous.
- **A4 absolute floor kept at the spec's suggested 4.0** — tunable constant `A4_MIN_EI` in `acceptance.py`.
- **Refusal detection: regex heuristics over the first 500 chars** — cheap and transparent; refusals still cluster normally (pinned), the flag only feeds the report. An LLM refusal classifier would add judge-model noise to a reporting-only signal.
- **Strategy C (NLI bidirectional entailment) not implemented** — not cheap to add locally: no NLI model is available through Ollama, so it would drag in a separate inference stack (transformers/torch) for a strategy the spec expects to fail on long-form output. Optional per SPEC 4.4.

## Interface

- **T5 export: two JSON files** — a blind labeling sheet (`human_label` to fill in) and a separate answer key joined by `pair_id`, so the labeler never sees strategy verdicts (SPEC 8).
- **Judge-audit artifact set: A2, A3, code:L1, code:L3, business:L1, how-to:L2** — spans low/high ambiguity and multiple families so the ~30 pairs stratify meaningfully.
- **T3 stability artifacts: A2, code:L1, code:L3** — one anchor, one L1, one L3 as pinned.
- **`calibrate` prints a PENDING_REVIEW warning** — T2 results are not counted as real until a human approves ladder content (SPEC 8.1).

## Assumptions

- **Anthropic Messages API has no seed parameter** — sample independence comes from temperature sampling; the Ollama client does pass a per-sample seed.
- **A1's "after whitespace-trim, all samples identical" is asserted as k = 1 and EI = 1.0** — the clustering strategies are the mechanism; a strategy that cannot merge trivially identical outputs should fail T1, which is the point of the anchor.
- **This session was build + unit-test only** — no live calibration was run; the acceptance table against real models is still outstanding, along with both human gates (ladder review, T5 labels).
