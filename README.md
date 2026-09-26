# Forgotten-State Contrastive Routing (FSCR)

Compact anonymous artifact for **Memory Is Not Control: Recovering Explicit
Forgetting in Reasoning-Distilled Models via Contrastive Routing**.

FSCR compares completed base and reasoning-model answers against forgotten
content. It keeps the reasoning-model answer unless all three frozen margins
support switching to the base answer. The router uses no correctness labels.

## Run

Python 3.10+; standard library only. No installation, GPU, API key, or model
download is required for the offline checks.

```bash
python scripts/reproduce.py
python -m fscr < data/examples.jsonl
# Optional: reproduce the original seeded dialogue-clustered inference.
python scripts/reproduce.py --clustered --output output/recomputed.json
```

To route your own candidates, supply one JSON object per line with
`base_answer`, `distill_answer`, and a `forgotten` list of strings. An optional
`eval_id` is returned unchanged. For dialogue-derived state, use
`python -m fscr --state dialogue` and provide `conversation` (turns with
`turn_id`, `role`, `content`) and `target_turn` instead of `forgotten`.
The extractor requires an explicit forget directive before the target.

Annotated thresholds are `(0.50, 0.40, 0.15)`; dialogue-derived thresholds
are `(0.40, 0.50, 0.00)`. Scoring and extraction are extracted from the
experimental implementation. The small CLI and reproduction script are
portable release wrappers.

## Contents

| Path | Contents |
|---|---|
| `fscr/` | Lexical scorer, frozen router, and dialogue-only state extractor |
| `data/dialogues/` | All 994 source dialogues, state units, original ICF context, and three ICF-DSM forgetting targets per dialogue |
| `data/main.csv` | All 15,904 primary benchmark/pair records, cached leakage scores, labels, and three fresh votes on each routed candidate |
| `data/state_inputs.csv` | All 5,208 held-out pair/probe records for the three state inputs and routing controls |
| `data/mquake_scores.csv` | All 5,996 aligned MQuAKE pair/example records used by the external table |
| `data/examples.jsonl` | 16 deterministic text fixtures: first two routed IDs in lexical order per primary benchmark/pair |
| `results/` | Reference counts, judge agreement, and dialogue-clustered intervals |
| `scripts/` | Offline route replay, paired metrics, and optional clustered inference |

## Scope and evaluation

This is a **core method and offline score-reproduction release**, not the
complete research archive. It covers the primary FSCR table, state-input and
routing-control comparisons, judge vote agreement, and MQuAKE paired effects.
The main score table retains every probe, including non-routed cases and
regressions. No rows are filtered by whether they help the method.

Primary routed candidates use the recorded two-of-three vote majority;
non-routed candidates retain the cached labels. State-input comparisons use
their separately cached labels, so their numbers must not replace the primary
fresh-vote results. Full sets include development dialogues. The common
held-out tail is source ID `>=562`: 434 dialogues / 1,302 probes per model pair.
Dialogue-derived thresholds were selected on source IDs 60–159.

MQuAKE uses the existing candidate/label intersection (998–1,000 examples per
pair) and its cached labels, rather than the primary three-vote protocol.
Its scores are provided for metric replay; its full candidate texts are not
included. The text fixtures check lexical implementation and are not an
evaluation subset. Bulk candidate generations, model weights, new GPU/API
execution, mechanism experiments, and human-audit records are outside this
compact package. Replaying cached scores does not independently validate the
judges or regenerate the answers.

The lexical rule can miss paraphrases or favor non-answers. Task and model
coverage are limited to the named settings; this package does not establish
universal improvement across benchmarks or model families.

## Data provenance

ICF-DSM derives from [ICF-Bench](https://github.com/qianyuli123/ICF-Bench).
MQuAKE-derived score records refer to
[MQuAKE](https://github.com/princeton-nlp/MQuAKE). Upstream datasets and their
attribution remain with their original authors. This release does not grant
a new license over third-party material. Dataset text is retained for
reproduction; author/account names and machine paths are not release metadata.

`MANIFEST.json` records file checksums. `python scripts/reproduce.py` verifies
them before replaying the results. No external calls are made by these scripts.

## Release validation

The complete offline check, both CLI modes, and all eight seeded primary
dialogue-clustered bootstrap/sign-flip results were verified against the
recorded aggregates. These checks reroute cached evidence and recompute
statistics; they do not regenerate model answers or obtain new judge labels.
