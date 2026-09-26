"""Original seeded dialogue-clustered inference for the primary table."""
from __future__ import annotations
import random
from typing import Any
BOOTSTRAP_REPLICATES = 20_000
PERMUTATION_REPLICATES = 100_000

def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = int(probability * (len(ordered) - 1))
    return ordered[position]

def summarize(
    contributions: dict[int, int],
    dialogue_ids: list[int],
    *,
    seed: int,
) -> dict[str, Any]:
    observed_net = sum(contributions.get(dialogue_id, 0) for dialogue_id in dialogue_ids)
    denominator = 3 * len(dialogue_ids)

    bootstrap_rng = random.Random(seed)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_net = sum(
            contributions.get(bootstrap_rng.choice(dialogue_ids), 0)
            for _ in dialogue_ids
        )
        bootstrap.append(100 * sampled_net / denominator)

    nonzero = [
        contributions.get(dialogue_id, 0)
        for dialogue_id in dialogue_ids
        if contributions.get(dialogue_id, 0) != 0
    ]
    permutation_rng = random.Random(seed + 1)
    observed_abs = abs(observed_net)
    extreme = 0
    for _ in range(PERMUTATION_REPLICATES):
        permuted = sum(
            value if permutation_rng.random() < 0.5 else -value
            for value in nonzero
        )
        extreme += abs(permuted) >= observed_abs

    return {
        "dialogues": len(dialogue_ids),
        "probes": denominator,
        "nonzero_dialogue_clusters": len(nonzero),
        "net": observed_net,
        "gain_pp": 100 * observed_net / denominator,
        "clustered_bootstrap_95ci_pp": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "cluster_sign_flip_two_sided_p": (extreme + 1) / (PERMUTATION_REPLICATES + 1),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
    }
