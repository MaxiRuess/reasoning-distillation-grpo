"""Paired problem-level inference; training-seed variability is reported separately."""

import numpy as np
from scipy.stats import binomtest


def wilson(correct, total):
    if not 0 <= correct <= total or total == 0:
        raise ValueError("Valid nonzero denominator required")
    z = 1.95996398454
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - half), float(center + half)]


def paired_comparison(left, right, seed=42, replicates=10000):
    if not left or len(left) != len(right):
        raise ValueError("Both complete paired result lists are required")
    left_ids = [str(row["problem_id"]) for row in left]
    right_ids = [str(row["problem_id"]) for row in right]
    if left_ids != right_ids or len(set(left_ids)) != len(left_ids):
        raise ValueError("Paired problem IDs must be unique and in identical order")
    if any(a["ground_truth"] != b["ground_truth"] for a, b in zip(left, right, strict=True)):
        raise ValueError("Paired ground truths differ")
    if any(a.get("scorer_version") != b.get("scorer_version") for a, b in zip(left, right, strict=True)):
        raise ValueError("Paired scoring policies differ")
    a = np.array([bool(row["correct"]) for row in left])
    b = np.array([bool(row["correct"]) for row in right])
    diff = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 500):
        indices = rng.integers(len(diff), size=(min(500, replicates - start), len(diff)))
        draws.extend(diff[indices].mean(axis=1).tolist())
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "num_paired": len(a),
        "left_correct": int(a.sum()),
        "right_correct": int(b.sum()),
        "delta": float(diff.mean()),
        "paired_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "left_only_correct": wins,
        "right_only_correct": losses,
        "mcnemar_exact_p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
    }


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, pvalues[i] * (len(pvalues) - rank)))
        adjusted[i] = previous
    return adjusted


def factorial_interaction(a, d, e, f, seed=42, replicates=10000):
    # These paired checks enforce shared questions, truths and scoring policies.
    for other in [d, e, f]:
        paired_comparison(a, other, seed, replicates=1)
    diff = np.array(
        [
            int(ff["correct"]) - int(dd["correct"]) - int(ee["correct"]) + int(aa["correct"])
            for aa, dd, ee, ff in zip(a, d, e, f, strict=True)
        ]
    )
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 500):
        draws.extend(diff[rng.integers(len(diff), size=(min(500, replicates - start), len(diff)))].mean(axis=1))
    return {"interaction": float(diff.mean()), "paired_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist()}


def analyze_campaign(campaign, comparisons):
    """campaign maps arm -> seed -> complete first-sample prediction records."""
    summaries, paired, interactions = {}, {}, {}
    for arm, seeds in campaign.items():
        accuracies = {seed: sum(row["correct"] for row in rows) / len(rows) for seed, rows in seeds.items()}
        summaries[arm] = {
            "per_seed_accuracy": accuracies,
            "mean_accuracy": float(np.mean(list(accuracies.values()))),
            "training_seed_std": float(np.std(list(accuracies.values()), ddof=1)) if len(seeds) > 1 else None,
        }
    primary_seeds = set(campaign.get("controlled_verified", {}))
    if primary_seeds != {"42", "43", "44"}:
        raise ValueError("All three predeclared training seeds are required")
    for seed in sorted(primary_seeds):
        family = {}
        for left, right in comparisons:
            family[left + "__vs__" + right] = paired_comparison(campaign[left][seed], campaign[right][seed])
        adjusted = holm_adjust([result["mcnemar_exact_p"] for result in family.values()])
        for result, p in zip(family.values(), adjusted, strict=True):
            result["holm_adjusted_p_primary_family"] = p
        paired[seed] = family
        interactions[seed] = factorial_interaction(
            campaign["base_model"]["42"],
            campaign["controlled_verified"][seed],
            campaign["controlled_grpo"][seed],
            campaign["controlled_verified_grpo"][seed],
        )
    secondary = {
        seed: paired_comparison(
            campaign["controlled_verified_tokens"][seed], campaign["controlled_unfiltered_tokens"][seed]
        )
        for seed in sorted(primary_seeds)
    }
    return {
        "arms": summaries,
        "primary_paired_by_seed": paired,
        "factorial_interaction_by_seed": interactions,
        "secondary_token_matched_by_seed": secondary,
        "uncertainty": "Bootstrap intervals resample test questions within a seed; training variation is reported separately.",
    }
