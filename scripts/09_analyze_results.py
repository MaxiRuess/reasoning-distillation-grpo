"""Analyze archived historical outcomes without pretending they validate the checkpoints."""

import argparse
import json
from pathlib import Path

from src.artifacts import atomic_json
from src.statistics import analyze_campaign, holm_adjust, paired_comparison, wilson


def historical_analysis(archive):
    conditions = ["base_model", "baseline", "sft_traces", "grpo_only", "sft_then_grpo", "re_distill"]
    rows, summaries = {}, {}
    for name in conditions:
        data = json.loads((Path(archive) / "eval_checkpoints" / f"{name}_gsm8k_final.json").read_text())
        records = data["results"]
        if len(records) != 1319 or [r["idx"] for r in records] != list(range(1319)):
            raise ValueError("Historical records have missing or duplicate IDs")
        correct = sum(bool(r["correct"]) for r in records)
        if correct != data["num_correct"]:
            raise ValueError("Historical count does not reconcile")
        summaries[name] = {
            "num_correct": correct,
            "num_total": 1319,
            "accuracy": correct / 1319,
            "wilson_95": wilson(correct, 1319),
        }
        rows[name] = [
            {
                "problem_id": str(r["idx"]),
                "ground_truth": r["gt"],
                "correct": r["correct"],
                "scorer_version": "legacy-v1",
            }
            for r in records
        ]
    comparisons = {}
    for a, b in [
        ("re_distill", "sft_traces"),
        ("grpo_only", "base_model"),
        ("sft_then_grpo", "sft_traces"),
        ("sft_traces", "base_model"),
        ("re_distill", "grpo_only"),
    ]:
        comparisons[a + "__vs__" + b] = paired_comparison(rows[a], rows[b])
    adjusted = holm_adjust([result["mcnemar_exact_p"] for result in comparisons.values()])
    for result, p in zip(comparisons.values(), adjusted, strict=True):
        result["holm_adjusted_p_exploratory_family"] = p
    return {
        "status": "historical_outcomes_only",
        "checkpoint_validation": "pending",
        "scorer": "legacy-v1",
        "raw_responses_available": False,
        "caution": "These paired outcomes do not repair checkpoint reconstruction or support causal attribution.",
        "conditions": summaries,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--historical-archive")
    source.add_argument("--campaign-manifest", help="JSON mapping arm -> seed -> local predictions.jsonl")
    parser.add_argument("--output", default="results/recovery/historical-paired-analysis.json")
    args = parser.parse_args()
    if args.historical_archive:
        result = historical_analysis(args.historical_archive)
    else:
        from src.data import load_config

        manifest = json.loads(Path(args.campaign_manifest).read_text())
        campaign = {
            arm: {
                seed: [json.loads(line) for line in Path(path).read_text().splitlines()] for seed, path in seeds.items()
            }
            for arm, seeds in manifest.items()
        }
        result = analyze_campaign(campaign, load_config()["study"]["primary_comparisons"])
    atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
