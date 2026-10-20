import pytest

from src.statistics import factorial_interaction, holm_adjust, paired_comparison


def records(values):
    return [
        {"problem_id": str(i), "ground_truth": "42", "correct": v, "scorer_version": "v2"} for i, v in enumerate(values)
    ]


def test_paired_statistics_and_holm():
    result = paired_comparison(records([True, True, False, True]), records([False, True, True, False]), replicates=500)
    assert result["left_only_correct"] == 2
    assert result["right_only_correct"] == 1
    assert result["delta"] == 0.25
    assert result["paired_bootstrap_95"][0] <= 0.25 <= result["paired_bootstrap_95"][1]
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    with pytest.raises(ValueError):
        paired_comparison(records([True]), records([False, True]))
    zeros = records([False] * 4)
    ones = records([True] * 4)
    assert factorial_interaction(zeros, ones, zeros, ones, replicates=100)["interaction"] == 0


def test_campaign_reports_training_variation_separately():
    from src.statistics import analyze_campaign

    arms = [
        "controlled_verified",
        "controlled_answer_only",
        "controlled_unfiltered",
        "controlled_grpo",
        "controlled_verified_grpo",
        "controlled_verified_tokens",
        "controlled_unfiltered_tokens",
    ]
    campaign = {arm: {seed: records([True, False, True, False]) for seed in ["42", "43", "44"]} for arm in arms}
    campaign["base_model"] = {"42": records([True, False, True, False])}
    result = analyze_campaign(campaign, [["controlled_verified", "controlled_unfiltered"]])
    assert result["arms"]["controlled_verified"]["training_seed_std"] == 0
    assert result["primary_paired_by_seed"]["42"]["controlled_verified__vs__controlled_unfiltered"]["delta"] == 0
