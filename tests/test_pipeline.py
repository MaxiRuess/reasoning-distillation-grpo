import pytest

from src.data import load_config
from src.pipeline import validate_training_request
from src.reward import score_response


def test_invalid_conditions_and_missing_parent_are_rejected_before_loading():
    config = load_config()
    for condition, phase, parent in [
        ("unknown", "sft", None),
        ("grpo_only", "sft", None),
        ("sft_then_grpo", "grpo", None),
        ("grpo_only", "grpo", "wrong-parent"),
    ]:
        with pytest.raises(ValueError):
            validate_training_request(config, condition, phase, parent, check_artifact=False)
    validate_training_request(config, "re_distill", "sft")


def test_symbolic_verification_and_legacy_policy_are_explicit():
    assert score_response(r"\boxed{\frac{1}{2}+7}", "7.5", "math500").correct
    assert not score_response(r"\boxed{\frac{1}{2}+7}", "0.5", "math500").correct
    assert score_response(r"\boxed{1,234}", "1234").correct
    assert score_response("The answer is 1,234.", "1234", scorer="legacy-v1").correct is False
    assert score_response("<think>42", "42").status == "unfinished_thinking"


def test_grpo_token_accounting_uses_state_not_the_unused_sft_counter():
    from types import SimpleNamespace

    from src.pipeline import processed_training_tokens

    trainer = SimpleNamespace(_total_train_tokens=0, state=SimpleNamespace(num_input_tokens_seen=54492))
    assert processed_training_tokens(trainer, "grpo") == 54492
    trainer._total_train_tokens = 521338
    assert processed_training_tokens(trainer, "sft") == 521338


def test_weighted_reward_diagnostics_ignore_disabled_format_component():
    from src.pipeline import RewardDiagnostics

    diagnostics = RewardDiagnostics([1, 0], 2)
    diagnostics.record([0, 0], [0.5, 1])
    first = diagnostics.take_logs()
    assert first["objective/reward_mean"] == 0
    assert first["objective/zero_variance_fraction"] == 1
    diagnostics.record([0, 1], [1, 0])
    assert diagnostics.take_logs()["objective/reward_mean"] == 0.5
    assert diagnostics.take_logs() == {}
    assert diagnostics.summary()["rollouts"] == 4
    assert diagnostics.summary()["zero_variance_groups"] == 1
