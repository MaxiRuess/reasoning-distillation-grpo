import copy
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data import chat_row, load_config
from src.distillation_experiment import (
    analyze_distillation,
    build_plan,
    build_quote,
    distillation_rows,
    execute_campaign,
    numeric_final_score,
    study_config,
)
from src.reward import final_response
from src.study import validate_quote


def test_matched_targets_keep_teacher_answer_and_common_prompt():
    source = [chat_row("Question?", "<think>Work</think>Final: \\boxed{\\frac{1}{2}}", "q", verified=True)]
    answer, trace = distillation_rows(source)
    assert answer[0]["prompt"] == trace[0]["prompt"]
    assert answer[0]["source_id"] == trace[0]["source_id"]
    assert answer[0]["completion"][0]["content"] == "\\boxed{\\frac{1}{2}}"
    assert trace[0]["completion"][0]["content"] == "<think>\nWork\n</think>\n\n\\boxed{\\frac{1}{2}}"
    assert final_response(trace[0]["completion"][0]["content"])[0].strip() == answer[0]["completion"][0]["content"]
    assert "boxed" in answer[0]["prompt"][0]["content"]
    assert "mathematical expression" in answer[0]["prompt"][0]["content"]
    assert "numerical value" not in answer[0]["prompt"][0]["content"]


@pytest.mark.parametrize(
    "response,correct,status",
    [
        ("<think>work</think>\\boxed{2} then \\boxed{2.0}", True, "ok"),
        ("\\boxed{2} or \\boxed{3}", False, "conflicting_final_answers"),
        ("<think>unfinished \\boxed{2}", False, "unfinished_thinking"),
        ("\\boxed{2} then \\boxed{", False, "malformed_final_answer"),
        ("The answer is 2", False, "missing_final_answer"),
        ("\\boxed{2.000001}", False, "ok"),
    ],
)
def test_prospective_numeric_endpoint(response, correct, status):
    score = numeric_final_score(response, "2")
    assert score["correct"] is correct
    assert score["status"] == status


def test_every_job_reserved_within_remaining_envelope():
    config = study_config(load_config())
    assert all(not arm["sft"]["packing"] and arm["sft"]["padding_free"] for arm in config["conditions"].values())
    protocol = {
        "protocol_id": "test",
        "config": config,
        "subsets": {
            "2048": {"problem_indices": list(range(256)), "rows_digest": "x"},
            "4096": {"problem_indices": list(range(64)), "rows_digest": "y"},
        },
    }
    plan = build_plan(protocol)
    quote = build_quote(plan)
    assert sum(n["kind"] == "sft" for n in plan["nodes"]) == 6
    assert sum(n["kind"] == "evaluate" for n in plan["nodes"]) == 14
    assert validate_quote(plan, quote, 18.13) + quote["non_runtime_allowance_usd"] <= 18.13
    with pytest.raises(ValueError):
        validate_quote(plan, quote, 5)


def test_item_uncertainty_does_not_count_three_seeds_as_more_questions():
    def rows(values):
        return [
            {
                "problem_id": str(i),
                "ground_truth": "2",
                "response": "\\boxed{2}" if value else "\\boxed{3}",
                "correct": value,
                "scorer_version": "boxed-final-v1",
                "finish_reason": "eos",
                "generated_tokens": 5,
            }
            for i, value in enumerate(values)
        ]

    campaign = {"base": rows([False, True])}
    for seed in (42, 43, 44):
        campaign[f"answer_{seed}"] = rows([False, True])
        campaign[f"trace_{seed}"] = rows([True, True])
    result = analyze_distillation(campaign)
    assert result["num_distinct_questions"] == 2
    assert result["primary"]["mean_delta"] == 0.5
    assert result["primary"]["training_seed_delta_std"] == 0
    broken = copy.deepcopy(campaign)
    broken["trace_44"][0]["ground_truth"] = "9"
    with pytest.raises(ValueError):
        analyze_distillation(broken)


def test_disjoint_execution_reserves_all_work_and_never_relaunches(tmp_path):
    config = study_config(load_config())
    plan = build_plan(
        {
            "protocol_id": "test",
            "subsets": {str(tokens): {"problem_indices": [0], "rows_digest": str(tokens)} for tokens in (2048, 4096)},
        }
    )
    quote = build_quote(plan)
    calls = []

    def prepare(**kwargs):
        calls.append("prepare")
        return {}

    def train(condition, seed, **kwargs):
        calls.append((condition, seed))
        return condition + str(seed)

    def evaluate(condition, max_new_tokens, **kwargs):
        calls.append((condition, max_new_tokens))
        return {"benchmarks": {"gsm8k": {"results_path": condition + str(max_new_tokens)}}}

    def verify(result_path, **kwargs):
        return {"summary": {"results_path": result_path}}

    callbacks = {
        "prepare": prepare,
        "prepare_runtime": prepare,
        "sft": train,
        "grpo": prepare,
        "evaluate": evaluate,
        "verify": verify,
    }
    ledger = execute_campaign(plan, config, quote, 17.45, tmp_path, **callbacks)
    assert ledger["status"] == "complete"
    assert len(ledger["results"]) == 37
    assert len(calls) == 23  # Three CPU preparations, six trainings, fourteen evaluations.
    assert execute_campaign(plan, config, quote, 17.45, tmp_path, **callbacks) == ledger
    assert len(calls) == 23


def test_failed_paid_chain_is_not_retried(tmp_path):
    config = study_config(load_config())
    plan = build_plan(
        {
            "protocol_id": "test",
            "subsets": {str(tokens): {"problem_indices": [0], "rows_digest": str(tokens)} for tokens in (2048, 4096)},
        }
    )
    quote = build_quote(plan)
    calls = []

    def cpu(**kwargs):
        return {}

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("paid failure")

    callbacks = {"prepare": cpu, "prepare_runtime": cpu, "sft": fail, "grpo": fail, "evaluate": fail, "verify": cpu}
    with pytest.raises(RuntimeError, match="paid chains failed"):
        execute_campaign(plan, config, quote, 17.45, tmp_path, **callbacks)
    assert len(calls) == 4
    with pytest.raises(ValueError, match="no automatic retries"):
        execute_campaign(plan, config, quote, 17.45, tmp_path, **callbacks)
    assert len(calls) == 4


@pytest.mark.parametrize("fail", [False, True])
def test_cli_persists_closed_app_state_for_success_and_failure(tmp_path, monkeypatch, fail):
    spec = importlib.util.spec_from_file_location("distillation_cli", Path("scripts/17_distillation_study.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proposal = tmp_path / "proposal"
    proposal.mkdir()
    (proposal / "quote.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["study", "--execute", "--proposal", str(proposal), "--budget-usd", "17.85"])
    monkeypatch.setattr(module, "load_config", dict)
    monkeypatch.setattr(
        module,
        "validate_proposal",
        lambda *a: ({"protocol_id": "test", "config": {}}, {"nodes": []}, {"non_runtime_allowance_usd": 0.4}),
    )
    app = SimpleNamespace(run=lambda: nullcontext(), app_id="ap-test")
    monkeypatch.setitem(
        sys.modules,
        "modal_train",
        SimpleNamespace(
            app=app,
            prepare_runtime=None,
            prepare_distillation_data=None,
            train_distillation=None,
            run_evaluation=None,
            verify_volume_result=None,
        ),
    )

    def execute(*args, **kwargs):
        if fail:
            raise RuntimeError("paid chain failed")
        return {"status": "complete"}

    monkeypatch.setattr(module, "execute_campaign", execute)
    monkeypatch.setattr(module, "collect_study", lambda *a: {"status": "complete"})
    if fail:
        with pytest.raises(RuntimeError, match="paid chain failed"):
            module.main()
    else:
        module.main()
    state = json.loads(Path("outputs/distillation-active.json").read_text())
    assert state["app_id"] == "ap-test"
    assert state["status"] == (
        "failed_app_context_closed_reconcile_before_retry" if fail else "jobs_complete_app_closed"
    )
