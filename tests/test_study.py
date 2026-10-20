import json

import pytest

from src.artifacts import fingerprint
from src.data import load_config
from src.study import build_study_plan, execute_study


def test_complete_dag_and_seed_parentage():
    cfg = load_config()
    plan = build_study_plan(cfg)
    seen = set()
    for node in plan["nodes"]:
        assert set(node["depends_on"]) <= seen
        seen.add(node["id"])
        if node["kind"] == "grpo" and node["condition"] == "controlled_verified_grpo":
            assert node["parent"] == f"train:controlled_verified:{node['seed']}:sft"
    assert len([node for node in plan["nodes"] if node["kind"] in {"sft", "grpo"}]) == 21
    legacy = build_study_plan(cfg, "legacy")
    assert any(node["condition"] == "re_distill" and node["kind"] == "sft" for node in legacy["nodes"])


def test_budget_refuses_launch_and_resume_does_not_double_charge(tmp_path):
    config = load_config()
    plan = {"nodes": [{"id": "prepare", "kind": "prepare", "condition": "a", "phase": "sft", "depends_on": []}]}
    quote = {
        "plan_digest": fingerprint(plan),
        "pilot_evidence": ["measured"],
        "pricing_source": "dated official rates",
        "tasks": {"prepare": {"rate_per_second": 1.0, "timeout_seconds": 2, "overhead_seconds": 0}},
    }
    quote_path = tmp_path / "quote.json"
    quote_path.write_text(json.dumps(quote))
    calls = []
    call = lambda **kwargs: calls.append(kwargs) or {"rows": 1}
    with pytest.raises(ValueError, match="exceed"):
        execute_study(plan, config, str(quote_path), 1, call, call, call, call, call, tmp_path / "ledger")
    assert not calls
    execute_study(plan, config, str(quote_path), 3, call, call, call, call, call, tmp_path / "ledger")
    execute_study(plan, config, str(quote_path), 3, call, call, call, call, call, tmp_path / "ledger")
    assert len(calls) == 1


def test_main_quote_rejects_stale_rewards_and_zero_token_counters():
    from copy import deepcopy

    from src.study import validate_pilot_measurements

    config = load_config()
    metadata = {
        "grpo": {
            "reward_objective": {"scorer_version": config["reward_scorer"]},
            "exposure": {"processed_train_tokens": 100, "completion_tokens": 80},
        }
    }
    evaluation = {"scorer_version": config["evaluation"]["scorer"]}
    validate_pilot_measurements(config, metadata, evaluation)
    stale = deepcopy(metadata)
    stale["grpo"]["reward_objective"]["scorer_version"] = "final-response-v2"
    with pytest.raises(ValueError, match="stale"):
        validate_pilot_measurements(config, stale, evaluation)
    metadata["grpo"]["exposure"]["processed_train_tokens"] = 0
    with pytest.raises(ValueError, match="token exposure"):
        validate_pilot_measurements(config, metadata, evaluation)


def test_quote_generator_preserves_existing_approval_artifacts(tmp_path, monkeypatch):
    import runpy
    import shutil
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ["config.yaml", "modal-pricing-2026-10-19.json"]:
        shutil.copyfile(root / "configs" / name, tmp_path / "configs" / name)
    original = tmp_path / "results/recovery/pilot-quote.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"immutable_approved_quote": true}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["11_quote_pilot.py", "--output-root", str(tmp_path / "proposals")])
    runpy.run_path(str(root / "scripts/11_quote_pilot.py"), run_name="__main__")
    assert original.read_text() == '{"immutable_approved_quote": true}'


def test_main_quote_rejects_changed_source_before_loading_or_downloading(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from unittest.mock import patch

    from src.study import build_pilot_plan

    root = Path(__file__).resolve().parents[1]
    config = load_config()
    plan = build_pilot_plan(config)
    plan["implementation_digest"] = "stale"
    ledger = {"identity": {"plan": plan}, "results": {node["id"]: {"status": "complete"} for node in plan["nodes"]}}
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(ledger))
    monkeypatch.setattr(sys, "argv", ["12_quote_study.py", "--pilot-ledger", str(path)])
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("Must reject stale source first")
        ),
        pytest.raises(ValueError, match="source"),
    ):
        runpy.run_path(str(root / "scripts/12_quote_study.py"), run_name="__main__")
