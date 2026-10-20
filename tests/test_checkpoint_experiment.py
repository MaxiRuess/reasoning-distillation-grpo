import copy
from unittest.mock import patch

import pytest

from src.artifacts import fingerprint
from src.evaluation import benchmark_rows


def test_subset_preserves_original_ids_and_checks_frozen_contents():
    examples = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(6)]
    config = {"evaluation": {"problem_indices": [2, 5]}}
    with patch("src.data.load_benchmark", return_value=examples):
        rows = benchmark_rows("gsm8k", config)
        assert [row["problem_id"] for row in rows] == ["2", "5"]
        config["evaluation"]["expected_rows_digest"] = fingerprint(rows)
        assert benchmark_rows("gsm8k", config) == rows
        examples[2]["question"] = "changed question"
        with pytest.raises(ValueError, match="Frozen benchmark"):
            benchmark_rows("gsm8k", config)


@pytest.mark.parametrize("indices", [[2, 2], [-1], [6], [True], []])
def test_invalid_frozen_subset_is_rejected(indices):
    with (
        patch("src.data.load_benchmark", return_value=[{"question": "q", "answer": "#### 1"}] * 6),
        pytest.raises(ValueError, match="indices"),
    ):
        benchmark_rows("gsm8k", {"evaluation": {"problem_indices": indices}})


def test_protocol_freezes_random_selection_and_exact_parent():
    from src.checkpoint_experiment import freeze_protocol

    rows = [{"problem_id": str(i), "question": f"q{i}", "answer": str(i)} for i in range(300)]
    sources = {
        "sft": {"path": "outputs/sft_then_grpo/checkpoint-117", "files": {"adapter": "a"}},
        "grpo": {"path": "outputs/sft_then_grpo", "files": {"adapter": "b"}},
    }
    config = {"model": {"name": "test", "revision": "fixed"}, "datasets": {"gsm8k": {"revision": "fixed"}}}
    protocol = freeze_protocol(config, rows, [str(i) for i in range(32)], sources, {"code": "frozen"})
    assert protocol == freeze_protocol(config, rows, [str(i) for i in range(32)], sources, {"code": "frozen"})
    assert len(protocol["problem_indices"]) == 256
    assert not set(protocol["problem_indices"]) & set(range(32))
    assert protocol["sources"]["sft"]["path"] == "outputs/sft_then_grpo/checkpoint-117"
    assert protocol["primary_comparison"] == ["reconstructed", "missing_sft"]
    assert protocol["secondary_comparison"] == ["reconstructed", "sft_parent"]


def test_quote_covers_all_work_and_cannot_exceed_twenty():
    from src.checkpoint_experiment import build_experiment_plan, build_experiment_quote
    from src.data import load_config
    from src.study import validate_quote

    config = load_config()
    protocol = {"protocol_id": "frozen", "problem_indices": list(range(32, 288)), "rows_digest": "rows"}
    plan = build_experiment_plan(config, protocol)
    quote = build_experiment_quote(plan)
    assert 10 < validate_quote(plan, quote, 20) < 12
    assert sum(n["kind"] == "evaluate" for n in plan["nodes"]) == 3
    assert not any(n["kind"] in {"sft", "grpo"} for n in plan["nodes"])
    bad = copy.deepcopy(quote)
    bad["tasks"]["eval:reconstructed"]["timeout_seconds"] = 43200
    with pytest.raises(ValueError, match="exceed"):
        validate_quote(plan, bad, 20)


def test_analysis_rejects_incomplete_or_misaligned_arms():
    from src.checkpoint_experiment import analyze_experiment

    protocol = {"problem_ids": ["1", "2"], "protocol_id": "test"}

    def records(values):
        return [
            {
                "problem_id": str(i + 1),
                "sample_id": 0,
                "ground_truth": "1",
                "correct": value,
                "scorer_version": "boxed-final-v1",
                "response": "\\boxed{1}",
                "generated_tokens": 5,
                "finish_reason": "eos",
                "status": "ok",
                "predicted": "1",
            }
            for i, value in enumerate(values)
        ]

    arms = {
        "sft_parent": records([True, False]),
        "missing_sft": records([False, False]),
        "reconstructed": records([True, True]),
    }
    result = analyze_experiment(protocol, arms)
    assert result["primary"]["delta"] == 1
    assert result["secondary"]["delta"] == 0.5
    arms["missing_sft"].pop()
    with pytest.raises(ValueError):
        analyze_experiment(protocol, arms)


def test_real_three_construction_exports_and_wrong_parent_rejection(tmp_path):
    import json

    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from src.artifacts import file_hash, read_model_manifest
    from src.checkpoint_experiment import reconstruct_arm

    base = tmp_path / "base"
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(base)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tokenizer.save_pretrained(base)
    sources = {}
    for label, contribution in [("sft", 0.1), ("grpo", 0.2)]:
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(contribution)
        folder = tmp_path / label
        model.save_pretrained(folder)
        sources[label] = {
            "path": label,
            "files": {name: file_hash(folder / name) for name in ["adapter_config.json", "adapter_model.safetensors"]},
        }
        model = model.merge_and_unload()
    config = {"model": {"name": str(base), "tokenizer": str(base), "revision": "local"}}
    protocol = {"protocol_id": "frozen", "sources": sources, "model": config["model"]}
    outputs = tmp_path / "exports"
    parent = reconstruct_arm("sft_parent", protocol, config, outputs, str(tmp_path))
    missing = reconstruct_arm("missing_sft", protocol, config, outputs, str(tmp_path))
    corrected = reconstruct_arm("reconstructed", protocol, config, outputs, str(tmp_path), parent_path=parent)
    assert read_model_manifest(corrected)["parent"]["artifact_id"] == read_model_manifest(parent)["artifact_id"]
    assert read_model_manifest(missing)["metadata"]["intentional_missing_sft_control"] is True
    for arm in ["sft_parent", "missing_sft", "reconstructed"]:
        assert json.loads((outputs / (arm + "-reload-proof.json")).read_text())["exact_cpu_logits_equal"]
    weights = [
        AutoModelForCausalLM.from_pretrained(path).state_dict()["model.layers.0.self_attn.q_proj.weight"]
        for path in [parent, missing, corrected]
    ]
    assert not torch.equal(weights[0], weights[2])
    assert not torch.equal(weights[1], weights[2])
    with pytest.raises(ValueError, match="different SFT parent"):
        reconstruct_arm("reconstructed", protocol, config, outputs, str(tmp_path), parent_path=missing)


def test_frozen_proposal_rejects_changed_source_and_tampering_before_execution(tmp_path):
    import json

    from src.checkpoint_experiment import (
        build_experiment_plan,
        build_experiment_quote,
        freeze_protocol,
        immutable_json,
        source_identity,
        validate_proposal,
    )
    from src.data import load_config

    config = load_config()
    rows = [{"problem_id": str(i), "question": str(i), "answer": str(i)} for i in range(300)]
    protocol = freeze_protocol(config, rows, [], {}, source_identity())
    plan = build_experiment_plan(config, protocol)
    for name, value in [("protocol", protocol), ("plan", plan), ("quote", build_experiment_quote(plan))]:
        immutable_json(tmp_path / (name + ".json"), value)
    validate_proposal(tmp_path, config, 20)
    with (
        patch("src.checkpoint_experiment.source_identity", return_value={"different": "code"}),
        pytest.raises(ValueError, match="implementation changed"),
    ):
        validate_proposal(tmp_path, config, 20)
    protocol["problem_indices"][0] = 299
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Frozen protocol"):
        validate_proposal(tmp_path, config, 20)


def test_executor_passes_subset_and_parent_and_never_retries_failed_paid_call(tmp_path):
    import json

    from src.checkpoint_experiment import build_experiment_plan, build_experiment_quote
    from src.data import load_config
    from src.study import execute_study

    config = load_config()
    protocol = {"protocol_id": "fixed", "problem_indices": [50, 60], "rows_digest": "frozen"}
    plan = build_experiment_plan(config, protocol)
    quote = tmp_path / "quote.json"
    quote.write_text(json.dumps(build_experiment_quote(plan)))
    calls = []

    def reconstruct(condition, parent_path, **kwargs):
        if condition == "reconstructed":
            assert parent_path == "/models/sft_parent"
        calls.append(condition)
        return "/models/" + condition

    def evaluate(**kwargs):
        assert kwargs["problem_indices"] == [50, 60]
        assert kwargs["expected_rows_digest"] == "frozen"
        calls.append("paid_evaluation")
        raise RuntimeError("interrupted paid task")

    callbacks = {
        "prepare": None,
        "sft": None,
        "grpo": None,
        "evaluate": evaluate,
        "verify": None,
        "prepare_runtime": lambda **kwargs: {},
        "reconstruct": reconstruct,
        "ledger_root": tmp_path / "ledgers",
    }
    with pytest.raises(RuntimeError, match="interrupted paid"):
        execute_study(plan, config, quote, 20, **callbacks)
    with pytest.raises(ValueError, match="Reconcile"):
        execute_study(plan, config, quote, 20, **callbacks)
    assert calls == ["sft_parent", "missing_sft", "reconstructed", "paid_evaluation"]
