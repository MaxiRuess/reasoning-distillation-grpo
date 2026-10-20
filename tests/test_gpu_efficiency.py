from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts import EvaluationStore
from src.evaluation import evaluate_dataset


def test_sampled_resume_replays_the_canonical_batch(tmp_path):
    rows = [{"problem_id": str(i), "question": str(i), "answer": "1"} for i in range(5)]
    cfg = {"batch_size": 2, "num_samples_per_problem": 2, "temperature": 0.6, "max_new_tokens": 8}
    calls = []

    def generate(model, tokenizer, questions, ids, cfg):
        calls.append(list(ids))
        # A batch-dependent answer emulates a seeded batched sampler.
        return [
            [{"response": f"{len(ids)}", "generated_tokens": 1, "finish_reason": "eos"} for _ in range(2)] for i in ids
        ]

    with patch("src.evaluation.generate_batch", side_effect=generate):
        expected = evaluate_dataset(None, None, rows, "gsm8k", cfg)
        store = EvaluationStore(tmp_path, {"test": "fixed"}, [r["problem_id"] for r in rows], 2)
        store.append(expected["records"][:2])  # Crash after one whole question of a two-question batch.
        calls.clear()
        result = evaluate_dataset(None, None, rows, "gsm8k", cfg, store)
        assert result == expected
        assert calls[0] == ["0", "1"]


def test_cpu_preprocessing_preserves_the_real_trl_masks_and_boundaries(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from src.preprocessing import load_sft_tokens, prepare_sft_tokens

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "test": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% if m['role']=='assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "test"}],
                "completion": [{"role": "assistant", "content": "test test"}],
            }
        ]
        * 3
    )
    raw = {"path": str(tmp_path), "content_hash": "test-raw"}
    cfg = {"max_seq_length": 32, "packing": True}
    meta = prepare_sft_tokens(ds, tok, raw, cfg)
    prepared, loaded = load_sft_tokens(tok, raw, cfg)
    assert loaded == meta
    assert "seq_lengths" in prepared.column_names
    assert any(any(row["completion_mask"]) for row in prepared)
    for row in prepared:
        assert sum(row["seq_lengths"]) == len(row["input_ids"])
        assert len(row["input_ids"]) <= 32
    arrow = next((Path(meta["path"]) / "dataset").glob("*.arrow"))
    with arrow.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_sft_tokens(tok, raw, cfg)


def test_generation_batches_all_question_sample_groups_once():
    from src.evaluation import generate_batch, inference_batch_size

    cfg = {
        "batch_size": 32,
        "num_samples_per_problem": 8,
        "temperature": 0.6,
        "max_new_tokens": 2048,
        "max_generation_sequences": 32,
        "max_batch_generated_tokens": 65536,
    }
    assert inference_batch_size(cfg) == 4
    assert inference_batch_size({**cfg, "num_samples_per_problem": 1, "max_new_tokens": 8192}) == 8

    class Tokenizer:
        padding_side = "right"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

    with patch("src.evaluation._generate", return_value=["sampled"]) as generator:
        assert generate_batch(None, Tokenizer(), ["a", "b"], ["0", "1"], cfg) == ["sampled"]
        assert generator.call_count == 1
        assert generator.call_args.args[2] == ["prompt", "prompt"]
        assert generator.call_args.args[-1] == 8


def test_volume_commits_are_periodic_and_final_flush_is_explicit():
    from src.evaluation import PeriodicCommit

    calls = []
    clock = [0.0]
    commit = PeriodicCommit(lambda: calls.append(clock[0]), batches=4, seconds=60, clock=lambda: clock[0])
    for _ in range(3):
        commit.tick()
    assert calls == []
    commit.tick()
    assert len(calls) == 1
    clock[0] = 61
    commit.tick()
    assert len(calls) == 2
    commit.flush()
    assert len(calls) == 3


def test_runtime_cpu_cache_is_a_dependency_before_gpu_evaluation():
    from src.data import load_config
    from src.study import build_pilot_plan, build_study_plan

    for plan in [build_study_plan(load_config()), build_pilot_plan(load_config())]:
        assert plan["nodes"][0]["kind"] == "prepare_runtime"
        completed = set()
        for node in plan["nodes"]:
            assert set(node["depends_on"]) <= completed
            completed.add(node["id"])


def test_real_sampled_generation_is_identical_after_mid_batch_resume(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(16)}, unk_token="t0")),
        unk_token="t0",
        eos_token="t1",
        pad_token="t1",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=True,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    rows = [{"problem_id": str(i), "question": f"t{i + 2}", "answer": "1"} for i in range(3)]
    cfg = {"batch_size": 2, "num_samples_per_problem": 2, "temperature": 0.6, "max_new_tokens": 3, "seed": 42}
    expected = evaluate_dataset(model, tokenizer, rows, "gsm8k", cfg)
    store = EvaluationStore(tmp_path, {"test": "cpu-real-model"}, [r["problem_id"] for r in rows], 2)
    store.append(expected["records"][:2])
    actual = evaluate_dataset(model, tokenizer, rows, "gsm8k", cfg, store)
    assert actual == expected


def test_warmup_downloads_are_pinned_and_cpu_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.runtime_cache import prepare_runtime_cache

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"test-weights")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    config = {"model": {"name": "model", "revision": "a" * 40}, "evaluation": {"benchmarks": ["gsm8k", "math500"]}}
    with (
        patch("src.runtime_cache.snapshot_download", return_value=str(snapshot)) as download,
        patch("src.runtime_cache.load_benchmark", return_value=SimpleNamespace(_fingerprint="fixed")),
    ):
        result = prepare_runtime_cache(config)
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert result["weights_sha256"]["model.safetensors"]
    assert set(result["benchmark_fingerprints"]) == {"gsm8k", "math500"}


def test_preprocessed_trainer_does_not_repack_document_boundaries(tmp_path):
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from src.preprocessing import load_sft_tokens, prepare_sft_tokens
    from src.training import build_sft_trainer

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "test": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% if m['role']=='assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
    ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "test"}], "completion": [{"role": "assistant", "content": "test"}]}]
        * 3
    )
    cfg = {
        "seed": 42,
        "sft": {
            "output_dir": str(tmp_path / "trainer"),
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "optim": "adamw_torch",
            "max_seq_length": 32,
            "packing": True,
            "use_liger_kernel": False,
            "bf16": False,
            "report_to": "none",
        },
    }
    raw = {"path": str(tmp_path), "content_hash": "fixed"}
    prepare_sft_tokens(ds, tok, raw, cfg["sft"])
    tokens, meta = load_sft_tokens(tok, raw, cfg["sft"])

    def model():
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
            )
        )

    lora = LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM")
    reference = build_sft_trainer(model(), tok, ds, cfg, lora)
    actual = build_sft_trainer(model(), tok, tokens, cfg, lora, prepared_manifest=meta)
    assert actual.train_dataset.to_list() == reference.train_dataset.to_list()
    assert len(actual.train_dataset[0]["seq_lengths"]) == 3


def test_cpu_monitor_does_not_claim_gpu_measurements():
    from src.performance import GpuMonitor

    with patch("torch.cuda.is_available", return_value=False):
        with GpuMonitor() as monitor:
            pass
        profile = monitor.summary()
    assert not profile["cuda_available"]
    assert profile["sampled_utilization_mean_percent"] is None
    assert profile["torch_peak_allocated_bytes"] is None


def test_packed_gpu_training_refuses_sdpa_fallback():
    from types import SimpleNamespace

    from src.training import build_sft_trainer

    model = SimpleNamespace(device=SimpleNamespace(type="cuda"), config=SimpleNamespace(_attn_implementation="sdpa"))
    with pytest.raises(ValueError, match="requires FlashAttention"):
        build_sft_trainer(model, None, None, {"sft": {"packing": True}}, None)


def test_pilot_executor_warms_cpu_cache_before_allocating_any_gpu(tmp_path):
    import json

    from src.artifacts import fingerprint
    from src.data import load_config
    from src.study import build_pilot_plan, execute_study

    config = load_config()
    plan = build_pilot_plan(config)
    quote = {
        "plan_digest": fingerprint(plan),
        "pilot_evidence": ["smoke reservation"],
        "pricing_source": "test",
        "tasks": {
            node["id"]: {"rate_per_second": 0.001, "timeout_seconds": 1, "overhead_seconds": 0}
            for node in plan["nodes"]
        },
    }
    path = tmp_path / "quote.json"
    path.write_text(json.dumps(quote))
    calls = []

    def callback(kind, output):
        def run(**kwargs):
            calls.append((kind, kwargs))
            return output

        return run

    execute_study(
        plan,
        config,
        str(path),
        1,
        prepare=callback("prepare", {}),
        sft=callback("sft", "/sft"),
        grpo=callback("grpo", "/grpo"),
        evaluate=callback("eval", {"benchmarks": {"gsm8k": {"results_path": "/eval"}}}),
        verify=callback("verify", {"summary": {"results_path": "/eval"}}),
        prepare_runtime=callback("runtime", {}),
        ledger_root=tmp_path / "ledgers",
    )
    assert [kind for kind, _ in calls] == ["runtime", "prepare", "prepare", "sft", "grpo", "eval", "verify"]
    assert calls[4][1]["sft_checkpoint"] == "/sft"
