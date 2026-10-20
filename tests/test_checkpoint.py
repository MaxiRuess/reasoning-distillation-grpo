import tempfile
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from src.training import load_model_artifact, save_trained_model


def tiny_model():
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def tokenizer():
    tok = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "test": 2}, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="[EOS]", unk_token="[UNK]")


def add_adapter(model):
    model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0.0),
    )
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if "lora_" in name:
                weight.fill_(0.15)
    return model.eval()


def test_two_updates_survive_real_peft_export_reload():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base"
        original = tiny_model()
        original.save_pretrained(base_path)
        config = {"model": {"name": str(base_path), "revision": "local"}, "quantization": {"enabled": False}}
        sft = add_adapter(tiny_model())
        ids = torch.tensor([[2, 3, 4]])
        sft_dir = Path(tmp) / "sft"
        save_trained_model(sft, tokenizer(), sft_dir, config)
        parent, _ = load_model_artifact(str(sft_dir), config, device_map=None, dtype=torch.float32)
        grpo = add_adapter(parent)
        expected = grpo(ids).logits.detach()
        assert not torch.allclose(expected, original(ids).logits, atol=1e-5)
        final = Path(tmp) / "grpo"
        save_trained_model(grpo, tokenizer(), final, config, parent_path=str(sft_dir))
        actual, _ = load_model_artifact(str(final), config, device_map=None, dtype=torch.float32)
        torch.testing.assert_close(actual(ids).logits, expected, atol=1e-5, rtol=1e-5)
        restored_adapter, _ = load_model_artifact(str(final / "adapter"), config, device_map=None, dtype=torch.float32)
        torch.testing.assert_close(restored_adapter(ids).logits, expected, atol=1e-5, rtol=1e-5)


def test_legacy_adapter_requires_explicit_import():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "adapter_config.json").write_text("{}")
        with pytest.raises((FileNotFoundError, ValueError), match="manifest|artifact"):
            load_model_artifact(tmp, {})


def test_sft_trainer_uses_real_processor_and_complete_target(tmp_path):
    from datasets import Dataset

    from src.training import build_sft_trainer

    tok = tokenizer()
    tok.pad_token = tok.eos_token
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% if m['role']=='assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "test"}], "completion": [{"role": "assistant", "content": "test"}]}]
    )
    cfg = {
        "seed": 42,
        "sft": {
            "output_dir": str(tmp_path),
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "optim": "adamw_torch",
            "max_seq_length": 64,
            "packing": False,
            "use_liger_kernel": False,
            "bf16": False,
            "report_to": "none",
            "max_steps": 1,
        },
    }
    trainer = build_sft_trainer(
        tiny_model(), tok, dataset, cfg, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM")
    )
    assert trainer.processing_class is tok
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert result.metrics["train_loss"] >= 0


def test_real_cpu_grpo_reports_tokens_and_weighted_objective(tmp_path):
    from copy import deepcopy

    from datasets import Dataset

    from src.data import load_config
    from src.pipeline import RewardDiagnostics, processed_training_tokens
    from src.training import build_grpo_trainer

    cfg = deepcopy(load_config()["conditions"]["controlled_grpo"])
    cfg["grpo"].update(
        output_dir=str(tmp_path),
        max_steps=1,
        num_generations=2,
        gradient_accumulation_steps=2,
        use_vllm=False,
        use_liger_kernel=False,
        max_completion_length=3,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        logging_steps=1,
        save_strategy="no",
    )
    model = tiny_model()
    model.config.eos_token_id = model.generation_config.eos_token_id = 1
    tok = tokenizer()
    tok.model_input_names = ["input_ids", "attention_mask"]
    tok.pad_token = tok.eos_token
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
    data = Dataset.from_list([{"prompt": [{"role": "user", "content": "test"}], "answer": "0"}])
    diagnostics = RewardDiagnostics([1, 0], 2)

    def correctness(completions, **kwargs):
        values = [0.0] * len(completions)
        diagnostics.record(values, [1.0] * len(values))
        return values

    def formatting(completions, **kwargs):
        return [1.0] * len(completions)

    trainer = build_grpo_trainer(model, tok, data, cfg, [correctness, formatting], diagnostics=diagnostics)
    trainer.train()
    assert processed_training_tokens(trainer, "grpo") > 0
    log = next(row for row in trainer.state.log_history if "objective/reward_mean" in row)
    assert log["objective/reward_mean"] == 0
    assert log["reward"] == 1  # The upstream raw component sum is a different quantity.
    assert not trainer.args.vllm_enable_sleep_mode
