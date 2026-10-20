"""Explicit trainer construction and lossless artifact lineage for sequential adaptation."""

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from src.artifacts import read_model_manifest, write_model_manifest


def _get_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _configure_tokenizer(tokenizer, config, generation=False):
    if config["model"].get("eos_token"):
        tokenizer.eos_token = config["model"]["eos_token"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if generation else "right"
    tokenizer.truncation_side = "left" if generation else "right"
    return tokenizer


def _load_base(spec, config, device_map="auto", dtype=torch.bfloat16):
    options = {
        "device_map": device_map,
        "dtype": dtype,
        "attn_implementation": _get_attn_implementation(),
        "revision": spec.get("revision"),
    }
    quant = config.get("quantization", {})
    if quant.get("enabled", False):
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
    return AutoModelForCausalLM.from_pretrained(spec["name"], **options)


def load_model(config):
    set_seed(config.get("seed", 42))
    model = _load_base(config["model"], config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("tokenizer", config["model"]["name"]),
        revision=config["model"].get("revision"),
        **({"fix_mistral_regex": False} if config["model"]["name"].startswith("Qwen/") else {}),
    )
    return model, _configure_tokenizer(tokenizer, config)


def load_model_artifact(model_path, config, device_map="auto", dtype=torch.bfloat16, _ancestors=None):
    """Load verified local exports; never guess a historical adapter's parent."""
    path = Path(model_path).resolve()
    ancestors = set() if _ancestors is None else _ancestors
    if path in ancestors:
        raise ValueError("Cycle in adapter parent lineage")
    ancestors = ancestors | {path}
    if not (path / "artifact.json").exists():
        raise ValueError("Model artifact manifest missing; explicitly import legacy checkpoints first")
    manifest = read_model_manifest(path)
    if manifest["format"] == "full":
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map=device_map, dtype=dtype, attn_implementation=_get_attn_implementation()
        )
    else:
        parent = manifest.get("parent")
        if parent:
            parent_manifest = read_model_manifest(parent["path"], verify_files=False)
            if parent_manifest["artifact_id"] != parent["artifact_id"]:
                raise ValueError("Adapter parent identity mismatch")
            model, _ = load_model_artifact(parent["path"], config, device_map, dtype, ancestors)
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
        else:
            model = _load_base(manifest["base"], config, device_map, dtype)
        model = PeftModel.from_pretrained(model, path)
    tokenizer = AutoTokenizer.from_pretrained(
        path, **({"fix_mistral_regex": False} if model.config.model_type == "qwen3" else {})
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def save_trained_model(model, tokenizer, output_dir, config, parent_path=None, metadata=None):
    """Retain the adapter and export all effective weights, including prior SFT."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "artifact.json").exists():
        raise FileExistsError("Refusing to overwrite a model artifact")
    parent = None
    if parent_path:
        parent_manifest = read_model_manifest(parent_path, verify_files=False)
        parent = {"path": str(Path(parent_path).resolve()), "artifact_id": parent_manifest["artifact_id"]}
    if isinstance(model, PeftModel):
        if config.get("quantization", {}).get("enabled"):
            raise ValueError("Export bf16 weights before merging quantized adapters")
        adapter_path = path / "adapter"
        if adapter_path.exists():
            raise FileExistsError("Refusing to overwrite an adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        write_model_manifest(adapter_path, config, "adapter", parent, metadata)
        model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return write_model_manifest(path, config, "full", parent, metadata)


def load_sft_checkpoint(checkpoint_path, config):
    model, tokenizer = load_model_artifact(checkpoint_path, config)
    if isinstance(model, PeftModel):
        model = model.merge_and_unload(safe_merge=True)
    return model, _configure_tokenizer(tokenizer, config)


def get_lora_config(config):
    return LoraConfig(**config["lora"])


def build_sft_trainer(model, tokenizer, dataset, condition_config, lora_config, prepared_manifest=None):
    cfg = condition_config["sft"]
    if (
        (cfg.get("packing") or cfg.get("padding_free"))
        and model.device.type == "cuda"
        and model.config._attn_implementation
        not in {
            "flash_attention_2",
            "flash_attention_3",
        }
    ):
        raise ValueError("Packed GPU SFT requires FlashAttention; refusing an unsafe SDPA fallback")
    seed = condition_config.get("seed", 42)
    set_seed(seed)
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=int(cfg.get("warmup_steps", 10)),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        optim=cfg["optim"],
        max_length=cfg["max_seq_length"],
        packing=cfg.get("packing", False),
        padding_free=cfg.get("padding_free", False),
        packing_strategy="bfd",
        completion_only_loss=True,
        shuffle_dataset=False,
        eos_token=tokenizer.eos_token,
        use_liger_kernel=cfg.get("use_liger_kernel", False),
        bf16=cfg.get("bf16", True),
        max_steps=cfg.get("max_steps", -1),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=2,
        dataset_kwargs={"skip_prepare_dataset": True} if prepared_manifest else None,
        report_to=cfg.get("report_to", "wandb"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        data_seed=seed,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, peft_config=lora_config, processing_class=tokenizer
    )
    if prepared_manifest:
        if "input_ids" not in dataset.column_names or prepared_manifest["rows"] != len(dataset):
            raise ValueError("Invalid preprocessed training dataset")
    else:
        from src.preprocessing import validate_sft_rows

        validate_sft_rows(trainer.train_dataset, tokenizer, cfg["max_seq_length"])
    return trainer


class StudyGRPOTrainer(GRPOTrainer):
    def log(self, logs, start_time=None):
        diagnostics = getattr(self, "reward_diagnostics", None)
        if diagnostics is not None and self.model.training:
            logs = {**logs, **diagnostics.take_logs()}
        super().log(logs, start_time)


def build_grpo_trainer(model, tokenizer, dataset, condition_config, reward_fns, lora_config=None, diagnostics=None):
    cfg = condition_config["grpo"]
    seed = condition_config.get("seed", 42)
    set_seed(seed)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    args = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg.get("max_steps", 500),
        reward_weights=cfg["reward_weights"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 1.0),
        top_k=cfg.get("top_k", 0),
        loss_type=cfg["loss_type"],
        beta=cfg["beta"],
        scale_rewards=cfg["scale_rewards"],
        num_iterations=1,
        remove_unused_columns=False,
        generation_batch_size=cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        vllm_importance_sampling_correction=True,
        optim=cfg["optim"],
        use_liger_kernel=cfg.get("use_liger_kernel", False),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_total_limit=2,
        save_steps=100,
        report_to=cfg.get("report_to", "wandb"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=10,
        seed=seed,
        data_seed=seed,
        shuffle_dataset=False,
        log_completions=cfg.get("log_completions", False),
        use_vllm=cfg.get("use_vllm", True),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_max_model_length=cfg.get("vllm_max_model_length"),
        # TRL 0.28 synchronizes weights before its generate() wake-up. Sleeping
        # buffers caused a real CUDA invalid-argument failure in the pilot.
        # Keep both small models resident; measure the additional memory on GPU.
        vllm_enable_sleep_mode=False,
    )
    trainer = StudyGRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.reward_diagnostics = diagnostics
    return trainer
