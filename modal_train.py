"""Modal cloud training orchestration for all experimental conditions."""

import modal

app = modal.App("reasoning-distillation-grpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "trl>=0.28.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.5.0",
        "accelerate>=1.5.0",
        "wandb>=0.17.0",
        "pyyaml",
        "liger-kernel",
        "sympy",
        "flash-attn",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("configs/config.yaml", remote_path="/root/configs/config.yaml")
)

volume = modal.Volume.from_name("reasoning-distillation-vol", create_if_missing=True)

VOLUME_PATH = "/vol"
SECRETS = [
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("huggingface-secret"),
]


@app.function(
    image=image,
    gpu="L40S",
    timeout=6 * 3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def train_sft(condition: str = "sft_traces"):
    """Run SFT training on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root")

    from src.data import get_dataset_for_condition, load_config
    from src.training import build_sft_trainer, get_lora_config, load_model

    import wandb

    config = load_config("/root/configs/config.yaml")
    condition_config = config["conditions"][condition]

    # Override output dir to write to volume
    output_dir = f"{VOLUME_PATH}/outputs/{condition}"
    condition_config["sft"]["output_dir"] = output_dir

    wandb.init(
        project=config.get("wandb", {}).get("project", "reasoning-distillation"),
        name=f"sft-{condition}",
        config=condition_config,
    )

    model, tokenizer = load_model(config)
    lora_config = get_lora_config(config)

    result = get_dataset_for_condition(condition, config, tokenizer)
    dataset = result[0] if isinstance(result, tuple) else result
    print(f"[{condition}] Training on {len(dataset)} examples")

    trainer = build_sft_trainer(model, tokenizer, dataset, condition_config, lora_config)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()

    wandb.finish()
    print(f"[{condition}] SFT training complete. Saved to {output_dir}")
    return output_dir


@app.function(
    image=image,
    gpu="H100",
    timeout=12 * 3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def train_grpo(condition: str = "grpo_only", sft_checkpoint: str | None = None):
    """Run GRPO training on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root")

    from src.data import format_numinamath_for_grpo, load_config
    from src.reward import binary_reward_fn
    from src.training import (
        build_grpo_trainer,
        get_lora_config,
        load_model,
        load_sft_checkpoint,
    )

    import wandb

    config = load_config("/root/configs/config.yaml")
    condition_config = config["conditions"][condition]

    output_dir = f"{VOLUME_PATH}/outputs/{condition}"
    condition_config["grpo"]["output_dir"] = output_dir

    wandb.init(
        project=config.get("wandb", {}).get("project", "reasoning-distillation"),
        name=f"grpo-{condition}",
        config=condition_config,
    )

    if sft_checkpoint:
        print(f"[{condition}] Loading SFT checkpoint from {sft_checkpoint}")
        model, tokenizer = load_sft_checkpoint(sft_checkpoint, config)
    else:
        model, tokenizer = load_model(config)

    lora_config = get_lora_config(config)
    dataset = format_numinamath_for_grpo(tokenizer)
    print(f"[{condition}] Training on {len(dataset)} examples")

    trainer = build_grpo_trainer(
        model, tokenizer, dataset, condition_config, lora_config,
        reward_fns=[binary_reward_fn],
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()

    wandb.finish()
    print(f"[{condition}] GRPO training complete. Saved to {output_dir}")
    return output_dir


@app.function(
    image=image,
    gpu="L40S",
    timeout=3 * 3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def run_evaluation(model_path: str, condition: str):
    """Run evaluation benchmarks on a trained model."""
    import sys
    sys.path.insert(0, "/root")

    from src.data import load_config
    from src.evaluation import run_all_evaluations

    config = load_config("/root/configs/config.yaml")
    results = run_all_evaluations(model_path, config, condition)
    volume.commit()

    return results


@app.local_entrypoint()
def run_full_pipeline():
    """Orchestrate the full experiment across all conditions.

    Execution order:
    1. SFT jobs (baseline, sft_traces, sft_then_grpo phase 1) — run in parallel
    2. GRPO jobs (grpo_only, sft_then_grpo phase 2) — grpo_only in parallel, sft_then_grpo waits for SFT
    3. Evaluation — all conditions in parallel
    """
    print("=" * 60)
    print("Starting full experimental pipeline")
    print("=" * 60)

    # Phase 1: SFT training (3 conditions can run in parallel)
    print("\n--- Phase 1: SFT Training ---")
    baseline_handle = train_sft.spawn("baseline")
    sft_traces_handle = train_sft.spawn("sft_traces")
    sft_then_grpo_sft_handle = train_sft.spawn("sft_then_grpo")

    # Also start grpo_only in parallel (it doesn't need SFT)
    grpo_only_handle = train_grpo.spawn("grpo_only")

    # Wait for SFT results
    baseline_path = baseline_handle.get()
    sft_traces_path = sft_traces_handle.get()
    sft_then_grpo_sft_path = sft_then_grpo_sft_handle.get()
    grpo_only_path = grpo_only_handle.get()

    print(f"Baseline SFT done: {baseline_path}")
    print(f"SFT traces done: {sft_traces_path}")
    print(f"SFT+GRPO SFT phase done: {sft_then_grpo_sft_path}")
    print(f"GRPO only done: {grpo_only_path}")

    # Phase 2: GRPO for sft_then_grpo (needs SFT checkpoint)
    print("\n--- Phase 2: GRPO Training (sft_then_grpo) ---")
    sft_then_grpo_path = train_grpo.remote("sft_then_grpo", sft_checkpoint=sft_then_grpo_sft_path)
    print(f"SFT+GRPO complete: {sft_then_grpo_path}")

    # Phase 3: Evaluation (all conditions in parallel)
    print("\n--- Phase 3: Evaluation ---")
    eval_handles = [
        run_evaluation.spawn(baseline_path, "baseline"),
        run_evaluation.spawn(sft_traces_path, "sft_traces"),
        run_evaluation.spawn(grpo_only_path, "grpo_only"),
        run_evaluation.spawn(sft_then_grpo_path, "sft_then_grpo"),
    ]

    for handle in eval_handles:
        result = handle.get()
        condition = result["condition"]
        for bench_name, bench_data in result["benchmarks"].items():
            print(f"  [{condition}] {bench_name}: {bench_data['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Pipeline complete! Results saved to volume.")
    print("=" * 60)
