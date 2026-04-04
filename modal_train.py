"""Modal cloud training orchestration for all experimental conditions."""

import modal

app = modal.App("reasoning-distillation-grpo")

# Two separate images: SFT (pinned torch for flash-attn) and GRPO (vLLM needs its own torch)
# vLLM bundles flash-attn internally, so no separate install needed for GRPO.

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

sft_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
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
        FLASH_ATTN_WHEEL,
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("configs/config.yaml", remote_path="/root/configs/config.yaml")
)

grpo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl[vllm]>=0.28.0",  # Pulls compatible torch + vLLM versions
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.5.0",
        "accelerate>=1.5.0",
        "wandb>=0.17.0",
        "pyyaml",
        "liger-kernel",
        "sympy",
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
    image=sft_image,
    gpu="L40S",
    timeout=6 * 3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def train_sft(condition: str = "sft_traces"):
    """Run SFT training on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root")

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    image=grpo_image,
    gpu="H100",
    timeout=12 * 3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def train_grpo(condition: str = "grpo_only", sft_checkpoint: str | None = None):
    """Run GRPO training on a cloud GPU with vLLM colocate mode."""
    import os
    import sys
    sys.path.insert(0, "/root")

    # Required for vLLM colocate mode on single GPU (from Modal's official example)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    from src.data import format_gsm8k_for_grpo, load_config
    from src.reward import binary_reward_fn, format_reward_fn
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
    dataset = format_gsm8k_for_grpo(tokenizer)
    print(f"[{condition}] Training on {len(dataset)} GSM8K examples (LoRA r={lora_config.r})")

    trainer = build_grpo_trainer(
        model, tokenizer, dataset, condition_config,
        reward_fns=[binary_reward_fn, format_reward_fn],
        lora_config=lora_config,
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()

    wandb.finish()
    print(f"[{condition}] GRPO training complete. Saved to {output_dir}")
    return output_dir


@app.function(
    image=sft_image,
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


@app.function(
    image=sft_image,
    gpu="L40S",
    timeout=3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def quick_test(model_path: str = "/vol/outputs/sft_traces"):
    """Generate a few sample solutions to sanity-check a trained model."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import load_config

    config = load_config("/root/configs/config.yaml")

    print(f"Loading model from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token = config["model"].get("eos_token")
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    test_questions = [
        "What is 15% of 200?",
        "A train travels 120 miles in 2 hours. What is its average speed in miles per hour?",
        "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?",
        "Sally has 3 red marbles and 5 blue marbles. What fraction of her marbles are red?",
        "A store sells apples for $2 each. If John buys 7 apples and pays with a $20 bill, how much change does he get?",
    ]

    print("\n" + "=" * 80)
    print("SAMPLE GENERATIONS FROM TRAINED MODEL")
    print("=" * 80)

    for i, question in enumerate(test_questions):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\n--- Question {i + 1}: {question}")
        print(f"--- Response:\n{response[:1500]}")
        if len(response) > 1500:
            print(f"... [truncated, {len(response)} chars total]")
        print()

    return "Quick test complete"


@app.function(
    image=sft_image,
    gpu="L40S",
    timeout=3600,
    secrets=SECRETS,
    volumes={VOLUME_PATH: volume},
)
def spot_check_gsm8k(model_path: str = "/vol/outputs/sft_traces", num_problems: int = 20, start_from: int = 0):
    """Run a quick GSM8K spot-check: solve N problems and report accuracy."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import load_config
    from src.reward import answers_match, extract_answer_auto, extract_gsm8k_answer

    config = load_config("/root/configs/config.yaml")

    print(f"Loading model from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token = config["model"].get("eos_token")
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load GSM8K test set
    ds = load_dataset("openai/gsm8k", "main", split="test")
    end_idx = min(num_problems, len(ds))
    ds = ds.select(range(start_from, end_idx))
    print(f"Evaluating Q{start_from+1} to Q{end_idx} ({len(ds)} problems)")

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        question = example["question"]
        gt_answer = extract_gsm8k_answer(example["answer"])

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer_auto(response)

        is_correct = pred_answer is not None and gt_answer is not None and answers_match(pred_answer, gt_answer)
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Q{start_from+i+1}: pred={pred_answer}, gt={gt_answer}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"GSM8K Spot Check: {correct}/{total} = {accuracy:.1%}")
    print(f"{'=' * 50}")

    return {"correct": correct, "total": total, "accuracy": accuracy}


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
