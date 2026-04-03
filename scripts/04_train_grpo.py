"""Run GRPO training for a given condition."""

import argparse

import wandb

from src.data import format_gsm8k_for_grpo, load_config
from src.reward import binary_reward_fn, format_reward_fn
from src.training import (
    build_grpo_trainer,
    get_lora_config,
    load_model,
    load_sft_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["grpo_only", "sft_then_grpo"],
        help="Training condition",
    )
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Path to SFT checkpoint (required for sft_then_grpo)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    args = parser.parse_args()

    if args.condition == "sft_then_grpo" and args.sft_checkpoint is None:
        parser.error("--sft-checkpoint is required for sft_then_grpo condition")

    config = load_config(args.config)
    condition_config = config["conditions"][args.condition]

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "reasoning-distillation"),
        entity=wandb_cfg.get("entity"),
        name=f"grpo-{args.condition}",
        config=condition_config,
    )

    print(f"Starting GRPO training for condition: {args.condition}")

    # Load model — either from base or SFT checkpoint
    if args.condition == "sft_then_grpo":
        print(f"Loading SFT checkpoint from: {args.sft_checkpoint}")
        model, tokenizer = load_sft_checkpoint(args.sft_checkpoint, config)
    else:
        model, tokenizer = load_model(config)

    lora_config = get_lora_config(config)

    # Load GRPO dataset
    dataset = format_gsm8k_for_grpo(tokenizer, max_samples=args.max_samples)
    print(f"Training on {len(dataset)} GSM8K examples (LoRA r={lora_config.r})")

    # Build trainer and train
    trainer = build_grpo_trainer(
        model, tokenizer, dataset, condition_config,
        reward_fns=[binary_reward_fn, format_reward_fn],
        lora_config=lora_config,
    )
    trainer.train()

    # Save final adapter and tokenizer
    output_dir = condition_config["grpo"]["output_dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
