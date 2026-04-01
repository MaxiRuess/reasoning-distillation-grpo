"""Run SFT training for a given condition."""

import argparse
import os

import wandb

from src.data import get_dataset_for_condition, load_config
from src.training import build_sft_trainer, get_lora_config, load_model


def main():
    parser = argparse.ArgumentParser(description="Train SFT model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["baseline", "sft_traces", "sft_then_grpo"],
        help="Training condition",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    condition_config = config["conditions"][args.condition]

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "reasoning-distillation"),
        entity=wandb_cfg.get("entity"),
        name=f"sft-{args.condition}",
        config=condition_config,
    )

    print(f"Starting SFT training for condition: {args.condition}")
    print(f"Output dir: {condition_config['sft']['output_dir']}")

    # Load model and tokenizer
    model, tokenizer = load_model(config)
    lora_config = get_lora_config(config)

    # Load dataset
    result = get_dataset_for_condition(args.condition, config, tokenizer)
    dataset = result[0] if isinstance(result, tuple) else result
    print(f"Training on {len(dataset)} examples")

    # Build trainer and train
    trainer = build_sft_trainer(model, tokenizer, dataset, condition_config, lora_config)
    trainer.train()

    # Save final adapter and tokenizer
    output_dir = condition_config["sft"]["output_dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
