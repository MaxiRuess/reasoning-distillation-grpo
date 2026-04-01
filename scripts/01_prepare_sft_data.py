"""Download, format, and cache SFT datasets for training."""

import argparse
from pathlib import Path

from src.data import get_dataset_for_condition, load_config
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["baseline", "sft_traces", "sft_then_grpo"],
        help="Training condition to prepare data for",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"])

    print(f"Preparing SFT data for condition: {args.condition}")

    result = get_dataset_for_condition(args.condition, config, tokenizer)

    # For sft_then_grpo, get_dataset_for_condition returns a tuple
    dataset = result[0] if isinstance(result, tuple) else result

    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    # Show sample examples
    print("\n--- Sample Examples ---")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        prompt_text = example["prompt"][0]["content"] if example.get("prompt") else "N/A"
        completion_text = example["completion"][0]["content"] if example.get("completion") else "N/A"
        print(f"\n[Example {i + 1}]")
        print(f"  Prompt: {prompt_text[:200]}...")
        print(f"  Completion: {completion_text[:200]}...")

    # Compute token length statistics
    print("\n--- Token Length Stats ---")
    lengths = []
    for example in dataset:
        prompt = example["prompt"][0]["content"] if example.get("prompt") else ""
        completion = example["completion"][0]["content"] if example.get("completion") else ""
        full_text = prompt + completion
        tokens = tokenizer.encode(full_text)
        lengths.append(len(tokens))

    print(f"  Mean tokens: {sum(lengths) / len(lengths):.0f}")
    print(f"  Max tokens: {max(lengths)}")
    print(f"  Min tokens: {min(lengths)}")

    # Save processed dataset
    save_dir = Path(f"data/processed/{args.condition}")
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_dir))
    print(f"\nSaved processed dataset to {save_dir}")


if __name__ == "__main__":
    main()
