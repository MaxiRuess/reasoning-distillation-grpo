"""Download, validate, and cache GRPO datasets for training."""

import argparse
from pathlib import Path

from src.data import format_numinamath_for_grpo, load_config
from src.reward import extract_answer_auto
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"])

    print("Preparing GRPO data from NuminaMath-RL-Verifiable")

    dataset = format_numinamath_for_grpo(tokenizer, max_samples=args.max_samples)
    print(f"\nDataset size after formatting: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    # Validate answer parseability
    parseable = 0
    unparseable = 0
    for example in dataset:
        answer = example["answer"]
        extracted = extract_answer_auto(answer)
        if extracted is not None:
            parseable += 1
        else:
            unparseable += 1

    print(f"\n--- Answer Validation ---")
    print(f"  Parseable answers: {parseable} ({100 * parseable / len(dataset):.1f}%)")
    print(f"  Unparseable answers: {unparseable} ({100 * unparseable / len(dataset):.1f}%)")

    # Show sample examples
    print("\n--- Sample Examples ---")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        prompt_text = example["prompt"][0]["content"] if example.get("prompt") else "N/A"
        print(f"\n[Example {i + 1}]")
        print(f"  Problem: {prompt_text[:200]}...")
        print(f"  Ground truth answer: {example['answer']}")

    # Save processed dataset
    save_dir = Path("data/processed/grpo")
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_dir))
    print(f"\nSaved processed dataset to {save_dir}")


if __name__ == "__main__":
    main()
