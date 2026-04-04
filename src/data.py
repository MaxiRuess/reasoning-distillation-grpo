"""Dataset loading and formatting for all training conditions."""

from pathlib import Path

import yaml
from datasets import Dataset, load_dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_s1k_for_sft(tokenizer) -> Dataset:
    """Format s1K-1.1 dataset for SFT with reasoning traces.

    The s1K dataset contains problems with DeepSeek-R1 reasoning traces.
    Each example is formatted as a chat conversation with <think> tags wrapping
    the reasoning trace in the assistant response.
    """
    ds = load_dataset("simplescaling/s1K-1.1", split="train")

    def format_example(example):
        question = example["question"]

        # s1K-1.1 columns: deepseek_thinking_trajectory (str), deepseek_attempt (str)
        thinking = example.get("deepseek_thinking_trajectory", "")
        answer = example.get("deepseek_attempt", example.get("solution", ""))

        assistant_content = f"<think>\n{thinking}\n</think>\n\n{answer}" if thinking else answer

        return {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": assistant_content}],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def format_openr1_for_sft(tokenizer, max_samples: int = 1000) -> Dataset:
    """Format Open-R1 dataset for re-distillation SFT.

    Uses reasoning traces from RL-trained models (verified correct by math
    verification). This tests whether RL-improved traces transfer better
    than raw DeepSeek-R1 traces (used in sft_traces condition).

    Selects the first verified-correct generation per problem.
    """
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")

    # Filter for examples with at least one math-verified correct generation
    ds = ds.filter(lambda x: x["correctness_math_verify"] is not None and any(x["correctness_math_verify"]))

    # Subsample to match sft_traces data size
    ds = ds.select(range(min(max_samples, len(ds))))

    def format_example(example):
        problem = example["problem"]

        # Pick the first verified-correct generation
        generations = example["generations"]
        correctness = example["correctness_math_verify"]
        correct_trace = None
        for gen, is_correct in zip(generations, correctness):
            if is_correct:
                correct_trace = gen
                break

        if correct_trace is None:
            correct_trace = generations[0]  # Fallback (shouldn't happen after filter)

        return {
            "prompt": [{"role": "user", "content": problem}],
            "completion": [{"role": "assistant", "content": correct_trace}],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def format_orca_math_for_sft(tokenizer, max_samples: int | None = None) -> Dataset:
    """Format Orca Math dataset for baseline SFT (answer-only, no reasoning traces)."""
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_example(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        return {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": answer}],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def format_numinamath_for_grpo(tokenizer, max_samples: int | None = None) -> Dataset:
    """Format NuminaMath-RL-Verifiable for GRPO training.

    GRPOTrainer expects a `prompt` column (list of message dicts) and an `answer`
    column that gets forwarded to the reward function as a keyword argument.
    """
    ds = load_dataset("nlile/NuminaMath-1.5-RL-Verifiable", split="train")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_example(example):
        # NuminaMath-1.5-RL-Verifiable columns: problem, solution, answer
        problem = example["problem"]
        answer = example["answer"]  # Final numerical answer string

        return {
            "prompt": [{"role": "user", "content": problem}],
            "answer": str(answer),
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # Filter out examples with empty answers
    ds = ds.filter(lambda x: len(x["answer"].strip()) > 0)
    return ds


def format_gsm8k_for_grpo(tokenizer, max_samples: int | None = None) -> Dataset:
    """Format GSM8K train set for GRPO training.

    GSM8K answers contain reasoning + #### final_number. We extract the
    final number as ground truth for the binary reward function.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_example(example):
        question = example["question"]
        # Extract final numerical answer after ####
        full_answer = example["answer"]
        final_answer = full_answer.split("####")[-1].strip() if "####" in full_answer else full_answer

        return {
            "prompt": [{"role": "user", "content": question}],
            "answer": final_answer,
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["answer"].strip()) > 0)
    return ds


def format_gsm8k_for_eval() -> Dataset:
    """Load GSM8K test set for evaluation.

    Returns dataset with `question` and `answer` columns.
    The answer column contains the full solution with #### delimiter.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")

    ds = ds.rename_column("question", "question")
    # GSM8K has 'answer' column with reasoning + #### final_answer
    return ds


def format_numinamath_for_eval(max_samples: int = 500) -> Dataset:
    """Load a held-out split of NuminaMath-CoT for MATH-style evaluation.

    Returns dataset with `question` and `answer` columns where answers
    use \\boxed{} format.
    """
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    # Use last N examples as a pseudo-test split
    total = len(ds)
    start_idx = max(0, total - max_samples)
    ds = ds.select(range(start_idx, total))

    def format_example(example):
        # NuminaMath-CoT columns: problem, solution
        return {
            "question": example["problem"],
            "answer": example["solution"],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def get_dataset_for_condition(
    condition_name: str, config: dict, tokenizer
) -> Dataset | tuple[Dataset, Dataset]:
    """Dispatcher: return the appropriate formatted dataset(s) for a training condition.

    For `sft_then_grpo`, returns a tuple of (sft_dataset, grpo_dataset).
    """
    condition = config["conditions"][condition_name]

    if condition_name == "baseline":
        return format_orca_math_for_sft(tokenizer, max_samples=50000)

    elif condition_name == "sft_traces":
        return format_s1k_for_sft(tokenizer)

    elif condition_name == "grpo_only":
        return format_numinamath_for_grpo(tokenizer)

    elif condition_name == "re_distill":
        return format_openr1_for_sft(tokenizer, max_samples=1000)

    elif condition_name == "sft_then_grpo":
        sft_ds = format_s1k_for_sft(tokenizer)
        grpo_ds = format_numinamath_for_grpo(tokenizer)
        return sft_ds, grpo_ds

    else:
        raise ValueError(f"Unknown condition: {condition_name}")
