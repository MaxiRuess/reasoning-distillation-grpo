"""Answer extraction and binary reward function for GRPO training."""

import math
import re


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer after the #### delimiter.

    GSM8K format: reasoning steps followed by #### final_answer
    """
    if "####" not in text:
        return None

    # Take everything after the last ####
    answer = text.split("####")[-1].strip()

    # Clean up common formatting
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    answer = answer.strip().rstrip(".")

    return answer if answer else None


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} with proper brace-depth counting.

    Handles nested braces like \\boxed{\\frac{1}{2}} correctly.
    """
    # Find the last occurrence of \boxed{
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    # Start after \boxed{
    start = idx + len("\\boxed{")
    depth = 1
    i = start

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    # Extract content (excluding the final closing brace)
    return text[start : i - 1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, handles common math formatting, and attempts
    numeric parsing for robust comparison.
    """
    answer = answer.strip()
    answer = answer.rstrip(".")
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")

    # Try to simplify LaTeX fractions like \frac{1}{2} -> 1/2
    frac_match = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", answer)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return str(num / den)
        except ValueError:
            pass

    # Try numeric parse
    try:
        val = float(answer)
        # Represent integers without decimal point
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    return answer


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Compare two answers after normalization.

    Tries numeric comparison first (with tolerance), then exact string match.
    """
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Try numeric comparison
    try:
        pred_val = float(pred_norm)
        truth_val = float(truth_norm)
        return math.isclose(pred_val, truth_val, rel_tol=1e-6, abs_tol=1e-8)
    except ValueError:
        pass

    # Fall back to exact string match
    return pred_norm == truth_norm


def extract_answer_auto(text: str) -> str | None:
    """Try multiple extraction methods in order of specificity.

    1. \\boxed{} (most specific, MATH format)
    2. #### delimiter (GSM8K format)
    3. Last number in text (fallback)
    """
    # Try \boxed{} first
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer

    # Try #### delimiter
    answer = extract_gsm8k_answer(text)
    if answer is not None:
        return answer

    # Fallback: extract last number from text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def _extract_completion_text(completion) -> str:
    """Extract plain text from a completion, handling both string and message dict formats."""
    if isinstance(completion, list):
        return " ".join(
            msg["content"] for msg in completion if isinstance(msg, dict) and "content" in msg
        )
    elif not isinstance(completion, str):
        return str(completion)
    return completion


def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """Reward for producing structured reasoning output.

    Gives partial credit for format compliance, providing gradient signal
    even when the answer is wrong. This bootstraps the base model into
    learning the <think>...</think> + answer structure before correctness
    reward can kick in.

    Scoring (0.0 to 1.0):
    - 0.5 for <think>...</think> block
    - 0.25 for \boxed{...} answer
    - 0.25 for any number appearing after </think>
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        score = 0.0

        # Check for <think>...</think> block
        if re.search(r"<think>.*?</think>", text, re.DOTALL):
            score += 0.5

        # Check for \boxed{} answer
        if "\\boxed{" in text:
            score += 0.25

        # Check for a number after </think> (or anywhere if no think block)
        after_think = text.split("</think>")[-1] if "</think>" in text else text
        if re.search(r"\d+", after_think):
            score += 0.25

        rewards.append(score)
    return rewards


def binary_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    """Binary reward function for GRPOTrainer.

    Returns 1.0 if the extracted answer matches ground truth, 0.0 otherwise.

    TRL passes these arguments to reward functions:
    - prompts: list of prompt strings or message dicts
    - completions: list of completion strings or message dicts
    - answer: list of ground truth strings (from dataset's `answer` column)
    - **kwargs: completion_ids, trainer_state, log_extra, log_metric, etc.

    In TRL 1.0, completions may be passed as lists of message dicts
    (e.g. [{"role": "assistant", "content": "..."}]) rather than plain strings.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        text = _extract_completion_text(completion)
        predicted = extract_answer_auto(text)
        if predicted is not None and answers_match(predicted, ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
