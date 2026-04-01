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


def binary_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    """Binary reward function for GRPOTrainer.

    Returns 1.0 if the extracted answer matches ground truth, 0.0 otherwise.

    TRL passes these arguments to reward functions:
    - prompts: list of prompt strings
    - completions: list of generated completion strings
    - answer: list of ground truth strings (from dataset's `answer` column)
    - **kwargs: completion_ids, trainer_state, log_extra, log_metric, etc.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        predicted = extract_answer_auto(completion)
        if predicted is not None and answers_match(predicted, ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
