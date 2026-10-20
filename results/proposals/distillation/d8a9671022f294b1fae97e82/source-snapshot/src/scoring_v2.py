"""Versioned final-response extraction and benchmark-specific correctness rewards."""

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext

SCORER_VERSION = "final-response-v2"
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VALUE = rf"(?:{_NUMBER}\s*/\s*{_NUMBER}|{_NUMBER}|[+-]?\\(?:d?frac)\{{[^{{}}]+\}}\{{[^{{}}]+\}})"


@dataclass(frozen=True)
class Score:
    predicted: str | None
    correct: bool
    status: str
    scorer_version: str = SCORER_VERSION

    def to_dict(self):
        return asdict(self)


def final_response(text: str) -> tuple[str | None, str]:
    """Require balanced thinking tags, then exclude all thinking from scoring."""
    depth = 0
    end = 0
    for match in re.finditer(r"</?think>", text):
        if match.group() == "<think>":
            if depth:
                return None, "malformed_thinking"
            depth = 1
        else:
            if not depth:
                return None, "malformed_thinking"
            depth = 0
            end = match.end()
    if depth:
        return None, "unfinished_thinking"
    return text[end:].strip(), "ok"


def extract_boxed_answer(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index].strip() or None
    return None


def _numeric(answer: str | None) -> Decimal | None:
    if not isinstance(answer, str):
        return None
    text = answer.strip().strip("$").strip().rstrip(".").rstrip("%").strip()
    if not re.fullmatch(_VALUE, text):
        return None
    text = text.replace(",", "")
    frac = re.fullmatch(r"([+-]?)\\d?frac\{([^{}]+)\}\{([^{}]+)\}", text)
    try:
        with localcontext() as context:
            context.prec = 40
            if frac:
                value = Decimal(frac[2]) / Decimal(frac[3])
                if frac[1] == "-":
                    value = -value
            elif "/" in text:
                numerator, denominator = text.split("/")
                value = Decimal(numerator.strip()) / Decimal(denominator.strip())
            else:
                value = Decimal(text)
        return value if value.is_finite() and math.isfinite(float(value)) else None
    except (InvalidOperation, ArithmeticError, ValueError, OverflowError):
        return None


def normalize_answer(answer: str | None) -> str:
    value = _numeric(answer)
    if value is None:
        return ""
    # Avoid enormous fixed-point strings for scientific notation.
    return (
        str(int(value))
        if value == value.to_integral_value() and abs(value) < Decimal("1e30")
        else str(value.normalize())
    )


def answers_match(predicted: str | None, ground_truth: str | None) -> bool:
    predicted_value, truth_value = _numeric(predicted), _numeric(ground_truth)
    if predicted_value is None or truth_value is None:
        return False
    return math.isclose(float(predicted_value), float(truth_value), rel_tol=1e-6, abs_tol=1e-8)


def extract_gsm8k_answer(text: str) -> str | None:
    return normalize_answer(text.rsplit("####", 1)[1]) or None if "####" in text else None


def _extract_final(text: str, symbolic: bool = False) -> str | None:
    text, status = final_response(text)
    if status != "ok" or not text:
        return None
    if "\\boxed{" in text:
        answer = extract_boxed_answer(text)
    elif "####" in text:
        answer = text.rsplit("####", 1)[1].strip()
    else:
        markers = list(
            re.finditer(
                r"\b(?:(?:final\s+answer|the\s+answer|answer)\s*(?:is\b|[:=])|(?:gets?|receives?|equals?|takes?|is|are)\b)\s*(.+)",
                text,
                re.IGNORECASE,
            )
        )
        answer = markers[-1][1] if markers else text.splitlines()[-1].strip()
        answer = answer.replace("**", "").replace("`", "").strip()
        if markers:
            # Units may follow a single complete value; another numeric expression may not.
            match = re.fullmatch(rf"\$?\s*({_VALUE})\s*\$?\s*%?(?:\s+[A-Za-z][A-Za-z\s-]*)?[.!]?", answer)
            if match:
                answer = match[1]
    return answer if symbolic and answer else normalize_answer(answer) or None


def extract_answer_auto(text: str) -> str | None:
    return _extract_final(text)


def score_response(text: str, ground_truth: str, benchmark: str = "gsm8k", scorer: str = SCORER_VERSION) -> Score:
    if scorer == "legacy-v1":
        from src import legacy_reward

        predicted = legacy_reward.extract_answer_auto(text)
        try:
            correct = predicted is not None and legacy_reward.answers_match(predicted, ground_truth)
        except (ValueError, OverflowError):
            correct = False
        return Score(predicted, correct, "legacy", scorer)
    if scorer != SCORER_VERSION:
        raise ValueError(f"Unknown scorer: {scorer}")
    _, status = final_response(text)
    if status != "ok":
        return Score(None, False, status)
    predicted = _extract_final(text, symbolic=benchmark == "math500")
    if predicted is None:
        return Score(None, False, "unparseable")
    if benchmark == "gsm8k":
        return Score(predicted, answers_match(predicted, ground_truth), "ok")
    if benchmark != "math500":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    from math_verify import LatexExtractionConfig, parse, verify

    gold = parse("$" + ground_truth + "$", extraction_config=[LatexExtractionConfig()])
    pred = parse("$" + predicted + "$", extraction_config=[LatexExtractionConfig()])
    return Score(predicted, bool(gold and pred and verify(gold, pred)), "ok" if pred else "unparseable")


def _extract_completion_text(completion) -> str:
    if isinstance(completion, list):
        return " ".join(str(msg.get("content", "")) for msg in completion if isinstance(msg, dict))
    return str(completion)


def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        answer, status = final_response(text)
        if status != "ok":
            rewards.append(0.0)
            continue
        thinking = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        score = 0.5 if thinking and thinking[1].strip() else 0.0
        score += 0.25 if extract_boxed_answer(answer or "") else 0.0
        score += 0.25 if extract_answer_auto(text) is not None else 0.0
        rewards.append(score)
    return rewards


def binary_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    if len(completions) != len(answer):
        raise ValueError("Completions and ground truths must align")
    return [
        float(score_response(_extract_completion_text(text), str(truth)).correct)
        for text, truth in zip(completions, answer, strict=True)
    ]
