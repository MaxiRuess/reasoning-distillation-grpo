"""Explicit final-answer scoring; archived prose heuristics remain reproducible."""

import importlib
import re
from dataclasses import asdict, dataclass

from src.scoring_v6 import (
    _numeric,
    extract_answer_auto,
    extract_boxed_answer,
    extract_gsm8k_answer,
    final_response,
    normalize_answer,
)
from src.scoring_v6 import answers_match as _approximate_match

__all__ = [
    "SCORER_VERSION",
    "SCORER_VERSIONS",
    "Score",
    "answers_match",
    "binary_reward_fn",
    "extract_answer_auto",
    "extract_boxed_answer",
    "extract_gsm8k_answer",
    "final_response",
    "format_reward_fn",
    "normalize_answer",
    "score_response",
]

SCORER_VERSION = "boxed-final-v1"
SCORER_VERSIONS = (SCORER_VERSION, *(f"final-response-v{n}" for n in range(6, 1, -1)), "legacy-v1")


@dataclass(frozen=True)
class Score:
    predicted: str | None
    correct: bool
    status: str
    scorer_version: str = SCORER_VERSION

    def to_dict(self):
        return asdict(self)


def answers_match(predicted, ground_truth):
    value, truth = _numeric(predicted), _numeric(ground_truth)
    if value is None or truth is None:
        return False
    if truth == truth.to_integral_value():
        return value == truth
    return _approximate_match(predicted, ground_truth)


def score_response(text, ground_truth, benchmark="gsm8k", scorer=SCORER_VERSION):
    if scorer in SCORER_VERSIONS[1:]:
        module = "scoring_v2" if scorer == "legacy-v1" else "scoring_v" + scorer.rsplit("v", 1)[1]
        old = importlib.import_module("src." + module).score_response(text, ground_truth, benchmark, scorer=scorer)
        return Score(**old.to_dict())
    if scorer != SCORER_VERSION or benchmark not in {"gsm8k", "math500"}:
        raise ValueError("Unknown scorer or benchmark")
    answer, status = final_response(text)
    if status != "ok":
        return Score(None, False, status)
    if r"\boxed{" not in (answer or ""):
        return Score(None, False, "missing_final_answer")
    if answer.count(r"\boxed{") != 1:
        return Score(None, False, "ambiguous_final_answer")
    boxed = extract_boxed_answer(answer)
    if boxed is None:
        return Score(None, False, "malformed_final_answer")
    if benchmark == "gsm8k":
        predicted = normalize_answer(boxed) or None
        return Score(
            predicted,
            answers_match(predicted, ground_truth),
            "ok" if predicted is not None else "unparseable_final_answer",
        )
    from math_verify import LatexExtractionConfig, parse, verify

    gold = parse("$" + ground_truth + "$", extraction_config=[LatexExtractionConfig()])
    predicted = parse("$" + boxed + "$", extraction_config=[LatexExtractionConfig()])
    return Score(
        boxed if predicted else None,
        bool(gold and predicted and verify(gold, predicted)),
        "ok" if predicted else "unparseable_final_answer",
    )


def _completion_text(completion):
    if isinstance(completion, list):
        return " ".join(str(message.get("content", "")) for message in completion if isinstance(message, dict))
    return str(completion)


def format_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        answer, status = final_response(text)
        if status != "ok":
            rewards.append(0.0)
            continue
        thinking = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        reward = 0.5 if thinking and thinking[1].strip() else 0.0
        boxed = extract_boxed_answer(answer or "") if (answer or "").count(r"\boxed{") == 1 else None
        reward += 0.25 if boxed else 0.0
        reward += 0.25 if boxed and normalize_answer(boxed) else 0.0
        rewards.append(reward)
    return rewards


def binary_reward_fn(prompts, completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("Completions and ground truths must align")
    return [
        float(score_response(_completion_text(text), str(truth)).correct)
        for text, truth in zip(completions, answer, strict=True)
    ]
