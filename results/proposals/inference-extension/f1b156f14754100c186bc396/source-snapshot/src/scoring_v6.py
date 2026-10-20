"""Versioned final-response extraction and benchmark-specific correctness rewards."""

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext

SCORER_VERSION = "final-response-v6"
SCORER_VERSIONS = (
    SCORER_VERSION,
    "final-response-v5",
    "final-response-v4",
    "final-response-v3",
    "final-response-v2",
    "legacy-v1",
)
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SMALL = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ],
        range(20),
        strict=True,
    )
)
_TENS = dict(
    zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10), strict=True)
)
_SCALES = {"thousand": 1000, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_WORD = "(?:" + "|".join([*_SMALL, *_TENS, "hundred", *_SCALES]) + ")"
_WORD_NUMBER = rf"(?:(?:minus|negative)\s+)?{_WORD}(?:[\s-]+(?:and[\s-]+)?{_WORD})*"
_VALUE = rf"(?:{_NUMBER}\s*/\s*{_NUMBER}|{_NUMBER}|[+-]?\\(?:d?frac)\{{[^{{}}]+\}}\{{[^{{}}]+\}}|{_WORD_NUMBER})"


def _cardinal_value(text):
    tokens = text.lower().replace("-", " ").split()
    sign = -1 if tokens[0] in {"minus", "negative"} else 1
    if sign == -1:
        tokens = tokens[1:]
    total, group, last_scale, kind = 0, 0, float("inf"), None
    for token in tokens:
        if token == "and":
            if kind not in {"hundred", "scale"}:
                return None
            kind = "and"
        elif token in _SMALL:
            value = _SMALL[token]
            if kind == "small" or (kind == "tens" and not 1 <= value <= 9):
                return None
            group += value
            kind = "small"
        elif token in _TENS:
            if kind in {"small", "tens"}:
                return None
            group += _TENS[token]
            kind = "tens"
        elif token == "hundred":
            if kind != "small" or not 1 <= group <= 9:
                return None
            group *= 100
            kind = "hundred"
        elif token in _SCALES:
            scale = _SCALES[token]
            if not group or scale >= last_scale or kind == "and":
                return None
            total += group * scale
            group, last_scale, kind = 0, scale, "scale"
        else:
            return None
    return sign * (total + group) if kind not in {None, "and"} else None


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
    if re.fullmatch(_WORD_NUMBER, text, re.IGNORECASE):
        value = _cardinal_value(text)
        return Decimal(value) if value is not None else None
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


def _extract_final_v2(text: str, symbolic: bool = False) -> str | None:
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


def _numeric_claim(claim: str, anchored=False) -> str | None:
    """Accept a complete numeric assertion, never a numeric prefix of an expression."""
    values = list(re.finditer(rf"(?<![\w.,/\\])({_VALUE})(?![\w.,/])", claim.rstrip(".! "), re.IGNORECASE))
    if not values:
        return None
    if anchored and claim[: values[0].start()].strip(" $:"):
        return None
    normalized = {normalize_answer(match[1]) for match in values}
    if "" in normalized or len(normalized) != 1:
        return None
    # Even a single valid number must not be part of an unfinished expression,
    # alternative answer, malformed thousands separator, or deliberation.
    remainder = claim
    for match in reversed(values):
        remainder = remainder[: match.start()] + " " + remainder[match.end() :]
    if re.search(
        r"[\d/+*=^<>\\]|\b(?:or|either|maybe|perhaps|consider\w*|guess\w*|tried|trying)\b", remainder, re.IGNORECASE
    ):
        return None
    if "?" in claim:
        return None
    if re.search(
        r"\b(?:(?:more|less|higher|lower|greater|fewer|larger|smaller)\s+than|at\s+(?:least|most)|between)\b",
        claim,
        re.IGNORECASE,
    ):
        return None
    prefix = claim[: values[0].start()].strip(" $:")
    if re.search(r"\b(?:I|we)\s+(?:check(?:ed|ing)?|consider\w*|tried|try|guess\w*)\b", prefix, re.IGNORECASE):
        return None
    return next(iter(normalized))


def _asserted_value(sentence: str) -> str | None:
    # Parse the value asserted by a predicate, rather than incidental quantities
    # in its subject. Trailing temporal/comparison context is not the answer.
    if "=" in sentence:
        answer = _contextual_value(sentence.rsplit("=", 1)[1])
        if answer is not None:
            return answer
    predicates = list(
        re.finditer(
            r"\b(?:is|are|was|were|be|equals?|gets?|receives?|takes?|earns?|makes?|needs?|requires?|"
            r"pays?|paid|costs?|spends?|spent|has|have|remains?|runs?|sells?|buys?|contains?|"
            r"ordered|born|gives?|total\s+(?:of|is)|profit\s+of)\s+",
            sentence,
            re.IGNORECASE,
        )
    )
    for predicate in reversed(predicates):
        tail = sentence[predicate.end() :]
        if not re.match(rf"\$?\s*{_VALUE}", tail, re.IGNORECASE):
            continue
        answer = _contextual_value(tail)
        if answer is not None:
            return answer
    return _contextual_claim(sentence, anchored=False)


def _contextual_value(text):
    return _contextual_claim(text, anchored=True)


def _contextual_claim(text, anchored):
    text = text.rstrip(".! ")
    value = re.search(rf"(?<![\w.,/\\]){_VALUE}(?![\w.,/])", text, re.IGNORECASE)
    if value is None:
        return None
    tail = text[value.end() :]
    context = re.search(
        rf"(?:\s+|[,;(]\s*)(?:after|before|when|while|for|from|if|assuming|given|where|since|because|which|compared\s+to|so\s+that|such\s+that)\b"
        rf"|\s+to\s+(?!{_WORD_NUMBER}\b)[A-Za-z]+"
        rf"|\s+(?:[A-Za-z][A-Za-z-]*\s+)+over\b",
        tail,
        re.IGNORECASE,
    )
    claim = text[: value.end() + context.start()] if context else text
    return _numeric_claim(claim, anchored=anchored)


def _extract_final(text: str, symbolic: bool = False) -> str | None:
    if symbolic:
        return _extract_final_v2(text, symbolic=True)
    text, status = final_response(text)
    if status != "ok" or not text:
        return None
    if "\\boxed{" in text:
        return normalize_answer(extract_boxed_answer(text)) or None
    if "####" in text:
        return normalize_answer(text.rsplit("####", 1)[1].strip()) or None
    text = text.replace("**", "").replace("`", "")
    # Numeric answer statements and answer headings outrank explanatory prose.
    # A nonnumeric "answer is straightforward" is not an answer statement.
    markers = list(re.finditer(r"\b(?:final\s+answer|the\s+answer|answer)\s*(?:is\b|[:=])\s*", text, re.IGNORECASE))
    for marker in reversed(markers):
        tail = text[marker.end() :]
        first = re.split(r"(?<=[.!?])\s+|\n+", tail, maxsplit=1)[0]
        if re.match(rf"\$?\s*{_VALUE}", first, re.IGNORECASE):
            return _contextual_value(first)
        if marker[0].rstrip().endswith((":", "=")):
            return _prose_assertion(re.split(r"\n\s*\n", tail, maxsplit=1)[0])
    # Work with the concluding paragraph, not numbers inside preceding working.
    paragraph = re.split(r"\n\s*\n", text)[-1]
    return _prose_assertion(paragraph)


def _prose_assertion(paragraph):
    if re.search(r"\b(?:cannot|can't|unable\s+to)\s+(?:calculate|determine|answer)\b", paragraph, re.IGNORECASE):
        return None
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", paragraph) if p.strip()]
    conclusions = [s for s in sentences if re.match(r"(?:thus|therefore|hence|so|in\s+conclusion)\b", s, re.IGNORECASE)]
    totals = [s for s in sentences if re.search(r"\btotal\s+(?:of|is)\b", s, re.IGNORECASE)]
    for group in [conclusions[::-1], totals[::-1], sentences]:
        for sentence in group:
            answer = _asserted_value(sentence)
            if answer is not None:
                return answer
    return None


def extract_answer_auto(text: str) -> str | None:
    return _extract_final(text)


def score_response(text: str, ground_truth: str, benchmark: str = "gsm8k", scorer: str = SCORER_VERSION) -> Score:
    if scorer == "final-response-v5":
        from src.scoring_v5 import score_response as score_v5

        return Score(**score_v5(text, ground_truth, benchmark).to_dict())
    if scorer == "final-response-v4":
        from src.scoring_v4 import score_response as score_v4

        return Score(**score_v4(text, ground_truth, benchmark).to_dict())
    if scorer == "final-response-v3":
        from src.scoring_v3 import score_response as score_v3

        return Score(**score_v3(text, ground_truth, benchmark).to_dict())
    if scorer == "final-response-v2":
        from src.scoring_v2 import score_response as score_v2

        return Score(**score_v2(text, ground_truth, benchmark).to_dict())
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
