"""Analyze actual raw responses from v2 evaluation directories."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.artifacts import atomic_json
from src.evaluation import summarize_records

PATTERNS = {
    "arithmetic": r"\d+\s*[+*/-]\s*\d+",
    "verification": r"(?i)(double.check|let me check|verify)",
    "backtracking": r"(?i)(wait|actually|that's wrong|mistake)",
}


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "evaluation.json").read_text())
    records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    summary = summarize_records(records, manifest["problem_ids"], manifest["num_samples"])
    patterns = (
        {
            name: sum(bool(re.search(pattern, row["response"])) for row in records) / len(records)
            for name, pattern in PATTERNS.items()
        }
        if records
        else {}
    )
    return {
        "summary": summary,
        "pattern_frequencies": patterns,
        "interpretation": "Lexical markers are descriptive, not evidence of faithful reasoning or causal self-correction.",
        "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/evaluations")
    args = parser.parse_args()
    for path in Path(args.results_dir).rglob("evaluation.json"):
        result = analyze(path.parent)
        atomic_json(path.parent / "trace_analysis.json", result)
        print(path.parent, json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
