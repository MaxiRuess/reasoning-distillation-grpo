"""Post-hoc analysis of reasoning traces across all conditions."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REASONING_PATTERNS = {
    "arithmetic": r"\d+\s*[\+\-\*\/]\s*\d+",
    "step_by_step": r"(?i)(step\s+\d|first|second|third|next|then|finally)",
    "backtracking": r"(?i)(wait|actually|no,|let me reconsider|that's wrong|mistake)",
    "verification": r"(?i)(let me check|verify|double.check|to confirm|makes sense)",
    "exploration": r"(?i)(let me think|consider|what if|alternatively|another approach)",
    "equation_setup": r"(?i)(let\s+\w+\s*=|equation|formula|substitute)",
}


def classify_reasoning_patterns(trace: str) -> dict[str, bool]:
    """Classify which reasoning patterns appear in a trace."""
    return {
        name: bool(re.search(pattern, trace))
        for name, pattern in REASONING_PATTERNS.items()
    }


def load_eval_results(results_dir: Path) -> dict[str, dict]:
    """Load evaluation results from all conditions."""
    all_results = {}
    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir():
            continue
        results_file = condition_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results[condition_dir.name] = json.load(f)
    return all_results


def analyze_trace_lengths(results: dict[str, dict], output_dir: Path):
    """Analyze and plot trace length distributions across conditions."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (condition, data) in zip(axes, results.items()):
        for benchmark_name, benchmark_data in data.get("benchmarks", {}).items():
            if "results" not in benchmark_data:
                continue
            lengths = [r["trace_length"] for r in benchmark_data["results"]]
            ax.hist(lengths, bins=50, alpha=0.7, label=benchmark_name)

        ax.set_title(condition)
        ax.set_xlabel("Trace Length (chars)")
        ax.legend()

    axes[0].set_ylabel("Count")
    plt.suptitle("Trace Length Distributions by Condition")
    plt.tight_layout()
    plt.savefig(output_dir / "trace_lengths.png", dpi=150)
    plt.close()
    print(f"Saved trace length plot to {output_dir / 'trace_lengths.png'}")


def analyze_length_vs_correctness(results: dict[str, dict], output_dir: Path):
    """Analyze correlation between trace length and correctness."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (condition, data) in zip(axes, results.items()):
        for benchmark_name, benchmark_data in data.get("benchmarks", {}).items():
            if "results" not in benchmark_data:
                continue

            correct_lengths = []
            incorrect_lengths = []
            for r in benchmark_data["results"]:
                if any(r.get("correct", [])):
                    correct_lengths.append(r["trace_length"])
                else:
                    incorrect_lengths.append(r["trace_length"])

            positions = [1, 2]
            data_to_plot = [correct_lengths, incorrect_lengths]
            bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(["Correct", "Incorrect"])

        ax.set_title(condition)

    axes[0].set_ylabel("Trace Length (chars)")
    plt.suptitle("Trace Length vs Correctness")
    plt.tight_layout()
    plt.savefig(output_dir / "length_vs_correctness.png", dpi=150)
    plt.close()
    print(f"Saved length vs correctness plot to {output_dir / 'length_vs_correctness.png'}")


def analyze_reasoning_patterns(results: dict[str, dict], output_dir: Path):
    """Classify and compare reasoning patterns across conditions."""
    pattern_data = {}

    for condition, data in results.items():
        pattern_counts = {name: 0 for name in REASONING_PATTERNS}
        total = 0

        for benchmark_data in data.get("benchmarks", {}).values():
            if "results" not in benchmark_data:
                continue
            for r in benchmark_data["results"]:
                # Use the first prediction as the trace to analyze
                if r.get("predictions"):
                    trace = str(r["predictions"][0]) if r["predictions"][0] else ""
                    patterns = classify_reasoning_patterns(trace)
                    for name, present in patterns.items():
                        if present:
                            pattern_counts[name] += 1
                    total += 1

        if total > 0:
            pattern_data[condition] = {
                name: count / total for name, count in pattern_counts.items()
            }

    if not pattern_data:
        print("No trace data available for pattern analysis")
        return

    # Create comparison table
    df = pd.DataFrame(pattern_data).T
    df = df.round(3)
    print("\n--- Reasoning Pattern Frequencies ---")
    print(df.to_string())

    # Plot
    df.plot(kind="bar", figsize=(10, 5))
    plt.title("Reasoning Pattern Frequencies by Condition")
    plt.ylabel("Frequency")
    plt.xlabel("Condition")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "reasoning_patterns.png", dpi=150)
    plt.close()
    print(f"Saved reasoning patterns plot to {output_dir / 'reasoning_patterns.png'}")


def print_summary_table(results: dict[str, dict]):
    """Print a markdown-formatted summary table."""
    print("\n## Results Summary\n")
    print(f"| {'Condition':<20} | {'Benchmark':<10} | {'Accuracy':<10} | {'Avg Trace Len':<15} |")
    print(f"|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 17}|")

    for condition, data in results.items():
        for bench_name, bench_data in data.get("benchmarks", {}).items():
            acc = bench_data.get("accuracy", 0)
            avg_len = bench_data.get("avg_trace_length", 0)
            print(f"| {condition:<20} | {bench_name:<10} | {acc:<10.4f} | {avg_len:<15.0f} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze reasoning traces across conditions")
    parser.add_argument(
        "--results-dir",
        default="outputs",
        help="Directory containing condition subdirectories with eval_results.json",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_eval_results(results_dir)

    if not results:
        print(f"No evaluation results found in {results_dir}")
        print("Run evaluation first with: python scripts/05_evaluate.py")
        return

    print(f"Found results for conditions: {list(results.keys())}")

    print_summary_table(results)
    analyze_trace_lengths(results, output_dir)
    analyze_length_vs_correctness(results, output_dir)
    analyze_reasoning_patterns(results, output_dir)

    print(f"\nAll analysis outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
