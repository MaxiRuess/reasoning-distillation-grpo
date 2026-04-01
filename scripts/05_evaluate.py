"""Run evaluation benchmarks on a trained model."""

import argparse
import json

from src.data import load_config
from src.evaluation import run_all_evaluations


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model-path", required=True, help="Path to trained model adapter")
    parser.add_argument("--condition", required=True, help="Condition name (for labeling results)")
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,math",
        help="Comma-separated list of benchmarks to run",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override evaluation config with CLI args
    config["evaluation"]["benchmarks"] = args.benchmarks.split(",")
    config["evaluation"]["num_samples_per_problem"] = args.num_samples

    print(f"Evaluating model: {args.model_path}")
    print(f"Condition: {args.condition}")
    print(f"Benchmarks: {config['evaluation']['benchmarks']}")
    print(f"Samples per problem: {args.num_samples}")

    results = run_all_evaluations(args.model_path, config, args.condition)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<12} {'Accuracy':<12} {'Correct':<12} {'Total':<12}")
    print("-" * 60)
    for name, res in results["benchmarks"].items():
        print(f"{name:<12} {res['accuracy']:<12.4f} {res['num_correct']:<12} {res['num_total']:<12}")
    print("=" * 60)


if __name__ == "__main__":
    main()
