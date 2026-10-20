"""Shared CLI contracts for preparation, training and evaluation."""

import argparse
import json

from src.data import load_config


def prepare_main(phase):
    from src.pipeline import prepare_phase

    parser = argparse.ArgumentParser(description=f"Prepare validated {phase.upper()} data on CPU")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", required=phase == "sft", default="grpo_only")
    parser.add_argument("--data-root", default="data/prepared")
    args = parser.parse_args()
    result = prepare_phase(load_config(args.config), args.condition, phase, args.data_root)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "path": result["manifest"]["path"],
                "quarantine_counts": result["manifest"]["quarantine_counts"],
            },
            indent=2,
        )
    )


def training_main(phase):
    from src.pipeline import train_phase

    parser = argparse.ArgumentParser(description=f"Train {phase.upper()} using prepared, fingerprinted data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/training")
    parser.add_argument("--data-root", default="data/prepared")
    args = parser.parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed
    if phase == "sft" and args.sft_checkpoint:
        parser.error("--sft-checkpoint is a GRPO-only argument")
    print(train_phase(config, args.condition, phase, args.sft_checkpoint, args.output_root, args.data_root))


def evaluation_main():
    from src.evaluation import run_all_evaluations
    from src.reward import SCORER_VERSION, SCORER_VERSIONS

    parser = argparse.ArgumentParser(description="Resumable evaluation retaining raw responses")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path")
    model.add_argument("--base-model-only", action="store_true")
    parser.add_argument("--benchmarks", default="gsm8k")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer", choices=SCORER_VERSIONS, default=SCORER_VERSION)
    parser.add_argument("--resume-dir")
    parser.add_argument("--output-root", default="outputs/evaluations")
    args = parser.parse_args()
    config = load_config(args.config)
    config["evaluation"].update(
        benchmarks=args.benchmarks.split(","),
        num_samples_per_problem=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        scorer=args.scorer,
        output_root=args.output_root,
        resume_dir=args.resume_dir,
    )
    print(
        json.dumps(
            run_all_evaluations(
                config["model"]["name"] if args.base_model_only else args.model_path, config, args.condition
            ),
            indent=2,
        )
    )
