"""Price the study from a completed pilot ledger and locally prepared data manifests.

No GPU jobs are started. The resulting reservation remains a proposal until its
cap is explicitly supplied to the study runner. Check the dated rates first.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint
from src.data import load_config, prepare_condition
from src.study import build_pilot_plan, build_study_plan, validate_pilot_measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-ledger", required=True)
    parser.add_argument("--pricing", default="configs/modal-pricing-2026-10-19.json")
    parser.add_argument("--data-root", default="data/prepared")
    parser.add_argument("--output", default="results/recovery/study-quote.json")
    args = parser.parse_args()
    ledger_path = Path(args.pilot_ledger)
    ledger = json.loads(ledger_path.read_text())
    if ledger["identity"]["plan"]["experiment"] != "pilot" or any(
        row["status"] != "complete" for row in ledger["results"].values()
    ):
        raise ValueError("A complete, persisted pilot is required; missing measurements are never guessed")
    config = load_config()
    if ledger["identity"]["plan"]["config_digest"] != fingerprint(config):
        raise ValueError("Pilot configuration differs from the proposed study")
    if ledger["identity"]["plan"]["implementation_digest"] != build_pilot_plan(config)["implementation_digest"]:
        raise ValueError("Pilot source differs from the proposed study; revalidate before pricing")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"], revision=config["model"]["revision"])
    pricing = json.loads(Path(args.pricing).read_text())
    cpu_rate = (
        pricing["cpu_core_per_second"] * pricing["physical_cores_limit"]
        + pricing["memory_gib_per_second"] * pricing["memory_gib_limit"]
    )
    cache = ledger_path.parent / "pilot-metadata"
    cache.mkdir(exist_ok=True)
    metadata, profiles = {}, {}
    for phase in ["sft", "grpo"]:
        row = ledger["results"]["pilot:" + phase]
        local = cache / (phase + ".json")
        remote = row["output"].removeprefix("/vol/") + "/training_metadata.json"
        if not local.exists():
            subprocess.run(
                [sys.executable, "-m", "modal", "volume", "get", "reasoning-distillation-vol", remote, str(local)],
                check=True,
            )
        metadata[phase] = json.loads(local.read_text())
        profile = cache / (phase + "-performance.json")
        remote_profile = row["output"].removeprefix("/vol/").rsplit("/", 1)[0] + "/" + phase + "-performance.json"
        if not profile.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "volume",
                    "get",
                    "reasoning-distillation-vol",
                    remote_profile,
                    str(profile),
                ],
                check=True,
            )
        profiles[phase] = json.loads(profile.read_text())
    eval_row = ledger["results"]["pilot:eval"]
    eval_summary = eval_row["output"]["benchmarks"]["gsm8k"]
    validate_pilot_measurements(config, metadata, eval_summary)
    measured_tokens = eval_summary["mean_generated_tokens"] * eval_summary["num_total"]
    if not measured_tokens:
        raise ValueError("Pilot generated no measurable tokens")
    eval_profile = cache / "eval-performance.json"
    remote_eval = eval_summary["results_path"].removeprefix("/vol/") + "/performance.json"
    if not eval_profile.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                "reasoning-distillation-vol",
                remote_eval,
                str(eval_profile),
            ],
            check=True,
        )
    eval_performance = json.loads(eval_profile.read_text())
    measured_tokens = eval_performance["generated_tokens_this_call"]
    if measured_tokens <= 0:
        raise ValueError("A fresh pilot evaluation must generate tokens to estimate throughput")
    eval_seconds_per_token = eval_performance["generation_seconds"] / measured_tokens
    eval_fixed = max(0, eval_row["elapsed_seconds"] - eval_performance["generation_seconds"])
    plan = build_study_plan(config)
    tasks = {}
    for node in plan["nodes"]:
        kind = node["kind"]
        if kind in {"prepare", "prepare_runtime"}:
            seconds = 300  # Main preparation reuses the completed pilot pool; this remains a timeout, not a claim of measured speed.
        elif kind == "verify":
            seconds = 60
        elif kind == "evaluate":
            problems = 500 if node["benchmark"] == "math500" else 1319
            seconds = eval_fixed + eval_seconds_per_token * problems * node["num_samples"] * node["max_new_tokens"]
        else:
            pilot = ledger["results"]["pilot:" + kind]
            per_step = profiles[kind]["train_seconds"] / metadata[kind]["exposure"]["optimizer_steps"]
            fixed = max(0, pilot["elapsed_seconds"] - profiles[kind]["train_seconds"])
            if kind == "grpo":
                steps = config["conditions"][node["condition"]]["grpo"]["max_steps"]
            else:
                _dataset, manifest = prepare_condition(
                    node["condition"], config, tokenizer, "sft", args.data_root, require_existing=True
                )
                settings = config["conditions"][node["condition"]]["sft"]
                from src.preprocessing import load_sft_tokens

                _tokens, token_manifest = load_sft_tokens(tokenizer, manifest, settings)
                sequences = token_manifest["rows"]
                steps = (
                    math.ceil(
                        sequences / (settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"])
                    )
                    * settings["num_train_epochs"]
                )
            seconds = fixed + per_step * steps
        timeout = math.ceil(seconds * 1.25)
        if timeout > 43200:
            raise ValueError(f"{node['id']} exceeds the per-job limit; partition and re-price before execution")
        tasks[node["id"]] = {
            "rate_per_second": cpu_rate + pricing["gpu_per_second"].get(node["gpu"], 0),
            "timeout_seconds": timeout,
            "overhead_seconds": pricing["overhead_seconds_per_task"],
        }
    quote = {
        "plan_digest": fingerprint(plan),
        "tasks": tasks,
        "pricing_source": pricing,
        "pilot_evidence": [{"path": str(ledger_path), "sha256": file_hash(ledger_path)}],
        "status": "proposal_not_approved",
        "method": "Measured pilot compute time plus one-time setup/export costs, padded by 25%; token-limit upper bound for evaluation. Batch-size differences and symbolic scoring remain estimation limits. Includes startup reserves, not provider billing guarantees.",
    }
    atomic_json(Path(args.output), quote)
    print(json.dumps(quote, indent=2))


if __name__ == "__main__":
    main()
