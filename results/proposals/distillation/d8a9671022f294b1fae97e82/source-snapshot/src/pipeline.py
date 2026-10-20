"""Shared local/Modal phase execution. Data preparation always precedes GPU allocation."""

import math
import re
import time
from copy import deepcopy

from src.artifacts import atomic_json, new_training_directory, runtime_metadata
from src.data import prepare_condition
from src.reward import SCORER_VERSION, binary_reward_fn, format_reward_fn


def processed_training_tokens(trainer, phase):
    value = (
        getattr(trainer.state, "num_input_tokens_seen", None)
        if phase == "grpo"
        else getattr(trainer, "_total_train_tokens", None)
    )
    if value is None or value < 0:
        raise ValueError(f"Missing valid {phase} token exposure counter")
    return int(value)


class RewardDiagnostics:
    """Record the configured weighted objective independently of vendor raw-sum logs."""

    def __init__(self, weights, generations):
        if len(weights) != 2 or generations < 2 or not all(math.isfinite(w) for w in weights):
            raise ValueError("Expected two finite reward weights and a generation group")
        self.weights, self.generations = weights, generations
        self.batches = []
        self.logged = 0

    def record(self, correctness, formatting):
        if len(correctness) != len(formatting) or not correctness or len(correctness) % self.generations:
            raise ValueError("Reward vectors must contain complete aligned generation groups")
        weighted = [self.weights[0] * a + self.weights[1] * b for a, b in zip(correctness, formatting, strict=True)]
        if not all(math.isfinite(v) for v in weighted):
            raise ValueError("Nonfinite objective reward")
        groups = [weighted[i : i + self.generations] for i in range(0, len(weighted), self.generations)]
        self.batches.append(
            {
                "rollouts": len(weighted),
                "groups": len(groups),
                "reward_sum": sum(weighted),
                "zero_variance_groups": sum(max(g) == min(g) for g in groups),
            }
        )

    def summary(self, batches=None):
        rows = self.batches if batches is None else batches
        totals = {
            key: sum(row[key] for row in rows) for key in ["rollouts", "groups", "reward_sum", "zero_variance_groups"]
        }
        return {
            **totals,
            "reward_mean": totals["reward_sum"] / totals["rollouts"] if totals["rollouts"] else 0,
            "zero_variance_fraction": totals["zero_variance_groups"] / totals["groups"] if totals["groups"] else 0,
        }

    def take_logs(self):
        if self.logged == len(self.batches):
            return {}
        summary = self.summary(self.batches[self.logged :])
        self.logged = len(self.batches)
        return {
            "objective/reward_mean": summary["reward_mean"],
            "objective/zero_variance_fraction": summary["zero_variance_fraction"],
        }


def validate_training_request(config, condition, phase, checkpoint=None, check_artifact=True):
    if condition not in config["conditions"] or phase not in config["conditions"][condition]:
        raise ValueError(f"{condition} does not support phase {phase}")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model"].get("revision", "")):
        raise ValueError("Training requires an immutable starting model revision")
    if phase == "grpo" and config.get("reward_scorer", SCORER_VERSION) != SCORER_VERSION:
        raise ValueError("New GRPO training must use the current validated reward scorer")
    needs_parent = (
        config["conditions"][condition].get("requires_sft")
        or config["conditions"][condition]["type"] == "sft_then_grpo"
    )
    if phase == "grpo" and bool(checkpoint) != bool(needs_parent):
        raise ValueError("SFT checkpoint is required exactly for a post-SFT GRPO condition")
    if checkpoint and check_artifact:
        from src.artifacts import read_model_manifest

        parent = read_model_manifest(checkpoint, verify_files=False)
        expected = config["conditions"][condition].get("sft_parent", condition)
        metadata = parent.get("metadata", {})
        if parent["base"] != config["model"] or metadata.get("condition") != expected or metadata.get("phase") != "sft":
            raise ValueError("The GRPO parent must be the declared SFT condition on the same starting model")
        if metadata.get("seed") != config.get("seed", 42):
            raise ValueError("SFT and GRPO training seeds must match")


def prepare_phase(config, condition, phase, data_root):
    if condition not in config["conditions"] or phase not in config["conditions"][condition]:
        raise ValueError("Unsupported condition/phase")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"], revision=config["model"]["revision"])
    dataset, manifest = prepare_condition(condition, config, tokenizer, phase, data_root)
    result = {"rows": len(dataset), "manifest": manifest}
    if phase == "sft":
        from src.preprocessing import prepare_sft_tokens

        result["tokenized_manifest"] = prepare_sft_tokens(
            dataset, tokenizer, manifest, config["conditions"][condition]["sft"]
        )
    return result


def _train_phase(
    config,
    condition,
    phase,
    checkpoint=None,
    output_root="outputs/training",
    data_root="data/prepared",
    performance=None,
):
    validate_training_request(config, condition, phase, checkpoint)
    from transformers import AutoTokenizer

    from src.training import (
        build_grpo_trainer,
        build_sft_trainer,
        get_lora_config,
        load_model,
        load_sft_checkpoint,
        save_trained_model,
    )

    config = deepcopy(config)
    seed = config.get("seed", 42)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"], revision=config["model"]["revision"])
    dataset, manifest = prepare_condition(
        condition, config, tokenizer, phase, data_root, require_existing=True, verify_content=phase != "sft"
    )
    tokenized = None
    if phase == "sft":
        from src.preprocessing import load_sft_tokens

        dataset, tokenized = load_sft_tokens(tokenizer, manifest, config["conditions"][condition]["sft"])
    output = new_training_directory(
        output_root, condition, phase, {"config": config, "dataset": manifest["content_hash"], "parent": checkpoint}
    )
    performance["output_dir"] = str(output)
    settings = config["conditions"][condition]
    settings[phase]["output_dir"] = str(output)
    settings["seed"] = seed
    if phase == "grpo":
        if not manifest.get("max_prompt_tokens"):
            raise ValueError("Re-prepare GRPO data with measured prompt lengths")
        total = manifest["max_prompt_tokens"] + settings[phase]["max_completion_length"]
        settings[phase]["vllm_max_model_length"] = ((total + 127) // 128) * 128
    start = time.monotonic()
    model, tokenizer = load_sft_checkpoint(checkpoint, config) if checkpoint else load_model(config)
    performance["model_load_seconds"] = time.monotonic() - start
    setup_start = time.monotonic()
    observed = {"rollouts": 0, "prompt_groups": 0, "source_ids": set(), "completion_tokens": 0}
    diagnostics = (
        RewardDiagnostics(settings[phase]["reward_weights"], settings[phase]["num_generations"])
        if phase == "grpo"
        else None
    )

    def correctness(prompts, completions, answer, source_id=None, completion_ids=None, **kwargs):
        if source_id is None:
            raise ValueError("GRPO must forward source_id for exposure accounting")
        observed["rollouts"] += len(completions)
        observed["prompt_groups"] += len(completions) // settings["grpo"]["num_generations"]
        observed["source_ids"].update(source_id or [])
        if completion_ids is not None:
            observed["completion_tokens"] += sum(len(ids) for ids in completion_ids)
        scores = binary_reward_fn(prompts, completions, answer, **kwargs)
        diagnostics.record(scores, format_reward_fn(prompts, completions))
        return scores

    tracking = None
    report_to = settings[phase].get("report_to", "wandb") or []
    if "wandb" in report_to:
        import wandb

        tracking = wandb.init(
            project=config.get("wandb", {}).get("project", "reasoning-distillation"),
            entity=config.get("wandb", {}).get("entity"),
            name=f"{condition}-{seed}-{phase}-{output.parent.name}",
            config=config,
            reinit="finish_previous",
        )
    try:
        if phase == "sft":
            trainer = build_sft_trainer(
                model, tokenizer, dataset, settings, get_lora_config(config), prepared_manifest=tokenized
            )
        else:
            trainer = build_grpo_trainer(
                model,
                tokenizer,
                dataset,
                settings,
                [correctness, format_reward_fn],
                get_lora_config(config),
                diagnostics,
            )
        performance["trainer_setup_seconds"] = time.monotonic() - setup_start
        train_start = time.monotonic()
        train_result = trainer.train()
        performance["train_seconds"] = time.monotonic() - train_start
        trainer.save_state()
        stats = {
            **observed,
            "source_ids": sorted(observed["source_ids"]),
            "unique_prompts": len(observed["source_ids"]) if phase == "grpo" else None,
            "available_prepared_rows": len(dataset),
            "optimizer_steps": trainer.state.global_step,
            "elapsed_seconds": time.monotonic() - start,
            "train_metrics": train_result.metrics,
            "processed_train_tokens": processed_training_tokens(trainer, phase),
            "processed_train_tokens_scope": "prompt_plus_completion" if phase == "grpo" else "packed_input",
            "processed_train_sequences": len(trainer.train_dataset),
            "epochs_completed": trainer.state.epoch,
        }
        metadata = {
            "condition": condition,
            "phase": phase,
            "seed": seed,
            "config": config,
            "resolved_arguments": trainer.args.to_dict(),
            "data_manifest": manifest,
            "tokenized_manifest": tokenized,
            "exposure": stats,
            "wandb_run_id": None,
            "runtime": runtime_metadata(),
            "reward_objective": {
                "scorer_version": SCORER_VERSION,
                "weights": settings[phase]["reward_weights"],
                "diagnostics": diagnostics.summary(),
                "vendor_reward_log": "unweighted component sum",
            }
            if diagnostics
            else None,
        }
        metadata["wandb_run_id"] = tracking.id if tracking else None
        atomic_json(output / "training_metadata.json", metadata)
        export_start = time.monotonic()
        exported = save_trained_model(
            trainer.model, trainer.processing_class, output, config, parent_path=checkpoint, metadata=metadata
        )
        performance["export_seconds"] = time.monotonic() - export_start
        performance["artifact_id"] = exported["artifact_id"]
        performance["optimizer_steps"] = trainer.state.global_step
        performance["step_metrics"] = trainer.state.log_history
        performance["processed_train_tokens"] = stats["processed_train_tokens"]
        return str(output)
    finally:
        if tracking:
            tracking.finish()


def train_phase(config, condition, phase, checkpoint=None, output_root="outputs/training", data_root="data/prepared"):
    from pathlib import Path

    from src.performance import GpuMonitor

    performance = {}
    monitor = GpuMonitor(
        1 if config.get("pilot") else config.get("performance", {}).get("telemetry_interval_seconds", 5)
    )
    try:
        with monitor:
            result = _train_phase(config, condition, phase, checkpoint, output_root, data_root, performance)
    except Exception as error:
        performance["status"] = "failed"
        performance["error_type"] = type(error).__name__
        if performance.get("output_dir"):
            performance["gpu"] = monitor.summary()
            atomic_json(Path(performance["output_dir"]).parent / (phase + "-performance.json"), performance)
        raise
    performance["status"] = "complete"
    performance["gpu"] = monitor.summary()
    atomic_json(Path(result).parent / (phase + "-performance.json"), performance)
    return result
