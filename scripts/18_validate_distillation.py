"""Audit downloaded distillation training artifacts and reconstruct question-batch token totals. No remote calls."""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_from_disk

from src.artifacts import file_hash, fingerprint
from src.checkpoint_experiment import immutable_json


def audit(proposal):
    proposal = Path(proposal)
    protocol = json.loads((proposal / "protocol.json").read_text())
    source_digest = fingerprint(
        {
            k: v
            for k, v in protocol["implementation"].items()
            if k.startswith("src/") or k in {"modal_train.py", "configs/config.yaml"}
        }
    )
    results, weight_hashes = {}, []
    for condition in ("distill_answer", "distill_trace"):
        expected = protocol["prepared"][condition]
        candidates = []
        for path in Path("data/distillation-prepared", condition, "sft").glob("*/data_manifest.json"):
            raw = json.loads(path.read_text())
            if (
                raw["content_hash"] == expected["content_hash"]
                and raw["spec"].get("matched_supervision_implementation")
                == protocol["implementation"]["src/distillation_experiment.py"]
            ):
                candidates.append((path, raw))
        if len(candidates) != 1:
            raise ValueError("The exact frozen local training preparation is required")
        path, raw = candidates[0]
        data = list(load_from_disk(str(path.parent / "dataset")))
        if fingerprint(data) != expected["content_hash"] or raw["source_ids"] != expected["source_ids"]:
            raise ValueError("Local training data differs from the frozen protocol")
        caches = list((path.parent / "sft-tokens").glob("*/tokens.json"))
        if len(caches) != 1:
            raise ValueError("One exact prepared token cache is required")
        token_meta = json.loads(caches[0].read_text())
        for name, digest in token_meta["files"].items():
            if file_hash(caches[0].parent / "dataset" / name) != digest:
                raise ValueError("Prepared token files changed")
        lengths = [len(row["input_ids"]) for row in load_from_disk(str(caches[0].parent / "dataset"))]
        if len(lengths) != 500 or sum(lengths) != expected["tokens"]["processed_tokens"]:
            raise ValueError("Question rows or token exposure changed")
        for seed in protocol["config"]["study"]["seeds"]:
            key = f"train_{condition}_{seed}"
            directory = proposal / "collected" / "training" / key
            metadata = json.loads((directory / "training_metadata.json").read_text())
            artifact = json.loads((directory / "artifact.json").read_text())
            profile = json.loads((directory / "performance.json").read_text())
            if artifact["files"]["training_metadata.json"] != file_hash(directory / "training_metadata.json"):
                raise ValueError("Training metadata differs from the exported manifest")
            if (
                artifact["format"] != "full"
                or artifact["parent"] is not None
                or artifact["base"] != protocol["config"]["model"]
                or artifact["metadata"] != metadata
            ):
                raise ValueError("Export lineage differs from the declared original student")
            exposure, args = metadata["exposure"], metadata["resolved_arguments"]
            if metadata["condition"] != condition or metadata["seed"] != seed:
                raise ValueError("Condition or seed identity mismatch")
            if metadata["runtime"]["source_digest"] != source_digest:
                raise ValueError("Training source differs from the frozen protocol")
            if (
                exposure["optimizer_steps"] != 64
                or exposure["epochs_completed"] != 2
                or exposure["processed_train_tokens"] != 2 * sum(lengths)
                or metadata["data_manifest"]["content_hash"] != expected["content_hash"]
            ):
                raise ValueError("Observed training exposure differs from the experiment")
            if not (
                args["seed"] == args["data_seed"] == seed
                and args["per_device_train_batch_size"] == 4
                and args["gradient_accumulation_steps"] == 4
                and args["padding_free"]
                and not args["packing"]
                and args["accelerator_config"]["use_seedable_sampler"]
            ):
                raise ValueError("Question-batch or sampler settings differ")
            observed = {row["step"]: int(row["num_tokens"]) for row in profile["step_metrics"] if "num_tokens" in row}
            expected_totals, batch_ids, cumulative, step = {}, [], 0, 0
            for epoch in range(2):
                order = torch.randperm(500, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
                for start in range(0, 500, 16):
                    selected = order[start : start + 16]
                    cumulative += sum(lengths[i] for i in selected)
                    step += 1
                    expected_totals[step] = cumulative
                    batch_ids.append([expected["source_ids"][i] for i in selected])
            if observed != expected_totals:
                raise ValueError("Actual per-update token stream differs from the seeded question batches")
            for row in profile["step_metrics"]:
                if any(not math.isfinite(float(row[name])) for name in ("loss", "grad_norm") if name in row):
                    raise ValueError("Nonfinite optimization metric")
            weights = artifact["files"]["model.safetensors"]
            weight_hashes.append(weights)
            results[key] = {
                "artifact_id": artifact["artifact_id"],
                "weights_sha256": weights,
                "optimizer_steps": 64,
                "epochs": 2,
                "input_tokens": cumulative,
                "supervised_tokens_from_prepared_masks_and_completed_epochs": 2
                * expected["tokens"]["supervised_tokens"],
                "all_64_step_token_totals_match": True,
                "question_batch_ids_digest": fingerprint(batch_ids),
                "train_seconds": profile["train_seconds"],
                "gpu_peak_memory_mib": profile["gpu"]["sampled_peak_device_memory_mib"],
                "gpu_utilization_full_call_mean_percent": profile["gpu"]["sampled_utilization_mean_percent"],
            }
    if len(set(weight_hashes)) != 6:
        raise ValueError("All six trained exports must have distinct weights")
    for seed in protocol["config"]["study"]["seeds"]:
        if (
            results[f"train_distill_answer_{seed}"]["question_batch_ids_digest"]
            != results[f"train_distill_trace_{seed}"]["question_batch_ids_digest"]
        ):
            raise ValueError("Corresponding arms have different reconstructed question batches")
    result = {
        "protocol_id": protocol["protocol_id"],
        "validator_sha256": file_hash(Path(__file__)),
        "all_six_exports_distinct": True,
        "training": results,
        "batch_check_scope": "Saved token totals agree with the seeded sampler at every update; source IDs are reconstructed, not independently logged by each GPU batch",
    }
    immutable_json(proposal / "collected" / "training-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    print(json.dumps(audit(parser.parse_args().proposal), indent=2))
