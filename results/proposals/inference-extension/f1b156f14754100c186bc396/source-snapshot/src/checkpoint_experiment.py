"""A frozen, evaluation-only ablation of the historical sequential-loading defect."""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

from src.artifacts import file_hash, fingerprint
from src.evaluation import benchmark_rows, summarize_records
from src.reward import SCORER_VERSION, score_response
from src.statistics import paired_comparison, wilson

ARMS = ("sft_parent", "missing_sft", "reconstructed")
HISTORICAL_SOURCES = {
    "sft": "outputs/sft_then_grpo/checkpoint-117",
    "grpo": "outputs/sft_then_grpo",
}
PILOT = Path("results/recovery/pilot-final-outcome.json")


def source_identity():
    paths = [
        *sorted(Path("src").glob("*.py")),
        Path("modal_train.py"),
        Path("configs/config.yaml"),
        Path("scripts/16_checkpoint_ablation.py"),
    ]
    return {str(path): file_hash(path) for path in paths}


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change frozen artifact: {path}")
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def freeze_protocol(config, rows, excluded_ids, sources, implementation):
    ids = [row["problem_id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(excluded_ids) <= set(ids):
        raise ValueError("Benchmark IDs or pilot exclusion IDs are inconsistent")
    eligible = [i for i, row in enumerate(rows) if row["problem_id"] not in set(excluded_ids)]
    indices = sorted(random.Random(2026).sample(eligible, 256))
    selected = [rows[i] for i in indices]
    protocol = {
        "schema_version": 1,
        "question": "Effect of omitting the exact SFT parent when reloading one historical SFT-to-GRPO run",
        "model": config["model"],
        "dataset": config["datasets"]["gsm8k"],
        "population_digest": fingerprint(rows),
        "selection_seed": 2026,
        "excluded_pilot_ids": sorted(set(excluded_ids)),
        "problem_indices": indices,
        "problem_ids": [row["problem_id"] for row in selected],
        "rows_digest": fingerprint(selected),
        "rows": selected,
        "sources": sources,
        "implementation": implementation,
        "primary_comparison": ["reconstructed", "missing_sft"],
        "secondary_comparison": ["reconstructed", "sft_parent"],
        "evaluation": {
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "num_samples_per_problem": 1,
            "enable_thinking": True,
            "batch_size": 32,
            "scorer": SCORER_VERSION,
        },
        "statistics": {"bootstrap_seed": 42, "replicates": 10000, "secondary_is_exploratory": True},
        "limits": [
            "One historical training run; uncertainty covers test items, not training seeds",
            "The original base revision is unknown; all reconstructions use the same declared pinned revision",
            "New boxed-answer prompting differs from historical evaluation; historical scores remain unchanged",
            "No claims about general RL efficacy, reasoning emergence, or verification causality",
            "Question selection excludes the recent pilot but is not a previously unseen benchmark",
        ],
    }
    return {**protocol, "protocol_id": fingerprint(protocol)}


def build_experiment_plan(config, protocol):
    nodes = [{"id": "prepare:runtime", "kind": "prepare_runtime", "depends_on": [], "gpu": None}]
    for arm in ARMS:
        parent = "reconstruct:sft_parent" if arm == "reconstructed" else None
        nodes.append(
            {
                "id": "reconstruct:" + arm,
                "kind": "reconstruct",
                "condition": arm,
                "depends_on": ["prepare:runtime"] + ([parent] if parent else []),
                "parent": parent,
                "gpu": None,
            }
        )
    for arm in ARMS:
        key = "eval:" + arm
        nodes.append(
            {
                "id": key,
                "kind": "evaluate",
                "condition": "checkpoint_ablation_" + arm,
                "parent": "reconstruct:" + arm,
                "depends_on": ["reconstruct:" + name for name in ARMS],
                "gpu": "L40S",
                "seed": 42,
                "benchmark": "gsm8k",
                "max_new_tokens": 2048,
                "num_samples": 1,
                "temperature": 0.0,
                "problem_indices": protocol["problem_indices"],
                "expected_rows_digest": protocol["rows_digest"],
            }
        )
        nodes.append(
            {
                "id": key + ":verify",
                "kind": "verify",
                "parent": key,
                "depends_on": [key],
                "gpu": None,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "checkpoint_ablation",
        "protocol_id": protocol["protocol_id"],
        "config_digest": fingerprint(config),
        "nodes": nodes,
    }


def build_experiment_quote(plan):
    pricing = json.loads(Path("configs/modal-pricing-2026-10-19.json").read_text())
    cpu = pricing["physical_cores_limit"] * pricing["cpu_core_per_second"]
    cpu += pricing["memory_gib_limit"] * pricing["memory_gib_per_second"]
    timeouts = {"prepare_runtime": 600, "reconstruct": 1800, "evaluate": 5400, "verify": 90}
    return {
        "status": "bounded_checkpoint_ablation_proposal",
        "plan_digest": fingerprint(plan),
        "pricing_source": pricing,
        "pilot_evidence": [{"path": str(PILOT), "sha256": file_hash(PILOT)}],
        "tasks": {
            n["id"]: {
                "timeout_seconds": timeouts[n["kind"]],
                "overhead_seconds": 120,
                "rate_per_second": cpu + pricing["gpu_per_second"].get(n["gpu"], 0),
            }
            for n in plan["nodes"]
        },
        "budget_usd": 20,
        "additional_allowances_usd": {"setup_and_builds": 2, "recovery_contingency": 6},
        "method": "Three L40S calls capped at 90 minutes; CPU jobs bounded separately. No automatic retries. "
        "Pilot throughput extrapolates to about $2-8 for evaluation; timeouts are not a billing guarantee.",
    }


def prepare_proposal(config, output_root="results/proposals/checkpoint-ablation"):
    rows = benchmark_rows("gsm8k", config)
    pilot = Path("outputs/pilot-evidence/boxed/evaluation/predictions.jsonl")
    exclusions = [json.loads(line)["problem_id"] for line in pilot.read_text().splitlines()]
    sources = {}
    for name, source in HISTORICAL_SOURCES.items():
        folder = Path("outputs/historical/2026-10-19") / source
        sources[name] = {
            "path": source,
            "files": {
                filename: file_hash(folder / filename)
                for filename in ["adapter_config.json", "adapter_model.safetensors"]
            },
        }
    protocol = freeze_protocol(config, rows, exclusions, sources, source_identity())
    protocol["pilot_exclusion_evidence"] = {"path": str(pilot), "sha256": file_hash(pilot)}
    protocol.pop("protocol_id")
    protocol["protocol_id"] = fingerprint(protocol)
    plan = build_experiment_plan(config, protocol)
    quote = build_experiment_quote(plan)
    from src.study import validate_quote

    maximum = validate_quote(plan, quote, 20)
    if maximum + sum(quote["additional_allowances_usd"].values()) > 20:
        raise ValueError("Runtime plus setup and contingency exceeds the agreed budget")
    path = Path(output_root) / protocol["protocol_id"][:24]
    for name, value in [("protocol", protocol), ("plan", plan), ("quote", quote)]:
        immutable_json(path / (name + ".json"), value)
    return path


def validate_proposal(path, config, budget):
    path = Path(path)
    protocol, plan, quote = [
        json.loads((path / (name + ".json")).read_text()) for name in ["protocol", "plan", "quote"]
    ]
    body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if protocol["protocol_id"] != fingerprint(body) or protocol["implementation"] != source_identity():
        raise ValueError("Frozen protocol or implementation changed; create a new proposal before execution")
    if plan != build_experiment_plan(config, protocol) or quote != build_experiment_quote(plan):
        raise ValueError("Frozen plan, configuration, pilot evidence, or pricing changed")
    from src.study import validate_quote

    maximum = validate_quote(plan, quote, budget)
    if not 0 < budget <= 20 or maximum + sum(quote["additional_allowances_usd"].values()) > budget:
        raise ValueError("Experiment requires its agreed total cap, at most $20 including allowances")
    if fingerprint(protocol["rows"]) != protocol["rows_digest"]:
        raise ValueError("Frozen question contents changed")
    return protocol, plan, quote


def reconstruct_arm(arm, protocol, config, output_root, source_root="/vol", parent_path=None):
    """Export on CPU, then require exact logits after loading the complete saved weights."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.artifacts import read_model_manifest
    from src.training import save_trained_model

    if arm not in ARMS or bool(parent_path) != (arm == "reconstructed"):
        raise ValueError("Only the corrected arm requires the exact reconstructed SFT parent")
    for source in protocol["sources"].values():
        for filename, expected in source["files"].items():
            if file_hash(Path(source_root) / source["path"] / filename) != expected:
                raise ValueError("Historical source adapter changed")
    if config["model"] != protocol["model"] or config.get("quantization", {}).get("enabled"):
        raise ValueError("Reconstruction requires the frozen unquantized base")
    parent = read_model_manifest(parent_path) if parent_path else None
    if parent and parent["metadata"]["checkpoint_ablation"] != {
        "protocol_id": protocol["protocol_id"],
        "arm": "sft_parent",
        "sources": protocol["sources"],
    }:
        raise ValueError("Correct reconstruction was given a different SFT parent")
    torch.set_num_threads(2)
    options = {"dtype": torch.bfloat16, "device_map": None, "attn_implementation": "sdpa"}
    model = AutoModelForCausalLM.from_pretrained(
        parent_path or config["model"]["name"],
        **({} if parent_path else {"revision": config["model"]["revision"]}),
        **options,
    )
    source = protocol["sources"]["sft" if arm == "sft_parent" else "grpo"]
    model = PeftModel.from_pretrained(model, Path(source_root) / source["path"]).merge_and_unload(safe_merge=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer"],
        revision=config["model"]["revision"],
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    inputs = tokenizer("Reload parity probe: 2 + 2 =", return_tensors="pt")
    with torch.inference_mode():
        expected = model(**inputs).logits.float().clone()
    output = Path(output_root) / arm
    manifest = save_trained_model(
        model,
        tokenizer,
        output,
        config,
        parent_path=parent_path,
        metadata={
            "checkpoint_ablation": {"protocol_id": protocol["protocol_id"], "arm": arm, "sources": protocol["sources"]},
            "historical_base_revision_verified": False,
            "intentional_missing_sft_control": arm == "missing_sft",
        },
    )
    del model
    gc.collect()
    reloaded = AutoModelForCausalLM.from_pretrained(output, **options).eval()
    with torch.inference_mode():
        actual = reloaded(**inputs).logits.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    read_model_manifest(output)
    immutable_json(
        Path(output_root) / (arm + "-reload-proof.json"),
        {
            "artifact_id": manifest["artifact_id"],
            "protocol_id": protocol["protocol_id"],
            "exact_cpu_logits_equal": True,
            "maximum_absolute_difference": float((actual - expected).abs().max()),
            "parent_artifact_id": parent["artifact_id"] if parent else None,
        },
    )
    return str(output)


def analyze_experiment(protocol, arms):
    if set(arms) != set(ARMS):
        raise ValueError("All three complete constructions are required")
    summaries = {}
    for arm, records in arms.items():
        summaries[arm] = summarize_records(records, protocol["problem_ids"], 1)
        if any(row.get("scorer_version") != SCORER_VERSION for row in records):
            raise ValueError("Experiment rows use a different scoring policy")
        summaries[arm]["wilson_95"] = wilson(summaries[arm]["num_correct"], len(records))
    return {
        "protocol_id": protocol["protocol_id"],
        "status": "complete_single_run_subset_ablation",
        "arms": summaries,
        "primary": paired_comparison(arms["reconstructed"], arms["missing_sft"]),
        "secondary": paired_comparison(arms["reconstructed"], arms["sft_parent"]),
        "secondary_is_exploratory": True,
        "uncertainty": "Paired test-item uncertainty only; one training run, one prompt and token budget",
        "limits": protocol.get("limits", []),
    }


def collect_experiment(proposal, ledger, output_root):
    protocol = json.loads((Path(proposal) / "protocol.json").read_text())
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    arms, identities = {}, {}

    def download(remote, local):
        if not local.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "volume",
                    "get",
                    "reasoning-distillation-vol",
                    remote.removeprefix("/vol/"),
                    str(local),
                ],
                check=True,
            )

    for arm in ARMS:
        row = ledger["results"]["eval:" + arm]
        verification = ledger["results"]["eval:" + arm + ":verify"]
        if row["status"] != "complete" or verification["status"] != "complete":
            raise ValueError("Cannot analyze an incomplete experiment")
        summary = row["output"]["benchmarks"]["gsm8k"]
        path = root / arm
        path.mkdir(exist_ok=True)
        for name in ["evaluation.json", "summary.json", "predictions.jsonl", "performance.json"]:
            download(summary["results_path"] + "/" + name, path / name)
        if (
            file_hash(path / "predictions.jsonl") != verification["output"]["predictions_sha256"]
            or json.loads((path / "summary.json").read_text()) != summary
        ):
            raise ValueError("Downloaded records do not match independent persistence verification")
        identity = json.loads((path / "evaluation.json").read_text())["identity"]
        identities[arm] = {k: v for k, v in identity.items() if k != "model"}
        arms[arm] = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        truths = {row["problem_id"]: row["answer"] for row in protocol["rows"]}
        for row in arms[arm]:
            if row["ground_truth"] != truths[row["problem_id"]]:
                raise ValueError("Persisted ground truth differs from the frozen protocol")
            rescored = score_response(row["response"], row["ground_truth"], "gsm8k", SCORER_VERSION).to_dict()
            if any(row.get(key) != value for key, value in rescored.items()):
                raise ValueError("Saved primary score does not reproduce on CPU")
        model_path = ledger["results"]["reconstruct:" + arm]["output"]
        download(model_path + "/artifact.json", path / "artifact.json")
        download(str(Path(model_path).parent / (arm + "-reload-proof.json")), path / "reload-proof.json")
        proof = json.loads((path / "reload-proof.json").read_text())
        if proof["protocol_id"] != protocol["protocol_id"] or proof["artifact_id"] != summary["model_identity"]:
            raise ValueError("Reload proof does not identify the evaluated artifact")
    if any(value != identities["sft_parent"] for value in identities.values()):
        raise ValueError("Arms differ in evaluation protocol or runtime")
    analysis = analyze_experiment(protocol, arms)
    diagnostic, review, key = [], [], {}
    for arm, records in arms.items():
        for row in records:
            legacy = score_response(row["response"], row["ground_truth"], "gsm8k", "legacy-v1")
            diagnostic.append(
                {
                    "arm": arm,
                    "problem_id": row["problem_id"],
                    "primary_correct": row["correct"],
                    "legacy_correct": legacy.correct,
                }
            )
            review_id = fingerprint({"protocol": protocol["protocol_id"], "arm": arm, "problem_id": row["problem_id"]})[
                :16
            ]
            key[review_id] = {"arm": arm, "problem_id": row["problem_id"]}
            question = next(q["question"] for q in protocol["rows"] if q["problem_id"] == row["problem_id"])
            review.append(
                {
                    "review_id": review_id,
                    "question": question,
                    "ground_truth": row["ground_truth"],
                    "response": row["response"],
                    "finish_reason": row["finish_reason"],
                    "reviewed_correct": "",
                    "notes": "",
                }
            )
    analysis["legacy_rescore_diagnostic"] = {
        arm: sum(d["primary_correct"] != d["legacy_correct"] for d in diagnostic if d["arm"] == arm) for arm in ARMS
    }
    analysis["legacy_rescore_caution"] = "Same new prompted responses; not a reproduction of historical scores"
    immutable_json(root / "analysis.json", analysis)
    immutable_json(root / "legacy-rescore-diagnostic.json", diagnostic)
    immutable_json(root / "review-key.json", key)
    review.sort(key=lambda row: row["review_id"])
    if not (root / "blinded-review.csv").exists():
        with (root / "blinded-review.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    return analysis
