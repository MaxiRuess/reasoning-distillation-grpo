"""Frozen, inference-only follow-up of existing reasoning-distillation checkpoints."""

import json
import math
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.artifacts import file_hash, fingerprint
from src.checkpoint_experiment import immutable_json
from src.data import load_config
from src.distillation_experiment import ENDPOINT, SEEDS, execute_campaign, numeric_final_score
from src.statistics import paired_comparison
from src.study import validate_quote

CAP = 30.0
ORIGINAL = Path("results/proposals/distillation/d8a9671022f294b1fae97e82")
ORIGINAL_LEDGER = Path("outputs/distillation/f1f80a202340d80f68faadfc/ledger.json")
MODELS = ("base", "trace_42", "trace_43", "trace_44")
BUDGETS = (2048, 4096, 8192)
RATES = {"gpu_per_second": 0.000542, "cpu_core_per_second": 0.0000131, "memory_gib_per_second": 0.00000222}


def implementation_identity():
    return {
        str(p): file_hash(p)
        for p in [
            *sorted(Path("src").glob("*.py")),
            Path("modal_train.py"),
            Path("configs/config.yaml"),
            Path("requirements/sft.lock"),
            Path("scripts/20_inference_extension.py"),
        ]
    }


def jobs_for(models):
    jobs = {}
    for tokens in BUDGETS:
        for model in MODELS:
            jobs[f"eval:{tokens}:{model}"] = {
                "model_key": model,
                "model_path": models[model]["path"],
                "generation": {
                    "max_new_tokens": tokens,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "enable_thinking": True,
                    "seed": 20261023,
                    "num_samples_per_problem": 1,
                },
            }
    for policy, temperature, top_p in (("sampled", 0.7, 0.8), ("greedy", 0.0, 0.95)):
        jobs[f"eval:2048:nothink_{policy}:base"] = {
            "model_key": "base",
            "model_path": models["base"]["path"],
            "generation": {
                "max_new_tokens": 2048,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 20,
                "enable_thinking": False,
                "seed": 20261023,
                "num_samples_per_problem": 1,
            },
        }
    return jobs


def freeze_protocol():
    old = json.loads((ORIGINAL / "protocol.json").read_text())
    ledger = json.loads(ORIGINAL_LEDGER.read_text())
    if ledger["status"] != "complete":
        raise ValueError("The original study must be complete")
    models = {
        "base": {
            "path": old["config"]["model"]["name"],
            "artifact_id": fingerprint(old["config"]["model"]),
            "weights_sha256": ledger["results"]["prepare:runtime"]["output"]["weights_sha256"],
        }
    }
    for seed in SEEDS:
        manifest_path = ORIGINAL / "collected/training" / f"train_distill_trace_{seed}" / "artifact.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["format"] != "full" or manifest["base"] != old["config"]["model"]:
            raise ValueError("Unexpected trace model lineage")
        models[f"trace_{seed}"] = {
            "path": ledger["results"][f"train:distill_trace:{seed}"]["output"],
            "artifact_id": manifest["artifact_id"],
            "manifest_sha256": file_hash(manifest_path),
        }
    selected = sorted(random.Random(20261022).sample(range(256), 128))
    subset = {
        "rows": [old["subsets"]["2048"]["rows"][i] for i in selected],
        "problem_indices": [old["subsets"]["2048"]["problem_indices"][i] for i in selected],
    }
    subset["rows_digest"] = fingerprint(subset["rows"])
    config = deepcopy(load_config())
    if config["model"] != old["config"]["model"]:
        raise ValueError("The original pinned student must be retained")
    config["evaluation"].pop("max_problems", None)
    config["evaluation"].pop("resume_dir", None)
    body = {
        "question": "Do existing trace checkpoints outperform the unchanged student under sampled thinking at 8K, and how do completion and accuracy vary across output limits?",
        "original_protocol_id": old["protocol_id"],
        "original_ledger_sha256": file_hash(ORIGINAL_LEDGER),
        "original_analysis_sha256": file_hash(ORIGINAL / "collected/analysis.json"),
        "config": config,
        "models": models,
        "subset": subset,
        "jobs": jobs_for(models),
        "implementation": implementation_identity(),
        "cap_usd": CAP,
        "primary_endpoint": ENDPOINT,
        "primary_comparison": "Mean trace-minus-unchanged accuracy at 8192 tokens on the same 128 questions, conditional on three trained seeds and one decoding seed",
        "secondary": "2K and 4K comparisons, length/completion diagnostics, sampled no-thinking policy and greedy no-thinking versus original answer-only outputs; descriptive",
        "selection_seed": 20261022,
        "decoding_seed": 20261023,
        "statistics": "10000 paired item-bootstrap resamples, seed42; three training seeds share128 questions; conditional item intervals and separate seed SD",
        "limits": [
            "Follow-up designed after seeing original results; these are previously used questions, not a new hidden benchmark",
            "One decoding seed; this does not estimate variability across sampled generations",
            "Effective batch sizes32/16/8 change with2K/4K/8K limits; no isolated causal output-length claim",
            "8K remains below Qwen's recommended32768 output tokens; no unconstrained-capability claim",
            "No training, second benchmark, outcome-dependent extension or automatic paid retry",
            "No-thinking recommended sampling differs from old greedy outputs; greedy diagnostic is retained for decoder-matched comparison",
            "Conditional accuracy among finished outputs selects different questions and is not a causal measure of reasoning",
        ],
    }
    return {**body, "protocol_id": fingerprint(body)}


def build_plan(protocol):
    nodes = [{"id": "prepare:runtime", "kind": "prepare_runtime", "depends_on": [], "gpu": None}]
    for job_id, job in sorted(protocol["jobs"].items()):
        generation = job["generation"]
        nodes.extend(
            [
                {
                    "id": job_id,
                    "kind": "evaluate",
                    "condition": job_id,
                    "model_key": job["model_key"],
                    "model_path": job["model_path"],
                    "depends_on": ["prepare:runtime"],
                    "gpu": "L40S",
                    "benchmark": "gsm8k",
                    "max_new_tokens": generation["max_new_tokens"],
                    "temperature": generation["temperature"],
                    "num_samples": 1,
                    "seed": generation["seed"],
                    "problem_indices": protocol["subset"]["problem_indices"],
                    "expected_rows_digest": protocol["subset"]["rows_digest"],
                },
                {"id": job_id + ":verify", "kind": "verify", "parent": job_id, "depends_on": [job_id], "gpu": None},
            ]
        )
    return {"experiment": "sampled-inference-extension", "protocol_id": protocol["protocol_id"], "nodes": nodes}


def build_quote(plan):
    cpu = 2 * RATES["cpu_core_per_second"] + 16 * RATES["memory_gib_per_second"]
    tasks = {}
    for node in plan["nodes"]:
        if node["kind"] == "evaluate":
            timeout = 600 if "nothink" in node["id"] else {2048: 1200, 4096: 2400, 8192: 7200}[node["max_new_tokens"]]
        else:
            timeout = 600 if node["kind"] == "prepare_runtime" else 90
        tasks[node["id"]] = {
            "timeout_seconds": timeout,
            "overhead_seconds": 120,
            "rate_per_second": cpu + (RATES["gpu_per_second"] if node["gpu"] else 0),
        }
    quote = {
        "plan_digest": fingerprint(plan),
        "tasks": tasks,
        "pricing_source": "https://modal.com/pricing",
        "pricing_rechecked": "2026-10-20",
        "rates": RATES,
        "pilot_evidence": "Original study ledger and 2K/4K L40S performance profiles; 8K is extrapolated",
        "non_runtime_allowance_usd": 1.0,
        "caution": "Timeout reservations are not expected usage or a provider invoice; no credits assumed",
    }
    quote["reserved_runtime_usd"] = validate_quote(plan, quote, CAP - 1)
    return quote


def validate_proposal(proposal, budget):
    protocol, plan, quote = [
        json.loads((Path(proposal) / f"{name}.json").read_text()) for name in ("protocol", "plan", "quote")
    ]
    if not math.isfinite(budget) or budget != CAP:
        raise ValueError("The exact separately authorized $30 cap is required")
    if fingerprint({k: v for k, v in protocol.items() if k != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("Protocol identity changed")
    if protocol != freeze_protocol() or plan != build_plan(protocol) or quote != build_quote(plan):
        raise ValueError("Source, model lineage, subset, config, plan or quote changed after freezing")
    validate_quote(plan, quote, budget - quote["non_runtime_allowance_usd"])
    return protocol, plan, quote


def execute(protocol, preflight, evaluate, verify, root="outputs/inference-extension"):
    def forbidden(**kwargs):
        raise ValueError("Training is outside the inference-only authorization")

    plan = build_plan(protocol)
    quote = build_quote(plan)
    return execute_campaign(
        plan,
        protocol["config"],
        quote,
        CAP - quote["non_runtime_allowance_usd"],
        root,
        prepare_runtime=preflight,
        prepare=forbidden,
        sft=forbidden,
        grpo=forbidden,
        evaluate=evaluate,
        verify=verify,
    )


def contrast(left, right):
    paired = [paired_comparison(a, b) for a, b in zip(left, right, strict=True)]
    differences = np.array(
        [
            [int(a["correct"]) - int(b["correct"]) for a, b in zip(aa, bb, strict=True)]
            for aa, bb in zip(left, right, strict=True)
        ]
    )
    items = differences.mean(axis=0)
    rng = np.random.default_rng(42)
    draws = np.concatenate([items[rng.integers(len(items), size=(500, len(items)))].mean(axis=1) for _ in range(20)])
    return {
        "mean_delta": float(items.mean()),
        "paired_item_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "num_distinct_questions": len(items),
        "per_seed": paired,
        "training_seed_delta_std": float(differences.mean(axis=1).std(ddof=1)) if len(paired) > 1 else None,
        "uncertainty": "Item interval conditional on observed trained models and decoding seed; shared items are not independent replications",
    }


def score_rows(rows):
    return [
        {**r, **numeric_final_score(r["response"], r["ground_truth"]), "strict_correct": r["correct"]} for r in rows
    ]


def summarize(rows):
    eos = [r for r in rows if r["finish_reason"] == "eos"]
    not_unfinished = [r for r in rows if r["status"] != "unfinished_thinking"]

    def accuracy(rr):
        return sum(r["correct"] for r in rr) / len(rr) if rr else None

    return {
        "num_total": len(rows),
        "num_correct": sum(r["correct"] for r in rows),
        "accuracy": accuracy(rows),
        "strict_accuracy": sum(r["strict_correct"] for r in rows) / len(rows),
        "eos_count": len(eos),
        "output_limit_count": len(rows) - len(eos),
        "unfinished_thinking_count": len(rows) - len(not_unfinished),
        "without_unfinished_thinking_count": len(not_unfinished),
        "final_status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "accuracy_among_eos": accuracy(eos),
        "accuracy_among_without_unfinished_thinking": accuracy(not_unfinished),
        "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
        "median_generated_tokens": float(np.median([r["generated_tokens"] for r in rows])),
    }


def analyze(campaigns):
    scored = {str(b): {m: score_rows(campaigns[str(b)][m]) for m in MODELS} for b in BUDGETS}
    budgets = {}
    reference = scored["8192"]["base"]
    for budget, arms in scored.items():
        for rows in arms.values():
            paired_comparison(rows, reference, replicates=1)
        budgets[budget] = {
            "arms": {m: summarize(rows) for m, rows in arms.items()},
            "trace_minus_unchanged": contrast([arms[f"trace_{s}"] for s in SEEDS], [arms["base"]] * 3),
        }
    return {
        "primary_8192": budgets["8192"]["trace_minus_unchanged"],
        "budgets": budgets,
        "length_check_8192_minus_2048": {m: paired_comparison(scored["8192"][m], scored["2048"][m]) for m in MODELS},
        "conditional_completion_caution": "EOS and nonunfinished subsets select model-dependent questions; descriptive only",
    }


def collect(proposal, ledger):
    from src.evaluation import SAMPLING_POLICY, inference_batch_size
    from src.reward import score_response

    proposal = Path(proposal)
    protocol = json.loads((proposal / "protocol.json").read_text())
    if ledger["status"] != "complete" or ledger["identity"]["plan"] != build_plan(protocol):
        raise ValueError("A complete matching campaign is required")
    campaigns = {str(b): {} for b in BUDGETS}
    diagnostics = {}
    hashes = {}
    for key, job in protocol["jobs"].items():
        summary = ledger["results"][key]["output"]["benchmarks"]["gsm8k"]
        proof = ledger["results"][key + ":verify"]["output"]
        folder = proposal / "collected" / key.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("predictions.jsonl", "summary.json", "evaluation.json", "performance.json"):
            dest = folder / name
            if not dest.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modal",
                        "volume",
                        "get",
                        "reasoning-distillation-vol",
                        (summary["results_path"] + "/" + name).removeprefix("/vol/"),
                        str(dest),
                    ],
                    check=True,
                )
        if (
            file_hash(folder / "predictions.jsonl") != proof["predictions_sha256"]
            or json.loads((folder / "summary.json").read_text()) != summary
        ):
            raise ValueError("Persistence hash or summary mismatch")
        rows = [json.loads(x) for x in (folder / "predictions.jsonl").read_text().splitlines()]
        if len(rows) != 128:
            raise ValueError("Incomplete follow-up evaluation")
        for row, gold in zip(rows, protocol["subset"]["rows"], strict=True):
            if (row["problem_id"], row["sample_id"], row["question"], row["ground_truth"]) != (
                gold["problem_id"],
                0,
                gold["question"],
                gold["answer"],
            ):
                raise ValueError("Frozen question identity mismatch")
            if any(row[k] != v for k, v in score_response(row["response"], row["ground_truth"]).to_dict().items()):
                raise ValueError("CPU strict-score mismatch")
        identity = json.loads((folder / "evaluation.json").read_text())["identity"]
        if identity["model"] != protocol["models"][job["model_key"]]["artifact_id"]:
            raise ValueError("Wrong evaluated checkpoint")
        for k, v in job["generation"].items():
            if identity["generation"].get(k) != v:
                raise ValueError("Wrong evaluated generation policy")
        if identity["dataset"] != protocol["subset"]["rows_digest"]:
            raise ValueError("Wrong evaluated question texts")
        bundled = {
            k: v
            for k, v in protocol["implementation"].items()
            if k.startswith("src/") or k in {"modal_train.py", "configs/config.yaml"}
        }
        expected_generation = {**protocol["config"]["evaluation"], **job["generation"]}
        if (
            identity["runtime"]["source_digest"] != fingerprint(bundled)
            or identity["sampling_policy"] != SAMPLING_POLICY
            or identity["effective_batch_size"] != inference_batch_size(expected_generation)
            or identity["generation"]["scorer"] != "boxed-final-v1"
            or identity["dataset_spec"] != protocol["config"]["datasets"]["gsm8k"]
        ):
            raise ValueError("Wrong runtime, sampling policy, batch size, scorer or dataset revision")
        scored = score_rows(rows)
        immutable_json(folder / "numerical-scores.json", scored)
        hashes[key] = {"raw_sha256": file_hash(folder / "predictions.jsonl"), "rows": len(rows)}
        if "nothink" in key:
            diagnostics[key] = summarize(scored)
        else:
            campaigns[str(job["generation"]["max_new_tokens"])][job["model_key"]] = rows
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "complete",
        **analyze(campaigns),
        "no_thinking_diagnostics": diagnostics,
        "raw_files": hashes,
        "limits": protocol["limits"],
    }
    old = {
        m: [json.loads(x) for x in (ORIGINAL / "collected/2048" / m / "predictions.jsonl").read_text().splitlines()]
        for m in ("base", "answer_42", "answer_43", "answer_44")
    }
    ids = {r["problem_id"] for r in protocol["subset"]["rows"]}
    old = {m: score_rows([r for r in rr if r["problem_id"] in ids]) for m, rr in old.items()}
    result["original_greedy_same_subset"] = {m: summarize(rows) for m, rows in old.items()}
    immutable_json(proposal / "collected/analysis.json", result)
    return result
