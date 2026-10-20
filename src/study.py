"""Predeclared study DAG and conservative, sequential spending reservations."""

import json
import math
import time
from pathlib import Path

from filelock import FileLock

from src.artifacts import atomic_json, file_hash, fingerprint


def build_study_plan(config, experiment="controlled"):
    if experiment not in {"legacy", "controlled"}:
        raise ValueError("Choose legacy or controlled")
    nodes = [
        {
            "id": "prepare:runtime",
            "kind": "prepare_runtime",
            "condition": "runtime_cache",
            "gpu": None,
            "depends_on": [],
        }
    ]
    if experiment == "legacy":
        seeds = [config["seed"]]
        arms = ["baseline", "sft_traces", "grpo_only", "sft_then_grpo", "re_distill"]
    else:
        seeds = config["study"]["seeds"]
        arms = [
            "controlled_answer_only",
            "controlled_unfiltered",
            "controlled_verified",
            "controlled_grpo",
            "controlled_verified_grpo",
            "controlled_unfiltered_tokens",
            "controlled_verified_tokens",
        ]
    prepared = set()

    def preparation(arm, phase):
        key = f"prepare:{arm}:{phase}"
        if key not in prepared:
            nodes.append(
                {
                    "id": key,
                    "kind": "prepare",
                    "condition": arm,
                    "phase": phase,
                    "gpu": None,
                    "depends_on": ["prepare:runtime"],
                }
            )
            prepared.add(key)
        return key

    def evaluation(arm, seed, parent, benchmark="gsm8k", tokens=2048, n=1, temp=0):
        key = f"eval:{arm}:{seed}:{benchmark}:{tokens}:{n}"
        nodes.append(
            {
                "id": key,
                "kind": "evaluate",
                "condition": arm,
                "seed": seed,
                "benchmark": benchmark,
                "max_new_tokens": tokens,
                "num_samples": n,
                "temperature": temp,
                "gpu": "L40S",
                "parent": parent,
                "depends_on": ["prepare:runtime"] + ([parent] if parent else []),
            }
        )
        nodes.append(
            {
                "id": key + ":verify",
                "kind": "verify",
                "condition": arm,
                "seed": seed,
                "parent": key,
                "gpu": None,
                "depends_on": [key],
            }
        )

    evaluation("base_model", 42, None)
    for seed in seeds:
        parents = {}
        for arm in arms:
            settings = config["conditions"][arm]
            if settings["type"] in {"sft", "sft_then_grpo"}:
                prep = preparation(arm, "sft")
                key = f"train:{arm}:{seed}:sft"
                nodes.append(
                    {
                        "id": key,
                        "kind": "sft",
                        "condition": arm,
                        "seed": seed,
                        "gpu": "L40S",
                        "parent": None,
                        "depends_on": [prep],
                    }
                )
                parents[arm] = key
            if settings["type"] in {"grpo", "sft_then_grpo"}:
                prep = preparation(arm, "grpo")
                parent = parents.get(arm) or parents.get(settings.get("sft_parent"))
                if settings.get("requires_sft") and not parent:
                    raise ValueError("A required SFT parent is not scheduled before GRPO")
                key = f"train:{arm}:{seed}:grpo"
                nodes.append(
                    {
                        "id": key,
                        "kind": "grpo",
                        "condition": arm,
                        "seed": seed,
                        "gpu": "H100",
                        "parent": parent,
                        "depends_on": [prep] + ([parent] if parent else []),
                    }
                )
                parents[arm] = key
            evaluation(arm, seed, parents[arm])
        if experiment == "controlled" and seed == config["study"]["secondary_seed"]:
            for arm in ["base_model"] + arms[:5]:
                parent = parents.get(arm)
                for tokens in [4096, 8192]:
                    evaluation(arm, seed, parent, tokens=tokens)
                evaluation(arm, seed, parent, benchmark="math500", tokens=8192)
                evaluation(arm, seed, parent, n=8, temp=0.6)
    return {
        "schema_version": 1,
        "experiment": experiment,
        "config_digest": fingerprint(config),
        "implementation_digest": fingerprint(
            {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "nodes": nodes,
        "primary_comparisons": config["study"]["primary_comparisons"],
        "selection_seed": config["data_seed"],
        "new_runs_only": True,
    }


def validate_quote(plan, quote, budget):
    if not math.isfinite(budget) or budget <= 0 or quote.get("plan_digest") != fingerprint(plan):
        raise ValueError("A positive agreed cap and quote matching this exact plan are required")
    if not quote.get("pilot_evidence") or not quote.get("pricing_source"):
        raise ValueError("A quote requires measured pilot evidence and a dated pricing source")
    if set(quote["tasks"]) != {node["id"] for node in plan["nodes"]}:
        raise ValueError("Quote does not cover every task")
    for item in quote["tasks"].values():
        if (
            not math.isfinite(item["rate_per_second"])
            or item["rate_per_second"] <= 0
            or (not isinstance(item["timeout_seconds"], int) or not 1 <= item["timeout_seconds"] <= 43200)
            or not isinstance(item.get("overhead_seconds", 120), int)
            or item.get("overhead_seconds", 120) < 0
        ):
            raise ValueError("Invalid all-in rate or hard runtime limit")
    maximum = sum(
        item["rate_per_second"] * (item["timeout_seconds"] + item.get("overhead_seconds", 120))
        for item in quote["tasks"].values()
    )
    if maximum > budget:
        raise ValueError(f"Hard runtime reservations ${maximum:.2f} exceed cap ${budget:.2f}")
    return maximum


def validate_pilot_measurements(config, metadata, evaluation):
    from src.reward import SCORER_VERSION

    expected = config["reward_scorer"]
    if expected != SCORER_VERSION or config["evaluation"]["scorer"] != expected:
        raise ValueError("Main-study rewards and evaluation must share the current answer contract")
    if metadata["grpo"].get("reward_objective", {}).get("scorer_version") != expected:
        raise ValueError("Pilot GRPO reward policy is missing or stale")
    if evaluation.get("scorer_version") != config["evaluation"]["scorer"]:
        raise ValueError("Pilot evaluation scorer differs from the study")
    exposure = metadata["grpo"]["exposure"]
    if not exposure.get("processed_train_tokens") or exposure["processed_train_tokens"] < exposure.get(
        "completion_tokens", 0
    ):
        raise ValueError("Pilot GRPO token exposure is missing or inconsistent")


def _execute_study(
    plan,
    config,
    quote_path,
    budget,
    prepare,
    sft,
    grpo,
    evaluate,
    verify,
    ledger_root="outputs/studies",
    prepare_runtime=None,
    reconstruct=None,
):
    with open(quote_path) as stream:
        quote = json.load(stream)
    maximum = validate_quote(plan, quote, budget)
    path = Path(ledger_root) / fingerprint(plan)[:24] / "ledger.json"
    identity = {"plan": plan, "quote": quote, "budget_usd": budget}
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger["identity"] != identity:
            raise ValueError("Ledger identity changed")
        if any(result["status"] != "complete" for result in ledger["results"].values()):
            raise ValueError(
                "Reconcile an interrupted/failed paid task before resuming; no automatic duplicate launches"
            )
    else:
        ledger = {"identity": identity, "reserved_total_usd": maximum, "results": {}}
    for node in plan["nodes"]:
        key = node["id"]
        if key in ledger["results"]:
            continue
        if any(parent not in ledger["results"] for parent in node["depends_on"]):
            raise ValueError("Study dependencies are incomplete")
        task = quote["tasks"][key]
        reserved = task["rate_per_second"] * (task["timeout_seconds"] + task.get("overhead_seconds", 120))
        consumed = sum(result["reserved_usd"] for result in ledger["results"].values())
        if consumed + reserved > budget:
            raise ValueError("Next task exceeds remaining reserved budget")
        ledger["results"][key] = {"status": "running", "reserved_usd": reserved}
        atomic_json(path, ledger)
        start = time.monotonic()
        try:
            parent = ledger["results"][node["parent"]]["output"] if node.get("parent") else None
            common = {"condition": node.get("condition"), "_timeout": task["timeout_seconds"]}
            if node["kind"] == "prepare_runtime":
                if prepare_runtime is None:
                    raise ValueError("Runtime cache preparation callback is required")
                output = prepare_runtime(_timeout=task["timeout_seconds"])
            elif node["kind"] == "prepare":
                output = prepare(**common, phase=node["phase"])
            elif node["kind"] == "sft":
                output = sft(**common, seed=node["seed"], pilot=node.get("pilot", False))
            elif node["kind"] == "grpo":
                output = grpo(**common, seed=node["seed"], sft_checkpoint=parent, pilot=node.get("pilot", False))
            elif node["kind"] == "verify":
                result_path = next(iter(parent["benchmarks"].values()))["results_path"]
                output = verify(result_path=result_path, _timeout=task["timeout_seconds"])
                if output["summary"] != next(iter(parent["benchmarks"].values())):
                    raise ValueError("Independent persisted summary differs from the producer")
            elif node["kind"] == "reconstruct":
                if reconstruct is None:
                    raise ValueError("Checkpoint reconstruction callback is required")
                output = reconstruct(**common, parent_path=parent)
            else:
                subset = {name: node[name] for name in ["problem_indices", "expected_rows_digest"] if name in node}
                output = evaluate(
                    **common,
                    **subset,
                    model_path=node.get("model_path") or parent or config["model"]["name"],
                    benchmark=node["benchmark"],
                    max_new_tokens=node["max_new_tokens"],
                    num_samples=node["num_samples"],
                    temperature=node["temperature"],
                    seed=node["seed"],
                    pilot=node.get("pilot", False),
                )
            ledger["results"][key].update(status="complete", output=output, elapsed_seconds=time.monotonic() - start)
        except Exception as error:
            ledger["results"][key].update(status="failed", error=type(error).__name__)
            atomic_json(path, ledger)
            raise
        atomic_json(path, ledger)
    return ledger


def build_pilot_plan(config):
    nodes = [
        {
            "id": "prepare:runtime",
            "kind": "prepare_runtime",
            "condition": "runtime_cache",
            "gpu": None,
            "depends_on": [],
        },
        {
            "id": "pilot:prepare:sft",
            "kind": "prepare",
            "condition": "controlled_verified",
            "phase": "sft",
            "depends_on": ["prepare:runtime"],
            "gpu": None,
        },
        {
            "id": "pilot:prepare:grpo",
            "kind": "prepare",
            "condition": "controlled_verified_grpo",
            "phase": "grpo",
            "depends_on": ["prepare:runtime"],
            "gpu": None,
        },
        {
            "id": "pilot:sft",
            "kind": "sft",
            "condition": "controlled_verified",
            "seed": 42,
            "pilot": True,
            "parent": None,
            "depends_on": ["pilot:prepare:sft"],
            "gpu": "L40S",
        },
        {
            "id": "pilot:grpo",
            "kind": "grpo",
            "condition": "controlled_verified_grpo",
            "seed": 42,
            "pilot": True,
            "parent": "pilot:sft",
            "depends_on": ["pilot:sft", "pilot:prepare:grpo"],
            "gpu": "H100",
        },
        {
            "id": "pilot:eval",
            "kind": "evaluate",
            "condition": "pilot_verified_grpo",
            "seed": 42,
            "pilot": True,
            "parent": "pilot:grpo",
            "depends_on": ["pilot:grpo"],
            "gpu": "L40S",
            "benchmark": "gsm8k",
            "max_new_tokens": 2048,
            "num_samples": 1,
            "temperature": 0,
        },
        {
            "id": "pilot:verify",
            "kind": "verify",
            "condition": "pilot_verified_grpo",
            "parent": "pilot:eval",
            "depends_on": ["pilot:eval"],
            "gpu": None,
        },
    ]
    return {
        "schema_version": 1,
        "experiment": "pilot",
        "config_digest": fingerprint(config),
        "implementation_digest": fingerprint(
            {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "nodes": nodes,
    }


def execute_study(
    plan,
    config,
    quote_path,
    budget,
    prepare,
    sft,
    grpo,
    evaluate,
    verify,
    ledger_root="outputs/studies",
    prepare_runtime=None,
    reconstruct=None,
):
    path = Path(ledger_root) / fingerprint(plan)[:24]
    path.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path / "ledger.lock"), timeout=0):
        return _execute_study(
            plan,
            config,
            quote_path,
            budget,
            prepare,
            sft,
            grpo,
            evaluate,
            verify,
            ledger_root,
            prepare_runtime,
            reconstruct,
        )
