"""Prospective matched-question reasoning distillation under a reserved compute cap."""

import json
import logging
import math
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.artifacts import atomic_json, file_hash, fingerprint
from src.checkpoint_experiment import immutable_json
from src.data import chat_row
from src.prompting import answer_prompt
from src.reward import _numeric, answers_match, extract_boxed_answer, final_response, score_response
from src.statistics import paired_comparison
from src.study import execute_study, validate_quote

SEEDS = (42, 43, 44)
CONDITIONS = ("distill_answer", "distill_trace")
ENDPOINT = "completed-consistent-numeric-boxes-v1"
CAP = 17.65  # Keep $2.14 for the earlier run and $0.18 for stopped CPU preparation.


def study_config(original):
    config = deepcopy(original)
    settings = deepcopy(config["conditions"]["controlled_verified"]["sft"])
    settings.update(
        num_train_epochs=2, report_to="none", save_strategy="no", logging_steps=1, packing=False, padding_free=True
    )
    config["conditions"] = {
        name: {"type": "sft", "dataset": source, "sft": {**settings, "output_dir": "outputs/" + name}}
        for name, source in zip(CONDITIONS, ("controlled_answer_only", "controlled_verified"), strict=True)
    }
    config["study"] = {
        "seeds": list(SEEDS),
        "num_sft_questions": 500,
        "max_length": 16128,
        "selection_seed": 2026,
        "matched_teacher_answers": True,
    }
    return config


def distillation_rows(source):
    answers, traces = [], []
    for row in source:
        text = row["completion"][0]["content"]
        final, status = final_response(text)
        value = extract_boxed_answer(final or "")
        thinking = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if (
            status != "ok"
            or value is None
            or not thinking
            or not thinking[1].strip()
            or row.get("verified") is not True
        ):
            raise ValueError("Matched supervision requires a complete verified teacher answer")
        common = {k: v for k, v in row.items() if k not in {"prompt", "completion", "source_id"}}
        prompt = answer_prompt(row["prompt"][0]["content"], benchmark="math500")
        final_target = "\\boxed{" + value + "}"
        answers.append(chat_row(prompt, final_target, row["source_id"], **common))
        trace_target = "<think>\n" + thinking[1].strip() + "\n</think>\n\n" + final_target
        traces.append(chat_row(prompt, trace_target, row["source_id"], **common))
    return answers, traces


def numeric_final_score(text, truth):
    """Require completed thinking and mutually consistent complete numeric final boxes."""
    final, status = final_response(text)
    predicted = None
    if status == "ok":
        starts = [m.start() for m in re.finditer(re.escape(r"\boxed{"), final)]
        if not starts:
            status = "missing_final_answer"
        else:
            values = [
                extract_boxed_answer(final[start : starts[i + 1] if i + 1 < len(starts) else None])
                for i, start in enumerate(starts)
            ]
            if any(value is None for value in values):
                status = "malformed_final_answer"
            elif any(_numeric(value) is None for value in values):
                status = "unparseable_final_answer"
            elif len({_numeric(value) for value in values}) != 1:
                status = "conflicting_final_answers"
            else:
                predicted = values[0]
    return {
        "predicted": predicted,
        "correct": status == "ok" and answers_match(predicted, truth),
        "status": status,
        "scorer_version": ENDPOINT,
    }


def implementation_identity():
    paths = [
        *sorted(Path("src").glob("*.py")),
        Path("modal_train.py"),
        Path("configs/config.yaml"),
        Path("requirements/sft.lock"),
        Path("scripts/17_distillation_study.py"),
    ]
    return {str(path): file_hash(path) for path in paths}


def freeze_protocol(config, prepared):
    from src.data import load_config
    from src.evaluation import benchmark_rows

    rows = benchmark_rows("gsm8k", load_config())
    old = json.loads(Path("results/proposals/checkpoint-ablation/6b95572ffeae5d5f98176ffd/protocol.json").read_text())
    excluded = set(old["problem_ids"]) | set(old["excluded_pilot_ids"])
    eligible = [i for i, row in enumerate(rows) if row["problem_id"] not in excluded]
    indices = sorted(random.Random(20261020).sample(eligible, 256))
    long_indices = sorted(random.Random(20261021).sample(indices, 64))
    subsets = {}
    for budget, selected in ((2048, indices), (4096, long_indices)):
        selected_rows = [rows[i] for i in selected]
        subsets[str(budget)] = {
            "problem_indices": selected,
            "rows": selected_rows,
            "rows_digest": fingerprint(selected_rows),
        }
    protocol = {
        "question": "Does reasoning-trace SFT improve fixed-budget mathematical accuracy beyond matched final-answer SFT?",
        "config": config,
        "implementation": implementation_identity(),
        "prepared": prepared,
        "subsets": subsets,
        "excluded_prior_question_ids": sorted(excluded),
        "primary_endpoint": ENDPOINT,
        "secondary_endpoint": "boxed-final-v1",
        "primary_comparison": "Mean trace-minus-answer accuracy across seeds 42,43,44 at 2048 tokens",
        "statistics": "Paired question bootstrap of seed-averaged differences; per-seed deltas and sample SD separately",
        "training_comparison": "Matched questions, final answers, epochs and question minibatches; unequal token exposure and compute",
        "target_normalization": "Both arms have the identical final boxed value. Trace arm additionally contains the teacher's explicit think block; post-think prose and other boxes are removed from both arms.",
        "length_check": "All seven models on a preselected common 64-question subset at 4096; descriptive secondary",
        "budget": {
            "envelope_usd": 20,
            "previous_metered_usd": 1.8600004,
            "previous_run_reserve_usd": 2.14,
            "stopped_cpu_preparation_reserve_usd": 0.18,
            "new_cap_usd": CAP,
        },
        "limits": [
            "Post-trained student; no reasoning-emergence or faithfulness claim",
            "One training-question sample and three seeds; small evaluation subset",
            "Previously used benchmark, with recent development questions excluded",
            "Teacher generation costs excluded: this uses an existing public dataset",
            "Matched-question effect is not an equal-compute or equal-token causal comparison",
            "No GRPO, checkpoint selection on test scores, or outcome-dependent retries",
        ],
    }
    return {**protocol, "protocol_id": fingerprint(protocol)}


def build_plan(protocol):
    nodes = [{"id": "prepare:runtime", "kind": "prepare_runtime", "depends_on": [], "gpu": None}]
    for condition in CONDITIONS:
        nodes.append(
            {
                "id": "prepare:" + condition,
                "kind": "prepare",
                "condition": condition,
                "phase": "sft",
                "depends_on": ["prepare:runtime"],
                "gpu": None,
            }
        )
    for condition in CONDITIONS:
        for seed in SEEDS:
            nodes.append(
                {
                    "id": f"train:{condition}:{seed}",
                    "kind": "sft",
                    "condition": condition,
                    "seed": seed,
                    "depends_on": ["prepare:" + name for name in CONDITIONS],
                    "gpu": "L40S",
                }
            )
    models = [("base", None)] + [
        (f"{name}_{seed}", f"train:{condition}:{seed}")
        for name, condition in zip(("answer", "trace"), CONDITIONS, strict=True)
        for seed in SEEDS
    ]
    trained = [n["id"] for n in nodes if n["kind"] == "sft"]
    for tokens in (2048, 4096):
        subset = protocol["subsets"][str(tokens)]
        for model, parent in models:
            key = f"eval:{tokens}:{model}"
            nodes.append(
                {
                    "id": key,
                    "kind": "evaluate",
                    "condition": "distillation_" + model,
                    "model_key": model,
                    "parent": parent,
                    "depends_on": trained,
                    "gpu": "L40S",
                    "benchmark": "gsm8k",
                    "max_new_tokens": tokens,
                    "num_samples": 1,
                    "temperature": 0.0,
                    "seed": 42,
                    "problem_indices": subset["problem_indices"],
                    "expected_rows_digest": subset["rows_digest"],
                }
            )
            nodes.append({"id": key + ":verify", "kind": "verify", "parent": key, "depends_on": [key], "gpu": None})
    return {
        "experiment": "bounded-reasoning-distillation",
        "protocol_id": protocol["protocol_id"],
        "execution": "CPU preparation then at most four independent model/seed chains; no paid retries",
        "nodes": nodes,
    }


def execute_campaign(plan, config, quote, budget, root="outputs/distillation", **callbacks):
    """Reuse the reservation executor for disjoint chains, with one global budget check."""
    from filelock import FileLock

    maximum = validate_quote(plan, quote, budget)
    root = Path(root)
    path = root / fingerprint(plan)[:24] / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"plan": plan, "quote": quote, "budget_usd": budget}
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists():
            old = json.loads(path.read_text())
            if old["identity"] != identity or old.get("status") != "complete":
                raise ValueError("Reconcile the existing campaign before any new paid calls; no automatic retries")
            return old
        ledger = {"identity": identity, "reserved_total_usd": maximum, "results": {}, "status": "running"}
        atomic_json(path, ledger)

        def run_chain(name, selected):
            ids = {node["id"] for node in selected}
            nodes = [
                {**node, "depends_on": [parent for parent in node["depends_on"] if parent in ids]} for node in selected
            ]
            chain = {
                "experiment": plan["experiment"],
                "protocol_id": plan["protocol_id"],
                "chain": name,
                "nodes": nodes,
            }
            chain_quote = {
                **quote,
                "plan_digest": fingerprint(chain),
                "tasks": {key: quote["tasks"][key] for key in ids},
            }
            location = path.parent / (name + "-quote.json")
            immutable_json(location, chain_quote)
            cap = sum(
                task["rate_per_second"] * (task["timeout_seconds"] + task["overhead_seconds"])
                for task in chain_quote["tasks"].values()
            )
            return execute_study(
                chain, config, str(location), cap + 1e-9, ledger_root=path.parent / "chains", **callbacks
            )

        try:
            cpu = [node for node in plan["nodes"] if node["kind"] in {"prepare_runtime", "prepare"}]
            ledger["results"].update(run_chain("prepare", cpu)["results"])
            atomic_json(path, ledger)
            chains = {}
            for name in ("base", *map(str, SEEDS)):

                def belongs(node, name=name):
                    if node["kind"] == "sft":
                        return str(node["seed"]) == name
                    if node["kind"] in {"evaluate", "verify"}:
                        suffix = ":" + name if name == "base" else "_" + name
                        return node["id"].removesuffix(":verify").endswith(suffix)
                    return False

                chains[name] = [node for node in plan["nodes"] if belongs(node)]
            errors = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(run_chain, name, nodes): name for name, nodes in chains.items()}
                for future in as_completed(futures):
                    try:
                        ledger["results"].update(future.result()["results"])
                        atomic_json(path, ledger)
                    except Exception as error:
                        logging.getLogger(__name__).exception("Study chain %s failed", futures[future])
                        errors.append(f"{futures[future]}: {type(error).__name__}: {error}")
            if errors:
                ledger["chain_errors"] = errors
                raise RuntimeError("One or more paid chains failed; reconcile their retained ledgers before any retry")
            if set(ledger["results"]) != {node["id"] for node in plan["nodes"]}:
                raise ValueError("Campaign chain partition lost a task")
            ledger["status"] = "complete"
        except Exception:
            ledger["status"] = "failed"
            atomic_json(path, ledger)
            raise
        atomic_json(path, ledger)
        return ledger


def build_quote(plan):
    prices = json.loads(Path("configs/modal-pricing-2026-10-19.json").read_text())
    cpu = 2 * prices["cpu_core_per_second"] + 16 * prices["memory_gib_per_second"]
    tasks = {}
    for node in plan["nodes"]:
        if node["kind"] == "sft":
            timeout = 2400 if node["condition"] == "distill_trace" else 300
        elif node["kind"] == "evaluate":
            timeout = 1400 if node["max_new_tokens"] == 2048 else 1150
        else:
            timeout = {"prepare_runtime": 600, "prepare": 900, "verify": 60}[node["kind"]]
        tasks[node["id"]] = {
            "timeout_seconds": timeout,
            "rate_per_second": cpu + (prices["gpu_per_second"]["L40S"] if node["gpu"] else 0),
            "overhead_seconds": 120,
        }
    quote = {
        "plan_digest": fingerprint(plan),
        "tasks": tasks,
        "pricing_source": prices["source"],
        "pricing_rechecked": "2026-10-20",
        "pilot_evidence": "results/recovery/pilot-final-outcome.json and completed checkpoint ablation profiles",
        "non_runtime_allowance_usd": 0.20,
        "expected_cost_caution": "Reservations are ceilings, not forecasts; 4K latency is extrapolated and bounded by timeout",
    }
    quote["reserved_runtime_usd"] = validate_quote(plan, quote, CAP)
    if quote["reserved_runtime_usd"] + quote["non_runtime_allowance_usd"] > CAP:
        raise ValueError("All runtime and non-runtime reservations must fit the remaining envelope")
    return quote


def validate_proposal(path, original, cap):
    path = Path(path)
    protocol, plan, quote = [
        json.loads((path / name).read_text()) for name in ("protocol.json", "plan.json", "quote.json")
    ]
    body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if fingerprint(body) != protocol["protocol_id"] or protocol["implementation"] != implementation_identity():
        raise ValueError("Protocol or implementation changed after freezing")
    if protocol["config"] != study_config(original) or plan != build_plan(protocol) or quote != build_quote(plan):
        raise ValueError("Frozen config, plan or quote changed")
    if not math.isfinite(cap) or cap != CAP:
        raise ValueError(f"Execution requires the exact remaining cap ${CAP:.2f}")
    validate_quote(plan, quote, cap - quote["non_runtime_allowance_usd"])
    return protocol, plan, quote


def analyze_distillation(campaign):
    expected = {"base"} | {f"{arm}_{seed}" for arm in ("answer", "trace") for seed in SEEDS}
    if set(campaign) != expected:
        raise ValueError("All seven complete models are required")
    robust = {
        key: [{**row, **numeric_final_score(row["response"], row["ground_truth"])} for row in rows]
        for key, rows in campaign.items()
    }
    ref = robust["base"]
    for rows in robust.values():
        paired_comparison(rows, ref, replicates=1)
    per_seed = {str(seed): paired_comparison(robust[f"trace_{seed}"], robust[f"answer_{seed}"]) for seed in SEEDS}
    diff = np.array(
        [
            [
                int(a["correct"]) - int(b["correct"])
                for a, b in zip(robust[f"trace_{seed}"], robust[f"answer_{seed}"], strict=True)
            ]
            for seed in SEEDS
        ]
    ).mean(axis=0)
    rng = np.random.default_rng(42)
    draws = np.concatenate([diff[rng.integers(len(diff), size=(500, len(diff)))].mean(axis=1) for _ in range(20)])
    deltas = [v["delta"] for v in per_seed.values()]
    return {
        "num_distinct_questions": len(ref),
        "endpoint": ENDPOINT,
        "arms": {
            key: {
                "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "strict_accuracy": sum(r["correct"] for r in campaign[key]) / len(rows),
                "unfinished_thinking_rate": sum(r["status"] == "unfinished_thinking" for r in rows) / len(rows),
                "output_limit_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
                "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
            }
            for key, rows in robust.items()
        },
        "primary": {
            "mean_delta": float(np.mean(deltas)),
            "per_seed": per_seed,
            "paired_item_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "training_seed_delta_std": float(np.std(deltas, ddof=1)),
            "uncertainty": "Item interval conditional on three trained seeds; seed SD separate; not 768 independent items",
        },
        "strict_secondary": {
            str(seed): paired_comparison(campaign[f"trace_{seed}"], campaign[f"answer_{seed}"]) for seed in SEEDS
        },
    }


def collect_study(proposal, ledger):
    proposal = Path(proposal)
    protocol = json.loads((proposal / "protocol.json").read_text())
    root = proposal / "collected"

    def download(remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
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

    campaigns = {"2048": {}, "4096": {}}
    identities = {}
    for node in ledger["identity"]["plan"]["nodes"]:
        result = ledger["results"].get(node["id"], {})
        if result.get("status") != "complete":
            raise ValueError("Cannot report a complete study with incomplete jobs")
        if node["kind"] == "sft":
            for name in ("artifact.json", "training_metadata.json", "performance.json"):
                download(result["output"] + "/" + name, root / "training" / node["id"].replace(":", "_") / name)
        if node["kind"] != "evaluate":
            continue
        tokens, model = str(node["max_new_tokens"]), node["model_key"]
        summary = result["output"]["benchmarks"]["gsm8k"]
        path = root / tokens / model
        for name in ("predictions.jsonl", "evaluation.json", "summary.json", "performance.json"):
            download(summary["results_path"] + "/" + name, path / name)
        proof = ledger["results"][node["id"] + ":verify"]["output"]
        if (
            proof["predictions_sha256"] != file_hash(path / "predictions.jsonl")
            or json.loads((path / "summary.json").read_text()) != summary
        ):
            raise ValueError("Downloaded evaluation differs from independently persisted result")
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        expected = protocol["subsets"][tokens]["rows"]
        if len(rows) != len(expected):
            raise ValueError("Missing evaluated questions")
        for row, gold in zip(rows, expected, strict=True):
            if row["problem_id"] != gold["problem_id"] or row["ground_truth"] != gold["answer"]:
                raise ValueError("Evaluation questions differ from the protocol")
            if any(
                row[k] != value for k, value in score_response(row["response"], row["ground_truth"]).to_dict().items()
            ):
                raise ValueError("Saved strict scores fail CPU reproduction")
        identity = json.loads((path / "evaluation.json").read_text())["identity"]
        common = {k: v for k, v in identity.items() if k != "model"}
        if tokens in identities and identities[tokens] != common:
            raise ValueError("Models have differing evaluation settings")
        identities[tokens] = common
        campaigns[tokens][model] = rows
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "complete",
        "primary_2048": analyze_distillation(campaigns["2048"]),
        "secondary_4096": analyze_distillation(campaigns["4096"]),
        "limits": protocol["limits"],
    }
    immutable_json(root / "analysis.json", result)
    atomic_json(Path("results/recovery/distillation-study-outcome.json"), result)
    return result
