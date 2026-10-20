from copy import deepcopy

import pytest

from src.artifacts import fingerprint
from src.data import load_config
from src.inference_extension import CAP, analyze, build_plan, build_quote, execute, jobs_for
from src.study import validate_quote


def protocol():
    rows = [{"problem_id": str(i), "question": f"q{i}", "answer": "1"} for i in range(128)]
    models = {"base": {"path": "Qwen/Qwen3-1.7B"}}
    models.update({f"trace_{s}": {"path": f"/vol/trace{s}"} for s in (42, 43, 44)})
    return {
        "protocol_id": "test",
        "config": load_config(),
        "models": models,
        "subset": {"rows": rows, "problem_indices": list(range(128)), "rows_digest": fingerprint(rows)},
        "jobs": jobs_for(models),
    }


def test_all_calls_reserved_with_exact_shared_questions_and_correct_models():
    p = protocol()
    plan = build_plan(p)
    quote = build_quote(plan)
    assert len(p["jobs"]) == 14
    assert sum(n["kind"] == "evaluate" for n in plan["nodes"]) == 14
    assert all(n["kind"] in {"prepare_runtime", "evaluate", "verify"} for n in plan["nodes"])
    assert validate_quote(plan, quote, CAP - quote["non_runtime_allowance_usd"]) < CAP - 1
    for n in plan["nodes"]:
        if n["kind"] == "evaluate":
            assert n["problem_indices"] == list(range(128))
            assert n["model_path"] == p["models"][n["model_key"]]["path"]
    with pytest.raises(ValueError):
        validate_quote(plan, quote, 10)
    assert p["jobs"]["eval:8192:trace_42"]["generation"]["temperature"] == 0.6
    assert p["jobs"]["eval:2048:nothink_sampled:base"]["generation"]["enable_thinking"] is False
    assert p["jobs"]["eval:2048:nothink_greedy:base"]["generation"]["temperature"] == 0


def test_actual_executor_keeps_model_paths_and_does_not_repeat_paid_calls(tmp_path):
    p = protocol()
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"benchmarks": {"gsm8k": {"results_path": kwargs["condition"]}}}

    def verify(result_path, **kwargs):
        return {"summary": {"results_path": result_path}}

    kwargs = {"preflight": lambda **kw: {}, "evaluate": evaluate, "verify": verify, "root": tmp_path}
    first = execute(p, **kwargs)
    assert len(first["results"]) == 29
    assert len(calls) == 14
    for c in calls:
        assert c["model_path"] == p["models"][p["jobs"][c["condition"]]["model_key"]]["path"]
    assert execute(p, **kwargs) == first
    assert len(calls) == 14


def test_analysis_distinguishes_eos_unfinished_and_missing_final_and_preserves_pairing():
    def rows(correct):
        return [
            {
                "problem_id": str(i),
                "sample_id": 0,
                "ground_truth": "1",
                "response": response,
                "correct": False,
                "finish_reason": finish,
                "generated_tokens": 10,
            }
            for i, (response, finish) in enumerate(
                [
                    (r"\boxed{1}" if correct else r"\boxed{2}", "eos"),
                    (r"<think>unfinished \boxed{1}", "length"),
                    ("<think>done</think> no final box", "length"),
                ]
            )
        ]

    campaigns = {
        str(b): {m: rows(m != "base") for m in ["base", "trace_42", "trace_43", "trace_44"]} for b in (2048, 4096, 8192)
    }
    result = analyze(campaigns)
    assert result["primary_8192"]["num_distinct_questions"] == 3
    assert result["primary_8192"]["mean_delta"] == pytest.approx(1 / 3)
    stats = result["budgets"]["8192"]["arms"]["trace_42"]
    assert stats["eos_count"] == 1
    assert stats["without_unfinished_thinking_count"] == 2
    assert stats["accuracy_among_eos"] == 1
    assert stats["accuracy_among_without_unfinished_thinking"] == 0.5
    bad = deepcopy(campaigns)
    bad["8192"]["trace_42"][0]["ground_truth"] = "9"
    with pytest.raises(ValueError):
        analyze(bad)


def test_collection_checks_real_saved_scores_and_rejects_changed_raw_hash(tmp_path, monkeypatch):
    import json

    import src.inference_extension as extension
    from src.artifacts import file_hash
    from src.evaluation import SAMPLING_POLICY, inference_batch_size
    from src.reward import score_response

    p = protocol()
    p["implementation"] = {"src/evaluation.py": "frozen", "modal_train.py": "frozen"}
    p["limits"] = []
    for model in p["models"].values():
        model["artifact_id"] = model["path"]
    proposal = tmp_path / "proposal"
    proposal.mkdir()
    (proposal / "protocol.json").write_text(json.dumps(p))
    results = {}
    for key, job in p["jobs"].items():
        folder = proposal / "collected" / key.replace(":", "_")
        folder.mkdir(parents=True)
        rows = []
        for gold in p["subset"]["rows"]:
            response = r"\boxed{1}"
            rows.append(
                {
                    "problem_id": gold["problem_id"],
                    "sample_id": 0,
                    "question": gold["question"],
                    "ground_truth": gold["answer"],
                    "response": response,
                    "finish_reason": "eos",
                    "generated_tokens": 5,
                    **score_response(response, gold["answer"]).to_dict(),
                }
            )
        (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = {"results_path": str(folder)}
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "performance.json").write_text("{}")
        generation = {**p["config"]["evaluation"], **job["generation"]}
        identity = {
            "model": p["models"][job["model_key"]]["artifact_id"],
            "generation": generation,
            "dataset": p["subset"]["rows_digest"],
            "runtime": {"source_digest": fingerprint(p["implementation"])},
            "sampling_policy": SAMPLING_POLICY,
            "effective_batch_size": inference_batch_size(generation),
            "dataset_spec": p["config"]["datasets"]["gsm8k"],
        }
        (folder / "evaluation.json").write_text(json.dumps({"identity": identity}))
        results[key] = {"output": {"benchmarks": {"gsm8k": summary}}}
        results[key + ":verify"] = {"output": {"predictions_sha256": file_hash(folder / "predictions.jsonl")}}
    original = tmp_path / "original"
    for model in ("base", "answer_42", "answer_43", "answer_44"):
        folder = original / "collected/2048" / model
        folder.mkdir(parents=True)
        (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(extension, "ORIGINAL", original)
    ledger = {"status": "complete", "identity": {"plan": build_plan(p)}, "results": results}
    result = extension.collect(proposal, ledger)
    assert sum(x["rows"] for x in result["raw_files"].values()) == 1792
    assert result["primary_8192"]["mean_delta"] == 0
    broken = proposal / "collected/eval_2048_base/predictions.jsonl"
    broken.write_text(broken.read_text().replace('"correct": true', '"correct": false', 1))
    with pytest.raises(ValueError, match="Persistence hash"):
        extension.collect(proposal, ledger)
