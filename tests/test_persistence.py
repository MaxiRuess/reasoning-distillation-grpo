import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.evaluation import run_all_evaluations


def test_saved_raw_responses_and_summary_can_resume_without_generation(tmp_path):
    config = {
        "model": {"name": "model", "revision": "a" * 40},
        "datasets": {"gsm8k": {"revision": "b" * 40}},
        "evaluation": {
            "benchmarks": ["gsm8k"],
            "max_new_tokens": 20,
            "temperature": 0,
            "batch_size": 1,
            "num_samples_per_problem": 1,
            "output_root": str(tmp_path),
        },
    }
    rows = [{"problem_id": "q", "question": "2+2?", "answer": "4"}]

    def generate(*args):
        return [[{"response": r"\boxed{4}", "generated_tokens": 5, "finish_reason": "eos"}]]

    with (
        patch("src.evaluation.load_model_for_eval", return_value=(None, SimpleNamespace(chat_template="template"))),
        patch("src.evaluation.benchmark_rows", return_value=rows),
        patch("src.evaluation.generate_batch", side_effect=generate) as generator,
    ):
        result = run_all_evaluations("model", config, "condition")
        path = result["benchmarks"]["gsm8k"]["results_path"]
        assert json.loads(Path(path + "/predictions.jsonl").read_text())["response"] == r"\boxed{4}"
        assert json.loads(Path(path + "/summary.json").read_text())["accuracy"] == 1
        profile = json.loads(Path(path + "/performance.json").read_text())
        assert profile["generated_tokens_this_call"] == 5
        assert profile["generation_seconds"] >= 0
        run_all_evaluations("model", config, "condition")
        assert json.loads(Path(path + "/performance.json").read_text())["generated_tokens_this_call"] == 0
        assert generator.call_count == 1
        config["evaluation"].update(resume_dir=path, max_new_tokens=21)
        with pytest.raises(ValueError, match="identity mismatch"):
            run_all_evaluations("model", config, "condition")


def test_cpu_rescore_preserves_original_predictions_and_rejects_duplicate_ids(tmp_path):
    from src.artifacts import atomic_json, file_hash
    from src.rescoring import rescore_saved_evaluation
    from src.reward import score_response

    source = tmp_path / "source"
    source.mkdir()
    row = {
        "problem_id": "0",
        "sample_id": 0,
        "question": "test",
        "response": "Ari spent $42 on supplies.",
        "ground_truth": "42",
        "generated_tokens": 8,
        "finish_reason": "eos",
        **score_response("Ari spent $42 on supplies.", "42", scorer="final-response-v2").to_dict(),
    }
    (source / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    atomic_json(source / "evaluation.json", {"problem_ids": ["0"], "num_samples": 1})
    atomic_json(source / "summary.json", {"run_id": "old", "benchmark": "gsm8k", "model_identity": "same_model"})
    before = file_hash(source / "predictions.jsonl")
    result = rescore_saved_evaluation(source, tmp_path / "rescored", scorer="final-response-v6")
    assert result["accuracy"] == 1 and result["model_identity"] == "same_model"
    assert result["generation_repeated"] is False and result["training_repeated"] is False
    assert file_hash(source / "predictions.jsonl") == before
    with pytest.raises(ValueError, match="answer contract"):
        rescore_saved_evaluation(source, tmp_path / "wrong_contract")
    (source / "predictions.jsonl").write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError):
        rescore_saved_evaluation(source, tmp_path / "invalid")
