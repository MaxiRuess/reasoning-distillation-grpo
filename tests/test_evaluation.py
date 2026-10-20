import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts import EvaluationStore
from src.evaluation import compute_pass_at_k, evaluate_dataset, summarize_records, validate_generation


def test_pass_at_k_and_accuracy_are_distinct():
    records = [
        {
            "problem_id": "q",
            "sample_id": i,
            "correct": i == 1,
            "predicted": "1",
            "generated_tokens": 1,
            "finish_reason": "eos",
        }
        for i in range(4)
    ]
    result = summarize_records(records, ["q"], 4)
    assert result["accuracy"] == 0
    assert result["pass@1"] == 0.25
    assert result["pass@4"] == 1
    with pytest.raises(ValueError):
        compute_pass_at_k([[True]], 4)
    with pytest.raises(ValueError):
        validate_generation({"num_samples_per_problem": 8, "temperature": 0, "batch_size": 1, "max_new_tokens": 20})


def test_interrupted_and_uninterrupted_records_match():
    rows = [{"problem_id": str(i), "question": f"Question {i}", "answer": str(i)} for i in range(4)]
    cfg = {"num_samples_per_problem": 1, "temperature": 0, "batch_size": 2, "max_new_tokens": 20}

    def generate(model, tokenizer, questions, ids, cfg):
        return [[{"response": "\\boxed{" + i + "}", "generated_tokens": 3, "finish_reason": "eos"}] for i in ids]

    with tempfile.TemporaryDirectory() as tmp, patch("src.evaluation.generate_batch", side_effect=generate):
        path = Path(tmp)
        store = EvaluationStore(path, {"model": "fixed"}, [r["problem_id"] for r in rows], 1)
        with pytest.raises(RuntimeError, match="interrupted"):
            evaluate_dataset(
                None,
                None,
                rows,
                "gsm8k",
                cfg,
                store,
                on_batch=lambda: (_ for _ in ()).throw(RuntimeError("interrupted")),
            )
        resumed = EvaluationStore(path, {"model": "fixed"}, [r["problem_id"] for r in rows], 1)
        result = evaluate_dataset(None, None, rows, "gsm8k", cfg, resumed)
        expected = evaluate_dataset(None, None, rows, "gsm8k", cfg)
        assert result == expected
        assert result["accuracy"] == 1
        assert all("response" in record for record in result["records"])


def test_single_sample_sampling_replays_canonical_batch_after_rng_changes():
    import torch

    from src.evaluation import generate_batch

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    cfg = {"temperature": 0.6, "num_samples_per_problem": 1, "batch_size": 2, "max_new_tokens": 20, "seed": 23}

    def sample(model, tokenizer, prompts, options, count):
        return torch.randint(0, 2**30, (len(prompts), count)).tolist()

    with patch("src.evaluation._generate", side_effect=sample):
        first = generate_batch(None, Tokenizer(), ["a", "b"], ["1", "2"], cfg)
        torch.rand(100)
        replay = generate_batch(None, Tokenizer(), ["a", "b"], ["1", "2"], cfg)
        other = generate_batch(None, Tokenizer(), ["a", "b"], ["1", "2"], {**cfg, "seed": 24})
    assert first == replay
    assert first != other
