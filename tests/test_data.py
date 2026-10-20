import pytest


def test_answer_contract_is_added_after_overlap_checks_and_matches_evaluation(tmp_path, monkeypatch):
    from unittest.mock import patch

    from datasets import Dataset

    from src.data import load_config, prepare_condition
    from src.evaluation import generate_batch
    from src.prompting import answer_prompt

    config = load_config()
    config["conditions"]["controlled_grpo"]["grpo"]["max_samples"] = 2
    source = Dataset.from_list(
        [
            {"question": q, "answer": "#### 2"}
            for q in ["leaked same question", "fresh question one", "fresh question two"]
        ]
    )
    monkeypatch.setattr("src.data.load_source", lambda *args, **kwargs: source)
    monkeypatch.setattr("src.data._heldout_questions", lambda *args: ["leaked same question"])

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=True, **kwargs):
            return [1] if tokenize else messages[0]["content"]

    dataset, manifest = prepare_condition("controlled_grpo", config, Tokenizer(), "grpo", tmp_path)
    assert set(dataset["source_id"]) == {"1", "2"}
    assert manifest["spec"]["answer_contract"] == config["reward_scorer"]
    with patch("src.evaluation._generate", return_value=[]) as generate:
        generate_batch(None, Tokenizer(), ["fresh question one"], ["1"], config["evaluation"])
    assert generate.call_args.args[2] == [answer_prompt("fresh question one")]
    assert answer_prompt("fresh question one") in [r["prompt"][0]["content"] for r in dataset]


from datasets import Dataset

from src.data import build_controlled_datasets, matched_token_dataset, overlap_reason


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(message["content"] for message in messages)
        return list(text) if tokenize else text


def source():
    return Dataset.from_list(
        [
            {
                "uuid": str(i),
                "problem": f"Compute item {i}",
                "answer": str(i),
                "generations": [
                    f"<think>compute</think> \\boxed{{{i}}}",
                    f"<think>bad guess</think> \\boxed{{{i + 1}}}",
                    "<think>unfinished",
                ],
                "correctness_math_verify": [True, False, True],
            }
            for i in range(8)
        ]
    )


def test_shared_questions_and_complete_targets():
    pools, quarantine = build_controlled_datasets(source(), ChatTokenizer(), count=5, seed=2026, max_length=200)
    ids = [pool["source_id"] for pool in pools.values()]
    assert all(values == ids[0] for values in ids)
    assert all(row["verified"] for row in pools["controlled_verified"])
    assert all("<think>" not in row["completion"][0]["content"] for row in pools["controlled_answer_only"])
    assert any(row["reason"] == "incomplete_or_overlength" for row in quarantine)
    again, _ = build_controlled_datasets(source(), ChatTokenizer(), count=5, seed=2026, max_length=200)
    assert pools["controlled_unfiltered"].to_list() == again["controlled_unfiltered"].to_list()


def test_overlap_and_alignment():
    assert overlap_reason("Hello    world", ["hello world"]) == "exact_test_overlap"
    assert (
        overlap_reason(
            "one two three four five six seven eight nine ten",
            ["one two three four five six seven eight nine ten eleven"],
        )
        == "near_duplicate_candidate"
    )
    ds = source().to_list()
    ds[0]["correctness_math_verify"] = [True]
    pools, quarantine = build_controlled_datasets(ds, ChatTokenizer(), count=7, max_length=200)
    assert len(pools["controlled_verified"]) == 7
    assert any(row["reason"] == "verification_alignment" for row in quarantine)


def test_matched_tokens_keep_whole_examples():
    pools, _ = build_controlled_datasets(source(), ChatTokenizer(), count=2, max_length=200)
    ds, budget = matched_token_dataset(pools["controlled_verified"], ChatTokenizer(), budget=100)
    assert budget["supervised_tokens"] >= 100
    assert budget["overshoot"] < 100
    assert all(row["completion"][0]["content"].endswith("}") for row in ds)
    with pytest.raises(ValueError):
        build_controlled_datasets(source(), ChatTokenizer(), count=20, max_length=200)
