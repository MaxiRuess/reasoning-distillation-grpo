"""Read-only audit probes; run from any directory with Python 3.10+.

These assert the current defects, not desired behavior. AST-isolated probes
exercise repository functions with test doubles; they are not GPU validation.
Use --local-data with the repository venv to inspect cached Arrow datasets.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reward import answers_match, binary_reward_fn, extract_answer_auto, format_reward_fn


def isolated(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-data", action="store_true")
    args = parser.parse_args()
    evidence = {"head_sha": "2c7bd86", "probe_scope": "CPU functions, AST test doubles, optional cached data"}

    cases = [
        ("comma", "The answer is 1,234.", "1234"),
        ("fraction", "The answer is 1/2.", "0.5"),
        ("rejected_thinking_answer", "<think>Candidate: \\boxed{10}. That is wrong.</think> Final answer: 12", "12"),
        ("unfinished_thinking", "<think>Try 40 then 42", "42"),
    ]
    evidence["answer_extraction"] = [
        {"case": name, "text": text, "truth": truth, "extracted": extract_answer_auto(text),
         "reward": binary_reward_fn(None, [text], [truth])[0]}
        for name, text, truth in cases
    ]
    assert [entry["reward"] for entry in evidence["answer_extraction"]] == [0.0, 0.0, 0.0, 1.0]
    evidence["empty_thinking_format_reward"] = format_reward_fn(None, ["<think></think> \\boxed{0}"])[0]
    assert evidence["empty_thinking_format_reward"] == 1.0
    evidence["fraction_suffix_ignored"] = answers_match(r"\frac{1}{2}+7", "0.5")
    assert evidence["fraction_suffix_ignored"]
    try:
        answers_match("inf", "1")
    except OverflowError:
        evidence["infinity_crash"] = "OverflowError"
    else:
        raise AssertionError("Expected existing infinity parsing defect")

    capture = lambda **kwargs: SimpleNamespace(**kwargs)
    ns = isolated("src/training.py", {"build_sft_trainer", "build_grpo_trainer"}, {
        "SFTConfig": capture, "GRPOConfig": capture, "SFTTrainer": capture, "GRPOTrainer": capture,
    })
    common = {"output_dir": "unused", "learning_rate": 1e-6, "num_train_epochs": 3,
              "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
              "optim": "unused", "max_seq_length": 16384, "num_generations": 8,
              "max_completion_length": 4096}
    tokenizer = object()
    trainers = [ns["build_sft_trainer"](object(), tokenizer, [], {"sft": common}, object()),
                ns["build_grpo_trainer"](object(), tokenizer, [], {"grpo": common}, [], object())]
    evidence["configured_tokenizer_forwarded_to_trainers"] = [
        getattr(trainer, "processing_class", None) is tokenizer for trainer in trainers
    ]
    assert evidence["configured_tokenizer_forwarded_to_trainers"] == [False, False]

    ns = isolated("src/evaluation.py", {"evaluate_gsm8k", "compute_pass_at_k", "run_all_evaluations"}, {
        "format_gsm8k_for_eval": lambda: {"question": ["a", "b"], "answer": ["#### 1", "#### 2"]},
        "generate_solutions": lambda *args, **kwargs: [["1", "0", "0", "0"], ["2", "0", "0", "0"]],
        "extract_gsm8k_answer": lambda text: text.split("####")[-1].strip(),
        "extract_answer_auto": extract_answer_auto, "answers_match": answers_match,
        "comb": math.comb, "Path": Path, "json": json,
    })
    result = ns["evaluate_gsm8k"](None, None, {"evaluation": {
        "max_new_tokens": 10, "temperature": 0.7, "batch_size": 2, "num_samples_per_problem": 4,
    }})
    evidence["multi_sample_accuracy"] = {"accuracy": result["accuracy"], "pass@1": result["pass@1"], "pass@4": result["pass@4"]}
    assert evidence["multi_sample_accuracy"] == {"accuracy": 1.0, "pass@1": 0.25, "pass@4": 1.0}
    evidence["pass_at_4_from_one_sample"] = ns["compute_pass_at_k"]([[True]], 4)
    assert evidence["pass_at_4_from_one_sample"] == 1.0
    ns["load_model_for_eval"] = lambda *args: (None, None)
    ns["evaluate_gsm8k"] = lambda *args: result
    cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                returned = ns["run_all_evaluations"]("unused", {"evaluation": {"benchmarks": ["gsm8k"]}}, "probe")
            saved = json.loads(Path("outputs/probe/eval_results.json").read_text())
            evidence["per_example_results_returned_but_not_saved"] = (
                "results" in returned["benchmarks"]["gsm8k"] and "results" not in saved["benchmarks"]["gsm8k"]
            )
            assert evidence["per_example_results_returned_but_not_saved"]
    finally:
        os.chdir(cwd)

    results = json.loads((ROOT / "results/gsm8k_results.json").read_text())["conditions"]
    z = 1.95996398454
    evidence["reported_counts_and_wilson95"] = {}
    for name, row in results.items():
        n, k = row["num_evaluated"], row["num_correct"]
        p = k / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        assert abs(p - row["accuracy"]) < 0.0005
        evidence["reported_counts_and_wilson95"][name] = {
            "correct": k, "n": n, "accuracy_pct": 100 * p,
            "wilson95_pct": [100 * (center - half), 100 * (center + half)],
        }

    if args.local_data:
        from datasets import Dataset, load_from_disk
        evidence["cached_datasets"] = {}
        cache = Path.home() / ".cache/huggingface/datasets/openai___gsm8k"
        test_files = list(cache.glob("**/*test*.arrow"))
        test = Dataset.from_file(str(test_files[0])) if test_files else None
        normalize = lambda text: re.sub(r"\W+", "", text.lower())
        test_questions = {normalize(row["question"]) for row in test} if test is not None else None
        tokenizer_paths = list((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots").glob("*/tokenizer.json"))
        tokenizer = None
        if tokenizer_paths:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tokenizer_paths[0]))
        for name in ["baseline", "sft_traces", "re_distill", "grpo"]:
            path = ROOT / "data/processed" / name
            if not path.exists():
                continue
            ds = load_from_disk(str(path))
            entry = {"rows": len(ds), "fingerprint": ds._fingerprint}
            if test_questions is not None and name != "grpo":
                entry["normalized_exact_test_overlaps"] = sum(normalize(row["prompt"][0]["content"]) in test_questions for row in ds)
            if "completion" in ds.column_names:
                texts = [row["completion"][0]["content"] for row in ds]
                entry.update({"mean_completion_chars": sum(map(len, texts)) / len(texts),
                              "missing_think_closures": sum("<think>" in text and "</think>" not in text for text in texts)})
            if tokenizer is not None and name in ["sft_traces", "re_distill"]:
                # Raw chat serialization, before TRL packing; no inference about actual truncation.
                lengths = []
                for row in ds:
                    text = ("<|im_start|>user\n" + row["prompt"][0]["content"] + "<|im_end|>\n"
                            + "<|im_start|>assistant\n" + row["completion"][0]["content"] + "<|im_end|>\n")
                    lengths.append(len(tokenizer.encode(text, add_special_tokens=False).ids))
                entry.update({"raw_chat_tokens": sum(lengths), "raw_chat_tokens_mean": sum(lengths) / len(lengths),
                              "raw_chat_tokens_max": max(lengths), "rows_over_16384": sum(length > 16384 for length in lengths)})
            evidence["cached_datasets"][name] = entry
        if test_questions is not None:
            example = "A store sells apples for $2 each. If John buys 7 apples and pays with a $20 bill, how much change does he get?"
            evidence["paper_apple_example_in_cached_gsm8k_test"] = normalize(example) in test_questions

    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
