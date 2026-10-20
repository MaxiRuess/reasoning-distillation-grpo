"""Benchmark evaluation with immutable identities, raw generations and safe resume."""

import time
from math import comb
from pathlib import Path

from filelock import FileLock

from src.artifacts import EvaluationStore, atomic_json, file_hash, fingerprint, read_model_manifest, runtime_metadata
from src.reward import SCORER_VERSION, extract_gsm8k_answer, score_response


def load_model_for_eval(model_path, config):
    from src.training import load_model, load_model_artifact

    if model_path == config["model"]["name"]:
        model, tokenizer = load_model(config)
        model.eval()
        tokenizer.padding_side = "left"
        return model, tokenizer
    return load_model_artifact(model_path, config)


def validate_generation(config):
    n = config.get("num_samples_per_problem", 1)
    if n < 1 or config.get("batch_size", 1) < 1 or config.get("max_new_tokens", 0) < 1:
        raise ValueError("Generation counts and lengths must be positive")
    if config.get("temperature", 0) < 0:
        raise ValueError("Temperature must be nonnegative")
    if n > 1 and config.get("temperature", 0) <= 0:
        raise ValueError("Multiple samples require temperature > 0; greedy repeats are not pass@k")


def compute_pass_at_k(results, k):
    if k < 1 or any(len(row) < k for row in results):
        raise ValueError("pass@k requires at least k samples per problem and k >= 1")
    scores = [1.0 - comb(len(row) - sum(row), k) / comb(len(row), k) for row in results]
    return sum(scores) / len(scores) if scores else 0.0


def summarize_records(records, problem_ids, num_samples):
    if len(records) != len(problem_ids) * num_samples:
        raise ValueError("Incomplete evaluation cannot be summarized as a full result")
    matrix = []
    for i, problem_id in enumerate(problem_ids):
        group = records[i * num_samples : (i + 1) * num_samples]
        if [(row["problem_id"], row["sample_id"]) for row in group] != [(problem_id, j) for j in range(num_samples)]:
            raise ValueError("Duplicate, missing or reordered samples")
        matrix.append([row["correct"] for row in group])
    total = len(problem_ids)
    correct = sum(row[0] for row in matrix)
    summary = {
        "accuracy": correct / total if total else 0.0,
        "num_correct": correct,
        "num_total": total,
        "num_samples_per_problem": num_samples,
        "mean_generated_tokens": sum(row["generated_tokens"] for row in records) / len(records) if records else 0,
        "truncation_rate": sum(row["finish_reason"] == "length" for row in records) / len(records) if records else 0,
        "extraction_failure_rate": sum(row["predicted"] is None for row in records) / len(records) if records else 0,
        "format_failure_rate": sum(
            row.get("status")
            in {"missing_final_answer", "malformed_final_answer", "unparseable_final_answer", "ambiguous_final_answer"}
            for row in records
        )
        / len(records)
        if records
        else 0,
        "unfinished_thinking_rate": sum(row.get("status") == "unfinished_thinking" for row in records) / len(records)
        if records
        else 0,
    }
    for k in [1, 4, 8]:
        if num_samples >= k:
            summary[f"pass@{k}"] = compute_pass_at_k(matrix, k)
    return summary


SAMPLING_POLICY = "canonical-batch-v2"


def inference_batch_size(cfg):
    n = cfg.get("num_samples_per_problem", 1)
    maximum = cfg.get("max_generation_sequences", 32)
    tokens = cfg.get("max_batch_generated_tokens", 65536)
    size = min(cfg["batch_size"], maximum // n, tokens // (cfg["max_new_tokens"] * n))
    if size < 1:
        raise ValueError("One sample group exceeds the inference resource envelope")
    return size


def generate_batch(model, tokenizer, questions, ids, cfg):
    from transformers import set_seed

    from src.prompting import ANSWER_CONTRACT, answer_prompt

    tokenizer.padding_side = "left"
    validate_generation(cfg)
    n = cfg.get("num_samples_per_problem", 1)
    if cfg.get("scorer", SCORER_VERSION) == ANSWER_CONTRACT:
        questions = [answer_prompt(question, cfg.get("benchmark", "gsm8k")) for question in questions]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.get("enable_thinking", True),
        )
        for q in questions
    ]
    if n > 1:
        seed = int(fingerprint({"seed": cfg.get("seed", 42), "problem_ids": ids, "policy": SAMPLING_POLICY})[:8], 16)
        set_seed(seed)
    return _generate(model, tokenizer, prompts, cfg, n)


def _generate(model, tokenizer, prompts, cfg, n):
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(model.device)
    if inputs["input_ids"].shape[1] + cfg["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("Prompt plus output budget exceeds context; refusing silent truncation")
    options = {
        "max_new_tokens": cfg["max_new_tokens"],
        "do_sample": cfg.get("temperature", 0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": n,
        "use_cache": True,
    }
    if options["do_sample"]:
        options.update(temperature=cfg["temperature"], top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20))
    else:
        options.update(temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **options)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    groups = [[] for _ in prompts]
    generated_ids = outputs[:, inputs["input_ids"].shape[1] :].cpu().tolist()
    for index, tokens in enumerate(generated_ids):
        stop = next((i for i, token in enumerate(tokens) if token in eos), None)
        if stop is not None:
            tokens = tokens[: stop + 1]
        groups[index // n].append(
            {
                "response": tokenizer.decode(tokens, skip_special_tokens=True),
                "generated_tokens": len(tokens),
                "finish_reason": "eos" if stop is not None else "length",
            }
        )
    return groups


def generate_solutions(model, tokenizer, questions, max_new_tokens=2048, temperature=0.0, batch_size=8, num_samples=1):
    cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "batch_size": batch_size,
        "num_samples_per_problem": num_samples,
    }
    validate_generation(cfg)
    solutions = []
    batch_size = inference_batch_size(cfg)
    for start in range(0, len(questions), batch_size):
        group = generate_batch(
            model,
            tokenizer,
            questions[start : start + batch_size],
            [str(i) for i in range(start, min(start + batch_size, len(questions)))],
            cfg,
        )
        solutions.extend([[row["response"] for row in samples] for samples in group])
    return solutions


def evaluate_dataset(model, tokenizer, rows, benchmark, cfg, store=None, on_batch=None, performance=None):
    cfg = {**cfg, "benchmark": benchmark}
    validate_generation(cfg)
    ids = [row["problem_id"] for row in rows]
    records = list(store.records) if store else []
    n, size = cfg.get("num_samples_per_problem", 1), inference_batch_size(cfg)
    # Dataset contents are part of the identity, and re-check answers on resume too.
    ground_truths = {row["problem_id"]: row["answer"] for row in rows}
    for record in records:
        if ground_truths[record["problem_id"]] != record["ground_truth"]:
            raise ValueError("Ground truth changed on resume")
    completed = len(records) // n
    # Recreate a partially committed canonical batch with the identical RNG seed.
    first = completed // size * size if completed < len(rows) else len(rows)
    for start in range(first, len(rows), size):
        batch = rows[start : start + size]
        generation_start = time.monotonic()
        generated = generate_batch(model, tokenizer, [row["question"] for row in batch], ids[start : start + size], cfg)
        if performance is not None:
            performance["generation_seconds"] = (
                performance.get("generation_seconds", 0) + time.monotonic() - generation_start
            )
            performance["generated_tokens_this_call"] = performance.get("generated_tokens_this_call", 0) + sum(
                sample["generated_tokens"] for group in generated for sample in group
            )
        if len(generated) != len(batch) or any(len(group) != n for group in generated):
            raise ValueError("Generator returned missing or extra samples")
        for offset, (row, samples) in enumerate(zip(batch, generated, strict=True)):
            if start + offset < completed:
                continue
            transaction = []
            for j, sample in enumerate(samples):
                score = score_response(sample["response"], row["answer"], benchmark, cfg.get("scorer", SCORER_VERSION))
                transaction.append(
                    {
                        "problem_id": row["problem_id"],
                        "sample_id": j,
                        "question": row["question"],
                        "ground_truth": row["answer"],
                        **sample,
                        **score.to_dict(),
                    }
                )
            if store:
                store.append(transaction)
            records.extend(transaction)
        if on_batch:
            on_batch()
    return {**summarize_records(records, ids, n), "benchmark": benchmark, "records": records}


def benchmark_rows(benchmark, config):
    from src.data import load_benchmark

    dataset = load_benchmark(benchmark, config)
    rows = []
    limit = config["evaluation"].get("max_problems")
    indices = config["evaluation"].get("problem_indices")
    if indices is not None and limit is not None:
        raise ValueError("Use frozen problem indices or a prefix limit, not both")
    if indices is not None and (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Invalid frozen problem indices")
    if limit is not None:
        if limit < 1:
            raise ValueError("Subset size must be positive")
        indices = list(range(min(limit, len(dataset))))
    for index in indices if indices is not None else range(len(dataset)):
        example = dataset[index]
        answer = extract_gsm8k_answer(example["answer"]) if benchmark == "gsm8k" else example["answer"]
        if answer is None:
            raise ValueError("Benchmark ground truth is unparseable")
        rows.append(
            {
                "problem_id": str(example.get("unique_id") or example.get("id") or index),
                "question": example.get("question", example.get("problem")),
                "answer": answer,
            }
        )
    expected = config["evaluation"].get("expected_rows_digest")
    if expected is not None and fingerprint(rows) != expected:
        raise ValueError("Frozen benchmark rows differ from the predeclared selection")
    return rows


def evaluate_gsm8k(model, tokenizer, config):
    return evaluate_dataset(model, tokenizer, benchmark_rows("gsm8k", config), "gsm8k", config["evaluation"])


def evaluate_math(model, tokenizer, config):
    return evaluate_dataset(model, tokenizer, benchmark_rows("math500", config), "math500", config["evaluation"])


def _run_all_evaluations(model_path, config, condition_name, on_batch=None, performance=None):
    cfg = config["evaluation"]
    committer = PeriodicCommit(on_batch, cfg.get("commit_every_batches", 4), cfg.get("commit_interval_seconds", 60))
    validate_generation(cfg)
    benchmarks = cfg["benchmarks"]
    if not benchmarks or any(name not in {"gsm8k", "math500"} for name in benchmarks):
        raise ValueError("Use gsm8k or math500; the old math name was an unrelated training-tail diagnostic")
    if cfg.get("resume_dir") and len(benchmarks) != 1:
        raise ValueError("An explicit resume directory requires one benchmark")
    identity = (
        fingerprint(config["model"])
        if model_path == config["model"]["name"]
        else read_model_manifest(model_path, verify_files=False)["artifact_id"]
    )
    load_start = time.monotonic()
    model, tokenizer = load_model_for_eval(model_path, config)
    if performance is not None:
        performance["model_load_seconds"] = time.monotonic() - load_start
    results = {"condition": condition_name, "benchmarks": {}}
    for benchmark in benchmarks:
        cfg = {**cfg, "benchmark": benchmark}
        rows = benchmark_rows(benchmark, config)
        key = {
            "model": identity,
            "answer_contract": SCORER_VERSION if cfg.get("scorer", SCORER_VERSION) == SCORER_VERSION else None,
            "sampling_policy": SAMPLING_POLICY,
            "effective_batch_size": inference_batch_size(cfg),
            "loading_config": config.get("quantization", {}),
            "model_dtype": str(getattr(model, "dtype", None)),
            "attention": getattr(getattr(model, "config", None), "_attn_implementation", None),
            "device_type": str(getattr(getattr(model, "device", None), "type", None)),
            "generation": {k: v for k, v in cfg.items() if k not in {"output_root", "resume_dir", "benchmarks"}},
            "benchmark": benchmark,
            "dataset": fingerprint(rows),
            "dataset_spec": config["datasets"][benchmark],
            "template": tokenizer.chat_template,
            "runtime": runtime_metadata(),
            "implementation": {
                name: file_hash(Path(__file__).with_name(name))
                for name in ["evaluation.py", "reward.py", "prompting.py"]
            },
        }
        root = Path(cfg.get("output_root", "outputs/evaluations"))
        path = (
            Path(cfg["resume_dir"])
            if cfg.get("resume_dir")
            else root / condition_name / benchmark / fingerprint(key)[:24]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".writer.lock", timeout=0):
            store = EvaluationStore(
                path, key, [row["problem_id"] for row in rows], cfg.get("num_samples_per_problem", 1)
            )
            if performance is not None:
                performance["results_paths"].append(str(path))
            result = evaluate_dataset(
                model, tokenizer, rows, benchmark, cfg, store, committer.tick if on_batch else None, performance
            )
            summary = {k: v for k, v in result.items() if k != "records"}
            summary.update(
                run_id=fingerprint(key),
                condition=condition_name,
                results_path=str(path),
                model_identity=identity,
                scorer_version=cfg.get("scorer", SCORER_VERSION),
                accuracy_definition="explicit_final_answer_accuracy"
                if cfg.get("scorer", SCORER_VERSION) == SCORER_VERSION
                else "archived_policy_accuracy",
                pilot_or_subset=bool(cfg.get("max_problems") or cfg.get("problem_indices")),
            )
            atomic_json(path / "summary.json", summary)
            committer.flush()
            results["benchmarks"][benchmark] = summary
    return results


class PeriodicCommit:
    def __init__(self, callback, batches=4, seconds=60, clock=time.monotonic):
        if batches < 1 or seconds <= 0:
            raise ValueError("Commit cadence must be positive")
        self.callback, self.batches, self.seconds, self.clock = callback, batches, seconds, clock
        self.pending = 0
        self.last = clock()

    def tick(self):
        self.pending += 1
        if self.pending >= self.batches or self.clock() - self.last >= self.seconds:
            self.flush()

    def flush(self):
        if self.callback:
            self.callback()
        self.pending = 0
        self.last = self.clock()


def run_all_evaluations(model_path, config, condition_name, on_batch=None):
    from src.performance import GpuMonitor

    performance = {
        "generation_seconds": 0.0,
        "generated_tokens_this_call": 0,
        "results_paths": [],
        "scope": "This function invocation, across benchmarks, including any replayed prefix",
        "status": "failed",
    }
    monitor = GpuMonitor(config["evaluation"].get("telemetry_interval_seconds", 5))
    try:
        with monitor:
            result = _run_all_evaluations(model_path, config, condition_name, on_batch, performance)
        performance["status"] = "complete"
    finally:
        performance["gpu"] = monitor.summary()
        for path in performance["results_paths"]:
            atomic_json(Path(path) / "performance.json", performance)
        if on_batch:
            on_batch()
    return result
