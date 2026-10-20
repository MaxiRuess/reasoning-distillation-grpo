"""Pinned data preparation, provenance, complete targets and matched-question controls."""

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset, load_from_disk

from src.artifacts import atomic_json, file_hash, fingerprint
from src.reward import extract_boxed_answer, final_response


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as stream:
        return yaml.safe_load(stream)


def load_source(key, config, split=None):
    spec = dict(config["datasets"][key])
    name, revision = spec.pop("name"), spec.pop("revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{key} must have an immutable dataset revision")
    subset = spec.pop("subset", None)
    dataset = load_dataset(name, subset, revision=revision, split=split or spec.pop("split", "train"))
    if not dataset.cache_files or any(revision not in Path(item["filename"]).parts for item in dataset.cache_files):
        raise ValueError("Dataset cache is not tied to the requested revision; refusing latest-cache fallback")
    return dataset


def load_benchmark(name, config):
    if name not in {"gsm8k", "math500"}:
        raise ValueError("Supported benchmarks: gsm8k, math500")
    return load_source(name, config, split="test")


def normalized_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).lower().split())


class HeldoutIndex:
    def __init__(self, questions):
        self.normalized = {normalized_question(q) for q in questions}
        self.tokens = [set(re.findall(r"\w+|[^\w\s]", q)) for q in self.normalized]
        self.inverted = {}
        for i, tokens in enumerate(self.tokens):
            for token in tokens:
                self.inverted.setdefault(token, []).append(i)

    def reason(self, question):
        normalized = normalized_question(question)
        if normalized in self.normalized:
            return "exact_test_overlap"
        tokens = set(re.findall(r"\w+|[^\w\s]", normalized))
        if len(tokens) < 8:
            return None
        counts = Counter(i for token in tokens for i in self.inverted.get(token, ()))
        for i, shared in counts.items():
            other = self.tokens[i]
            if len(other) >= 8 and shared / (len(tokens) + len(other) - shared) >= 0.85:
                return "near_duplicate_candidate"
        return None


def overlap_reason(question, heldout):
    index = heldout if isinstance(heldout, HeldoutIndex) else HeldoutIndex(heldout)
    return index.reason(question)


def chat_row(question, completion, source_id, **provenance):
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": completion}],
        "source_id": str(source_id),
        **provenance,
    }


def target_length(row, tokenizer):
    return len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))


def valid_trace(text):
    if not isinstance(text, str):
        return False
    final, status = final_response(text)
    thinking = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return (
        status == "ok"
        and thinking is not None
        and bool(thinking[1].strip())
        and bool(final)
        and extract_boxed_answer(final) is not None
    )


def build_controlled_datasets(source, tokenizer, count=1000, seed=2026, max_length=16384, heldout=()):
    """Freeze one question pool and independently selected raw/correct generations."""
    rng = random.Random(seed)
    heldout = heldout if isinstance(heldout, HeldoutIndex) else HeldoutIndex(heldout)
    rows = {"controlled_answer_only": [], "controlled_unfiltered": [], "controlled_verified": []}
    quarantine, seen = [], set()
    iterable = source.shuffle(seed=seed) if hasattr(source, "shuffle") else rng.sample(list(source), len(source))
    for example in iterable:
        source_id = str(example["uuid"])
        problem, answer = example["problem"], str(example["answer"] or "").strip()
        reason = overlap_reason(problem, heldout)
        if not answer or source_id in seen or reason:
            quarantine.append({"source_id": source_id, "reason": reason or "missing_answer_or_duplicate_id"})
            continue
        generations, correct = example["generations"] or [], example["correctness_math_verify"]
        if correct is None or len(generations) != len(correct):
            quarantine.append({"source_id": source_id, "reason": "verification_alignment"})
            continue
        eligible = []
        for i, text in enumerate(generations):
            row = chat_row(problem, text, source_id, generation_index=i, verified=correct[i] is True)
            if valid_trace(text) and target_length(row, tokenizer) <= max_length:
                eligible.append(row)
            else:
                quarantine.append({"source_id": source_id, "generation_index": i, "reason": "incomplete_or_overlength"})
        verified = [row for row in eligible if row["verified"]]
        if not verified:
            continue
        seen.add(source_id)
        raw, good = rng.choice(eligible), rng.choice(verified)
        rows["controlled_unfiltered"].append(raw)
        rows["controlled_verified"].append(good)
        rows["controlled_answer_only"].append(
            chat_row(problem, "\\boxed{" + answer + "}", source_id, generation_index=-1, verified=True)
        )
        if len(seen) == count:
            break
    if len(seen) != count:
        raise ValueError(f"Only {len(seen)} eligible shared questions; requested {count}")
    return {name: Dataset.from_list(values) for name, values in rows.items()}, quarantine


def matched_token_dataset(dataset, tokenizer, budget, seed=2026):
    """Repeat whole examples to a fixed supervised completion-token budget."""
    if budget < 1 or not len(dataset):
        raise ValueError("Positive token budget and nonempty dataset required")
    # Count completion supervision using the same chat prefix boundary as SFTTrainer.
    lengths = []
    for row in dataset:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        lengths.append(target_length(row, tokenizer) - len(prompt))
    if min(lengths) < 1:
        raise ValueError("No completion supervision")
    rng = random.Random(seed)
    indices, total = [], 0
    while total < budget:
        order = list(range(len(dataset)))
        rng.shuffle(order)
        for i in order:
            indices.append(i)
            total += lengths[i]
            if total >= budget:
                break
    return dataset.select(indices), {
        "requested_tokens": budget,
        "supervised_tokens": total,
        "overshoot": total - budget,
    }


def format_s1k_for_sft(tokenizer, config=None):
    config = config or load_config()
    ds = load_source("s1k", config)
    rows = []
    for i, ex in enumerate(ds):
        thinking, answer = (
            ex.get("deepseek_thinking_trajectory", ""),
            ex.get("deepseek_attempt", ex.get("solution", "")),
        )
        text = f"<think>\n{thinking}\n</think>\n\n{answer}" if thinking else answer
        rows.append(chat_row(ex["question"], text, str(i), generation_index=0))
    return Dataset.from_list(rows)


def format_orca_math_for_sft(tokenizer, max_samples=None, config=None):
    """Ordinary worked-solution SFT; Orca's answer field contains reasoning."""
    config = config or load_config()
    ds = load_source("orca_math", config).shuffle(seed=config.get("data_seed", 2026))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return Dataset.from_list([chat_row(ex["question"], ex["answer"], str(i)) for i, ex in enumerate(ds)])


def format_gsm8k_for_grpo(tokenizer, max_samples=None, config=None):
    config = config or load_config()
    ds = load_source("gsm8k", config, split="train")
    indices = list(range(len(ds)))
    random.Random(config.get("data_seed", 2026)).shuffle(indices)
    if max_samples:
        indices = indices[:max_samples]
    return Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": ds[i]["question"],
                    }
                ],
                "answer": ds[i]["answer"].rsplit("####", 1)[-1].strip(),
                "source_id": str(i),
            }
            for i in indices
        ]
    )


def format_gsm8k_for_eval():
    return load_benchmark("gsm8k", load_config())


def _heldout_questions(config):
    questions = []
    for name in ["gsm8k", "math500"]:
        for row in load_benchmark(name, config):
            questions.append(row.get("question", row.get("problem")))
    return questions


def prepare_controlled_pool(config, tokenizer, data_root):
    spec = {
        "sources": config["datasets"],
        "model": config["model"],
        "selection": config["study"],
        "selection_seed": config["data_seed"],
        "preparation_implementation": file_hash(Path(__file__)),
    }
    path = Path(data_root) / "controlled-pool" / fingerprint(spec)[:24]
    if (path / "pool.json").exists():
        manifest = json.loads((path / "pool.json").read_text())
        if manifest["spec"] != spec:
            raise ValueError("Controlled pool configuration mismatch")
        for name, expected in manifest["dataset_hashes"].items():
            if fingerprint(list(load_from_disk(str(path / name)))) != expected:
                raise ValueError("Controlled pool contents changed")
        return path
    datasets, quarantine = build_controlled_datasets(
        load_source("openr1", config),
        tokenizer,
        count=config["study"]["num_sft_questions"],
        seed=config["data_seed"],
        max_length=config["study"]["max_length"],
        heldout=HeldoutIndex(_heldout_questions(config)),
    )
    path.mkdir(parents=True, exist_ok=False)
    for name, dataset in datasets.items():
        dataset.save_to_disk(str(path / name))
    atomic_json(
        path / "pool.json",
        {
            "spec": spec,
            "quarantine": quarantine,
            "dataset_hashes": {name: fingerprint(list(dataset)) for name, dataset in datasets.items()},
        },
    )
    return path


def prepare_condition(
    condition_name, config, tokenizer, phase, data_root="data/prepared", require_existing=False, verify_content=True
):
    from src.prompting import ANSWER_CONTRACT

    condition = config["conditions"][condition_name]
    source_name = condition.get("sft_dataset", condition.get("dataset")) if phase == "sft" else "gsm8k"
    spec = {
        "condition": condition_name,
        "preparation_implementation": file_hash(Path(__file__)),
        "phase": phase,
        "source": source_name,
        "datasets": config["datasets"],
        "model": config["model"],
        "data_seed": config["data_seed"],
        "answer_contract": config.get("reward_scorer") if phase == "grpo" else None,
        "prompt_implementation": file_hash(Path(__file__).with_name("prompting.py"))
        if phase == "grpo" and config.get("reward_scorer") == ANSWER_CONTRACT
        else None,
        "settings": condition[phase],
        "study": config.get("study"),
    }
    if config.get("study", {}).get("matched_teacher_answers"):
        spec["matched_supervision_implementation"] = file_hash(Path(__file__).with_name("distillation_experiment.py"))
    # Paths and tracking preferences do not change the prepared examples.
    spec["settings"] = {
        k: v for k, v in spec["settings"].items() if k in {"max_seq_length", "token_matched", "max_samples"}
    }
    path = Path(data_root) / condition_name / phase / fingerprint(spec)[:24]
    if (path / "data_manifest.json").exists():
        manifest = json.loads((path / "data_manifest.json").read_text())
        dataset = load_from_disk(str(path / "dataset"))
        if manifest["spec"] != spec or (verify_content and manifest["content_hash"] != fingerprint(list(dataset))):
            raise ValueError("Prepared dataset changed; rebuild into a new directory")
        return dataset, manifest
    if require_existing:
        raise FileNotFoundError(f"Prepare data on CPU first: {path}")
    quarantine, token_budget = [], None
    if phase == "grpo":
        candidates = format_gsm8k_for_grpo(tokenizer, None, config)
        heldout = HeldoutIndex(_heldout_questions(config))
        selected = []
        requested = condition[phase].get("max_samples")
        for row in candidates:
            reason = heldout.reason(row["prompt"][0]["content"])
            if reason:
                quarantine.append({"source_id": row["source_id"], "reason": reason})
            else:
                selected.append(row)
            if requested and len(selected) == requested:
                break
        if requested and len(selected) != requested:
            raise ValueError("Not enough disjoint GRPO prompts")
        dataset = Dataset.from_list(selected)
        from src.prompting import ANSWER_CONTRACT, answer_prompt

        if config.get("reward_scorer") == ANSWER_CONTRACT:
            dataset = Dataset.from_list(
                [
                    {**row, "prompt": [{"role": "user", "content": answer_prompt(row["prompt"][0]["content"])}]}
                    for row in selected
                ]
            )
    elif source_name.startswith("controlled_"):
        pool = prepare_controlled_pool(config, tokenizer, data_root)
        dataset = load_from_disk(str(pool / source_name))
        if config.get("study", {}).get("matched_teacher_answers"):
            from src.distillation_experiment import distillation_rows

            answer_rows, trace_rows = distillation_rows(load_from_disk(str(pool / "controlled_verified")))
            selected = answer_rows if source_name == "controlled_answer_only" else trace_rows
            if source_name not in {"controlled_answer_only", "controlled_verified"}:
                raise ValueError("The bounded distillation study has only answer and trace SFT arms")
            dataset = Dataset.from_list(selected)
        if condition[phase].get("token_matched"):
            totals = []
            for arm in ["controlled_unfiltered", "controlled_verified"]:
                other = load_from_disk(str(pool / arm))
                total = sum(
                    target_length(row, tokenizer)
                    - len(tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True))
                    for row in other
                )
                totals.append(total)
            dataset, token_budget = matched_token_dataset(dataset, tokenizer, min(totals) * 3, config["data_seed"])
    elif source_name == "s1k":
        dataset = format_s1k_for_sft(tokenizer, config)
    elif source_name == "orca_math":
        dataset = format_orca_math_for_sft(tokenizer, max_samples=50000, config=config)
    elif source_name == "openr1":
        pool = prepare_controlled_pool(config, tokenizer, data_root)
        dataset = load_from_disk(str(pool / "controlled_verified"))
    else:
        raise ValueError(f"Unsupported SFT source: {source_name}")
    if phase == "sft":
        heldout = HeldoutIndex(_heldout_questions(config))
        accepted = []
        for row in dataset:
            _, status = final_response(row["completion"][0]["content"])
            reason = overlap_reason(row["prompt"][0]["content"], heldout)
            if status != "ok" or target_length(row, tokenizer) > condition[phase]["max_seq_length"] or reason:
                quarantine.append(
                    {"source_id": row["source_id"], "reason": reason or (status if status != "ok" else "overlength")}
                )
            else:
                accepted.append(row)
        dataset = Dataset.from_list(accepted) if accepted else None
    if dataset is None or not len(dataset):
        raise ValueError("No valid training examples remain")
    path.mkdir(parents=True, exist_ok=False)
    dataset.save_to_disk(str(path / "dataset"))
    manifest = {
        "schema_version": 1,
        "spec": spec,
        "rows": len(dataset),
        "content_hash": fingerprint(list(dataset)),
        "path": str(path),
        "quarantine": quarantine,
        "token_budget": token_budget,
        "source_ids": [row["source_id"] for row in dataset],
        "quarantine_counts": dict(Counter(row["reason"] for row in quarantine)),
        "chat_token_lengths": [target_length(row, tokenizer) for row in dataset] if phase == "sft" else [],
        "max_prompt_tokens": max(
            len(tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True))
            for row in dataset
        )
        if phase == "grpo"
        else None,
    }
    atomic_json(path / "data_manifest.json", manifest)
    return dataset, manifest


def get_dataset_for_condition(condition_name, config, tokenizer):
    condition = config["conditions"][condition_name]
    if condition["type"] == "sft_then_grpo":
        return (
            prepare_condition(condition_name, config, tokenizer, "sft")[0],
            prepare_condition(condition_name, config, tokenizer, "grpo")[0],
        )
    phase = "grpo" if condition["type"] == "grpo" else "sft"
    return prepare_condition(condition_name, config, tokenizer, phase)[0]
