"""Benchmark evaluation: GSM8K and MATH with pass@k metrics."""

import json
from math import comb
from pathlib import Path

import torch
from peft import PeftModel

from src.training import _get_attn_implementation
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data import format_gsm8k_for_eval, format_numinamath_for_eval
from src.reward import answers_match, extract_answer_auto, extract_boxed_answer, extract_gsm8k_answer


def load_model_for_eval(model_path: str, config: dict) -> tuple:
    """Load a trained model for inference.

    Supports both LoRA adapters (base + adapter) and full fine-tuned checkpoints.
    Detects which type by checking for adapter_config.json in the model path.
    Sets left-padding for batch generation.
    """
    from pathlib import Path

    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        # LoRA adapter — load base model then apply adapter
        quant_cfg = config["quantization"]
        use_quantization = quant_cfg.get("enabled", quant_cfg.get("load_in_4bit", False))

        if use_quantization:
            compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=compute_dtype,
                attn_implementation=_get_attn_implementation(),
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation=_get_attn_implementation(),
            )

        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Full fine-tuned checkpoint — load directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=_get_attn_implementation(),
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for batch generation

    return model, tokenizer


def generate_solutions(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    batch_size: int = 8,
    num_samples: int = 1,
) -> list[list[str]]:
    """Generate solutions for a list of questions.

    Returns list of lists — outer list is per-question, inner list contains
    `num_samples` generated solutions (for pass@k computation).
    """
    all_solutions = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
        batch_questions = questions[i : i + batch_size]

        # Format as chat messages and apply template
        messages_batch = [
            [{"role": "user", "content": q}] for q in batch_questions
        ]
        prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]

        for sample_idx in range(num_samples):
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the generated portion (strip prompt)
            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)

                if sample_idx == 0:
                    all_solutions.append([generated])
                else:
                    idx = i + j
                    if idx < len(all_solutions):
                        all_solutions[idx].append(generated)

    return all_solutions


def evaluate_gsm8k(model, tokenizer, config: dict) -> dict:
    """Run end-to-end GSM8K evaluation.

    Returns dict with accuracy, pass@k, trace length stats, and per-example results.
    """
    eval_cfg = config["evaluation"]
    ds = format_gsm8k_for_eval()

    questions = ds["question"]
    ground_truths = ds["answer"]

    solutions = generate_solutions(
        model,
        tokenizer,
        questions,
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
        batch_size=eval_cfg["batch_size"],
        num_samples=eval_cfg.get("num_samples_per_problem", 1),
    )

    results = []
    for q, gt, sols in zip(questions, ground_truths, solutions):
        gt_answer = extract_gsm8k_answer(gt)
        sample_correct = []

        for sol in sols:
            pred_answer = extract_answer_auto(sol)
            is_correct = pred_answer is not None and gt_answer is not None and answers_match(pred_answer, gt_answer)
            sample_correct.append(is_correct)

        results.append({
            "question": q,
            "ground_truth": gt_answer,
            "predictions": [extract_answer_auto(s) for s in sols],
            "correct": sample_correct,
            "trace_length": len(sols[0]) if sols else 0,
        })

    num_correct = sum(1 for r in results if any(r["correct"]))
    accuracy = num_correct / len(results) if results else 0.0
    avg_trace_length = sum(r["trace_length"] for r in results) / len(results) if results else 0.0

    # Compute pass@k for various k values
    correctness_matrix = [r["correct"] for r in results]
    pass_at = {}
    for k in [1, 4, 8]:
        if eval_cfg.get("num_samples_per_problem", 1) >= k:
            pass_at[f"pass@{k}"] = compute_pass_at_k(correctness_matrix, k)

    return {
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        **pass_at,
        "num_correct": num_correct,
        "num_total": len(results),
        "avg_trace_length": avg_trace_length,
        "results": results,
    }


def evaluate_math(model, tokenizer, config: dict) -> dict:
    """Run end-to-end MATH evaluation using held-out NuminaMath-CoT data."""
    eval_cfg = config["evaluation"]
    ds = format_numinamath_for_eval()

    questions = ds["question"]
    ground_truths = ds["answer"]

    solutions = generate_solutions(
        model,
        tokenizer,
        questions,
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
        batch_size=eval_cfg["batch_size"],
        num_samples=eval_cfg.get("num_samples_per_problem", 1),
    )

    results = []
    for q, gt, sols in zip(questions, ground_truths, solutions):
        gt_answer = extract_boxed_answer(gt) or extract_answer_auto(gt)
        sample_correct = []

        for sol in sols:
            pred_answer = extract_answer_auto(sol)
            is_correct = pred_answer is not None and gt_answer is not None and answers_match(pred_answer, gt_answer)
            sample_correct.append(is_correct)

        results.append({
            "question": q,
            "ground_truth": gt_answer,
            "predictions": [extract_answer_auto(s) for s in sols],
            "correct": sample_correct,
            "trace_length": len(sols[0]) if sols else 0,
        })

    num_correct = sum(1 for r in results if any(r["correct"]))
    accuracy = num_correct / len(results) if results else 0.0
    avg_trace_length = sum(r["trace_length"] for r in results) / len(results) if results else 0.0

    correctness_matrix = [r["correct"] for r in results]
    pass_at = {}
    for k in [1, 4, 8]:
        if eval_cfg.get("num_samples_per_problem", 1) >= k:
            pass_at[f"pass@{k}"] = compute_pass_at_k(correctness_matrix, k)

    return {
        "benchmark": "math",
        "accuracy": accuracy,
        **pass_at,
        "num_correct": num_correct,
        "num_total": len(results),
        "avg_trace_length": avg_trace_length,
        "results": results,
    }


def compute_pass_at_k(results: list[list[bool]], k: int) -> float:
    """Compute the unbiased pass@k estimator.

    For each problem with n total samples and c correct samples:
    pass@k = 1 - C(n-c, k) / C(n, k)

    This is the standard estimator from the Codex paper (Chen et al., 2021).
    """
    scores = []
    for sample_results in results:
        n = len(sample_results)
        c = sum(sample_results)

        if n < k:
            # Not enough samples — use empirical estimate
            scores.append(1.0 if c > 0 else 0.0)
        elif c == 0:
            scores.append(0.0)
        elif c >= n:
            scores.append(1.0)
        else:
            scores.append(1.0 - comb(n - c, k) / comb(n, k))

    return sum(scores) / len(scores) if scores else 0.0


def run_all_evaluations(model_path: str, config: dict, condition_name: str) -> dict:
    """Run all configured benchmarks and save results."""
    model, tokenizer = load_model_for_eval(model_path, config)
    eval_cfg = config["evaluation"]

    all_results = {"condition": condition_name, "benchmarks": {}}

    for benchmark in eval_cfg["benchmarks"]:
        if benchmark == "gsm8k":
            result = evaluate_gsm8k(model, tokenizer, config)
        elif benchmark == "math":
            result = evaluate_math(model, tokenizer, config)
        else:
            print(f"Unknown benchmark: {benchmark}, skipping")
            continue

        all_results["benchmarks"][benchmark] = result
        print(f"[{condition_name}] {benchmark}: accuracy={result['accuracy']:.4f}")

    # Save results to JSON
    output_dir = Path(f"outputs/{condition_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    # Remove per-example results for the saved summary (they can be large)
    summary = {
        "condition": condition_name,
        "benchmarks": {
            name: {k: v for k, v in res.items() if k != "results"}
            for name, res in all_results["benchmarks"].items()
        },
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    return all_results
