# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML research project comparing five training conditions for teaching reasoning to a small model (Qwen3 1.7B) via distillation from DeepSeek-R1 traces, GRPO reinforcement learning, and re-distillation from RL-verified traces. Uses TRL (SFTTrainer, GRPOTrainer) with LoRA (r=64, rsLoRA) in full bfloat16 precision.

## Commands

```bash
# Install (editable, venv recommended)
pip install -e .

# Data preparation
python scripts/01_prepare_sft_data.py --condition sft_traces
python scripts/01_prepare_sft_data.py --condition re_distill

# SFT training (requires GPU / Modal)
modal run modal_train.py::train_sft --condition sft_traces
modal run modal_train.py::train_sft --condition re_distill

# GRPO training
modal run modal_train.py::train_grpo --condition grpo_only
modal run modal_train.py::train_grpo --condition sft_then_grpo --sft-checkpoint /vol/outputs/sft_then_grpo

# Evaluation (spot check on N problems)
modal run modal_train.py::spot_check_gsm8k --model-path /vol/outputs/sft_traces --num-problems 100
modal run modal_train.py::spot_check_gsm8k --model-path /vol/outputs/sft_traces --num-problems 100 --start-from 33

# Quick test (5 sample questions, visual check)
modal run modal_train.py::quick_test --model-path /vol/outputs/sft_traces
```

## Architecture

```
configs/config.yaml     <- Single source of truth for all hyperparameters
        |
src/data.py             <- Loads HF datasets, formats per condition
src/reward.py           <- Answer extraction (####, \boxed{}), binary + format rewards
src/training.py         <- Model loading (bf16 or quantized), LoRA + trainer construction
src/evaluation.py       <- GSM8K/MATH benchmarks, pass@k computation
        |
scripts/01-06           <- CLI entry points for each pipeline phase
modal_train.py          <- Cloud orchestration (L40S for SFT, H100 for GRPO)
results/                <- Evaluation results and findings
paper/                  <- Literature comparison
```

**Config flow**: Scripts call `load_config()` -> extract `config["conditions"][condition_name]` -> pass to trainer builders which read the nested `["sft"]` or `["grpo"]` sub-key.

## Five Experimental Conditions

| Condition | Trains with | Script(s) |
|---|---|---|
| `baseline` | SFT on Orca Math (Q&A, no traces) | `train_sft` |
| `sft_traces` | SFT on s1K-1.1 (DeepSeek-R1 traces) | `train_sft` |
| `re_distill` | SFT on Open-R1 (RL-verified traces) | `train_sft` |
| `grpo_only` | GRPO from base model | `train_grpo` |
| `sft_then_grpo` | SFT on traces, then GRPO refinement | `train_sft` -> `train_grpo` |

## Critical Patterns

**Two Modal images** — `sft_image` uses torch 2.8 + flash-attn (pre-built wheel). `grpo_image` uses `trl[vllm]` which pulls its own torch version. They cannot be combined because vLLM's CUDA memory pool conflicts with flash-attn build requirements.

**GRPO requires vLLM** — Without vLLM, GRPO generation is unusably slow. Uses `vllm_mode="colocate"` with `vllm_enable_sleep_mode=True`. Requires distributed env vars (`RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1`).

**Reward functions** — TRL passes `prompts`, `completions`, and dataset columns as kwargs:
```python
def binary_reward_fn(prompts, completions, answer, **kwargs) -> list[float]
def format_reward_fn(prompts, completions, **kwargs) -> list[float]
```
Both handle TRL's message dict format (completions may be `[{"role": "assistant", "content": "..."}]`).

**Attention auto-detect** — `_get_attn_implementation()` returns `"flash_attention_2"` if flash-attn is installed, `"sdpa"` otherwise. Used in `load_model()`, `load_sft_checkpoint()`, and `load_model_for_eval()`.

**EOS token alignment** — Qwen3 base uses `<|endoftext|>` as eos_token, but its chat template uses `<|im_end|>` as end-of-turn. Config sets `eos_token: "<|im_end|>"` which `load_model()` applies to the tokenizer.

**Quantization is optional** — `config["quantization"]["enabled"]` controls whether to use 4-bit NF4 or full bf16. Currently disabled (full bf16).

**rsLoRA scaling** — LoRA rank is set to 64 with `use_rslora: true`. Standard LoRA scaling (alpha/r) causes gradient collapse at high ranks; rsLoRA scales by alpha/sqrt(r) instead.

**Dataset format conventions**:
- SFT datasets: `prompt` + `completion` columns, both lists of message dicts
- GRPO datasets: `prompt` (message dicts) + `answer` (plain string for reward function)

**SFT->GRPO checkpoint handoff**: `load_sft_checkpoint()` merges the LoRA adapter into base weights via `merge_and_unload()`, then loads tokenizer from base model (not checkpoint, to avoid cross-version tokenizer incompatibility).

**Evaluation**: `spot_check_gsm8k` supports `start_from` parameter for resumable evaluation after timeouts.

**Modal volumes**: Functions must call `volume.commit()` after writing outputs or data is lost.

## Linting

Ruff with line-length 120.
