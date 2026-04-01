# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML research project comparing four training conditions for teaching reasoning to a small model (Qwen3 1.7B) via distillation from DeepSeek-R1 traces and/or GRPO reinforcement learning. Uses TRL (SFTTrainer, GRPOTrainer) with LoRA (r=64, rsLoRA) in full bfloat16 precision.

## Commands

```bash
# Install (editable, venv recommended)
pip install -e .

# Data preparation
python scripts/01_prepare_sft_data.py --condition sft_traces
python scripts/02_prepare_grpo_data.py

# Training (requires GPU)
python scripts/03_train_sft.py --condition sft_traces
python scripts/04_train_grpo.py --condition grpo_only
python scripts/04_train_grpo.py --condition sft_then_grpo --sft-checkpoint outputs/sft_then_grpo/sft_phase

# Evaluation
python scripts/05_evaluate.py --model-path outputs/sft_traces --condition sft_traces

# Full cloud pipeline (all 4 conditions, parallelized)
modal run modal_train.py
```

## Architecture

```
configs/config.yaml     ← Single source of truth for all hyperparameters
        ↓
src/data.py             ← Loads HF datasets, formats per condition
src/reward.py           ← Answer extraction (####, \boxed{}), binary reward for GRPO
src/training.py         ← Model loading (bf16 or quantized), LoRA + trainer construction
src/evaluation.py       ← GSM8K/MATH benchmarks, pass@k computation
        ↓
scripts/01-06           ← CLI entry points for each pipeline phase
modal_train.py          ← Cloud orchestration (L40S for SFT, H100 for GRPO)
```

**Config flow**: Scripts call `load_config()` → extract `config["conditions"][condition_name]` → pass to trainer builders which read the nested `["sft"]` or `["grpo"]` sub-key.

## Four Experimental Conditions

| Condition | Trains with | Script(s) |
|---|---|---|
| `baseline` | SFT on Orca Math (Q&A, no traces) | `03_train_sft.py` |
| `sft_traces` | SFT on s1K-1.1 (DeepSeek-R1 traces) | `03_train_sft.py` |
| `grpo_only` | GRPO from base model | `04_train_grpo.py` |
| `sft_then_grpo` | SFT on traces, then GRPO refinement | `03_train_sft.py` → `04_train_grpo.py` |

## Critical Patterns

**Reward function signature** — TRL passes `prompts`, `completions`, and dataset columns as kwargs. The parameter name `answer` must exactly match the dataset column name:
```python
def binary_reward_fn(prompts, completions, answer, **kwargs) -> list[float]
```

**EOS token alignment** — Qwen3 base uses `<|endoftext|>` as eos_token, but its chat template uses `<|im_end|>` as end-of-turn. Config sets `eos_token: "<|im_end|>"` which `load_model()` applies to the tokenizer.

**Quantization is optional** — `config["quantization"]["enabled"]` controls whether to use 4-bit NF4 or full bf16. Currently disabled (full bf16) because the 1.7B model is small enough that quantization noise hurts more than the VRAM savings help. `load_model()`, `load_sft_checkpoint()`, and `load_model_for_eval()` all check this flag.

**rsLoRA scaling** — LoRA rank is set to 64 with `use_rslora: true`. Standard LoRA scaling (alpha/r) causes gradient collapse at high ranks; rsLoRA scales by alpha/sqrt(r) instead. Critical when r > 32.

**Dataset format conventions**:
- SFT datasets: `prompt` + `completion` columns, both lists of `{"role": ..., "content": ...}` message dicts
- GRPO datasets: `prompt` (message dicts) + `answer` (plain string for reward function)

**SFT→GRPO checkpoint handoff**: `load_sft_checkpoint()` merges the LoRA adapter into base weights via `merge_and_unload()` before GRPO applies a fresh adapter. LoRA adapters cannot stack in TRL.

**Tokenizer padding side**: Right-padding during training, left-padding during generation. `load_model_for_eval()` handles this switch.

**Answer extraction priority** (`extract_answer_auto`): `\boxed{}` (brace-depth counting) → `####` delimiter → last number regex fallback.

**Modal volumes**: Functions must call `volume.commit()` after writing outputs or data is lost.

## Linting

Ruff with line-length 120.
