# Reasoning Trace Distillation: Can Small Models Learn to Think?

How effectively do reasoning capabilities transfer from large "thinking" models to small models through distillation — and does adding RL (GRPO) on top of distillation meaningfully improve reasoning quality?

## Experimental Conditions

| Condition | Method | What it tests |
|---|---|---|
| `baseline` | SFT on Q&A pairs (no reasoning traces) | Control — answer-only training |
| `sft_traces` | SFT on DeepSeek-R1 reasoning traces | Pure distillation |
| `grpo_only` | GRPO with answer-correctness reward | Pure RL |
| `sft_then_grpo` | SFT on traces → GRPO refinement | Combined distillation + RL |

## Quick Start

```bash
# Install
pip install -e .

# Prepare data
python scripts/01_prepare_sft_data.py --condition sft_traces
python scripts/02_prepare_grpo_data.py

# Train locally
python scripts/03_train_sft.py --condition sft_traces
python scripts/04_train_grpo.py --condition grpo_only

# Evaluate
python scripts/05_evaluate.py --model-path outputs/sft_traces --condition sft_traces

# Analyze traces
python scripts/06_analyze_traces.py --results-dir outputs

# Run full pipeline on Modal
modal run modal_train.py
```

## Project Structure

```
├── configs/config.yaml          # All hyperparameters
├── src/
│   ├── data.py                  # Dataset loading & formatting
│   ├── reward.py                # Answer extraction & reward function
│   ├── training.py              # Model loading & trainer construction
│   └── evaluation.py            # Benchmark evaluation
├── scripts/
│   ├── 01_prepare_sft_data.py   # Format SFT datasets
│   ├── 02_prepare_grpo_data.py  # Format GRPO datasets
│   ├── 03_train_sft.py          # SFT training
│   ├── 04_train_grpo.py         # GRPO training
│   ├── 05_evaluate.py           # Run benchmarks
│   └── 06_analyze_traces.py     # Trace analysis
└── modal_train.py               # Cloud training orchestration
```

## Datasets

| Dataset | HuggingFace ID | Role |
|---|---|---|
| s1K-1.1 | `simplescaling/s1K-1.1` | SFT — reasoning traces from DeepSeek-R1 |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | MATH evaluation |
| NuminaMath-RL-Verifiable | `nlile/NuminaMath-1.5-RL-Verifiable` | GRPO training |
| Orca Math | `microsoft/orca-math-word-problems-200k` | Baseline SFT (answer-only) |
| GSM8K | `openai/gsm8k` | Evaluation benchmark |

## Training Details

- **Model**: Qwen3 1.7B in full bfloat16 (no quantization)
- **LoRA**: rank=64, rsLoRA scaling, 7 target modules (q/k/v/o + gate/up/down proj)
- **SFT**: lr=2e-4, 3 epochs, max_seq_len=16384, packing enabled
- **GRPO**: lr=5e-7, group_size=8, binary reward, KL=0.1
- **Compute**: Modal (L40S for SFT, H100 for GRPO)
- **Tracking**: Weights & Biases

### Design Rationale

**Full bf16 over QLoRA** — At 1.7B parameters, the model is only ~3.4 GB in bf16 vs ~1.2 GB in 4-bit. The extra ~2 GB is negligible on an L40S (48 GB), but eliminates quantization noise that measurably degrades fine-tuning quality ([QuAILoRA, arXiv:2410.14713](https://arxiv.org/abs/2410.14713)).

**LoRA r=64 with rsLoRA** — Ablation studies show Qwen peaks at r=64 for GSM8K-style math reasoning, with no improvement at r=128 ([arXiv:2512.15634](https://arxiv.org/abs/2512.15634)). Standard LoRA scaling (alpha/r) causes gradient collapse at high ranks; rsLoRA fixes this by scaling alpha/sqrt(r) ([arXiv:2312.03732](https://arxiv.org/abs/2312.03732)).

**16K sequence length** — s1K-1.1 reasoning traces average ~10K tokens. Qwen3-1.7B supports 32K natively. At 16K with batch_size=4, estimated VRAM is ~13 GB — well within the L40S budget.
