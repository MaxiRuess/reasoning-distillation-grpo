# Reasoning Trace Distillation: Can Small Models Learn to Think?

## Research Questions

1. **Can reasoning be transferred through imitation alone?** When a 1.7B model is fine-tuned on reasoning traces from a much larger "thinking" model (DeepSeek-R1, 671B), does it learn genuine multi-step reasoning — or just surface-level pattern mimicry?

2. **Does reinforcement learning add value beyond distillation?** If distillation works, can GRPO (Group Relative Policy Optimization) further refine the distilled reasoning — or does the RL objective conflict with the distilled knowledge?

3. **What is the right way to get RL benefits into small models?** If direct RL optimization fails, can we capture RL's improvements indirectly — by distilling the outputs of RL-trained models rather than RL-training the small model itself?

## Motivation

DeepSeek-R1 demonstrated that reinforcement learning can teach large models to reason, producing step-by-step `<think>` traces that dramatically improve math problem-solving. The subsequent distillation of R1's traces into smaller models (1.5B-70B) showed these capabilities can transfer across scale. But two questions remain open for practitioners working with sub-2B models: is the distillation alone sufficient, or does adding RL provide meaningful gains? And if the naive SFT→RL pipeline that works at 7B+ scale breaks down at smaller scales, what alternative captures RL's benefits?

## Approach

We fine-tune `Qwen3-1.7B` (base) under five controlled conditions using the same LoRA configuration (r=64, rsLoRA, full bfloat16) and evaluate on GSM8K (grade school math):

**Distillation Conditions:**
| Condition | Training Data | What We're Testing |
|---|---|---|
| `baseline` | 50K Orca Math (answers only) | Can answer memorization solve math? |
| `sft_traces` | 1K DeepSeek-R1 traces (s1K-1.1) | Does imitating reasoning traces teach reasoning? |
| `re_distill` | 1K Open-R1 RL-verified traces | Do RL-improved traces transfer better than raw traces? |

**RL Conditions:**
| Condition | Method | What We're Testing |
|---|---|---|
| `grpo_only` | GRPO from base model on GSM8K | Can RL discover reasoning without seeing examples? |
| `sft_then_grpo` | SFT on traces → GRPO refinement | Does RL refine or destroy distilled reasoning? |

## Findings

| Condition | GSM8K Accuracy | Training Data | Method |
|---|---|---|---|
| `baseline` | 67.0% | 50K answers | SFT |
| `sft_then_grpo` | 66.7% | 1K traces → 7.5K GSM8K | SFT → GRPO |
| `grpo_only` | 77.8% | 7.5K GSM8K | GRPO |
| `sft_traces` | 80.8% | 1K DeepSeek-R1 traces | SFT |
| **`re_distill`** | **86.0%** | 1K RL-verified traces | SFT |

For reference, Qwen3-1.7B base achieves 75.4% on GSM8K with 4-shot chain-of-thought prompting.

**What we found:**

- **Distillation works remarkably well.** 1,000 reasoning traces from DeepSeek-R1 (80.8%) outperform 50,000 answer-only examples (67.0%) by 13.8 percentage points. The model produces genuine `<think>` reasoning blocks with self-correction, backtracking, and verification — not just pattern mimicry. Quality of training signal dominates quantity.

- **Direct RL after distillation is destructive at 1.7B scale.** `sft_then_grpo` (66.7%) performs 14.1 points *worse* than `sft_traces` alone (80.8%). GRPO causes catastrophic forgetting of the distilled reasoning patterns. This is consistent with Niu et al.'s proof (arXiv:2601.07389) that SFT and RL objectives are mathematically non-decouplable, compounded by entropy collapse (arXiv:2602.02244), distribution mismatch (arXiv:2602.01058), and LoRA merge artifacts (arXiv:2410.21228).

- **Pure GRPO from base provides minimal benefit.** `grpo_only` (77.8%) gains only 2.4 points over the base model's 4-shot capability (75.4%). During training, 75-85% of steps had zero within-group reward variance (zero gradient). This confirms Yue et al.'s finding (arXiv:2504.13837) that RL redistributes sampling probability but does not create new reasoning capabilities, while "distillation can genuinely introduce new knowledge."

- **Re-distillation is the right approach.** `re_distill` (86.0%) outperforms direct distillation (80.8%) by 5.2 points using the same 1K examples, same model, same hyperparameters — the only difference is trace source. RL-verified traces from Open-R1 are higher quality because they've been filtered for mathematical correctness. This captures RL's benefits (better reasoning paths) without RL's risks (forgetting, instability, engineering complexity).

- **Trace quality follows a clear hierarchy.** Answer-only (67%) < base 4-shot (75%) < RL-redistributed (78%) < raw teacher traces (81%) < RL-verified traces (86%). The largest single jump (5.2 points) comes from correctness verification — **data quality filtering is the highest-leverage intervention** for reasoning distillation.

**Example: Distilled reasoning in action.** The `sft_traces` model on a GSM8K problem:

> **"A store sells apples for $2 each. If John buys 7 apples and pays with a $20 bill, how much change does he get?"**
> *sft_traces response:* `<think>` Okay, let me try to figure out this problem. So, John buys 7 apples, each costing $2, and he pays with a $20 bill. We need to find out how much change he gets. First, I know that the price per apple is $2. If he buys 7 apples, the total cost should be 7 multiplied by $2. Let me write that out: 7 * $2 = $14. John paid with a $20 bill. The change he should get back is $20 minus $14 = $6. Let me double-check: 7 apples at $2 each is $14. $20 minus $14 is $6. `</think>` John gets \boxed{6} dollars in change.

The model shows genuine step-by-step reasoning with self-verification — not template matching.

## Setup

```bash
git clone https://github.com/MaxiRuess/reasoning-distillation-grpo.git
cd reasoning-distillation-grpo

python -m venv venv
source venv/bin/activate
pip install -e .

# Modal setup (for cloud GPU training)
modal setup
modal secret create wandb-secret WANDB_API_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<token>
```

## Usage

```bash
# Step 1: Prepare datasets
python scripts/01_prepare_sft_data.py --condition sft_traces
python scripts/01_prepare_sft_data.py --condition re_distill

# Step 2: Train SFT conditions (Modal, L40S GPU)
modal run modal_train.py::train_sft --condition baseline
modal run modal_train.py::train_sft --condition sft_traces
modal run modal_train.py::train_sft --condition re_distill

# Step 3: Train GRPO conditions (Modal, H100 GPU with vLLM)
modal run modal_train.py::train_grpo --condition grpo_only
modal run modal_train.py::train_grpo --condition sft_then_grpo --sft-checkpoint /vol/outputs/sft_then_grpo

# Step 4: Evaluate on GSM8K
modal run modal_train.py::spot_check_gsm8k --model-path /vol/outputs/re_distill --num-problems 100

# Step 5: Quick sanity check (5 sample problems, visual inspection)
modal run modal_train.py::quick_test --model-path /vol/outputs/sft_traces
```

## Project Structure

```
├── configs/
│   └── config.yaml               # All hyperparameters for all conditions
├── src/
│   ├── data.py                    # Dataset loading & formatting (s1K, Open-R1, Orca, GSM8K)
│   ├── reward.py                  # Answer extraction & reward functions (binary + format)
│   ├── training.py                # Model loading, LoRA setup, SFT/GRPO trainer construction
│   └── evaluation.py             # GSM8K/MATH benchmarks, pass@k computation
├── scripts/                       # CLI entry points for each pipeline phase
├── modal_train.py                 # Modal cloud orchestration (SFT + GRPO + eval)
├── results/
│   ├── gsm8k_results.json         # Structured results
│   └── FINDINGS.md                # Detailed findings with literature comparison
├── paper/
│   └── literature_comparison.md   # Comprehensive literature review (30+ papers)
└── data/                          # Processed datasets (gitignored)
```

## Tech Stack

- **Training**: TRL (SFTTrainer, GRPOTrainer), PEFT (LoRA r=64, rsLoRA), Transformers
- **Base model**: Qwen3-1.7B in full bfloat16 (no quantization)
- **Data**: [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) (DeepSeek-R1 traces), [Open-R1 Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) (RL-verified traces), [Orca Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k), [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- **Evaluation**: GSM8K test set (~100 problems per condition, greedy decoding)
- **Compute**: Modal cloud — L40S for SFT (~$2/run), H100 for GRPO with vLLM colocate (~$40/run)
- **Tracking**: Weights & Biases

## Open Questions

**Does the SFT→RL degradation persist with better RL implementation?** Our GRPO setup faced significant engineering challenges (vLLM compatibility, dataset difficulty calibration, LoRA merge artifacts). The 14-point degradation may partly reflect suboptimal implementation rather than fundamental limitations. Entropy-preserving SFT (CurioSFT, arXiv:2602.02244), importance-weighted RL (PEAR, arXiv:2602.01058), or adapter composition methods could potentially avoid the catastrophic forgetting we observed. Whether any implementation can overcome the proven non-decouplability of SFT and RL objectives (arXiv:2601.07389) at sub-2B scale remains an open question.

**What is the optimal trace selection strategy for re-distillation?** Our `re_distill` condition uses the first math-verified correct solution from Open-R1. Alternative strategies — selecting the shortest correct trace, the most diverse trace set, or including negative examples (REDI, arXiv:2505.24850) — might further improve results. The 5.2-point gap between raw and verified traces suggests significant headroom from better data curation.

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reinforcement learning for reasoning in LLMs
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) — source of s1K-1.1 dataset
- [REDI](https://arxiv.org/abs/2505.24850) — reinforcement distillation with positive + negative traces
- [Non-decoupling of SFT and RL](https://arxiv.org/abs/2601.07389) — proof that SFT-then-RL causes degradation
- [Does RL Really Incentivize Reasoning?](https://arxiv.org/abs/2504.13837) — RL redistributes, doesn't create new capabilities
- [Dr. GRPO](https://arxiv.org/abs/2503.20783) — critical analysis of R1-Zero-like training
- [LoRA: Illusion of Equivalence](https://arxiv.org/abs/2410.21228) — LoRA's intruder dimensions degrade sequential adaptation
- [QuAILoRA](https://arxiv.org/abs/2410.14713) — quantization degrades fine-tuning quality
- [rsLoRA](https://arxiv.org/abs/2312.03732) — rank-stabilized LoRA scaling

## License

MIT
