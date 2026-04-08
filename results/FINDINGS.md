# Reasoning Trace Distillation: Can Small Models Learn to Think?

## Research Question

For sub-2B parameter models, is reasoning trace distillation sufficient for math reasoning — or does reinforcement learning provide additional benefit? And if RL helps, what is the right way to transfer its benefits to small models?

## Experimental Setup

- **Student model**: Qwen3-1.7B (base, not instruct) with LoRA r=64, rsLoRA, full bfloat16
- **Benchmark**: GSM8K test set (grade school math, full 1,319 problems per condition, greedy decoding)
- **Compute**: Modal cloud (L40S for SFT, H100 for GRPO with vLLM colocate)

### Training Conditions

| Condition | Method | Training Data |
|---|---|---|
| `baseline` | SFT on answer-only data | 50K Orca Math examples |
| `sft_traces` | SFT on reasoning traces | 1K DeepSeek-R1 traces (s1K-1.1) |
| `re_distill` | SFT on RL-verified traces | 1K Open-R1 math-verified traces |
| `grpo_only` | GRPO from base model | 7.5K GSM8K with binary + format rewards |
| `sft_then_grpo` | SFT on traces, then GRPO | 1K traces → 7.5K GSM8K GRPO |

## Results

| Condition | GSM8K Accuracy | N |
|---|---|---|
| `baseline` | **69.2%** | 1,319 |
| `sft_then_grpo` | **69.7%** | 1,319 |
| Base model | **72.2%** | 1,319 |
| `sft_traces` | **73.8%** | 1,319 |
| `grpo_only` | **75.7%** | 1,319 |
| `re_distill` | **78.5%** | 1,319 |

Evaluated on the full GSM8K test set (1,319 problems, greedy decoding, ±2.5pp confidence intervals). For reference, Qwen reports 75.4% with 4-shot chain-of-thought prompting (Qwen3 Technical Report, arXiv:2505.09388).

## Key Findings

### Finding 1: Re-distillation from RL-verified traces is the best approach

`re_distill` (78.5%) outperforms all other conditions — beating `grpo_only` (75.7%) by 2.8 points and `sft_traces` (73.8%) by **4.7 percentage points** using the same amount of data (1K examples), the same model, and identical hyperparameters. The only difference is the source of traces — math-verified correct solutions from the Open-R1 project (RL-trained model outputs) versus raw DeepSeek-R1 traces from s1K-1.1. With ±2.5pp confidence intervals, this gap is statistically significant.

This demonstrates that **the right way to get RL benefits into a small model is through better training data, not through direct RL optimization.** Let a larger model do the RL, verify the outputs for correctness, then distill the verified traces into the small model.

The REDI paper (arXiv:2505.24850) established this approach at scale: Qwen-REDI-1.5B achieves 83.1% on MATH-500 using 131K Open-R1 traces, matching DeepSeek's 800K-trace distillation. Our result extends this finding to the extreme low-data regime (1K traces) and confirms the mechanism — RL-verified traces are higher quality because they've been filtered for correctness, removing the noise and errors present in raw teacher traces.

### Finding 2: Direct GRPO marginally outperforms raw distillation

`grpo_only` (75.7%) outperforms `sft_traces` (73.8%) by 1.9 percentage points — a gap within our ±2.5pp confidence intervals and thus not conclusive, but notable because it challenges the assumption that distillation dominates RL at small scale. Both conditions exceed the base model (72.2%), confirming that both training signals provide genuine improvement.

During GRPO training, the reward signal was sparse: `frac_reward_zero_std` was 0.75-0.85 (75-85% of steps had zero gradient). Despite this, GRPO achieved comparable or slightly better results than distillation, suggesting that even sparse RL signal is sufficient for grade-school math at 1.7B scale.

### Finding 3: GRPO after distillation degrades performance

`sft_then_grpo` (69.7%) performs **4.1 percentage points worse** than `sft_traces` alone (73.8%), falling below the base model to near-baseline performance. Adding GRPO refinement to a distilled model is harmful at the 1.7B scale.

This finding is supported by theoretical proof. Niu et al. (arXiv:2601.07389) demonstrate on Qwen3-0.6B that SFT and RL objectives are mathematically non-decouplable: RL training necessarily increases SFT loss. Three compounding mechanisms explain the degradation:

1. **Distribution mismatch**: Zhang et al. (arXiv:2602.01058) show that stronger SFT checkpoints can significantly underperform weaker ones after RL, because the offline SFT data distribution mismatches on-policy RL rollouts.

2. **Entropy collapse**: Wang et al. (arXiv:2602.02244) demonstrate that standard SFT reduces generation diversity, leaving RL with a narrowed solution space. Our GRPO training showed 75-85% of steps with zero within-group reward variance, consistent with an entropy-collapsed policy.

3. **LoRA merge artifacts**: Our pipeline used `PeftModel.merge_and_unload()` to merge SFT LoRA weights before applying a fresh LoRA for GRPO. Shuttleworth et al. (arXiv:2410.21228) show LoRA produces "intruder dimensions" that degrade sequential adaptation.

This degradation is not widely reported in prior work because most SFT-then-RL pipelines use models at 7B+ scale, where model capacity buffers against forgetting. At 7B, AceReason-Nemotron (arXiv:2505.16400) shows RL improves distilled models by +14.6%. The critical insight is **scale-dependent**: SFT-then-RL is complementary at sufficient model size, but harmful at 1.7B.

### Finding 4: Distillation is data-efficient but not dominant

`sft_traces` (73.8%) outperforms `baseline` (69.2%) by 4.6 percentage points despite training on **50x fewer examples** (1,000 vs 50,000). The quality of the training signal — reasoning traces with `<think>` tags from DeepSeek-R1 — matters more than quantity. However, GRPO achieves comparable results (75.7%) without any reasoning traces, demonstrating that distillation is not the only path to improvement at this scale.

Interestingly, `baseline` (69.2%) performs *below* the unmodified base model (72.2%). Rajani et al. (arXiv:2507.10616) provide a mechanism: SFT modifies mid-layer MLPs aggressively, *replacing* existing capabilities. Answer-only SFT may teach the model to skip reasoning steps it would otherwise perform in-context, degrading its latent chain-of-thought ability.

### Finding 5: Trace quality follows a clear hierarchy

Our five conditions reveal a hierarchy of training signal quality:

| Training Signal | GSM8K | Quality Source |
|---|---|---|
| Answer-only (50K Orca Math) | 69.2% | No reasoning process |
| Base model (no training) | 72.2% | Zero-shot greedy |
| Raw teacher traces (1K s1K-1.1) | 73.8% | DeepSeek-R1 reasoning patterns |
| GRPO from base (7.5K GSM8K) | 75.7% | RL probability redistribution |
| RL-verified traces (1K Open-R1) | 78.5% | Filtered for correctness |

The largest single jump (4.7 points) comes from verifying trace correctness — suggesting that **data quality filtering is the single highest-leverage intervention** for reasoning distillation.

## Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|---|---|---|
| **H1**: Distillation sufficient for easy problems | **Supported** | 73.8% on GSM8K from 1K traces; re_distill pushes to 78.5% |
| **H2**: RL refines distilled reasoning | **Falsified** | 69.7% < 73.8% (−4.1pp); but re-distillation (78.5%) shows RL benefits transfer through data |
| **H3**: Pure RL needs scale | **Not supported** | GRPO-Only (75.7%) outperforms SFT-Traces (73.8%); RL is viable at 1.7B |
| **H4**: Trace compression emerges with RL | **Not observed** | GRPO degraded rather than compressed traces |

## Limitations

1. **Different trace sources**: `sft_traces` uses s1K-1.1 (curated 1K problems) while `re_distill` uses Open-R1 (filtered from 220K). The improvement may partly reflect different problem distributions, not solely trace quality.

2. **GRPO implementation challenges**: The GRPO results reflect significant engineering constraints (vLLM compatibility, memory management, dataset difficulty calibration). A better-optimized GRPO setup might produce different results.

3. **LoRA merge confound**: The `sft_then_grpo` degradation may be partly a LoRA merge artifact rather than a fundamental SFT-RL conflict. A controlled experiment with full fine-tuning or adapter composition would isolate this.

4. **Contamination**: GRPO-Only trains on the GSM8K training split and is evaluated on GSM8K test. While these are standard splits, the model learns the GSM8K distribution during training, which may inflate its score relative to conditions trained on different distributions.

5. **Single benchmark**: GSM8K measures grade-school math only. Results may not generalize to harder math (MATH, AIME) or other reasoning domains (code, logic).

6. **Greedy decoding only**: All results use temperature=0. Pass@k evaluation would reveal whether GRPO improves sampling efficiency even if not peak capability.

## Conclusion

For small language models (1-2B parameters), **data quality is the most effective lever for mathematical reasoning.**

Our five-condition experiment reveals a clear hierarchy: RL-verified reasoning traces (78.5%) > GRPO from base (75.7%) > raw teacher traces (73.8%) > base model (72.2%) > answer-only training (69.2%). The largest gains come from filtering traces for mathematical correctness before distillation.

Notably, direct GRPO (75.7%) marginally outperforms raw distillation (73.8%), challenging the assumption that distillation dominates RL at small scale. However, applying GRPO *after* distillation degrades performance by 4.1 points, consistent with theoretical work on the non-decouplability of SFT and RL objectives.

The practical recommendation for practitioners working with sub-2B models is clear: **use re-distillation.** Let a larger RL-trained model generate solutions, verify them for correctness, and SFT-distill the verified traces into your small model. This captures the benefits of RL (higher-quality reasoning paths) without the risks of direct RL optimization (forgetting, instability, engineering complexity). One thousand verified traces and a single SFT run (~$2 on Modal) outperform all other approaches at a fraction of the compute cost.
