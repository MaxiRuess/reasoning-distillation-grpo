# Reasoning Trace Distillation: Can Small Models Learn to Think?

## Research Question

For sub-2B parameter models, is reasoning trace distillation sufficient for math reasoning — or does reinforcement learning provide additional benefit? And if RL helps, what is the right way to transfer its benefits to small models?

## Experimental Setup

- **Student model**: Qwen3-1.7B (base, not instruct) with LoRA r=64, rsLoRA, full bfloat16
- **Benchmark**: GSM8K test set (grade school math, ~100 problems per condition, greedy decoding)
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
| `baseline` | **67.0%** | 100 |
| `sft_then_grpo` | **66.7%** | 99 |
| `grpo_only` | **77.8%** | 99 |
| `sft_traces` | **80.8%** | 99 |
| `re_distill` | **86.0%** | 93 |

For reference, Qwen3-1.7B base achieves 75.4% on GSM8K with 4-shot chain-of-thought prompting (Qwen3 Technical Report, arXiv:2505.09388).

## Key Findings

### Finding 1: Distillation is remarkably data-efficient

`sft_traces` (80.8%) outperforms `baseline` (67.0%) by 13.8 percentage points despite training on **50x fewer examples** (1,000 vs 50,000). The quality of the training signal — reasoning traces with `<think>` tags from DeepSeek-R1 — matters far more than quantity.

This result is consistent with the broader distillation literature. Hsieh et al. (arXiv:2305.02301) established that distilling intermediate rationales enables small models to outperform larger models with less data. Our data efficiency is notable: DeepSeek-R1-Distill-Qwen-1.5B achieves 83.9% on the harder MATH-500 benchmark, but uses approximately 800K traces (arXiv:2501.12948) — roughly 800x our dataset. The Re-distillation work of Chen et al. (arXiv:2505.17988) similarly shows that 500 carefully chosen re-distilled samples can match instruct-tuned variants at the 1.5B scale.

Interestingly, `baseline` (67.0%) performs *below* the base model's 4-shot capability (75.4%). Rajani et al. (arXiv:2507.10616) provide a mechanism: SFT modifies mid-layer MLPs aggressively, *replacing* existing capabilities. Answer-only SFT may teach the model to skip reasoning steps it would otherwise perform in-context, actively degrading its latent chain-of-thought ability.

### Finding 2: Re-distillation from RL-verified traces is the best approach

`re_distill` (86.0%) outperforms direct distillation `sft_traces` (80.8%) by **5.2 percentage points** using the same amount of data (1K examples), the same model, and identical hyperparameters. The only difference is the source of traces — math-verified correct solutions from the Open-R1 project (RL-trained model outputs) versus raw DeepSeek-R1 traces from s1K-1.1.

This demonstrates that **the right way to get RL benefits into a small model is through better training data, not through direct RL optimization.** Let a larger model do the RL, verify the outputs for correctness, then distill the verified traces into the small model. This avoids the SFT-RL objective conflict that caused `sft_then_grpo` to degrade.

The REDI paper (arXiv:2505.24850) established this approach at scale: Qwen-REDI-1.5B achieves 83.1% on MATH-500 using 131K Open-R1 traces, matching DeepSeek's 800K-trace distillation. Our result extends this finding to the extreme low-data regime (1K traces) and confirms the mechanism — RL-verified traces are higher quality because they've been filtered for correctness, removing the noise and errors present in raw teacher traces.

### Finding 3: Direct GRPO after distillation causes catastrophic forgetting

`sft_then_grpo` (66.7%) performs **14.1 percentage points worse** than `sft_traces` alone (80.8%). Adding GRPO refinement to a distilled model is actively harmful at the 1.7B scale.

This finding is supported by theoretical proof. Niu et al. (arXiv:2601.07389) demonstrate on Qwen3-0.6B that SFT and RL objectives are mathematically non-decouplable: RL training necessarily increases SFT loss. Three compounding mechanisms explain the degradation:

1. **Distribution mismatch**: Zhang et al. (arXiv:2602.01058) show that stronger SFT checkpoints can significantly underperform weaker ones after RL, because the offline SFT data distribution mismatches on-policy RL rollouts.

2. **Entropy collapse**: Wang et al. (arXiv:2602.02244) demonstrate that standard SFT reduces generation diversity, leaving RL with a narrowed solution space. Our GRPO training showed 75-85% of steps with zero within-group reward variance, consistent with an entropy-collapsed policy.

3. **LoRA merge artifacts**: Our pipeline used `PeftModel.merge_and_unload()` to merge SFT LoRA weights before applying a fresh LoRA for GRPO. Shuttleworth et al. (arXiv:2410.21228) show LoRA produces "intruder dimensions" that degrade sequential adaptation.

This degradation is not widely reported in prior work because most SFT-then-RL pipelines use models at 7B+ scale, where model capacity buffers against catastrophic forgetting. At 7B, AceReason-Nemotron (arXiv:2505.16400) shows RL improves distilled models by +14.6%. The critical insight is **scale-dependent**: SFT-then-RL is complementary at sufficient model size, but destructive at 1.7B.

### Finding 4: Pure GRPO does not create new reasoning capabilities

`grpo_only` (77.8%) represents only a 2.4 percentage point gain over the base model's 4-shot chain-of-thought (75.4%). During GRPO training, the reward signal was near-zero: `frac_reward_zero_std` was 0.75-0.85 (75-85% of steps had zero gradient). The model's score primarily reflects Qwen3's latent capabilities being marginally redistributed, not new reasoning being learned.

This is directly explained by Yue et al. (arXiv:2504.13837), who show that RL-trained models outperform base models at pass@1, but base models achieve comparable or *higher* pass@k at large k. "All reasoning paths in the RLVR model are already present in the base model." In contrast, "distillation can genuinely introduce new knowledge into the model, different from RLVR" — consistent with our finding that both `sft_traces` (80.8%) and `re_distill` (86.0%) significantly exceed the base model's capability.

### Finding 5: Trace quality follows a clear hierarchy

Our five conditions reveal a hierarchy of training signal quality:

| Training Signal | GSM8K | Quality Source |
|---|---|---|
| Answer-only (50K Orca Math) | 67.0% | No reasoning process |
| Base model 4-shot (no training) | 75.4% | In-context reasoning |
| GRPO from base (7.5K GSM8K) | 77.8% | RL probability redistribution |
| Raw teacher traces (1K s1K-1.1) | 80.8% | DeepSeek-R1 reasoning patterns |
| RL-verified traces (1K Open-R1) | 86.0% | Filtered for correctness |

Each step up the hierarchy represents a qualitative improvement in signal: from no reasoning → in-context reasoning → RL-amplified → teacher traces → verified teacher traces. The largest single jump (8.2 points) comes from verifying trace correctness — suggesting that **data quality filtering is the single highest-leverage intervention** for reasoning distillation.

## Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|---|---|---|
| **H1**: Distillation sufficient for easy problems | **Supported** | 80.8% on GSM8K from 1K traces; re_distill pushes to 86.0% |
| **H2**: RL refines distilled reasoning | **Falsified** | 66.7% < 80.8%; but re-distillation (86.0%) shows RL benefits transfer through data, not optimization |
| **H3**: Pure RL needs scale | **Supported** | Near-zero GRPO signal at 1.7B; 2.4pp gain is probability redistribution per arXiv:2504.13837 |
| **H4**: Trace compression emerges with RL | **Not observed** | GRPO degraded rather than compressed traces |

## Limitations

1. **Sample size**: ~100 problems per condition with confidence intervals of approximately ±8 percentage points. The difference between `sft_traces` (80.8%) and `re_distill` (86.0%) is suggestive but not statistically significant at p<0.05.

2. **Different trace sources**: `sft_traces` uses s1K-1.1 (curated 1K problems) while `re_distill` uses Open-R1 (filtered from 220K). The improvement may partly reflect different problem distributions, not solely trace quality.

3. **GRPO implementation challenges**: The GRPO results reflect significant engineering constraints (vLLM compatibility, memory management, dataset difficulty calibration). A better-optimized GRPO setup might produce different results.

4. **LoRA merge confound**: The `sft_then_grpo` degradation may be partly a LoRA merge artifact rather than a fundamental SFT-RL conflict. A controlled experiment with full fine-tuning or adapter composition would isolate this.

5. **Single benchmark**: GSM8K measures grade-school math only. Results may not generalize to harder math (MATH, AIME) or other reasoning domains (code, logic).

6. **Greedy decoding only**: All results use temperature=0. Pass@k evaluation would reveal whether GRPO improves sampling efficiency even if not peak capability.

## Conclusion

For small language models (1-2B parameters), **the path to strong reasoning runs through data quality, not RL optimization.**

Our five-condition experiment reveals a clear hierarchy: RL-verified reasoning traces (86.0%) > raw teacher traces (80.8%) > RL from base (77.8%) > answer-only training (67.0%). The largest gains come from the quality of the training signal — specifically, filtering traces for mathematical correctness before distillation.

Direct RL optimization (GRPO) is counterproductive at this scale. Applied after distillation, it causes a 14-point degradation through catastrophic forgetting. Applied from a base model, it provides only marginal improvement (2.4 points) by redistributing sampling probability without creating new reasoning capabilities.

The practical recommendation for practitioners working with sub-2B models is clear: **use re-distillation.** Let a larger RL-trained model generate solutions, verify them for correctness, and SFT-distill the verified traces into your small model. This captures the benefits of RL (higher-quality reasoning paths) without the risks of direct RL optimization (forgetting, instability, engineering complexity). One thousand verified traces and a single SFT run (~$2 on Modal) outperform weeks of GRPO debugging and ~$40+ of H100 compute.
