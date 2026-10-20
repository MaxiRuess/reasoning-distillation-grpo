# Findings from the reasoning-trace distillation study

The completed experiments show shorter observed responses and a numerical accuracy advantage for trace SFT at a 2K output limit. They do **not establish an accuracy advantage at 8K**. Native no-thinking inference also avoids the severe failure of the answer-only SFT recipe. The evidence supports a case study of output-budget sensitivity in an already post-trained Qwen3-1.7B student.

## Numerical accuracy depends on the evaluated output budget

The sampled follow-up evaluates the unchanged student and three existing trace checkpoints on the same 128 GSM8K questions. Thinking-mode sampling uses temperature 0.6, top-p 0.95 and top-k 20, with one fixed decoding seed.

| Output limit | Unchanged accuracy | Trace mean accuracy | Trace mean minus unchanged |
|---|---:|---:|---:|
| 2,048 tokens | 67.97% | 80.21% | +12.24 pp |
| 4,096 tokens | 85.94% | 84.64% | −1.30 pp |
| 8,192 tokens | 87.50% | 86.20% | −1.30 pp |

The follow-up's primary comparison is at 8K: **−1.30 percentage points**, with a 95% paired item-bootstrap interval of **−5.73 to +2.86** and training-seed difference SD of 0.45 points. This interval establishes neither an advantage nor equivalence. It conditions on the three trained checkpoints and the one decoding seed; there are 128 shared questions, not 384 independent observations. The 2K/4K contrasts are descriptive.

This follow-up was designed after reviewing the original experiment and reuses previously evaluated questions. Effective batch sizes change from 32 to 16 to 8 across limits, which also changes random-number streams. These are separately sampled trajectories, not truncated versions of a shared 8K rollout. The curve does not isolate a causal effect of output length alone.

## Answer-format scoring changes the 8K point estimate

The numerical primary rule accepts repeated, numerically identical final boxes. The strict secondary rule requires exactly one box. Both were retained before follow-up generation.

Five unchanged-student responses at 8K contain repeated, numerically consistent correct boxes. Its score is therefore 112/128 (87.50%) numerically and 107/128 (83.59%) under the strict rule. The trace scores are unchanged. Under the strict secondary rule, the trace-minus-unchanged difference is **+2.60 points**, with a conditional item interval of **−2.60 to +8.07**.

The point estimate changes sign, while neither scoring rule establishes an 8K advantage. The primary rule and original scores remain unchanged. The [secondary calculation](../paper/inference-extension-secondary.json) uses the same saved raw responses.

## Short-limit gains coincide with more frequent completion

In the sampled follow-up at 2K, 69.53% of unchanged-student responses terminate at an end-of-sequence token, versus 84.90% across trace checkpoints. Mean generated lengths are 1,451 and 994 tokens, respectively. At 8K, EOS rates are 97.66% and 97.92%; mean lengths are 2,012 and 1,520 tokens. These counts include generated thinking and final-response tokens and remain subject to each output limit.

A post hoc check of the original 256-question greedy outputs gives a related observation: 99 of 101 question–seed pairs where a trace checkpoint alone is correct involve an output-limited unchanged response. Conversely, 29 of 41 unchanged-only-correct pairs involve an output-limited trace response. Questions repeat across the three seed comparisons.

Among original outputs without unfinished thinking, accuracies are 93.26% unchanged and 93.46% pooled trace. Among actual EOS-terminated outputs, they are 97.83% and 93.60%. These denominators select different questions for each model. They cannot establish equal reasoning ability or reveal how truncated responses would have answered if continued.

The trace-arm targets average 5,270 supervised tokens per example, yet student responses are shorter on GSM8K. This is an observed length change across different training and test tasks. No teacher-length ablation identifies a learned compression mechanism.

## Native no-thinking does not reproduce answer-only SFT collapse

| Unchanged student at 2K on the same 128 questions | Correct | Accuracy | Mean generated tokens |
|---|---:|---:|---:|
| Native no-thinking, greedy | 99/128 | 77.34% | 239 |
| Native no-thinking, sampled | 101/128 | 78.91% | 237 |

Both native no-thinking runs terminate at EOS on all 128 questions. On these exact questions, the original answer-only checkpoints average 6.51% under greedy decoding. Merely disabling the native thinking channel does not reproduce that failure. The control does not identify knowledge forgetting or the cause of the training collapse; native no-thinking outputs can contain worked steps in their final responses.

Sampled native no-thinking reaches 78.91% with 237 mean tokens, compared with the trace mean of 80.21% and 994 tokens at 2K. These are descriptive results for different inference policies: no-thinking uses temperature 0.7/top-p 0.8, while thinking uses 0.6/0.95. The observations do not establish equivalence, general policy ordering, latency gains or proportional GPU-cost savings.

## The original training comparison remains a recipe-specific result

The original experiment trains three seeds per arm on the same 500 OpenR1 questions and final answers from DeepSeek-R1 generations, with two epochs and 64 optimizer updates. The trace arm additionally receives the teacher's thinking. Its supervised-token exposure and compute are much larger despite matched questions and updates.

On 256 GSM8K questions at 2K with greedy thinking, trace SFT averages 78.1%, unchanged 70.3%, and answer-only SFT 6.25%. The original primary trace-minus-answer difference is +71.88 points (conditional item interval +67.06 to +76.56). The secondary trace-minus-unchanged difference is +7.81 points (+3.39 to +12.50). All 768 answer-only outputs contain empty visible thinking blocks; their training targets supervise template-inserted empty thinking blocks. The failed recipe gives little evidence about trace SFT relative to a well-functioning answer-only method.

On the original preselected 64-question 4K check, trace SFT averages 83.85% versus 87.50% unchanged (−3.65 points, interval −8.33 to +1.04). These original greedy results are distinct from the sampled 128-question follow-up.

## Scope, evidence and compute

The study covers one student, one 500-question training sample, one answer-only recipe and one benchmark. GSM8K had been used in the repository before this study. Dataset overlap screening does not establish absence of pretraining contamination, and the 8K ceiling is below Qwen's recommended 32,768-token allowance. Final-answer scores do not assess the correctness or faithfulness of intermediate reasoning.

The original experiment retains 2,240 responses; the follow-up retains 1,792. Persistence hashes and local CPU strict scores were verified. All 29 follow-up tasks succeeded, with no training or paid retries. The app stopped with zero tasks and no remaining containers. Estimated follow-up resource-time cost is $13.34, or $15.47 with all extra allowances, within its separate $30 cap; this is not a reconciled provider invoice. This manuscript revision uses CPU only and adds no GPU work.

- [Sampled follow-up analysis](proposals/inference-extension/f1b156f14754100c186bc396/collected/analysis.json) and [results report](../docs/experiments/inference-extension-results-2026-10-20.md).
- [Original distillation results](../docs/experiments/distillation-results-2026-10-20.md) and [completion diagnostic](../paper/original-completion-diagnostic.json).
- [Completed compute record](recovery/inference-extension-outcome.json), [paper source](../paper/paper.tex) and [current PDF](../assets/paper.pdf).
- [Earlier findings and historical audit context](historical/FINDINGS-before-manuscript-rewrite-2026-10-20.md), including the separate checkpoint-loading ablation. Those legacy results are not pooled with the experiments above.
