# Sampled inference follow-up results — October 20, 2026

The frozen inference-only package completed 14 GPU evaluations and 14 separate persistence checks, after CPU preflight. It reused the original student and three trace checkpoints; no training was run. All 1,792 new raw responses passed hash, identity and CPU strict-score checks. The original experiment remains separate.

## Primary 8K comparison

Trace checkpoints average **86.20%**, versus **87.50%** unchanged. The mean difference is **-1.30 percentage points**, with a 95% conditional paired item-bootstrap interval of **-5.73 to +2.86** and seed-difference SD 0.45 points. The 8K numerical comparison establishes neither an accuracy advantage nor equivalence.

The interval conditions on three trained checkpoints and one decoding seed. There are 128 common questions, not 384 independent observations. The follow-up was designed after seeing the original findings and reuses previously evaluated questions. Eight thousand output tokens is still a constrained inference setting.

## Secondary sensitivity to answer formatting

The unchanged student's strict one-box accuracy at 8K is **83.59%**, versus **86.20%** for the trace mean. The strict trace-minus-unchanged difference is **+2.60 points** (95% conditional item interval **-2.60 to +8.07**). Five unchanged-student responses have repeated, numerically consistent correct boxes: they pass the numerical primary rule and fail the strict rule. The point estimate changes sign, while neither rule establishes an 8K advantage. This saved-output secondary analysis does not replace the primary endpoint.

## Same-question sampled results

| Checkpoint | 2K accuracy | 4K accuracy | 8K accuracy | 8K EOS /128 | 8K mean tokens |
|---|---:|---:|---:|---:|---:|
| base | 67.97% | 85.94% | 87.50% | 125 | 2012.4 |
| trace_42 | 80.47% | 82.81% | 86.72% | 126 | 1511.3 |
| trace_43 | 82.03% | 83.59% | 85.94% | 126 | 1575.4 |
| trace_44 | 78.12% | 87.50% | 85.94% | 124 | 1473.3 |

Thinking-mode sampling uses temperature 0.6, top-p 0.95, top-k 20 and disabled min-p, with seed 20261023. Effective batches are 32/16/8 across the limits. Canonical batch membership changes the RNG streams, so these are separately sampled trajectories, not truncations of a common rollout. The curve does not isolate a causal output-length effect. EOS, unfinished thinking and final-field failures are distinct.

## No-thinking controls and the answer-only collapse

| Original student at 2K, thinking disabled | Correct /128 | Numerical accuracy | Mean tokens |
|---|---:|---:|---:|
| greedy | 99 | 77.34% | 239.0 |
| sampled | 101 | 78.91% | 237.2 |

On precisely these 128 questions, the original answer-only checkpoints average **6.51%** under greedy decoding. The original unchanged thinking-mode greedy score is **74.22%**. Switching off the native thinking channel therefore does not reproduce the answer-only recipe's collapse. The control does not identify knowledge forgetting or the cause of training failure; native no-thinking outputs can still contain worked steps in the final response.

The sampled no-thinking policy uses temperature 0.7/top-p 0.8, while sampled thinking uses 0.6/0.95. Its 2K accuracy is 78.91% with 237.2 mean generated tokens, versus the trace mean of 80.21% and 993.8 tokens. This compares complete model-and-policy recipes and is descriptive; one run does not establish equivalence or general policy ordering.

## Original 256-question completion diagnostic

This new post hoc diagnostic uses only saved original outputs: 99 of 101 trace-only-correct question–seed pairs involve an unchanged-student response hitting the 2K limit. Conversely, 29 of 41 unchanged-only-correct pairs involve a trace response hitting the limit. Those pairs reuse questions across seeds. Among outputs without unfinished thinking, accuracy is 93.26% unchanged versus 93.46% pooled trace. With actual EOS termination, the corresponding rates are 97.83% and 93.60%. Conditional subsets select different questions and cannot isolate reasoning ability.

## Cost and limits

New client elapsed time priced at resource caps is estimated at **$13.34** (6.14 aggregate GPU-call hours). Adding separate startup allowance gives **$14.47**; adding the incidental allowance gives **$15.47**. The latter allowances can overlap observed time or remain unused. These are not reconciled provider invoices. Frozen reservations totaled $29.0453128 against the separate $30 cap. Credits and public teacher-generation costs are excluded from the budget justification and student-run accounting, respectively.

One student, one training sample, one benchmark, one decoding seed and changing batch size limit the conclusions. No general reasoning-capacity gain, causal compression mechanism or publication acceptance is established. Broader training and benchmark extensions were not run.

## Evidence

- Frozen proposal: `results/proposals/inference-extension/f1b156f14754100c186bc396`.
- [Completed analysis](../../results/proposals/inference-extension/f1b156f14754100c186bc396/collected/analysis.json), with per-job raw hashes and both scoring endpoints.
- [Paper source](../../paper/paper.tex), [PDF](../../assets/paper.pdf) and [follow-up figure](../../assets/inference-extension-2026-10-20.png).
- [Compute estimate](../../paper/inference-extension-cost.json) and [original completion diagnostic](../../paper/original-completion-diagnostic.json).
- [Strict secondary 8K analysis](../../paper/inference-extension-secondary.json), derived from the same saved raw outputs.
- [Protocol and execution scope](inference-extension-2026-10-20.md).
