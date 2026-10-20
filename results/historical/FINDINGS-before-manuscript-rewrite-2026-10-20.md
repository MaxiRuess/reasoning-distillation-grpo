# Findings from the reasoning-distillation study

## Completed sampled inference follow-up — October 20

The [follow-up](../docs/experiments/inference-extension-results-2026-10-20.md) evaluated the unchanged student and three existing trace checkpoints on the same 128 questions at 2K/4K/8K with recommended thinking-mode sampling. Trace mean accuracy is 80.21%/84.64%/86.20%, versus 67.97%/85.94%/87.50% unchanged. The primary 8K difference is −1.30 percentage points (95% conditional paired item interval −5.73 to +2.86; seed-difference SD 0.45 points). An 8K advantage is not established; the interval does not establish equivalence. Changing batch membership changes RNG streams, so this is not an isolated causal length intervention.

Native no-thinking at 2K scores 77.34% greedy and 78.91% sampled, with about 239/237 mean tokens. On these exact questions the original answer-only SFT checkpoints average 6.51% greedy. Switching off the native thinking channel therefore does not reproduce that recipe's collapse. Sampled trace SFT averages 80.21% at 2K with 994 tokens; comparisons with native no-thinking are descriptive because sampling parameters differ.

In a post hoc audit of the original 256-question outputs, 99 of 101 trace-only-correct question–seed pairs involve an output-limited unchanged response. Conversely, 29 of 41 unchanged-only-correct pairs involve an output-limited trace response. Among outputs without unfinished thinking, accuracies are 93.26% unchanged and 93.46% pooled trace; actual EOS-conditioned accuracies are 97.83% and 93.60%. These are different model-selected subsets, not a causal test of reasoning ability.

All 29 tasks and 1,792 new raw responses passed their checks. The app is stopped and the live container inventory is empty. Estimated resource-time cost is $13.34, or $15.47 with startup and incidental allowances, within the separate $30 cap; this is not a reconciled bill. The full suite passes 84 tests plus 11 subtests. No training, second benchmark or paid retries were run. Original evidence remains unchanged. See the [closing record](recovery/inference-extension-outcome.json). No further GPU spending is authorized.

## Completed matched-question distillation study — October 20

The [three-seed study](../docs/experiments/distillation-results-2026-10-20.md) completed six 500-question SFT runs, fourteen evaluations and 2,240 saved responses. At a 2K output limit, teacher-trace SFT averages 78.1% versus 70.3% for the unchanged student. The descriptive trace-minus-unchanged difference is +7.81 percentage points, with a 95% paired item-bootstrap interval of +3.39 to +12.50 conditional on the three trained seeds. This is a secondary comparison; the predeclared primary comparison remains trace versus answer-only SFT.

Answer-only SFT averages 6.25%. The primary trace-minus-answer difference is +71.88 points (conditional paired item interval +67.06 to +76.56). The answer-only recipe supervises template-inserted empty thinking blocks and collapses. Its large deficit describes the executed recipe and does not establish a general distillation benefit over an optimized answer-only method.

On the same preselected 64-question 4K check, trace SFT averages 83.85% versus 87.50% unchanged: −3.65 points (−8.33 to +1.04). No advantage over the unchanged student is demonstrated at that ceiling; this is not equivalence or evidence of a general reversal. Questions and updates are matched across training arms, while tokens and compute differ. One student, one training sample and one hyperparameter recipe limit generalization.

All experiment apps are stopped, with zero tasks and an empty final container inventory. Estimated new resource-time cost is $7.01 ($8.58 including extra startup allowance); conservative combined accounting including earlier-run reserves and all allowances is $11.10 against $20. These are not reconciled bills. All strict scores reproduce on CPU and raw hashes passed separate-container checks; 79 tests plus 11 subtests pass. The [closing record](recovery/distillation-study-close.json) indexes the evidence. No further GPU work is authorized.

## Earlier audit and historical evidence

The historical recipe rankings are preserved in `gsm8k_results.json`, explicitly marked unvalidated. The starting checkpoint is post-trained Qwen3-1.7B, and Orca Math contains worked reasoning solutions.

The recovered outcome files reconcile all six published counts. Their paired analysis is in `recovery/historical-paired-analysis.json`; it uses the legacy correctness flags, not newly generated or rescored responses. The archived files do not contain raw responses.

The strongest implementation defect is the missing SFT contribution when reloading the combined GRPO adapter. The original SFT checkpoint is available and has a nonzero update. Full-size reconstruction and a separately frozen 256-question re-evaluation are now complete; historical full-benchmark correction remains pending.

The Open-R1 recipe's +4.70-point historical advantage over s1K is statistically resolved at the question level, but does not isolate correctness filtering. The s1K-versus-starting-checkpoint +1.67-point difference is unresolved in the recovered paired comparison. Training-seed variability was not measured historically.

No result establishes reasoning emergence, an inevitable SFT/RL conflict, entropy collapse, faithful self-correction, or a model-size threshold. Those claims have been removed from the revised paper and README. See `docs/remediation/STATUS.md` for fixes, evidence and outstanding paid validation.

Pilot audits exposed prose-scorer failures, a GRPO token-counter defect and an unweighted vendor reward log. The pilot repairs passed 53 CPU tests and a fresh six-update/32-question GPU check. New arms share an explicit boxed-answer request; the scorer reads that field and reports format/unfinished-thinking failures separately. Integer gold values compare exactly. Tokens, weighted rewards, parent lineage, changed model weights and independent persistence hashes were verified. `recovery/pilot-final-outcome.json` indexes the evidence. Old prose policies and raw results remain unchanged and are not directly comparable as a training improvement. Full-study spending is still unapproved.

## Completed checkpoint-loading ablation

The separately authorized $20 experiment completed all three arms on the same 256 questions: exact SFT parent 191/256 (74.61%), SFT omitted 168/256 (65.63%), correct reconstruction 184/256 (71.88%). Restoration gains 6.25 percentage points under the predeclared strict answer contract, paired 95% interval +0.78 to +11.72, exact McNemar p=0.03648. Accepting four repeated identical correct boxes in a separate post hoc sensitivity yields +4.69 points, interval -0.39 to +9.77, p=0.11140. The corrected-versus-SFT difference is -2.73 points, interval -7.03 to +1.56, and does not resolve a harmful RL effect.

All 768 raw responses were saved, independently persisted and reproduced through CPU scoring. The exact SFT parent is used; all full exports have distinct weight hashes and exact reload-probe logits. Model-label-hidden final-field review covered 246 responses from 82 disagreement question groups, without changing primary scores. No independent human review was performed. Closing reported additional metered usage was $1.86 ($0 billed after credits in the snapshot); billing may adjust. All experiment apps are stopped. The expanded suite passes 65 tests plus 11 subtests.

This is a single-run, prompt- and budget-dependent reproducibility result. The original base revision and historical reward defects remain limitations. See [the results report](../docs/experiments/checkpoint-ablation-results-2026-10-19.md) and `recovery/checkpoint-ablation-outcome.json`. Original historical scores remain unchanged.
