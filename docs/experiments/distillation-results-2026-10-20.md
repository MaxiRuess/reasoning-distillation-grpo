# Completed bounded reasoning-distillation study — October 20, 2026

The authorized study is complete and all experiment apps are stopped. The final container inventory is empty. No additional GPU experiment, expansion or paid retry is planned. Estimated new resource-time cost is **$7.01**; the conservative total including startup and other allowances and earlier-run reserves is **$11.10 against the $20 envelope**. These figures are estimates, not a reconciled provider invoice.

The useful finding is bounded: teacher-trace SFT improved accuracy over the unchanged, already reasoning-capable student under a 2K output ceiling. The small 4K check did not demonstrate that advantage. The primary trace-versus-answer contrast is large because this particular answer-only recipe collapsed; it is not a general estimate of the benefit of reasoning distillation.

## Design actually executed

The [prospective protocol](distillation-study-2026-10-20.md) compares answer-only with explicit teacher-thinking supervision on the same 500 unique normalized OpenR1 questions and identical final boxed answers. Each arm has seeds 42/43/44, two epochs and 64 optimizer updates, using common seeded question minibatches. The student is pinned post-trained `Qwen/Qwen3-1.7B`, not `Qwen3-1.7B-Base`. No GRPO or new teacher generation was run.

The arms match question exposure, updates and hyperparameters, not token exposure or compute. Per trained model, the answer arm consumes 127,656 input tokens and 14,754 supervised tokens; the trace arm consumes 5,383,134 input tokens and 5,270,232 supervised tokens. Complete targets and end-of-turn tokens are retained. All 500 answer-only targets supervise the empty thinking block inserted by the student chat template; trace targets instead supervise populated teacher thinking.

The unchanged student and six trained models were evaluated on 256 common GSM8K questions with greedy thinking-enabled generation and a 2,048-token ceiling. A preselected subset of the same 64 questions was evaluated at 4,096 tokens. Effective batches are 32 and 16 respectively. Recent development and checkpoint-ablation questions are excluded; historical use of GSM8K still prevents treating this as an entirely new hidden benchmark.

The frozen numerical primary endpoint accepts repeated numerically identical final boxes, requires completed thinking, and rejects missing, malformed or conflicting final fields. The original exactly-one-box scorer is retained as a secondary endpoint and reproduced on CPU. Neither scorer measures reasoning faithfulness.

## Primary 2K outcomes

| Model | Seed | Correct / 256 | Numerical accuracy | Strict accuracy | Unfinished thinking | Mean generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged student | — | 180 | 70.31% | 68.75% | 24.61% | 1,436 |
| Answer only | 42 | 15 | 5.86% | 5.86% | 0.00% | 12 |
| Answer only | 43 | 16 | 6.25% | 6.25% | 0.00% | 12 |
| Answer only | 44 | 17 | 6.64% | 6.64% | 0.00% | 11 |
| Teacher trace | 42 | 197 | 76.95% | 76.95% | 15.23% | 974 |
| Teacher trace | 43 | 207 | 80.86% | 80.86% | 15.62% | 956 |
| Teacher trace | 44 | 196 | 76.56% | 76.56% | 18.36% | 1,034 |

Trace SFT averages **78.1%**, versus **6.25%** for answer-only and **70.3%** unchanged. The predeclared mean trace-minus-answer difference is **+71.88 percentage points**, with paired item-bootstrap 95% interval **+67.06 to +76.56**. The sample SD of the three seed differences is 2.44 points. The interval conditions on these three trained seeds; they do not turn 256 shared questions into 768 independent items.

The separate descriptive trace-minus-unchanged difference averages **+7.81 points**, conditional item interval **+3.39 to +12.50**, with seed-difference SD 2.38 points. Trace responses average 987.9 useful generated tokens and 16.4% unfinished thinking, versus 1,436.2 and 24.6% unchanged. Those are descriptive completion patterns, not isolated proof of greater reasoning capacity, faithful compression or proportional GPU-cost savings.

All 768 answer-only 2K outputs contain empty visible thinking blocks, generally followed by incorrect numeric guesses. This is a behavioral collapse of the specified recipe, rather than an unfinished-thinking scoring penalty. Without an original-student no-thinking baseline, alternate template or hyperparameter sweep, the experiment cannot separate loss of visible deliberation, knowledge forgetting and other recipe effects. It does not show that answer-only distillation generally fails.

## Preselected 64-question length check

Both columns below use the **same 64 questions**; comparing the full 256-question 2K rate with the 64-question 4K rate would confound question composition.

| Model | Seed | 2K correct / 64 | 4K correct / 64 | 2K accuracy | 4K accuracy |
|---|---:|---:|---:|---:|---:|
| Unchanged student | — | 46 | 56 | 71.88% | 87.50% |
| Answer only | 42 | 7 | 8 | 10.94% | 12.50% |
| Answer only | 43 | 6 | 6 | 9.38% | 9.38% |
| Answer only | 44 | 5 | 5 | 7.81% | 7.81% |
| Teacher trace | 42 | 50 | 53 | 78.12% | 82.81% |
| Teacher trace | 43 | 52 | 56 | 81.25% | 87.50% |
| Teacher trace | 44 | 48 | 52 | 75.00% | 81.25% |

At 4K, trace SFT averages **83.85%** versus **87.50%** unchanged. The descriptive mean trace-minus-unchanged difference is **−3.65 points**, conditional item interval **−8.33 to +1.04**, with seed-difference SD 3.25 points. This small check does not demonstrate a trace-training advantage at 4K, and does not establish equivalence or a general reversal. Batch size also changes, so it does not purely isolate output ceiling. One answer-only prediction changes despite no truncation, reinforcing that qualification.

## Cost and shutdown evidence

| Accounting item | USD | Interpretation |
|---|---:|---|
| Original combined envelope | 20.00 | Authorized ceiling |
| Earlier ablation reported metered increment | 1.86 | Historical closing snapshot; subject to adjustment |
| Earlier ablation reserve | 2.14 | Conservative hold, used in combined accounting |
| Stopped CPU-only preparation reserve | 0.18 | Includes setup allowance; no model outcomes existed then |
| New study cap | 17.65 | Final frozen authorization |
| New task-timeout plus startup reservations | 17.4413 | Prelaunch worst-case reservations |
| New non-runtime allowance | 0.20 | Separate allowance |
| Total prelaunch reservations including prior holds | 19.9613 | Below the combined ceiling |
| Observed new client elapsed time at resource caps | **7.0080** | Resource-time estimate, including CPU preparation and verification |
| Additional startup allowance | 1.5748 | Can overlap elapsed time already observed |
| New elapsed estimate plus startup allowance | **8.5828** | Conservative estimate, not an invoice |
| New estimate plus startup, non-runtime and prior holds | **11.1028** | Conservative envelope accounting |

Elapsed-time estimates price each job at its own configured GPU/CPU/memory caps. Rates were rechecked against [Modal's published pricing](https://modal.com/pricing) on October 20, 2026. L40S plus the bounded two CPU cores and 16 GiB is $2.173392/hour. Credits were not used to justify the ceiling. Teacher-generation costs, historical training costs and a new full study are excluded.

Billing was not refreshed: automatic approval review rejected the optional browser GitHub OAuth sign-in. No sign-in workaround was attempted. The existing authenticated CLI executed the authorized jobs and verified shutdown; actual billed usage remains unreconciled.

App `ap-aZd7ZnQUdqEZTp3NSR75oy` stopped at 07:50:11 Europe/Berlin. All three earlier setup apps are also stopped with zero tasks. The [final app inventory](../../results/recovery/distillation-apps-final-cost-check.json) and [empty container inventory](../../results/recovery/distillation-containers-final-cost-check.json) record the closing state. The unrelated deployed sandbox has zero tasks and was left untouched. Remaining funds do not authorize another experiment.

## Verification and reproducibility

All **37 nodes** completed: one runtime preparation, two data preparations, six SFT jobs, fourteen evaluations and fourteen separate persistence checks. All **2,240 responses** are retained: 1,792 primary and 448 long-ceiling responses. Raw hashes passed separate-container verification; strict scores reproduce on CPU and the prospective numerical scores are retained separately. Correct model/evaluation identities and paired question IDs were checked.

All six full exports have distinct hashes and the original student lineage. Two epochs and 64 updates, including cumulative token counts at every update, agree with the reconstructed seedable question sampler. This is not an independently logged GPU source-ID trace. An isolated one-token corruption was rejected by the training auditor. The full local suite passes **79 tests plus 11 subtests**; scoped Ruff and whitespace checks pass. The length-table regression test deliberately varies question composition to detect an invalid unpaired comparison. Review ran sequentially in the primary context; no independent-model or human review is claimed.

After all model jobs finished, two bookkeeping fixes corrected the collector's training-profile path and the forward label for unpacked token scope. Frozen scientific source and original outputs remain preserved. Removing the six local profiles and downloading them through the corrected CPU-only collector reproduced every original hash and left the analysis hash unchanged. The strict launch source guard consequently rejects the historical proposal under current source, including CLI collection. Do not weaken it or rerun the completed proposal; the already collected artifacts and source snapshot support audit.

Artifacts:

- [Frozen protocol and source snapshot](../../results/proposals/distillation/d8a9671022f294b1fae97e82/), full identifier `d8a9671022f294b1fae97e829d3c651d9a79a2bc62c1544115952f3f7edf04fd`.
- [Final analysis](../../results/proposals/distillation/d8a9671022f294b1fae97e82/collected/analysis.json), [descriptive paired length/baseline analysis](../../results/proposals/distillation/d8a9671022f294b1fae97e82/collected/descriptive-secondary-analysis.json) and [4K parent comparison](../../results/proposals/distillation/d8a9671022f294b1fae97e82/collected/parent-comparisons-4096.json).
- [Training audit](../../results/proposals/distillation/d8a9671022f294b1fae97e82/collected/training-audit.json), [answer-only behavior audit](../../results/proposals/distillation/d8a9671022f294b1fae97e82/collected/answer-only-behavior-audit.json) and [500-row data audit](../../results/recovery/distillation-data-audit.json).
- [Closing cost/provenance record](../../results/recovery/distillation-study-close.json), [post-run code-fix record](../../results/recovery/distillation-postrun-code-fixes.json) and [collection regression evidence](../../results/recovery/distillation-collection-fix-validation.json).
- [Revised paper PDF](../../assets/paper.pdf), [source](../../paper/paper.tex) and [results figure](../../assets/distillation-2026-10-20.png).

The PDF compiles to eight pages; every page was rendered and visually inspected. Complete model weights/adapters remain on the persistent experiment volume and have not been published. Changes remain local and uncommitted. This supports a transparent, limited empirical report; it does not establish a new distillation method, broad superiority, reasoning emergence or publication acceptance.
