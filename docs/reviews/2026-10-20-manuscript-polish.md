# Manuscript revision using completed findings

The paper is now titled *Reasoning-Trace Distillation under Output Limits: A Study of Qwen3-1.7B*. The abstract, introduction, methods, results and discussion have been rewritten around the completed study. Historical audits and cost accounting remain in the appendix. No new experiment, GPU allocation, commit or publication was performed.

## Claim and evidence alignment

| Finding | Treatment in the revised paper | Retained evidence |
|---|---|---|
| Trace SFT exceeds the unchanged student at 2K: 78.12% versus 70.31%, descriptive difference +7.81 points, conditional interval +3.39 to +12.50 | Leads the abstract and results; explicitly remains secondary | Frozen analysis and descriptive baseline comparison |
| The preselected 64-question 4K check does not demonstrate an advantage: 83.85% versus 87.50%, difference −3.65 points, interval −8.33 to +1.04 | Reported in the abstract, results and discussion; no equivalence or general reversal claim | Same-question paired length analysis and 4K parent comparison |
| Answer-only SFT collapses to 6.25%; primary trace-minus-answer difference +71.88 points, interval +67.06 to +76.56 | Primary status preserved; recipe-specific interpretation and supervised empty thinking blocks explained | Frozen primary analysis and behavior audit |
| Questions and updates match; supervision tokens and compute differ | Exact token table and explicit interpretation limits | Prepared data and training audits |

The figure now shows the original seven-checkpoint 2K comparison alongside the trace and unchanged-student results on the same 64 questions at both limits. Its caption distinguishes question sets and vertical scales. Every numerical data row in both generated results tables is unchanged. The original predeclared endpoint, scoring policies and statistics remain intact; editorial emphasis does not promote the unchanged-student comparison to a primary outcome.

## Verification

All 142 protected evidence-file hashes match the pre-edit snapshot. The reporting script's data-validation and statistical-analysis code is unchanged. Its existing synthetic regression test passes, including the deliberately mismatched full-sample/subset accuracy case. Scoped Ruff and whitespace checks pass.

The final PDF has eight pages, all rendered and visually inspected. It compiles without warnings, unresolved references or overfull/underfull boxes. Table values reconcile with the previous revision, and updated PDF/source/figure hashes are recorded in [the revision record](../../results/recovery/distillation-manuscript-polish.json) and [the closing record](../../results/recovery/distillation-study-close.json). The previous source, figure and PDF are retained in ignored local storage under `outputs/paper-polish-before-2026-10-20`.

This is an editorial and artifact-consistency review in the primary context, not independent peer review. The paper retains the limitations of the actual experiment and makes no new empirical claim.
