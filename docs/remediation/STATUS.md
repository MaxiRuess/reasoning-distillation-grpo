# Remediation status — October 20, 2026

The separately authorized [sampled inference follow-up is complete](../experiments/inference-extension-results-2026-10-20.md): 29 successful tasks, 14 evaluations and 1,792 verified responses on 128 shared questions. At 8K, trace checkpoints average 86.20% versus 87.50% unchanged (−1.30 points, 95% conditional item interval −5.73 to +2.86); an advantage is not established. Native sampled no-thinking reaches 78.91% at 2K with 237 mean tokens and does not reproduce the answer-only SFT collapse. All experiment apps are stopped and the live container inventory is empty. Estimated new resource-time cost is $13.34, or $15.47 including all extra allowances, within the separate $30 cap; these are not reconciled bills. Full local verification passes 84 tests plus 11 subtests. No training, second benchmark, automatic paid retry or further GPU spending is included. The [closing record](../../results/recovery/inference-extension-outcome.json) identifies the evidence; original results remain unchanged.

The [bounded three-seed distillation study is complete](../experiments/distillation-results-2026-10-20.md), distinct from the unapproved comprehensive six-arm study below. All 37 nodes, six exports and 2,240 responses passed their artifact checks. At 2K, trace SFT averages 78.1% versus 70.3% unchanged; the answer-only recipe collapses to 6.25%, with supervised empty thinking blocks as a material limitation. The preselected 64-question 4K check does not demonstrate an advantage over the unchanged student. All experiment apps are stopped, zero tasks remain, and the final container inventory is empty. Estimated new resource-time cost is $7.01; the conservative envelope total with startup, other allowances and earlier-run reserves is $11.10 against $20, not a reconciled bill. Local verification passes 79 tests plus 11 subtests. No further GPU work is authorized.

The implementation on `fix/research-remediation` fixes the identified pilot defects and at the ablation stage passed 65 CPU tests plus 11 subtests. New runs share an explicit boxed-answer contract, with separate format/unfinished-thinking rates. All pilot and checkpoint-ablation apps are stopped. The earlier pilot's closing report recorded about $3.12 under its $10 approval; the separately approved $20 ablation completed with $1.86 reported additional metered usage, subject to billing adjustments. Historical evidence is preserved and full-study spending remains unapproved. See the [final repair report](pilot-defect-fixes-2026-10-19.md) and [ablation results](../experiments/checkpoint-ablation-results-2026-10-19.md).

## Implementation and evidence

| Finding | Disposition | Evidence / remaining gate |
|---|---|---|
| F01: combined checkpoint | Complete effective weights exported, intermediate adapters retained, explicit parent/seed validation | Tiny two-update test passes; all three historical full-size constructions pass exact CPU-logit export/reload checks and 256-question GPU evaluations. Historical full-benchmark correction remains pending. |
| F02: model identity | Paper, README, config and instructions identify post-trained Qwen | Pretrained-only cohort excluded as agreed |
| F03: control definition | Historical Orca relabelled worked-solution SFT; genuine final-answer control implemented | Shared question IDs and deterministic selection tested; original counts preserved |
| F04: scoring | Explicit shared final-answer contract replaces unreliable prose quantity selection; exact integer comparison | Fresh GPU pilot: 28 completed fields parsed, 4 unfinished, zero completed-response format failures; all 32 records CPU/hash verified; old policies/results preserved |
| F05: causal isolation | Six-arm shared-question protocol plus token-matched controls implemented | New study remains unrun; paper no longer claims verification is the isolated cause of the historical difference |
| F06: resume identity | Full run identities, atomic transactions, exclusive writers, mismatch rejection | Interruption/resume equivalence, corrupted tail and duplicate-ID tests pass |
| F07: trace retention | Raw JSONL, token counts, finish reasons and actual-response analysis | Saved-response integration test passes; old numeric predictions are not treated as traces |
| F08: persistence | All new Modal evaluations write below `/vol/evaluations-v2`; independent verification implemented | Pilot passed separate-container summary/hash verification and local raw-record reconciliation |
| F09: metrics | First-sample accuracy separated from sampled pass@k; invalid k and greedy repeats rejected | Hand-calculated metric fixtures pass |
| F10: benchmark identity | Pinned MATH-500 test set and Math-Verify replace ambiguous training-tail benchmark | Dataset fetched; symbolic equivalence tested; full model evaluation pending |
| F11: orchestration | Shared CLI/Modal logic, `re_distill` included, required SFT parents checked, CPU preparation mandatory | Complete DAG and invalid-request tests pass; legacy and controlled dry runs succeed |
| F12: tokenizer | Actual configured processor passed and saved | CPU training test plus full-size exported-Qwen tokenizer round-trip comparison pass |
| F13: environments | Separate locked images and runtime/source manifests | Images built and L40S SFT/H100 GRPO/L40S evaluation ran; image layer order fixed and vLLM sleep disabled after real failures; other protocols unvalidated |
| F14: exposure | Trainer-state token counts and weighted-objective diagnostics verified | Fresh full-size run: 50,392 total tokens, 45,344 completion tokens, 48 rollouts/6 groups; metadata/state/profile agree; old erratum preserved |
| F15: uncertainty | Paired bootstrap, exact McNemar, Holm correction, factorial interaction and separate training-seed summaries | Historical six-arm records reconciled and analyzed; bounded distillation three-seed estimates complete; full six-arm estimates pending |
| F16: mechanisms | Unsupported emergence, entropy, self-correction, merge-loss and scale-threshold claims removed | Revised paper distinguishes observations, hypotheses and missing measurements |
| F17: budgets/length | Complete targets, processed EOS supervision checks, token-matched datasets and 2K/4K/8K evaluation matrix | Original 2K/4K checks and four-checkpoint sampled 128-question 2K/4K/8K follow-up complete; broader six-arm matrix remains unrun |
| F18: malformed traces | Structurally invalid/overlength traces quarantined with source/generation IDs | Shared-pool preparation and malformed/alignment tests pass |

## Recovered historical evidence

- Immutable local archive: `outputs/historical/2026-10-19`; index/hashes: `results/recovery/archive-index.json`.
- 56 archived files include all six per-question evaluations, final adapters and the combined run's SFT checkpoint at step 117. Remote source files were not modified.
- `results/recovery/checkpoint-lineage.json` confirms a nonzero sampled SFT projection update and the final adapter's original-Qwen base reference.
- `results/recovery/recovered-runtimes.json` records model-card framework versions. SFT and GRPO used different PyTorch/Transformers versions; the new TRL 0.28.0 locks are not represented as historical replicas.
- `results/recovery/historical-paired-analysis.json` contains exploratory paired comparisons. The s1K-versus-starting-checkpoint interval includes zero. The Open-R1/s1K separation does not identify verification as its cause.
- Full-size local CPU reconstruction was initially deferred after the available-memory check. Subsequent cloud CPU reconstruction and the frozen 256-question GPU ablation are complete; historical full-benchmark correction remains pending.
- Original numerical results are preserved in `results/historical/gsm8k_results.original.json`; original paper source/PDF are also in the local archive.

## Local verification

- `python -m pytest tests -q`: 65 tests pass, including real CPU SFT/GRPO integration, three-construction export/reload, strict/archived scoring, frozen subset integrity, provenance, spending and retry guards; eleven additional subtests pass.
- `ruff check src scripts tests modal_train.py`: passes.
- Local dry runs include 21 controlled training jobs, 46 evaluations and 46 separate persistence checks, across the declared primary/secondary protocols; these are schedules, not completed experiments.
- Linux/Python 3.11 images built and the bounded GPU pilot completed. Long-context/sampled protocols and full-study resource choices remain unvalidated.
- Paper tables and the historical comparison figure are generated from the paired-analysis JSON. The revised PDF compiles locally and is rendered for visual inspection.
- CI is defined for CPU correctness; it has not yet run on a remote CI service.

The prepared answer-only arm packs into seven sequences (three optimizer steps over three epochs), versus 355/360 sequences for the unfiltered/verified trace arms. Controlled SFT therefore uses a common 10% proportional warmup; the original fixed ten-step warmup would span the entire answer-only run. This is an explicit new-study correction, not a change to the historical record.

The [GPU efficiency review](../reviews/2026-10-19-gpu-efficiency-review.md) adds CPU token preparation/cache warming, batched sampled inference, bounded vLLM context and profiles. Original verified SFT preparation contained 5,865,452 tokens in 360 sequences (99.44% packed capacity), with 104.6 seconds of pilot training. The final explicit-output protocol measured 89.8 seconds of GRPO training and 131.8 seconds of evaluation generation; its changed prompt/output workload prevents treating these as a code-only speedup. H100 cost optimality and the secondary GPU protocols remain unmeasured. New preparation identities must be used after protocol/source changes.

## Compute decision

The separate $20 checkpoint-loading ablation is complete. Its [outcome](../../results/recovery/checkpoint-ablation-outcome.json) records 191/256 for the exact SFT parent, 168/256 with SFT omitted and 184/256 after reconstruction. The predeclared +6.25-point effect has a paired 95% interval of +0.78 to +11.72 points. A post hoc sensitivity accepting four repeated identical correct boxes yields +4.69 points, interval -0.39 to +9.77. The corrected-versus-SFT difference remains unresolved. Primary scores and historical results are unchanged. No training or additional evaluation matrix was run.

`results/recovery/pilot-plan.json` and `pilot-quote.json` are the reviewed proposal artifacts. The user subsequently authorized the small pilot; `pilot-authorization.json` binds that approval to the quote hash. The original quote remains unchanged for ledger identity.

The pilot warms the runtime cache and prepares the common pool on CPU, runs two SFT steps, runs six post-SFT GRPO steps, evaluates 32 GSM8K questions, then verifies the result from a separate container. Runtime reservations total about **$2.55**, including the stated CPU/memory limits and startup allowance at the dated published rates. The user approved the **$10 pilot cap** to leave room for image building, initialization and incidental costs. Rates were rechecked before launch. The quoted runner applies per-job resource/time limits; this is not a provider billing guarantee. The pre-run billing snapshot is `pilot-billing-before.json`.

No main-study budget is inferred from pilot approval. The main quote generator requires completed pilot evidence and prepared data; it refuses to fabricate throughput estimates. The main study, reconstructed historical full evaluations and any extra re-evaluation batches require their own priced approval.

The original recovery ledger and counter erratum remain preserved. `pilot-final-outcome.json` / `pilot-boxed-outcome.json` index the successful final source/contract/lineage/token/hash checks. Prose variants are archived as diagnostics, not validated primary metrics. Current quotation rejects stale reward/evaluation contracts and inconsistent exposure. New proposals never overwrite original approval files; do not blindly relaunch an original failed ledger.

For a future separately agreed run, generate a new proposal and use its printed quote path:

```bash
python scripts/08_run_study.py --experiment pilot --execute \
  --quote PRINTED_QUOTE_PATH --budget-usd AGREED_CAP
```

The full controlled study is unrun. Historical full-benchmark corrections and repeated controlled training would be needed for broader empirical claims. The completed subset ablation supports a limited reproducibility report with its format sensitivity stated. Changes remain local and uncommitted for review.
