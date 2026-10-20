# Research and implementation audit

Reviewed 2026-10-19, checkout `2c7bd86`.

Historical audit capture: the findings and counterexample probe describe that checkout. The remediation changes its behavior; use the current regression tests and `docs/remediation/STATUS.md` to assess the repaired implementation, rather than expecting the old defect-asserting probe to pass.

The repository is a useful exploratory comparison of adaptation recipes, but its current implementation and experimental controls do not support several central causal conclusions. The most urgent issue is a checkpoint reconstruction bug that can invalidate the SFT→GRPO comparison. Correct that and validate model identity and scoring before spending money on additional experiments.

This is a report-only audit. Training code, configuration, results, paper, and Git history were not modified. The accompanying probes assert existing defects; their successful execution does not mean those defects are fixed.

## Scope and evidence

- Read the published `assets/paper.pdf`, local `paper/paper.tex`, README, findings, results JSON, all source modules, CLI scripts, and Modal orchestration.
- Inspected the locally cached processed datasets: 50,000 baseline examples, 1,000 s1K examples, 1,000 Open-R1 examples, and the older 131,063-example NuminaMath GRPO preparation output.
- Executed reward/scoring functions directly. Used AST-isolated repository functions with test doubles to verify trainer argument forwarding, multi-sample metrics, and output persistence without importing GPU packages.
- Checked primary model/dataset documentation, TRL v0.28.0 implementation, PEFT checkpoint documentation, and the central theoretical citation.
- Did not launch Modal jobs, download model weights, retrain, retrieve remote checkpoints/W&B logs, or reproduce the reported accuracies. Local PyTorch, PEFT, TRL, and pytest are absent. Historical runtime versions are not recorded in the checked-in results.
- Review ran sequentially in the main thread, as requested by the repository instructions. No independent model review was performed.

Evidence files: [reproduction script](2026-10-19-audit-checks.py), [machine-readable observations](2026-10-19-audit-evidence.json).

## Findings at a glance

P1 means a result or central interpretation needs correction or revalidation. P2 means a material reliability, methodology, or reproducibility gap.

| ID | Priority | Finding | Evidence status |
|---|---|---|---|
| F01 | P1 | SFT contribution is omitted when the final GRPO adapter is reloaded | Confirmed code path and upstream checkpoint semantics; remote artifact not inspected |
| F02 | P1 | Configured student is already post-trained, rather than a pretrained-only base | Confirmed checkpoint identity |
| F03 | P1 | The answer-only control contains worked reasoning solutions | Confirmed in cached training examples |
| F04 | P1 | Shared answer extraction mis-scores answers and accepts unfinished reasoning | Executed counterexamples |
| F05 | P1 | Verification and SFT/RL effects are not causally isolated | Confirmed experimental design |
| F06 | P1 | Resuming evaluation can combine outputs from different models/configurations | Confirmed checkpoint read/write logic |
| F07 | P2 | Traces are discarded, and trace analysis examines extracted answers | Confirmed persistence probe and source |
| F08 | P2 | Generic Modal evaluation writes results outside the persistent volume | Confirmed path mismatch |
| F09 | P2 | Multi-sample accuracy is pass@n; default pass@k repeats greedy outputs | Executed metric probe and source |
| F10 | P2 | The benchmark labelled MATH is an arbitrary NuminaMath training tail | Confirmed data loader |
| F11 | P2 | Full pipeline omits re_distill; CLI entry points disagree | Confirmed orchestration and CLI choices |
| F12 | P2 | Trainer builders ignore the configured tokenizer | Executed argument-capture probe; upstream fallback verified |
| F13 | P2 | Moving dependencies and implicit objective defaults prevent faithful reproduction | Confirmed specifications and TRL defaults |
| F14 | P2 | Dataset size is presented as examples consumed, overstating RL data usage | Conditional arithmetic under documented v0.28 defaults |
| F15 | P2 | Statistical claims do not account for paired predictions or training variability | Confirmed reporting gap |
| F16 | P2 | Proposed degradation mechanisms are asserted beyond the evidence | Primary-source comparison and logical counterexamples |
| F17 | P2 | Length, training exposure, and inference budgets confound comparisons | Cached token counts and configuration |
| F18 | P2 | Verified-correct traces still include malformed/incomplete thinking blocks | Confirmed in 3 cached Open-R1 examples |

## F01 — Preserve the SFT weights when saving and evaluating SFT→GRPO

Locations: `src/training.py:219–222`, `modal_train.py:156–175`, `modal_train.py:423–430`, `src/evaluation.py:29–57`, `scripts/04_train_grpo.py:72–75`.

`load_sft_checkpoint()` loads the original Qwen checkpoint, merges the SFT adapter into its weights, and returns that in-memory model. GRPO attaches a fresh adapter. `trainer.save_model()` then saves a PEFT checkpoint, whose weights contain the new adapter rather than the modified base. Both evaluation paths recreate the original Qwen model and attach only the GRPO adapter. PEFT explicitly documents that adapter checkpoints exclude base weights. [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

The intended model is `W_original + Δ_SFT + Δ_GRPO`; the evaluator constructs `W_original + Δ_GRPO`. This is a missing SFT contribution, distinct from numerical rounding during merging. Modal also writes both phases to `/vol/outputs/sft_then_grpo`, overwriting the root SFT artifact with the GRPO adapter.

**Consequence:** the reported 69.7% cannot be treated as evidence that GRPO destroyed the distilled policy until the exact historical artifact is reconstructed and re-evaluated. The code defect is confirmed; whether the historical run used precisely this artifact needs remote verification.

**Fix:** preserve separate phase directories. Either export the final fully merged model, or persist the merged SFT base and explicitly load the GRPO adapter against that base. Honor the artifact's model type and base provenance in every evaluator. Before further training, compare logits and greedy outputs before saving and after reloading on fixed prompts, with an explicit bf16 tolerance. Also compare SFT before/after merging. Do not infer the corrected accuracy from the existing count.

## F02 — Identify the student as a post-trained reasoning model

Locations: `configs/config.yaml:2–3`, `results/FINDINGS.md:9`, README Approach and Findings; paper Sections 1, 3.2, and 3.3.

The configured identifier is `Qwen/Qwen3-1.7B`. Its official card lists pretraining **and post-training** and describes built-in thinking capabilities. The separate pretrained-only checkpoint is `Qwen/Qwen3-1.7B-Base`. The findings explicitly describe the former as “base, not instruct.” [Configured checkpoint](https://huggingface.co/Qwen/Qwen3-1.7B), [pretrained-only checkpoint](https://huggingface.co/Qwen/Qwen3-1.7B-Base).

**Consequence:** these experiments measure additional adaptation of an existing reasoning model. They do not demonstrate that a reasoning-naive student learned to think, or that GRPO discovered reasoning without previous reasoning supervision. “Base” is acceptable as shorthand for a shared starting point only if its prior post-training is stated accurately.

**Fix:** relabel the existing study and its conclusions. If emergence from pretraining is the research question, run a separate cohort starting from `-Base`, with an explicitly chosen prompt format. Existing adapters cannot be assumed interchangeable between these checkpoints.

## F03 — The control is not answer-only

Location: `src/data.py:83–99`; paper Section 3.3, Baseline; README's data-efficiency and qualitative claims.

The loader copies Orca's complete `answer` field. Cached baseline row 1 includes defining an unknown, solving `x / 10 = 6`, obtaining `x = 60`, and then computing `60 - 15 = 45`. The public dataset also shows multi-step worked solutions. [Microsoft dataset](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k).

**Consequence:** the 69.2% versus 73.8% comparison is between different kinds, sources, and lengths of worked solutions. It cannot establish that reasoning supervision beats final-answer-only supervision. The paper's own baseline example also performs multiple arithmetic steps despite its empty thinking block.

**Fix:** label the existing arm ordinary worked-solution SFT. Build a genuine final-answer-only control from the **same questions and verified solutions** used by the reasoning arm, retaining only a validated final answer. Include matched-format and matched-budget controls so tags and extra tokens are not mistaken for reasoning acquisition.

## F04 — Fix the shared reward and evaluation verifier

Locations: `src/reward.py:66–84`, `src/reward.py:110–130`, `src/reward.py:159–175`.

Executed observations:

| Completion or comparison | Current behavior | Defect |
|---|---|---|
| `The answer is 1,234.` | Extracts `234.` | False negative for 1234 |
| `The answer is 1/2.` | Extracts `2.` | False negative for 0.5 |
| Thinking rejects `\boxed{10}`, final answer is 12 | Extracts 10 | A discarded candidate overrides the final answer |
| `<think>Try 40 then 42` with truth 42 | Correctness reward 1 | Unfinished reasoning earns answer credit |
| `<think></think> \boxed{0}` | Format reward 1 | Empty thinking earns maximum structure reward |
| `answers_match("\\frac{1}{2}+7", "0.5")` | True | Fraction regex ignores the suffix |
| `answers_match("inf", "1")` | Raises `OverflowError` | Non-finite numeric conversion can stop a run |

**Consequence:** reward optimization and benchmark measurement share the same biases. Effects can vary by condition because boxed answers, prose answers, fractions, and thinking lengths differ. The counterexamples establish defects, not their incidence in the historical scores.

**Fix:** define and version a benchmark-specific extraction policy. Parse the final response separately from thinking, handle commas/signs/fractions as complete expressions, reject non-finite values, and record extraction failures separately. Audit old-versus-new scores on retained raw generations. If rewarding structure, require a well-formed final response and distinguish empty thinking from substantive content; structure still does not prove reasoning quality.

## F05 — Remove causal attribution to verification and the claimed factorial design

Locations: `src/data.py:22–38`, `src/data.py:51–79`, `configs/config.yaml:112`, `configs/config.yaml:140`; paper Sections 1, 3.1, 4.1, and 6.

s1K and Open-R1 differ in questions, source distribution, sampling, trace length, and curation. Both provide DeepSeek-R1 outputs: Open-R1's own card identifies its generator as DeepSeek-R1. Thus “raw teacher” versus “RL-trained teacher” is not an isolated treatment. Selecting the first 1,000 passing Open-R1 rows also introduces order-dependent sampling. [Open-R1 dataset description](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k).

The stated 2×2 design is incomplete as an intervention comparison: the no-traces/no-RL arm receives 50K Orca SFT examples, while the no-traces/RL arm starts from untouched Qwen. Reward weights also change from `[0.5, 0.5]` to `[1.0, 0.0]`. One cannot attribute their interaction solely to the presence of traces and RL.

**Fix:** describe these as comparisons of recipes. For a verification experiment, sample verified/unverified generations from a shared prompt pool and teacher, stratifying or matching difficulty and length. For the primary SFT×RL comparison use shared initialization, no extra SFT in the untreated cell, and the same RL dataset/reward/procedure in both RL cells. Add the matched final-answer arm separately. Do not call a 4.7-point difference between different datasets the measured effect of filtering.

## F06 — Bind resumable evaluations to immutable experiment identity

Locations: `modal_train.py:394–408`, `modal_train.py:499–502`.

The resume key is only `{condition}_gsm8k.json`. Its contents record `condition`, `next_idx`, and results. They do not identify adapter bytes, starting checkpoint, dataset revision, prompt template, scorer, token budget, or decoding settings. Retraining an arm or changing its evaluator and reusing the condition name silently joins an old prefix with a new suffix. Base-only mode can also reuse a condition name.

**Fix:** hash model/adapter identities, dataset revision, prompt/scorer versions, and resolved generation config into a run ID. Refuse mismatched resumes. Validate unique indices, expected ground truths, result count, and `next_idx`; checkpoint with an atomic replacement. This needs a focused test that deliberately resumes with a different model/config and expects rejection.

## F07 — Save the actual generations and analyze those

Locations: `src/evaluation.py:165–170`, `src/evaluation.py:302–311`, `scripts/06_analyze_traces.py:115–119`, `modal_train.py:483–489`.

Generic evaluation returns per-example records but removes them from the only JSON it saves. The analysis script reads that JSON and skips benchmarks lacking `results`, yielding empty trace plots or no patterns. Even if per-example records are preserved, `predictions` contains extracted answers, and pattern analysis treats `predictions[0]` as the reasoning trace. It would analyze `"42"` rather than the generated solution. The full GSM8K evaluator keeps scores and character lengths but also discards raw responses.

**Fix:** retain per-example JSONL with question ID, raw text, token IDs or token counts, termination/truncation status, extracted answer, and correctness. Keep summaries separately. Analyze `response`/`thinking_text`, never extracted numeric predictions. Paired statistical analysis and retrospective scorer corrections need these same records.

## F08 — Persist generic Modal evaluation output on the mounted volume

Locations: `src/evaluation.py:298–311`, `modal_train.py:189–199`.

`run_all_evaluations()` writes relative `outputs/{condition}/eval_results.json`. Modal mounts its persistent volume at `/vol`, and this function neither redirects output there nor changes working directory. `volume.commit()` does not persist unrelated container files. The specialized full GSM8K evaluator correctly writes under `/vol`, so this bug concerns generic `run_evaluation()` and the full-pipeline entry point.

**Fix:** accept an explicit result directory and have the Modal caller pass a location under `/vol`. Verify the resulting file through a separate volume read after the producing container exits.

## F09 — Separate single-sample accuracy from pass@k and require sampling

Locations: `src/evaluation.py:105–116`, `src/evaluation.py:173–182`, `src/evaluation.py:231–239`, `src/evaluation.py:265–267`, `scripts/05_evaluate.py:30–32`.

With four solutions per problem and only one correct solution, the probe returns `accuracy=1.0`, `pass@1=0.25`, and `pass@4=1.0`. The field called accuracy changes meaning with sample count. The CLI's `--num-samples` does not change temperature, so its default generates the same greedy answer repeatedly. Finally, direct `compute_pass_at_k([[True]], 4)` returns 1.0 even though the estimator lacks four samples.

**Fix:** report first-sample accuracy, sampled pass@1, and pass@k as explicitly distinct metrics. Reject multi-sample runs at temperature zero unless explicitly requesting deterministic repeats, expose sampling parameters and seeds, and reject unsupported k. This does not by itself invalidate the published one-sample accuracy; it invalidates the current proposed pass@k extension.

## F10 — Do not label the NuminaMath training tail MATH

Locations: `src/data.py:169–189`, `src/evaluation.py:195–198`, `src/evaluation.py:241–242`.

The `math` benchmark selects the last 500 records from `AI-MO/NuminaMath-CoT`'s training split. It is neither MATH nor MATH-500. There is no enforced disjointness from related Numina/Open-R1/s1K source collections, and numeric/string normalization is insufficient for general symbolic mathematics.

**Fix:** use an explicitly identified benchmark release and immutable problem IDs; decontaminate against every training set. If retaining this diagnostic, name it `numinamath_train_tail_500` and avoid held-out claims without checking source overlap. Use validated mathematical equivalence for symbolic answers. The current paper's GSM8K-only result table is not evidence of MATH performance.

## F11 — Make every advertised arm executable through the supported paths

Locations: `modal_train.py:548–580`, `scripts/03_train_sft.py:18`, `scripts/02_prepare_grpo_data.py:6–28`, `modal_train.py:156–160`, README Usage.

The full pipeline trains/evaluates four conditions and omits `re_distill`, its strongest reported arm. Local SFT CLI choices also exclude it. The GRPO preparation script writes NuminaMath data, while actual GRPO training loads GSM8K directly; training does not consume the prepared cache. Modal GRPO also permits `condition="sft_then_grpo"` without an SFT checkpoint, silently running from Qwen. README's sequential commands omit training the SFT phase for `sft_then_grpo` before referring to its checkpoint.

**Fix:** centralize condition dispatch/validation, add the missing arm, document or automate the dependency order, and make preparation artifacts either consumed and fingerprint-checked or clearly optional diagnostics. Reject inconsistent condition/checkpoint combinations before provisioning training.

## F12 — Pass the tokenizer to both trainers

Locations: `src/training.py:120–125`, `src/training.py:175–181`.

Both builders accept `tokenizer` but omit `processing_class=tokenizer` from trainer construction. Argument capture reproduces the omission. TRL v0.28.0 creates its own processor/tokenizer when this argument is absent. The separately saved tokenizer can therefore differ from the trainer's processor. [SFT implementation](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/sft_trainer.py), [GRPO implementation](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py).

**Fix:** pass the tokenizer and configure GRPO padding/truncation explicitly. Save the actual trainer processor. Verify IDs and serialized prompts across preparation, training, generation, and reloading. Do not claim that this currently causes incorrect EOS behavior: the cached Qwen tokenizer already uses `<|im_end|>`, consistent with the current upstream tokenizer, so the supposed base-EOS correction is itself inaccurately documented.

## F13 — Pin the environment and record resolved training objectives

Locations: `pyproject.toml:8–26`, `modal_train.py:17–46`, `src/training.py:148–173`.

Only some SFT infrastructure is pinned. TRL, Transformers, PEFT, datasets, vLLM, and other dependencies move, and model/dataset revisions are unspecified. Important objective settings are left to defaults. In TRL v0.28.0, those include `loss_type="dapo"`, `beta=0.0`, group reward scaling, and vLLM importance-sampling correction. A label of GRPO alone does not specify this recipe. These v0.28 defaults are not proof of the versions used in the historical runs. [GRPO configuration source](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_config.py).

**Fix:** pin separate SFT/GRPO environments and image identities. Persist the resolved trainer arguments, model/dataset revisions, data fingerprints, code SHA, adapter/base lineage, generation template/config, initialization seed, and W&B run ID. Set the seed before adapter construction. There is currently no checked-in test suite or CI to catch the reproduced failures; add focused CPU tests plus a tiny optional GPU save/reload smoke check.

## F14 — Distinguish available dataset rows from prompts actually consumed

Locations: `configs/config.yaml:104–115`, `configs/config.yaml:132–143`; README and paper claim GRPO requires 7.5× more examples.

Under TRL v0.28's documented defaults on one GPU, `generation_batch_size = 1 × 1 × 8 = 8` completions, and `num_generations=8` means **one distinct prompt per generation batch**. With one generation batch per optimizer step, 500 steps expose about 500 prompt groups and 4,000 completions, not all 7,473 dataset rows. Exact historical consumption requires the runtime version and logs. [Configuration/default derivation](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_config.py), [repeat sampler setup](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py).

**Fix:** report available pool size, sampled/unique prompt count, rollout count, completion tokens, and optimizer steps separately. Increase steps or prompt groups deliberately if the intention is to expose the entire training set. Do not derive relative data efficiency from dataset cardinality alone.

## F15 — Use paired comparisons and repeat training

Locations: `results/FINDINGS.md`, paper Sections 4 and 5; `results/gsm8k_results.json`.

All reported correct counts agree with the rounded accuracies. Wilson intervals calculated directly from those counts are:

| Arm | Correct / 1,319 | Accuracy | 95% Wilson interval |
|---|---|---|---|
| Original Qwen checkpoint | 952 | 72.18% | 69.70–74.53% |
| Orca SFT | 913 | 69.22% | 66.68–71.65% |
| s1K SFT | 974 | 73.84% | 71.41–76.14% |
| GRPO only | 999 | 75.74% | 73.35–77.98% |
| SFT→GRPO | 919 | 69.67% | 67.14–72.09% |
| Open-R1 SFT | 1,036 | 78.54% | 76.25–80.68% |

An individual arm's ±2.5-point interval is not a test of the difference between two arms. The models solve the same questions, so paired bootstrap or McNemar comparisons should use per-question outcomes. The 1.67-point s1K gain represents only 22 additional correct answers and does not establish reliable improvement without that analysis and training repeats. The 4.70-point Open-R1/s1K gap is substantial; do not dismiss it just because individual intervals overlap. A statistically credible difference still does not identify verification as its cause.

**Fix:** recover paired records, preselect primary comparisons, report effect-size intervals, and run at least three independent training seeds as an initial variability check. State separately the uncertainty from test items and from training. Avoid turning one recipe's ranking into a general scale law.

## F16 — Reframe mechanism claims as hypotheses

Locations: paper Sections 4.3, 5.2, and 5.3; `results/FINDINGS.md` Finding 3.

Three arguments currently go beyond their evidence:

1. **Higher SFT loss does not prove lower GSM8K accuracy.** Niu et al.'s Theorem 3.1 concerns SFT likelihood under assumptions about the reference distribution and optimized RL objective; its empirical example uses Qwen3-0.6B on CoLA. It does not establish that RL necessarily reduces mathematical correctness. The repository does not measure the theorem's loss quantity across GRPO. [Original theorem and experiments](https://arxiv.org/html/2601.07389v1).
2. **Equal rewards do not imply identical text or entropy collapse.** Diverse solutions can all be correct or all wrong. `frac_reward_zero_std` measures a reward statistic, not token entropy or completion uniqueness; it is a fraction of samples/groups in the generation batch, rather than automatically the fraction of optimizer steps with zero gradient. [TRL metric definition](https://huggingface.co/docs/trl/v0.28.0/en/grpo_trainer).
3. **Intruder dimensions do not establish that merging drops those dimensions.** Algebraic LoRA merging adds `sBA` to `W`; it does not discard the update through a new low-rank approximation. bf16 rounding can occur and should be measured. Sequential-adaptation findings are not evidence of the paper's specific claimed merge-loss mechanism. [LoRA study](https://arxiv.org/abs/2410.21228). F01 is a separate concrete preservation bug.

The qualitative apple problem is one of `quick_test()`'s hand-written prompts and is absent from the cached GSM8K test set; the paper calls it a GSM8K problem. Both quoted responses compute the answer in multiple steps. Additional verbosity and “let me double-check” cannot by themselves prove causal self-correction or faithful reasoning. The proposed 1.7B–7B threshold also comes from incomparable models/benchmarks, rather than a within-family scale experiment.

**Fix:** separate measured behavior from proposed mechanisms. Measure held-out SFT loss, token entropy, unique solution paths, length/termination rates, merge/reload output parity, and actual correction events. Use controlled reasoning perturbations or ablations to assess whether self-verification changes accuracy.

## F17 — Account for trace length and inference limits

Locations: `configs/config.yaml:75–82`, `configs/config.yaml:92–98`, `configs/config.yaml:149–152`, `src/data.py:56–57`.

Using cached tokenizer revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` on raw serialized question/completion chat text:

| Dataset | Mean tokens | Total tokens before packing | Rows longer than 16,384 |
|---|---|---|---|
| s1K traces | 10,013.5 | 10,013,506 | 101 / 1,000 |
| Open-R1 traces | 6,233.6 | 6,233,574 | 26 / 1,000 |

These counts precede trainer chat rewriting and packing; they do not establish exactly which historical final answers were truncated or split. They establish unequal exposure and the need to inspect actual packed batches, masks, boundaries, and preserved final answers. Merely matching examples, epochs, and maximum length does not match optimization steps, supervised tokens, or task difficulty.

Evaluation allows 2,048 output tokens, versus 4,096 for RL generation and substantially longer SFT traces. Without token counts and finish reasons, correctness is mixed with the ability to finish within this budget. Qwen's card also advises sampling for thinking mode because greedy decoding can degrade behavior. Greedy evaluation is a legitimate fixed protocol, but conclusions should be scoped to it and tested for budget/decoding sensitivity. [Qwen generation guidance](https://huggingface.co/Qwen/Qwen3-1.7B).

**Fix:** inspect actual trainer-processed batches, preserve complete answer supervision, report truncated/completed fractions, and compare accuracy at fixed token budgets such as 2K/4K/8K. Include a sampled-decoding condition. Use token-matched ablations before interpreting data or compute efficiency; student-training cost should be distinguished from the upstream cost of generating/verifying teacher traces.

## F18 — Validate verified traces for completeness, not only answer correctness

Location: `src/data.py:53–79`.

Three cached Open-R1 completions contain `<think>` without `</think>`. A correctness flag alone does not validate format or whether the target will retain a complete final answer after processing. The loader also throws away original row identity and verification provenance when it removes all source columns.

**Fix:** check complete reasoning boundaries, final-answer presence, verifier provenance, and processed sequence completeness; retain stable source IDs. Quarantine/review malformed examples rather than assuming correctness verification covers these properties. Recompute the selected sample after applying a deterministic, documented quality policy.

## What remains supported

The six reported counts are arithmetically consistent. Open-R1 SFT has the highest **reported** accuracy in this particular run. The cached datasets match the stated SFT example counts. A normalized exact-match comparison of their questions against all 1,319 cached GSM8K test questions found zero matches in each of the baseline, s1K, and Open-R1 sets. This rules out that narrow overlap criterion only: no fuzzy/paraphrase or pretraining-contamination audit was performed.

The numerical ranking has not been independently reproduced. The SFT→GRPO row especially needs artifact reconstruction. Existing evidence does not establish reasoning emergence, the isolated effect of verification, inevitable SFT/RL conflict, or a model-size threshold.

## Recommended order of work

1. Recover remote checkpoint lineage and per-question records; preserve both SFT and GRPO phases before changing anything.
2. Fix model serialization/loading and scoring. Add targeted regression checks and a tiny GPU model save/reload comparison. Measure the no-update SFT→merge→save→reload path before retraining.
3. Re-evaluate existing artifacts using immutable evaluation identities, retained raw responses, and explicit token/decoding budgets. Reconstruct the combined model correctly; do not replace its reported number by a guessed adjustment.
4. Correct model/control labels and qualify causal/mechanistic claims in the canonical paper and README.
5. Pin environments and record resolved arguments, source revisions, dataset fingerprints, prompt exposure, and seeds.
6. Run the smallest decisive experiments: shared-question final-answer versus reasoning supervision; shared-pool verified versus unverified traces; consistent-reward SFT×RL comparison; repeat seeds; then add harder, genuinely held-out benchmarks.

## Reproducing the audit checks

From the repository root:

```bash
python3 docs/reviews/2026-10-19-audit-checks.py
venv/bin/python docs/reviews/2026-10-19-audit-checks.py --local-data
```

The first command needs only the standard library and the existing reward module. The optional local-data pass uses installed `datasets` and `tokenizers` and existing cache files; it does not download data or run a model. The evidence JSON was generated by the second command. These documentation files live under the repository's already-ignored `docs/` directory; no tracked source files were changed.
