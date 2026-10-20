# Matched-question reasoning distillation within the $20 envelope

## Research question

Does reasoning-trace SFT improve mathematical problem solving beyond final-answer SFT on the same teacher questions and final answers, in the already post-trained Qwen3-1.7B student?

This is a prospective small-data distillation study. The checkpoint-loading ablation remains supporting reproducibility evidence. GRPO, verification-filter comparisons, and the earlier comprehensive training matrix are outside this run.

## Training design

- One frozen selection of 500 complete, verified OpenR1 teacher generations; pinned student and source revisions remain in the protocol.
- Answer arm: only the final boxed value extracted from that exact teacher generation. Trace arm: the teacher's explicit thinking block followed by the identical final box. Post-thinking prose is removed so both arms have exactly the same final-answer field. Both use the same question and mathematical-expression instruction.
- Seeds 42, 43, 44 for each arm; two epochs, learning rate 0.0002, rsLoRA rank 64, 10% proportional warmup.
- Four questions per microbatch and four gradient-accumulation microbatches per update. Both arms retain 500 separate sequences and receive 64 optimizer updates. The seedable sampler preserves the same question minibatches in corresponding arms.
- Packing is disabled in both arms to preserve question minibatches. FlashAttention processes examples without padding, with position resets preventing cross-example attention. SDPA fallback is refused on this path.
- Full targets including EOS must survive tokenization. The source selection has a 16,128-token ceiling, leaving room for the common instruction under the 16,384-token training limit.

CPU preparation measures, per epoch:

| Arm | Questions | Input tokens | Supervised tokens |
|---|---:|---:|---:|
| Final answers | 500 | 63,828 | 7,377 |
| Reasoning traces | 500 | 2,691,567 | 2,635,116 |

The comparison holds questions, answers, question exposure, optimizer updates and hyperparameters constant. It does not hold supervised tokens or compute constant. It measures the effect of these two supervision recipes, not an equal-compute advantage or proof that traces are faithful reasoning.

The student chat template inserts an empty thinking block into answer-only assistant targets. The actual prepared completion masks supervise those markers in all 500 answer-only rows. This is a material feature of the chosen recipe; there is no original-student no-thinking baseline or alternate-template/hyperparameter sweep in the authorized experiment. The behavior audit is retained separately from the unchanged primary endpoint.

## Evaluation and statistics

The unchanged student and all six trained models receive the same 256 frozen random GSM8K test questions at 2,048 generated tokens, greedy decoding with thinking enabled. Selection excludes the previous 32 development questions and 256 checkpoint-ablation questions, using seed 20261020. This benchmark was used historically; the remaining questions are not a new independent benchmark.

A separate seed, 20261021, selects 64 of those questions before generation for the 4,096-token descriptive secondary check on all seven models. The existing generation-memory ceiling gives batches of 32 at 2K and 16 at 4K. This small sensitivity check is not a powered equivalence test or a pure isolation of token budget from batch effects.

The prospectively frozen primary endpoint, `completed-consistent-numeric-boxes-v1`, requires completed thinking and at least one complete numeric box in the final response. Repeated numerically identical answers are accepted; conflicting boxes, malformed boxes, missing final boxes and unfinished thinking fail. The unchanged strict `boxed-final-v1` endpoint is secondary. Raw generations, termination reasons, useful tokens and both scores are retained.

The primary effect is the mean trace-minus-answer accuracy across the three seeds at 2K. A paired bootstrap resamples distinct test questions and averages across the three trained seeds within each draw. Per-seed deltas and their sample standard deviation are reported separately. Three seeds on 256 shared questions do not become 768 independent questions. There is no checkpoint selection on test outcomes or outcome-dependent extension.

## Budget and execution

The total envelope is $20, including the previous experiment. Its reported metered increment was $1.8600004; $2.14 is held conservatively for that run. Another $0.18 covers the observed stopped CPU-only preparation attempts with startup allowance. This study's maximum remaining authorization is $17.65.

- Sum of task-timeout and startup/idle reservations: **$17.4413436**.
- Additional allowance for non-runtime costs: **$0.20**.
- Combined with the earlier-run and stopped-preparation reserves: **$19.9613436**.

These are reservations, not an expected invoice. The expected new runtime cost is approximately $10–13 based on measured SFT token throughput and 2K batch latency; 4K latency is an extrapolation. Each job has a fixed timeout, two physical CPU cores, a 16-GiB memory cap, two-second scale-down, and zero configured retries. Prices were rechecked against [Modal pricing](https://modal.com/pricing) on 2026-10-20. Credits are not used to justify the cap.

CPU cache and data preparation finish before GPU allocation. Up to four disjoint chains run concurrently: the unchanged checkpoint, plus one chain for each training seed. The global budget check covers all chains; each chain also has its own immutable quote and ledger. Failure or interruption requires reconciliation before any new paid call. Existing completed jobs are never silently repeated.

All evaluation files are verified from a separate container and downloaded with prediction-hash checks. Saved strict scores must reproduce on CPU. Export hashes, training exposure, versions, scorer identity, frozen question texts, dataset revisions and code hashes accompany results.

## Reproduction

CPU preparation only:

```sh
PYTHONPATH=. .venv/bin/python scripts/17_distillation_study.py
```

Execution requires the exact frozen proposal and cap:

```sh
PYTHONPATH=. .venv/bin/python scripts/17_distillation_study.py \
  --execute --proposal results/proposals/distillation/PROTOCOL_PREFIX --budget-usd 17.65
```

With source matching the frozen proposal, `--collect-only` downloads completed results without launching model jobs. The completed run's source snapshot is retained. Subsequent bookkeeping fixes changed current source, so the CLI source guard now rejects this historical proposal, including collection. Do not weaken the guard or rerun it; the already collected results and CPU-only collection regression evidence are retained. Earlier unexecuted preparation artifacts are not approval to execute them.

The final frozen proposal is [`d8a9671022f294b1fae97e82`](../../results/proposals/distillation/d8a9671022f294b1fae97e82/protocol.json), full protocol ID `d8a9671022f294b1fae97e829d3c651d9a79a2bc62c1544115952f3f7edf04fd`. The [500-row data audit](../../results/recovery/distillation-data-audit.json) verifies shared question IDs/prompts, identical single-box final fields, exact teacher-thinking preservation apart from outer whitespace, and complete targets. Teacher final answers comprise 313 numeric and 187 symbolic/other values.

## Verification and limits

The new tests exercise matched targets, permissive identical-box handling, rejection of conflicting/malformed/unfinished answers, the global reservation ceiling, complete disjoint-chain execution, duplicate-launch refusal and CLI success/failure bookkeeping. Existing tests cover real tiny Qwen/PEFT training and export/reload, preprocessing masks, canonical evaluation/resume and persistence.

Review and simplification were performed sequentially in the primary context under the supplied AGENTS.md instructions. No independent model review is claimed. The initial execution attempt encountered a local string-versus-Path bookkeeping error before any remote function was called; its app was verified stopped with zero tasks. That boundary is now covered by CLI tests. Original historical archives and approvals remain unchanged.

A second app was stopped during CPU preparation before GPU allocation to replace a numerical-only instruction with a mathematical-expression instruction appropriate to symbolic OpenR1 targets. Its stopped/zero-task state and CPU-only ledger are preserved. The supervised-data cache identity now binds the matching implementation so the old prompt cannot be silently reused. The final protocol must bind the corrected prepared data; old unexecuted/superseded proposal directories must not be launched.

The subsequent 500-row audit found 50 original teacher finals with multiple boxes. That CPU preparation was also stopped before training, and final fields were normalized identically across arms. The same $0.18 preparation reserve covers the observed CPU-only attempts; no GPU outcomes existed when the final protocol was frozen.

Even a positive result would be a bounded finding about one student, one teacher-data selection and two training epochs. A null result would not establish that reasoning distillation never helps. Publication novelty and broader generalization remain separate questions.

## Completion

The authorized study completed all 37 nodes and retained 2,240 independently hash-verified raw responses. All strict scores reproduce on CPU, all six training exports pass the exposure/lineage audit, and 79 tests plus 11 subtests pass. Every experiment app is stopped with zero tasks; the final container inventory is empty. Estimated new resource-time cost is $7.01, or $8.58 with separate startup allowance; conservative combined accounting including prior reserves and all allowances is $11.10 against $20. These are not reconciled bills. No extension using remaining funds is authorized. See the [completed results and limitations](distillation-results-2026-10-20.md) and [closing record](../../results/recovery/distillation-study-close.json).
