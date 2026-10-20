# Reasoning-Trace Distillation under Output Limits

**[Read the paper (PDF)](https://maxiruess.github.io/reasoning-distillation-grpo/assets/paper.pdf)** · [Findings](results/FINDINGS.md) · [Sampled results](docs/experiments/inference-extension-results-2026-10-20.md)

This project studies reasoning-trace distillation into **the already post-trained `Qwen/Qwen3-1.7B` checkpoint**. It does not test reasoning emergence from a pretrained-only model. Earlier supervised/RL recipe comparisons are preserved as historical evidence.

[Paper (PDF)](assets/paper.pdf) · [Audit](docs/reviews/2026-10-19-research-audit.md) · [GPU review](docs/reviews/2026-10-19-gpu-efficiency-review.md) · [Remediation status](docs/remediation/STATUS.md)

The [bounded distillation study is complete](docs/experiments/distillation-results-2026-10-20.md): answer-only and teacher-trace SFT used the same 500 questions and final answers, three seeds, and 64 updates over two epochs. On 256 frozen GSM8K questions at 2K, trace SFT averaged 78.1% versus 70.3% for the unchanged student; the answer-only recipe collapsed to 6.25%. Its template supervised empty thinking blocks, so the large trace-versus-answer gap does not isolate a general distillation benefit. On the preselected 64-question 4K check, trace SFT averaged 83.9% versus 87.5% unchanged, with no demonstrated advantage. All 2,240 raw responses were persisted and independently hash-verified; strict scores reproduce on CPU. All experiment apps are stopped. New resource-time cost is estimated at $7.01, or $8.58 with extra startup allowance; the conservative total including earlier-run reserves and all allowances is $11.10 within $20. This is not a reconciled bill. No further GPU work is authorized from the remaining budget.

The separately authorized [sampled inference follow-up is complete](docs/experiments/inference-extension-results-2026-10-20.md). On 128 shared questions, trace checkpoints average **86.20% at 8K versus 87.50% unchanged** (difference −1.30 points; 95% conditional item interval −5.73 to +2.86). The 2K trace advantage does not establish an advantage at 8K. Native sampled no-thinking reaches 78.91% at 2K with 237 mean tokens, versus 80.21% and 994 tokens for trace checkpoints; these policies use different sampling parameters. All 1,792 new responses were verified, and the app is stopped with no containers remaining. Estimated resource-time cost is **$13.34**, or **$15.47 including startup and incidental allowances**, within the separate **$30 cap**; this is not a reconciled bill. No training, second benchmark or further GPU spending is included. Original greedy results remain separate. The full CPU suite passes 84 tests plus 11 subtests.

The separately approved [checkpoint-loading ablation](docs/experiments/checkpoint-ablation-2026-10-19.md) compares the exact historical SFT parent, the missing-SFT loading control and the corrected sequential checkpoint on 256 frozen GSM8K questions. It performs no training and has its own $20 cap, quote and persistent ledger. Prepare its proposal with `PYTHONPATH=. python scripts/16_checkpoint_ablation.py`; execution requires the frozen proposal and explicit budget. The main controlled study remains unapproved.

The [completed ablation](docs/experiments/checkpoint-ablation-results-2026-10-19.md) scored 191/256 for the exact SFT parent, 168/256 with SFT omitted and 184/256 after correct reconstruction. The predeclared restoration effect is +6.25 percentage points; accepting four repeated identical correct boxes reduces it to +4.69 points with an interval including zero. The corrected-versus-SFT difference is unresolved. All apps stopped; closing reported additional metered usage was $1.86, subject to billing adjustments.

## Historical results and their limits

The six original evaluation files were recovered and their counts reconciled. These are **historical legacy-scored measurements**, not corrected or independently reproduced accuracies.

| Historical recipe | Correct / 1,319 | Accuracy |
|---|---:|---:|
| Original post-trained Qwen checkpoint | 952 | 72.2% |
| Orca **worked-solution** SFT | 913 | 69.2% |
| s1K trace SFT | 974 | 73.8% |
| GRPO from the original checkpoint | 999 | 75.7% |
| s1K SFT → GRPO **requiring checkpoint reconstruction** | 919 | 69.7% |
| Open-R1 verified-trace SFT | 1,036 | 78.5% |

The audit found a missing SFT contribution when reloading the final combined adapter, scorer defects, and confounded data comparisons. The intermediate SFT checkpoint remains recoverable. Original records contain extracted answers, not raw responses, so corrected scoring requires new generations.

[Recovered paired analysis](results/recovery/historical-paired-analysis.json) finds the s1K-versus-starting-checkpoint difference unresolved (paired 95% interval −0.76 to +4.09 percentage points). The Open-R1/s1K historical difference is +4.70 points (paired interval +2.50 to +6.97), but different question distributions and lengths prevent attributing it to verification alone. These tests do not validate checkpoint reconstruction or capture training-seed variability.

## Setup and local verification

Use Python 3.11 or 3.12. Linux GPU jobs use separate fully resolved Python 3.11 locks under `requirements/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest tests -q
ruff check src scripts tests modal_train.py
```

The new stack uses TRL 0.28.0. [Recovered historical cards](results/recovery/recovered-runtimes.json) identify TRL 1.0.0 with different SFT/GRPO PyTorch and Transformers versions; their complete transitive environments are unknown.

The [pilot repairs](docs/remediation/pilot-defect-fixes-2026-10-19.md) pass 53 CPU tests and a fresh GPU training/evaluation/persistence check. New runs use a shared explicit final-answer contract, correct GRPO token accounting and weighted-objective diagnostics. All pilot apps are stopped; reported metered usage across attempts is about $3.12 within the $10 approval. No full-study budget is approved.

## Preparing and training

Preparation preserves source IDs, pins revisions, quarantines malformed/overlength targets and overlap candidates, and fingerprints the exact data consumed by training. Prepare on CPU **before** allocating a training GPU.

SFT preparation also saves verified token IDs, completion masks and packing boundaries, using the pinned TRL preprocessor. GPU SFT consumes this cache without repeating tokenization. The Modal study first warms pinned weights and benchmarks on CPU into `/vol/hf-cache-v2`; GPU images run offline. Direct Modal training entry points require this runtime cache and prepared data already to exist.

```bash
python scripts/01_prepare_sft_data.py --condition controlled_verified
python scripts/02_prepare_grpo_data.py --condition controlled_grpo

# Local GPU training, after resource approval:
python scripts/03_train_sft.py --condition controlled_verified --seed 42
python scripts/04_train_grpo.py --condition controlled_verified_grpo --seed 42 \
  --sft-checkpoint outputs/training/controlled_verified/RUN/sft
```

The historical condition names remain available, including `re_distill`. Orca remains a worked-solution recipe; the genuine final-answer-only control is `controlled_answer_only`. Newly prepared historical recipes can differ from their original data selection because invalid targets and overlaps are now quarantined.

## Controlled study

The six primary arms are the untouched checkpoint, final-answer-only SFT, unfiltered-trace SFT, verified-trace SFT, GRPO from the untouched checkpoint, and the same GRPO procedure after verified-trace SFT. The SFT arms share 1,000 Open-R1 questions. Training seeds are 42/43/44, with separate token-matched unfiltered/verified ablations. No pretrained-only cohort is included. Controlled SFT arms use a common 10% proportional warmup: the final-answer-only dataset packs into seven sequences and only three optimizer steps over three epochs, so a fixed ten-step warmup would cover the entire run. Historical recipe settings retain their fixed warmup.

```bash
# Local dry run: no Modal import, image build, or remote job.
python scripts/08_run_study.py --experiment controlled
python scripts/08_run_study.py --experiment legacy
python scripts/08_run_study.py --experiment controlled --prepare-only

# Produce the small pilot's reservation proposal without running it.
python scripts/11_quote_pilot.py
```

Main-study execution requires a quote derived from measured pilots and an explicit budget. The runner executes sequentially, applies per-job timeouts/resource limits, persists reservations, and refuses ambiguous retries after interrupted paid calls. Reservations are **not a provider billing guarantee**: image builds, initialization, storage and egress need allowance, and a Modal workspace budget should be configured before paid work.

The quote generator prints new content-addressed proposal paths under `results/proposals`; it preserves original approval artifacts. After agreeing a cap for a new run, use its printed quote path:

```bash
python scripts/08_run_study.py --experiment pilot --execute \
  --quote PRINTED_QUOTE_PATH --budget-usd AGREED_CAP
```

The main quote can then be generated from the completed pilot ledger with `scripts/12_quote_study.py --pilot-ledger LEDGER`. Collect a completed main study with `scripts/13_collect_study.py --ledger LEDGER`, then analyze its `campaign.json` using `scripts/09_analyze_results.py --campaign-manifest CAMPAIGN --output NEW_ANALYSIS`. These tools refuse missing pilot measurements and incomplete study records.

The first pilot's prose-scoring defects are preserved in its [original outcome](results/recovery/pilot-outcome.json). The [final validated run](results/recovery/pilot-final-outcome.json) uses the explicit answer contract, reuses unchanged SFT, freshly prepares GRPO prompts, runs six updates and 32 evaluations, and verifies results independently. Old and new-protocol scores are not a training-improvement comparison. No full-study or hardware-comparison budget is approved.

## Evaluation

```bash
python scripts/05_evaluate.py --condition verified_seed42 --model-path MODEL_EXPORT --benchmarks gsm8k
python scripts/05_evaluate.py --condition qwen_original --base-model-only --benchmarks gsm8k
python scripts/05_evaluate.py --condition verified_sampled --model-path MODEL_EXPORT \
  --benchmarks gsm8k --num-samples 8 --temperature 0.6
python scripts/05_evaluate.py --condition verified_math --model-path MODEL_EXPORT \
  --benchmarks math500 --max-new-tokens 8192
python scripts/06_analyze_traces.py --results-dir outputs/evaluations
```

Each run retains `evaluation.json`, raw `predictions.jsonl`, and a reproducible `summary.json`. `accuracy` means first-sample accuracy; pass@1/4/8 use the complete sampled correctness matrix. The old ambiguous benchmark name `math` is rejected; it was a NuminaMath training-tail diagnostic.

New GRPO and evaluation use `boxed-final-v1`: a shared prompt requests one final boxed value, which the primary scorer reads. Missing/malformed/multiple boxes and unfinished thinking are reported separately. Integer gold answers compare exactly. See the [contract](docs/remediation/scoring-policy-boxed-final-v1.md). Frozen prose versions v2-v6 and `legacy-v1` remain diagnostic/reproduction options, not primary new-study metrics. `extract_answer_auto` is an archived diagnostic helper; production grading uses `score_response`.

The bounded distillation study retains these strict GPU scores as a secondary endpoint. Its prospectively frozen numerical primary endpoint is computed from the same saved raw responses and accepts repeated identical numeric final boxes while rejecting conflicts and unfinished thinking. It does not change archived scores or GRPO rewards.

CPU rescoring with `scripts/15_rescore_saved.py` preserves original files and writes new source-hashed artifacts. Choose an archived `--scorer` explicitly for old unprompted responses; applying the new contract retrospectively is rejected. Rescoring does not retrain the model or make historical and new-protocol scores directly comparable.

Evaluation batches are bounded by live sequence count and output token budget. Sampling uses a versioned canonical-batch seed policy; interrupted partial batches replay deterministically. Remote volume commits are periodic with a final flush. `performance.json` records timing, actual generated tokens and GPU samples for each function invocation. Training writes sibling `sft-performance.json`/`grpo-performance.json` files. These measurements must establish throughput and memory headroom before scaling the study.

Resume automatically uses the full immutable run identity, or accepts `--resume-dir` for one benchmark. Changing weights, dataset, scorer, template, decoding settings, or runtime identity refuses resume. The strict final-response scorer and historical `legacy-v1` policy are explicitly distinct.

## Historical recovery

The archive index records hashes; local model files remain in ignored storage. The original source/PDF and six original evaluation files were preserved before revision.

```bash
python scripts/07_recover_history.py --download-weights --destination outputs/historical/NEW_ARCHIVE
python scripts/09_analyze_results.py --historical-archive outputs/historical/2026-10-19

# Explicit reconstruction, after allowing for model download/RAM/disk:
python scripts/10_import_checkpoint.py \
  --source outputs/historical/2026-10-19/outputs/sft_then_grpo \
  --sft-parent outputs/historical/2026-10-19/outputs/sft_then_grpo/checkpoint-117 \
  --condition sft_then_grpo --output outputs/imported/combined
```

Legacy adapters are never silently attached to a guessed base. New complete exports load independently; intermediate adapters retain their parent identity. Historical revision uncertainty is recorded rather than erased.

## Research interpretation

The bounded three-seed distillation study and checkpoint-loading subset ablation are complete; the comprehensive six-arm controlled study and corrected historical full-benchmark evaluation remain unrun. Statistical separation does not establish a causal verification benefit. Equal rewards do not demonstrate entropy collapse; verification phrases do not establish faithful self-correction; and the historical combined-model failure does not prove that SFT and RL inherently conflict.

The paper source is tracked at `paper/paper.tex`. Rebuild with `scripts/build_paper.sh`; this uses the completed follow-up's saved proposal, analysis, raw outputs and execution ledger to regenerate the tables and prose on CPU before compiling the PDF. The published proposal includes `execution-ledger.json`, so a fresh checkout does not need private `outputs/` files or cloud access to rebuild the paper. The [rewritten findings](results/FINDINGS.md) also show answer-format sensitivity: the strict one-box secondary rule changes the 8K point estimate to +2.60 points (conditional interval −2.60 to +8.07), while the numerical primary remains −1.30 points (−5.73 to +2.86). Neither establishes an 8K advantage. Structured historical results remain preserved under `results/historical/`; new measurements must be stored separately.

## License

MIT. Upstream models and datasets retain their respective licenses and terms.
