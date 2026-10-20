# Sampled inference follow-up — October 20, 2026

The user approved implementation of the proposed 128-question evaluation package with a **new, additional $30 cap**. This does not reopen the completed $20 experiment or authorize training. The frozen proposal is `results/proposals/inference-extension/f1b156f14754100c186bc396`.

## Fixed scope

Use the unchanged pinned post-trained Qwen3-1.7B and the three existing trace SFT exports (seeds 42/43/44). Select 128 of the original 256 GSM8K questions with seed 20261022, without looking at new outputs. The question texts, answers, original indices and model artifact identities are frozen. These are previously used questions, not a new hidden test set.

All four models receive one sampled generation per question at each of 2,048, 4,096 and 8,192 output tokens. Thinking-mode sampling follows the current model card: temperature 0.6, top-p 0.95, top-k 20; min-p is disabled by the model/runtime default. The fixed decoding seed is 20261023. The evaluator now seeds every sampled canonical batch, including the single-sample case. Source identities and sampling-policy version distinguish these from the original greedy results.

Two small unchanged-student no-thinking diagnostics use the same 128 questions at 2K: recommended sampling (temperature 0.7, top-p 0.8, top-k 20), and the original greedy policy. The greedy diagnostic permits a decoder-matched comparison with the old answer-only outputs on precisely these questions; the sampled diagnostic describes the recommended no-thinking policy. The two policies must not be silently pooled.

The existing generated-token ceiling yields effective batches 32/16/8 at 2K/4K/8K. Batch size therefore changes with the output ceiling, and the study cannot isolate a pure causal effect of ceiling alone. The canonical-batch RNG seeds also depend on batch membership; the runs are separately sampled trajectories, not prefixes of a common 8K rollout. One decoding seed also does not estimate sampling variability. Eight thousand output tokens remains below Qwen's recommended 32,768; this is not an unconstrained capability evaluation.

## Outcomes and analysis

The prospective follow-up primary comparison is mean trace-minus-unchanged numerical accuracy at 8K across the three trained seeds. This is a new development-informed protocol, not a change to the original primary endpoint. Use the unchanged `completed-consistent-numeric-boxes-v1` numerical endpoint and retain strict `boxed-final-v1` scores as secondary. Every strict score must reproduce on CPU from the retained raw response.

Use 10,000 paired item-bootstrap draws with seed 42, averaging seed differences within each question. Report a conditional item interval and separate training-seed SD; there are 128 shared questions, not 384 independent observations. Results at 2K/4K, length changes, conditional accuracies and no-thinking comparisons are descriptive. Report end-of-sequence termination separately from unfinished thinking: closing the thinking block does not guarantee a completed response. Conditional accuracy selects model-dependent subsets and cannot establish a causal reasoning mechanism.

Preserve all raw responses, actual useful token counts, end reasons, model/runtime/template identities and hashes. Each of 14 GPU evaluations has a separate CPU persistence verification. CPU preflight validates exact source, frozen dataset texts, original-student weight hashes, all three complete exports and prompt/context headroom before GPU allocation. No SFT, GRPO, second benchmark, extra sampling seeds, automatic retry or outcome-dependent extension is included.

## Cost and execution

Prices were rechecked against https://modal.com/pricing on October 20. L40S GPU $0.000542/sec plus two physical CPU cores and 16 GiB at their cap rates totals $0.00060372/sec ($2.173392/hour). Credits do not justify this budget.

The 14 GPU calls,14 CPU verifications and CPU preflight reserve **$28.0453128** including 120-second startup allowances per task. A separate **$1** incidental allowance gives **$29.0453128**, below the additional **$30 cap**. Estimated expected cost remains approximately $16–25;8K and sampled-generation timing are extrapolated. Reservations are not an invoice or a provider billing guarantee.

GPU hard timeouts are 1,200/2,400/7,200 seconds at 2K/4K/8K and 600 seconds for each no-thinking diagnostic. CPU preflight is limited to 600 seconds, verification to 90 seconds. Two-second scale-down, zero configured retries and fixed CPU/memory caps apply. Four disjoint chains share one global reservation, with immutable per-chain ledgers and file locks. Any failure or ambiguous interruption requires reconciliation; the runner refuses to silently repeat paid work.

Run through `scripts/20_inference_extension.py`; the default freezes a proposal without cloud jobs. Paid execution requires `--execute --proposal EXACT_FROZEN_PROPOSAL --budget-usd 30`. Collection uses `--collect-only` and never launches model jobs. `outputs/inference-extension-active.json` records the active app and ledger. Original results remain preserved separately.

## Local verification

The expanded CPU suite passes 84 tests plus 11subtests. The single-sample RNG regression failed before the fix, then passed. New tests cover complete shared-question reservations, all14 correct model paths through the actual reservation executor, duplicate-call refusal, paired-question integrity and the distinction between EOS, unfinished thinking and missing final answers. Scoped Ruff and whitespace checks pass. Manual source review and simplification run sequentially under the supplied AGENTS.md mapping; no independent reviewer or separated-agent receipt is claimed.

## Completion

All 29 tasks completed successfully; all 1,792 new responses passed separate persistence and local CPU checks. The [results report](inference-extension-results-2026-10-20.md) retains the original protocol's scope and identifies the new primary result: trace mean 86.20% versus 87.50% unchanged at 8K (−1.30 points, conditional interval −5.73 to +2.86). Live inventory confirms the app stopped with zero tasks and no containers. Estimated resource-time cost is $13.34, or $15.47 including all extra allowances, within the additional $30 cap. These are estimates rather than reconciled bills. No extra GPU jobs, training or retries were added; unused funds do not authorize further work.
