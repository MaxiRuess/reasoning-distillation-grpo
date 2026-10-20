# Pilot defects repaired — October 19, 2026

The identified implementation defects are repaired and verified. The central scoring fix is an **explicit final-answer contract**, shared by GRPO and evaluation. Prose heuristics are archived for reproduction and diagnostics; they are no longer the primary metric. Original model, response and score artifacts remain unchanged.

The final validation ran six GRPO updates, 32 evaluations and an independent-container persistence check. All 28 completed answers supplied valid final fields; four responses stopped during thinking. There were no format failures among completed responses. These are development observations, not full-benchmark findings or evidence that training improved accuracy.

## Repairs

| Defect | Repair |
|---|---|
| Prose extraction chose inputs instead of a stated total | Shared prompts request exactly one boxed final value. The scorer reads that field after completed thinking; it never guesses a quantity from prose. |
| Large integer comparisons accepted off-by-one errors; tiny nonzero values matched zero | Integer ground truths compare exactly; finite decimal/fraction handling remains for noninteger values. |
| GRPO read an unused SFT token counter | Read trainer-state tokens with explicit prompt-plus-completion scope; observe completion tokens/groups independently. |
| Vendor reward log ignored configured weights | Add weighted objective means and zero-variance diagnostics alongside the raw-sum field. |
| Changed instructions could escape cache/run identity | Bind the answer-contract version and instruction hash. Check dataset overlap before appending instructions. |
| Old responses could be relabelled under a new contract | CPU rescoring writes new source-hashed artifacts and rejects treating old unprompted generations as new-contract results. |
| Old pilot evidence could calibrate a changed study | Reject stale reward/evaluation policies and zero/inconsistent exposure counters. |
| New proposals overwrote approval files | Write proposals to content-addressed directories; preserve identical files and refuse changed ones. |
| Modal image order and vLLM sleep/sync failed | Set environment layers before source mounts; keep sleep disabled in the pinned stack. |
| Distributed state and project deprecation noise remained | Explicit group cleanup, supported dtype/reinit arguments and cleared sampling-only flags for greedy evaluation. Vendor-internal warnings remain upstream. |

Policy: [boxed-final-v1](scoring-policy-boxed-final-v1.md). This is a **material new-run protocol clarification**: all new arms receive the same request; primary accuracy includes output-contract compliance, with format/unfinished-thinking rates separate. SFT source supervision is retained. Historical scores are not retroactively relabelled or compared as a training improvement.

## Verification

- **53 CPU tests pass**, including real SFT/GRPO training, checkpoint round trips, current/archived scoring, exact integer matching, shared prompts, overlap checks, interruption/resume, CPU rescoring, approval preservation and stale-source quote rejection. Ruff and whitespace checks pass.
- Fresh GPU validation used the unchanged verified-SFT export, freshly prepared GRPO prompts, **six updates**, **48 rollouts**, **six unique groups**, **50,392 total tokens** and **45,344 completion tokens**.
- Trainer state, metadata and profiles agree. Weighted objective logs match correctness-only weights; nonzero gradients occurred at nonzero learning rates.
- The complete GRPO export retains its SFT parent. Its model-weight hash differs from the parent, and evaluation loaded the verified export.
- All **32 predictions** reproduce on CPU. Their hash and summary match a separate container's read.
- All **500 GRPO source IDs** are unchanged. Only the shared instruction changed; its version/hash and measured prompt lengths are recorded.
- GPU source/configuration hashes match the reviewed implementation. All pilot apps are stopped. The canonical five-page PDF was rebuilt and every page visually inspected.

Evidence: [pilot-final-outcome.json](../../results/recovery/pilot-final-outcome.json). Completed ledger: `outputs/studies/670043ab83e41bfba8bcc140/ledger.json`. Successful [Modal app](https://modal.com/apps/maxiruess/main/ap-zwvnwtALx33P1B86kJydzO).

Review was targeted and manual in the primary context because the branch contains prior uncommitted work. It was not an independent-model review. File-wide simplification of overlapping edits was skipped. No commits or pushes were made.

## Spending and limits

Reported workspace metered usage increased approximately **$3.12** across all attempts and repairs, within the original **$10** approval. Billing can lag; credits cover billed cost in the snapshot. Completed jobs were reconciled using measured client-call time plus at least 120 seconds at bounded rates; failed attempts retained full reservations. A separate $1 image-build allowance remained. Per-task accounting and the final quote hash are in `pilot-boxed-authorization.json`; original approval files were preserved.

No full study, historical reconstruction evaluation or hardware comparison ran. The pilot does not establish optimal GPU/batch selection, long-context/sampled behavior, full-benchmark accuracy or the research hypothesis.

## Diagnosis and prevention

Prose policies v2-v6 reproduced their records but remained unsuitable as primary measurement: correct totals coexist with inputs and explanatory context. Broader parsing did not make a field authoritative. The final fix makes the output contract explicit, following [Math-Verify's guidance](https://github.com/huggingface/Math-Verify). Old policies/evidence remain available and labelled in `scorer-policy-status.json`.

The [development log](pilot-defect-investigation-log.md) preserves intermediate work. New production paths use `score_response`; `extract_answer_auto` is an archived v6 diagnostic utility only. Regression fixtures now cover the missing prose/contract and real GRPO accounting boundaries. Scorer fixtures and pilot questions are development evidence, not independent scorer-accuracy validation.
