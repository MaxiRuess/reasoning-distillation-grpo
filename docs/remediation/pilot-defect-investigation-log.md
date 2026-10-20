# Archived pilot defect investigation

Development chronology only. Superseded by the [final repair report](pilot-defect-fixes-2026-10-19.md). Pending-validation statements below describe intermediate states.

Scope: defects discovered by the first small GPU pilot. The existing uncommitted research remediation is preserved; no commits or pushes are part of this work.

## Root causes and repairs

| Defect | Repair | Verification |
|---|---|---|
| Clear prose/cardinal answers were rejected; explanatory/contextual wording and restated inputs obscured conclusions | Versioned v6 shared numeric/cardinal/context policy; measured older policies frozen | Prose/cardinal/context/ambiguity fixtures; all 128 retained responses replayed; original scores reproduced; immutable CPU rescoring |
| GRPO's generic SFT token counter stayed at zero | Read GRPO trainer-state tokens and label their scope | Real tiny CPU GRPO training test plus explicit SFT/GRPO counter tests |
| Vendor aggregate reward log ignored configured component weights | Independently measure and log weighted objective mean and zero-variance groups | Real CPU trainer integration proves disabled format reward does not affect the weighted diagnostic |
| Old pilot could be mistaken for calibration of the repaired reward policy | Check declared reward policy and exposure before main quotation | Stale-version and zero-counter rejection tests; old configuration/quote cannot authorize current code |
| vLLM sleep/sync ordering failed on GPU | Keep sleep disabled in the pinned stack | Earlier recovered six-update GPU run; retained regression assertion and corrected-policy GPU recheck |
| Image environment settings followed local file mounts | Build environment before source mounts | Earlier corrected image build; current source images build in the recheck |
| Qwen reload triggered a spurious Mistral-regex warning | Explicitly disable that unrelated patch in project Qwen loaders | Exported backend/template comparison plus real CPU checkpoint round trips; internal vLLM loads still emit the upstream warning without changing the tokenizer |
| Deprecated project dtype/tracking arguments and inherited greedy sampling flags generated noise | Use supported dtype/reinit arguments and clear sampling-only options for greedy evaluation | CPU tests and runtime recheck; deprecated dtype calls inside the pinned vendor integration are unchanged |
| GRPO left its initialized distributed group to process exit | Explicit group cleanup after volume commit | GPU recheck verifies normal shutdown |

Policy: [scoring-policy-v6.md](scoring-policy-v6.md). CPU rescoring CLI: `python scripts/15_rescore_saved.py --source SOURCE --output-root NEW_ROOT`.

The original model's saved 32 responses rescore from 3/32 under v2 to 15/32 under v3. This is a post-hoc diagnostic of the scorer change on the same generations, not new training evidence or a full-benchmark result. Original files remain unchanged. The retained unfinished-thinking outputs remain failures. The summary is `results/recovery/pilot-v3-rescore.json`.

## Review and validation

The causal chain is confirmed by the original raw responses, the frozen-v2 reproduction, trainer state and real CPU GRPO behavior. There is no external issue of record. The branch contains prior uncommitted work, so review is targeted and manual in the primary context; it is not an independent-model review. File-wide simplification of overlapping prior edits was skipped. Existing tests should have covered descriptive prose tails and real GRPO counters; those cases now have regression coverage.

Local verification: 49 tests pass, including real tiny SFT and GRPO training; Ruff and whitespace checks pass. All measured older scorer versions are preserved. The paper's validation status was updated, its five-page PDF rebuilt, and every page visually inspected. Known remaining research limits include long-context/sampled GPU protocols and unmeasured hardware/batch alternatives; these are not represented as implementation defects repaired by a small pilot.

## Bounded GPU revalidation

The corrected-policy recheck uses the remaining original $10 pilot allowance. Both prior full-plan reservations and an additional $1 overhead allowance are conservatively retained, leaving a $3.90504 runner cap. The new full-plan reservation is $2.54748. CPU preparation and the unchanged SFT export are reused; GRPO, generation and independent persistence verification are fresh. No full study or hardware comparison is authorized.

V3's fresh run completed and correctly recorded 53,736 total GRPO tokens, 49,840 completion tokens and configured weighted rewards. Its evaluation exposed two remaining clear prose false negatives. The measured v3 policy was frozen; v4 removes the verb-allowlist dependency and examines full answer sections. Original v2 and v3 score records are preserved. No training-performance improvement is inferred from post-hoc rescoring.

V4 revalidation is running from `outputs/studies/988d96fc9d984daf13206d83/ledger.json`. Budget reconciliation retains the **full reservations for every task actually launched** in earlier attempts, plus a $1 overhead allowance. Only reservations for never-launched tasks are released, leaving a $3.6429432 runner cap within the original $10 approval. The new full-plan reservation is $2.54748. Unchanged CPU preparation and SFT are reused. Plan, quote and detailed provenance are in the `results/recovery/pilot-v4-*` artifacts. No full study or hardware comparison is authorized.

V4 subsequently completed all operational checks. Its two unparseable answers were a model abstention and a valid answer written as “two.” V5 adds cardinal-number parsing and explicitly rejects abstentions with incidental quantities. The v4 implementation and records are preserved. V5 revalidation uses `outputs/studies/a7a500bdc685a4ca2b4b6b85/ledger.json`, a remaining $2.0152608 cap and a $1.72313688 reservation. Completed preparation/SFT are prepopulated for reuse, with one-second guard timeouts plus startup allowances in their quote; they are never dispatched again. Every prior launched task remains fully reserved. The combined reservations and $1 overhead allowance remain below the original $10 approval.

V5 completed, correctly recording 61,560 total and 57,664 completion tokens. Its evaluation exposed a correct salary followed by an if-clause. V6 uses one context-boundary rule across all assertion paths. Replay across the 128 retained responses also checks explanatory tails with cardinal quantities, purpose clauses and temporal units. Original model/score files remain unchanged.

For the final v6 check, all completed jobs were reconciled using measured end-to-end call time plus at least 120 seconds at their bounded CPU/memory/GPU rates. Failed jobs retain their full original reservations, and a separate $1 image-build allowance remains. This yields a $3.68446619 prior upper-bound reservation and a $5.31553381 remaining runner cap under the original $10 approval. The new full-plan reservation is $2.54748; completed SFT/preparation are reused. Detailed per-task settlement is in `pilot-v6-authorization.json`; the new ledger is `outputs/studies/fed166b37e3b11a95114ce3e/ledger.json`. Final v6 GPU validation is in progress. Reservations remain estimates with hard job limits, not a provider billing guarantee.
