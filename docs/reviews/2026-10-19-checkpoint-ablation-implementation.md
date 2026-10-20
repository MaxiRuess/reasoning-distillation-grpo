# Checkpoint-ablation implementation review

Reviewed: 2026-10-19

## Intent

Implement the accepted evaluation-only comparison of the exact historical SFT parent, intentionally missing-SFT reconstruction, and corrected sequential reconstruction. Freeze 256 questions excluding the recent pilot, retain raw evidence, and keep new spending within a separate $20 cap. Preserve historical measurements and prohibit training or automatic paid retries.

## Actionable findings

No unresolved implementation findings were retained in the scoped sequential review.

Resolved during implementation:

- The existing evaluator supported only prefix subsets. Frozen indices now preserve original problem IDs and check the selected row digest. Duplicate, negative, out-of-range, empty and Boolean indices are rejected.
- The executor previously trusted persistence verification without comparing the returned summary. It now requires equality with the producer's summary; the existing orchestration test fixture was updated to meet that real callback contract.
- Local collection now reproduces every saved score from raw responses on CPU and checks ground truths against the frozen protocol. This supplements independent-container hash checks.
- Repeated collection preserves an existing response-review CSV instead of overwriting annotations.

## Coverage

Reviewed the new experiment module and CLI plus changes to evaluator subset selection, shared executor callbacks, Modal entry points and corresponding tests. The review checked checkpoint lineage, CPU memory cleanup, source/protocol identities, paired analysis, persistent evidence, cost reservations, failure/retry behavior and preservation of historical results. Existing artifact, statistics and evaluation machinery is reused. Small repeated dictionaries are retained where they express different persisted contracts; no general experiment framework was added.

The real tiny Qwen/PEFT test exports all three constructions, verifies exact reload logits and rejects an incorrect SFT parent. Additional tests cover deterministic exclusion/selection, changed source/protocol rejection, all-in spending checks, subset dispatch and refusal to repeat an interrupted paid call. The initial new tests failed before implementation. The final full suite passed: 65 tests plus 11 subtests. Ruff and formatting checks passed.

Project instructions require sequential work in the primary context. This is a manual review applying correctness, standards, testing, maintainability and reliability lenses; it is not an independent model review or the multi-agent review workflow. No commits, pushes or PRs were authorized or performed.

## Verdict

Ready for the bounded experiment. CPU/GPU operational results are recorded separately after execution; passing software checks does not establish scientific conclusions or publication novelty. Runtime reservations total $10.4364324, plus $8 in explicit setup/recovery allowances, below the $20 cap. Provider billing may lag and is reconciled separately.
