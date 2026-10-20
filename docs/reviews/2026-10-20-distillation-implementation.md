# Bounded distillation implementation review

Scope: `src/distillation_experiment.py`, `scripts/17_distillation_study.py`, the matched-supervision branch in `src/data.py`, the padding-free option and FlashAttention guard in `src/training.py`, new Modal preparation/training functions, and `tests/test_distillation_experiment.py`.

Intent: answer the distillation question within the existing $20 envelope, including prior ablation spending. No GRPO, original-archive mutation, commits, pushes or external publication are included.

## Actionable findings

Resolved before scientific model execution:

1. Short answer targets packed into very few sequences would receive too few updates. Both arms now retain separate question rows, share seeded question minibatches and get 64 updates. Padding-free FlashAttention avoids paying for padded tokens while retaining question boundaries.
2. The numerical endpoint must accept repeated identical final answers prospectively while rejecting conflicts, unfinished thinking and malformed boxes. Versioned tests cover these cases. The existing strict scorer is preserved as a secondary endpoint and raw-generation audit.
3. Training seeds must not multiply the number of independent benchmark questions. The main interval resamples the shared questions after averaging seed differences; per-seed effects and their sample SD remain separate.
4. Independent execution chains must not each spend the whole cap. A global quote covers every job before dispatch, and each disjoint chain receives only its own reservation. Completed and failed-chain tests exercise all 37 nodes and refuse duplicate paid calls.
5. Initial CLI execution passed string paths to the Path-only atomic writer. It failed before any remote call; app `ap-wquVeB6J5sISSERXQ6R3DK` was verified stopped with zero tasks. The affected writes now use Paths and the CLI boundary is tested for both success and failure. No model-result-dependent retry occurred.
6. A numerical-only instruction conflicts with symbolic OpenR1 targets. App `ap-SWHIoHEMttHLW4x0DvKLmD` was stopped during CPU preparation with no GPU tasks launched. Both training arms now request a mathematical expression; the cache specification additionally binds the matched-supervision implementation. The stopped CPU reservations remain inside the original envelope.
7. Fifty original teacher finals contain multiple boxes. App `ap-iEbSoWAWgoSG51hH9sQkDT` was stopped during CPU preparation, before GPU allocation. The final protocol preserves the teacher's explicit thinking block and gives both arms the same single-box final field; a complete 500-row audit confirms this and unchanged thinking content. No model outcomes informed the change.

## Coverage

Source review, data/provenance checks, cost-bound arithmetic, failure scenarios and simplification ran sequentially in the primary context under the user-supplied AGENTS.md mapping. No independent model or cross-provider review ran. The compound review skill's prescribed separated-agent receipt was unavailable under that instruction; this is a manual source-review record, not a claim to such a receipt.

The implementation reuses the existing reservation executor, file locks, versioned raw evaluation path, complete model export, dataset preparation, paired statistics and immutable JSON writer. A small wrapper partitions independent chains; it does not introduce a second training or inference engine.

CPU tests cover matched targets, prospective scoring, reservations, chain execution, failed-call refusal, CLI bookkeeping and seed/item uncertainty. Existing real tiny Qwen/PEFT tests cover training, masks and export/reload. Ruff checks the executable source, scripts and tests. The broad `ruff check .` also reports two pre-existing findings in the preserved audit helper under `docs/reviews/2026-10-19-audit-checks.py`; that historical helper is outside this change.

## Verdict and limits

The separately authorized bounded run is complete. Full-size training exposure, export hashes, raw-response persistence, matching evaluation identities and CPU score reproduction passed their checks. The frozen source/configuration remained unchanged during model execution. Outcomes cannot establish reasoning emergence, faithfulness, equal-compute efficiency or generality across students and datasets.

Full-size training verification completed for all six exports: two epochs, 64 updates, 127,656 answer-arm input tokens or 5,383,134 trace-arm input tokens per seed. Export weights are all distinct and bind the declared original student. The saved token totals agree at every update with the expected seeded question batches for all six models. This reconstructs source IDs from the sampler and checks the observed token stream; it is not an independently logged per-GPU source-ID trace. See the CPU-only validator `scripts/18_validate_distillation.py` and the frozen proposal's `collected/training-audit.json`.

All fourteen evaluations and separate persistence checks completed; 2,240 raw responses have matching hashes and strict CPU scores. The 2K finding is qualified by the collapsed answer-only recipe, its supervised empty thinking blocks, and the absence of demonstrated improvement over the unchanged student in the small 4K check. The reporting regression test protects pairing of the same 64 questions across ceilings. Local checks pass 79 tests plus 11 subtests, scoped Ruff and whitespace checks.

After GPU completion, review corrected the collection path for sibling `sft-performance.json` and the forward unpacked-token scope label. Archived scientific source and outputs were preserved. A CPU-only collection regression removed all six local profiles, downloaded the correct remote files, and reproduced their hashes without changing final analysis. The strict source guard now rejects the original proposal under changed source; it must not be rerun. See `results/recovery/distillation-postrun-code-fixes.json` and `distillation-collection-fix-validation.json`.

The final app inventory shows all experiment apps stopped with zero tasks; the final container inventory is empty. New elapsed-time resource cost is estimated at $7.01, with conservative combined accounting of $11.10 against $20. No invoice reconciliation, independent-model review, commit or publication is claimed. The [final results report](../experiments/distillation-results-2026-10-20.md) records interpretation and evidence.
