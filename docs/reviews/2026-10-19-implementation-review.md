# Implementation review

Review performed sequentially in the primary context, following the repository's no-subagent instruction. This is not an independent-model review or a claim of full GPU validation.

## Actionable findings

The review found and corrected these issues while implementing the audit plan:

- Dataset libraries can fall back to the newest cached revision in offline mode. Preparation now rejects a cache not tied to the requested commit and binds prepared datasets to the preparation implementation.
- Intermediate adapters need their actual SFT base, not only a parent label. Retained adapters now resolve and verify their parent; final complete exports load independently. GRPO also checks the declared SFT condition and seed.
- Duplicate study/evaluation writers can defeat resume correctness and spending reservations. Both now use exclusive writer locks; interrupted paid calls cannot be blindly launched again.
- Per-model persistence checks also consume resources. They are separate quoted study nodes, and resource/time limits apply to all nodes. Startup allowances are reservations, not guaranteed billing limits.
- Pilot updates must target the requested phase, including conditions supporting both phases. SFT/GRPO pilot overrides now select the phase explicitly.
- Tokenizer forwarding, packed completion supervision, first-sample metrics and retained response text required actual behavioral checks. Real CPU model/trainer tests and saved-result integration tests cover these boundaries.
- Historical model cards disclosed partial runtime versions; those are recorded separately from the new locked stack rather than inferred to be identical.

- Visual QA caught a rendering-order defect: sorted JSON comparison keys were zipped with a separate label list. Rendering now joins by explicit comparison IDs, with a regression test checking both table values and plotted effects against independently calculated count differences.

## Coverage

Reviewed source, entry points, configuration, dataset selection, model serialization, statistical routines, budget/resume flow, generated paper artifacts and the revised documentation. Tests, lint, local protocol dry runs, historical count reconciliation and CPU data preparation provide local evidence. No external review provider was used.

## Verdict

The local repairs are implemented and CPU checks pass. Full-size GPU image compatibility, real GRPO exposure accounting, separate-container persistence and corrected research results are explicit outstanding gates in `docs/remediation/STATUS.md`. The study is not represented as empirically complete.
