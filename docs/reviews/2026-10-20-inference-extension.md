# Inference follow-up implementation review

Intent: execute the authorized 128-question inference-only package under a new $30 cap, using the four existing checkpoint identities. Preserve the completed study. Source review, simplification and failure analysis were performed sequentially in the primary context under the user's AGENTS.md mapping. No independent-model review or separated-agent Compound receipt is claimed.

## Actionable findings

Resolved before cloud execution:

- Single-sample stochastic generation was unseeded. The canonical-batch seed now applies whenever temperature is positive; sampling-policy v3 and source identities prevent silent mixing with old generations. A regression demonstrably failed before the change and passes afterward.
- The generic reservation executor always selected its parent or original student. Explicit frozen model paths are now supported; all 14 calls through the real executor are tested against their intended checkpoints, and the remote function rechecks frozen artifact IDs.
- JSON serialization sorts object keys. Sorting the jobs before constructing the plan keeps freeze/read/validate identical and avoids plan drift after serialization.
- Collection must bind source, sampling, batch size and dataset revision, not merely filenames and scores. All are checked alongside model identity, exact questions, separate persistence hashes and CPU strict-score reproduction.
- Empty thinking blocks and EOS completion are different events. Reporting retains both denominators instead of labelling every closed thinking block a finished answer.

## Coverage

Reuse the existing immutable JSON writer, global campaign reservation, disjoint chain ledgers, locks, no-retry guard, generation path, independent persistence verifier and raw-response scorer. New logic supplies a protocol, bounded quote, inference-only callbacks and follow-up analysis. It introduces no second model-serving or training engine.

Full local checks:84 tests and11 subtests pass, including original real CPU training/export tests. Scoped Ruff passes. The 14 GPU jobs and 14 verifications plus CPU preflight have a total worst-case reservation including incidental allowance of$29.0453128. Resource-time estimates remain distinct from a provider invoice. Completed and failed chain protections are inherited from the original tested executor; no outcome-dependent paid retries are permitted.

## Verdict

Local implementation is ready for the explicitly authorized bounded run. Full-size remote weight/data preflight and completed-generation persistence/identity checks remain runtime gates. The128question sample, one decoding seed, changing effective batch size and8K ceiling limit interpretation. Neither publication acceptance nor broad reasoning gains are promised. No commit, push or external publication is included.

The added saved-artifact integration test exercises all 14 local evaluation directories through the real collector and rejects an altered raw-response hash. The final local suite passes 84 tests plus 11 subtests.

## Runtime outcome

All preflight and collection gates passed: 29 successful tasks, all 14 checkpoint/policy identities, and 1,792 raw responses. The single fixed decoding-seed primary 8K interval includes zero; reporting makes no general reasoning or equivalence claim. Post hoc original-output diagnostics distinguish closed thinking from EOS and report both trace-only and unchanged-only correct pairs. Live shutdown inventory records the app stopped and no remaining containers. Estimated elapsed cost plus all extra allowances is $15.47 against $30; it is not a reconciled invoice. No scientific execution source was changed after freezing, and all 142 previously recorded original evidence files remain unchanged.
