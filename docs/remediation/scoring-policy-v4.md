# Final-response scoring policy v4

`final-response-v4` is frozen as measured. The current [v5 policy](scoring-policy-v5.md) adds English cardinal-number normalization. V4 retains v3's thinking exclusion, complete finite numeric values, explicit answer priority, numeric assertion/equation handling and ambiguity guards. It corrects two further defects found by fresh GPU evaluation:

- A complete single-value declarative statement does not need to contain a verb from a finite allowlist. Ordinary verbs such as “had” and “carried” are accepted. Deliberation, alternative answers, questions and unresolved bounds such as “higher than 12” remain rejected.
- An `Answer:` section is examined for its conclusive numeric assertion across sentences. Restating multiple input quantities in its first sentence does not hide a later explicit total.

Both v2 and the measured v3 implementation are frozen in `src/scoring_v2.py` and `src/scoring_v3.py`. All 32 original v2 records and all 32 measured v3 records reproduce through their explicit version dispatch. Current CPU tests include the newly exposed patterns, bounds and distractor checks, plus real SFT/GRPO trainer integration. Extraction does not inspect the ground truth.

See [v3's policy](scoring-policy-v3.md) for unchanged numerical/benchmark rules, provenance and weighted-objective reporting. New runs use v4 in both evaluation and GRPO rewards. CPU rescoring preserves the original scores and model identities; it does not repair rewards supplied to an older training run. Scorer fixtures and pilot questions are development evidence, not a held-out estimate of general scorer accuracy.
