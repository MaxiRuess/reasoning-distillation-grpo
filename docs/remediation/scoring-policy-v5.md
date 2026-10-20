# Final-response scoring policy v5

V5 is frozen as measured. The current [v6 policy](scoring-policy-v6.md) unifies contextual clause handling. V5 retains [v4](scoring-policy-v4.md) and adds English cardinal-number normalization, because fresh evaluation produced the correct final answer “two” in words.

Supported forms include zero through nineteen, tens with optional units, hundreds and descending thousand/million/billion/trillion groups, optional grammatical “and,” hyphenation, and initial “minus”/“negative.” Invalid sequences such as “one two,” “twenty thirty” and alternative answers remain unparseable. This is cardinal-number parsing, not unrestricted natural-language interpretation. Explicit abstentions do not receive credit for incidental quantities in their text. Numeric literals/fractions, thinking exclusion, bounds rejection and symbolic MATH-500 behavior retain the preceding policy.

Measured v2, v3 and v4 implementations are frozen and their saved records reproduce exactly under explicit version dispatch. Extraction never receives the ground truth. Development fixtures cover prose, answer sections, word numbers, malformed/ambiguous expressions, bounds, abstention and incomplete thinking; they do not establish a held-out accuracy estimate for arbitrary prose.

New GRPO rewards and evaluation declare `final-response-v5`. Metadata and `objective/*` logs identify the weighted reward policy. CPU rescoring creates new source-hashed artifacts; it does not retrain earlier models or overwrite their recorded scores. See the [defect-repair report](pilot-defect-fixes-2026-10-19.md) for validation and spending provenance.
