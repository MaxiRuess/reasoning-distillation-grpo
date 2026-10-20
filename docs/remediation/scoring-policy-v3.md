# Final-response scoring policy v3

This measured version is frozen. The current [v4 policy](scoring-policy-v4.md) corrects two additional prose cases found in v3's fresh pilot; original v3 results remain unchanged.

`final-response-v3` corrects the pilot's numeric prose extraction defects. Its rules do not inspect the ground truth while extracting a prediction. Numeric matching happens afterward. `final-response-v2` is frozen in `src/scoring_v2.py`, and `legacy-v1` remains available for explicit reproduction.

## Extraction rules

1. Require balanced, non-nested thinking tags. Exclude thinking from extraction; unfinished reasoning receives no numeric answer.
2. Prefer the final boxed answer or `####` answer. Require a complete finite numeric value, including supported fractions; do not accept a numeric prefix of an expression.
3. Recognize explicit numeric answer statements and `Answer:` labels followed by a numeric assertion. A descriptive phrase such as “the answer is straightforward” does not override a completed answer.
4. Otherwise examine the concluding paragraph. Prefer a concluding assertion or stated total; then examine its numeric assertions in order. Recognize values after ordinary predicates and the scalar right-hand side of a final equality. Temporal/comparison context following an asserted value does not replace that value.
5. Reject malformed numbers, nonfinite values, unresolved alternatives, unfinished expressions, questions and deliberation without a conclusive assertion. Continue to expose `unparseable` rather than invent a prediction for genuinely ambiguous output.

Markdown emphasis does not affect extraction. Numeric tolerance and MATH-500 symbolic equivalence retain the previous policy. MATH-500 extraction is unchanged; this revision targets GSM8K numeric prose.

The project policy is more restrictive than the [evaluation harness's flexible last-number extraction](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml). It preserves the separation between thinking and final answers. [Math-Verify](https://github.com/huggingface/Math-Verify) remains the symbolic verifier; this project does not infer that a number appearing somewhere in reasoning is the final answer.

## Verification and provenance

Regression fixtures include ordinary prose, explanatory tails, answer headings, currency, fractions, temporal context, Markdown and equations, plus ambiguous expressions, distractor numbers and incomplete thinking. They are development fixtures, not a held-out estimate of scorer accuracy. The three pilot false negatives are checked separately against saved generations. All 32 original v2 scores reproduce exactly through the frozen implementation.

CPU rescoring creates new predictions, summary and provenance under a new identity. It records the source prediction, manifest and summary hashes, scorer implementation and runtime. It never changes the original response/score files, and explicitly records that neither generation nor training was repeated. Post-hoc rescoring cannot repair the rewards supplied to an earlier GRPO run; corrected rewards require fresh training validation.

## Objective and token reporting

New GRPO requests declare `reward_scorer: final-response-v3`. The pipeline rejects stale reward-policy declarations. Metadata records the version, component weights and measured weighted-objective statistics. `objective/reward_mean` and `objective/zero_variance_fraction` accompany the upstream unweighted `reward` field in step logs; these are distinct quantities when a reward component has zero weight.

GRPO processed tokens come from `trainer.state.num_input_tokens_seen`, with explicit `prompt_plus_completion` scope. Completion tokens and rollout/group/source-ID counts remain independently observed in the reward callback. SFT retains its packed-input token counter. Main-study quotation rejects missing/stale reward policies, mismatched evaluation scorers and zero/inconsistent GRPO exposure counters.
