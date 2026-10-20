# Explicit final-answer contract

`boxed-final-v1` is the current policy for new training and evaluation. The shared prompt requests exactly one final answer in `\boxed{...}` after reasoning. GSM8K asks for only the numerical value, with no units/explanation inside the box; MATH-500 asks for the mathematical expression.

The scorer reads that field only after completed thinking. It does not select a number from prose. Missing, malformed, unparseable or multiple final boxes are format failures; unfinished thinking is reported separately. Integer ground truths compare exactly, preventing off-by-one errors at large magnitudes and tiny nonzero values matching zero. Finite decimal/fraction tolerance is retained for noninteger gold values. English cardinal values are accepted only when the entire boxed value parses. Symbolic equivalence uses Math-Verify.

This is a material protocol clarification, applied uniformly to every new arm. Primary `accuracy` is first-sample success under the requested output contract, not unrestricted semantic grading of arbitrary prose. Format-failure and unfinished-thinking rates accompany it. SFT source targets are retained; GRPO and evaluation share the same appended instruction. Question/source IDs are unchanged, overlap checks run before instruction insertion, and prepared-data/run identities bind the instruction implementation.

The design follows [Math-Verify's guidance to request boxed answers](https://github.com/huggingface/Math-Verify). Pilot audits showed why progressively broader prose heuristics were insufficient: a response can state both input quantities and a correct total. Numeric parsing alone cannot reliably select the intended field. The explicit output contract removes that ambiguity instead of assigning mathematical correctness to a guessed quantity.

Measured prose policies v2-v6 remain frozen for reproduction/diagnostics, not primary new-study grading. The re-exported `extract_answer_auto` is the archived v6 diagnostic utility; new training and evaluation use `score_response`, which never calls it. Old unprompted generations cannot be retrospectively treated as though the new instruction was present; CPU rescoring enforces this distinction.

Historical results remain unchanged. Old and new pilot scores are not directly comparable as a training-performance improvement. Pilot examples are development evidence; full-benchmark results, long-context/sampled GPU validation and cost-optimal hardware remain outstanding research work.
