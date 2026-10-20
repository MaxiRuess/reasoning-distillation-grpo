# Full manuscript review and findings rewrite

Reviewed artifact: [paper source](../../paper/paper.tex) and [current PDF](../../assets/paper.pdf), titled *Reasoning-Trace Distillation under Output Limits: A Qwen3-1.7B Case Study*. This is a sequential source, numerical and editorial review in the primary context, not an independent peer review. The work uses saved outputs and CPU only.

## Scientific changes

The paper now leads with the completed sampled follow-up. The primary numerical result remains trace mean 86.20% versus unchanged 87.50% at 8K: −1.30 points, conditional paired item interval −5.73 to +2.86. The 2K difference of +12.24 points is descriptive. The original greedy experiment and its trace-minus-answer-only primary comparison remain separate and fully reported.

The review exposed an omitted interpretation of the retained strict scores. Five unchanged-student 8K responses pass the numerical primary rule but fail the exactly-one-box secondary rule because they repeat numerically consistent correct boxes. The strict unchanged score is 107/128 (83.59%), compared with 112/128 (87.50%) numerically; trace scores are unchanged. The strict secondary difference is +2.60 points, interval −2.60 to +8.07. The point estimate changes sign, but neither rule establishes an 8K advantage. The paper now reports this sensitivity explicitly. The [new secondary calculation](../../paper/inference-extension-secondary.json) does not change the frozen primary analysis.

Completion and length are reported as observations. At 2K, EOS rates are 69.53% unchanged and 84.90% across trace checkpoints; at 8K they are 97.66% and 97.92%. The trace students average 1,520 generated tokens at 8K versus 2,012 unchanged, despite trace-arm targets averaging 5,270 supervised tokens. Training and test tasks differ, and no teacher-length ablation supports a causal compression claim.

The original post hoc diagnostic reports both directions of disagreement: 99/101 trace-only-correct question–seed pairs involve an output-limited unchanged response, while 29/41 unchanged-only-correct pairs involve an output-limited trace response. Its conditional-accuracy table now includes the actual denominators and distinguishes EOS from the absence of unfinished thinking. Model-selected subsets do not identify reasoning ability.

Native no-thinking results contextualize the failed control: 77.34% greedy and 78.91% sampled on the same 128 questions, versus 6.51% for the original answer-only checkpoints. The paper distinguishes the native thinking-channel switch from an absence of worked reasoning and from the effects of training. Comparisons between sampled native no-thinking and trace SFT remain descriptive because their sampling parameters differ.

## Editorial changes

The title identifies a case study. The abstract, introduction, methods, discussion, limitations and conclusion were rewritten around the observed results. Training details and exact prompts move to the appendix; process history and legacy recipe rankings remain in repository documentation. Limitations are consolidated, including the reused benchmark questions, one decoding seed, unequal training exposure, changing batch/RNG streams and output ceilings below the model-card guidance. Final-answer accuracy is distinguished from reasoning correctness or faithfulness.

The findings document was rewritten to match the paper. Its earlier version, including historical audit and checkpoint-ablation results, is [archived unchanged](../../results/historical/FINDINGS-before-manuscript-rewrite-2026-10-20.md). The paper build now regenerates current result fragments from the completed saved follow-up rather than running the unused historical figure generator. It does not launch cloud jobs.

## Verification

The [numeric audit](2026-10-20-manuscript-rewrite-numeric-audit.json) reproduces strict scores for all 4,032 responses in 28 raw files. It checks all 1,792 follow-up responses against the frozen question text, answer and ID, reconciles numerical/strict accuracy and token means, and separately recalculates both 8K item-bootstrap intervals from raw correctness vectors.

All 248 original/follow-up evidence files recorded before rewriting, including the completed execution ledgers and original closing record, remain unchanged. The frozen scientific execution source is unchanged. The current manuscript supersedes the earlier PDF version referenced by the experiment's closing record; that historical record is preserved, and the new manuscript revision has its own hashes.

The PDF builds in two LaTeX passes with no final warnings, unresolved references or overfull boxes. All nine pages were visually inspected. Paragraph boundaries around generated floats were corrected. Scoped Ruff and whitespace checks pass. The scientific runtime was not modified, so the previously passing 84-test suite was not rerun for this prose revision; the new secondary analysis was verified by the separate CPU recalculation above.

## Source checks

Related-work descriptions and bibliographic details were checked against these primary sources:

- [Hsieh et al., Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/): rationale supervision in a multitask framework; Findings of ACL 2023, pp. 8003–8017.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948): reasoning distillation into smaller students.
- [Li et al., Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143): small-model learnability and mixing supervision of different complexity.
- [Chen et al., Do NOT Think That Much for 2+3=?](https://arxiv.org/abs/2412.21187): overthinking and self-training for shorter reasoning.
- [Aggarwal and Welleck, L1](https://arxiv.org/abs/2503.04697): LCPO is the method in this paper, not a separate work; accepted at COLM 2025.
- [Muennighoff et al., s1](https://arxiv.org/abs/2501.19393): trace SFT combined with inference-time budget forcing.
- [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B#best-practices): distinct thinking/no-thinking sampling guidance, warning against greedy thinking, and recommended 32,768-token output allowance.
- [Pinned OpenR1 dataset card](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/e4e141ec9dea9f8326f4d347be56105859b2bd68/README.md): DeepSeek-R1 generation provenance and inherited verification labels.

## Remaining scientific limits

The revised manuscript supports a narrow empirical case study. It does not establish a general reasoning gain, a causal compression mechanism, a well-tuned answer-only baseline or generalization beyond this student and benchmark. The strict and numerical intervals leave uncertainty in both directions. Rewriting improves the account of the evidence; it does not add independent replications or resolve those experimental limits.
