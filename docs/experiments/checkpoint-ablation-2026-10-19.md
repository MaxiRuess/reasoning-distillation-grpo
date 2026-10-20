# Historical checkpoint reconstruction ablation

Created: 2026-10-19

This protocol implements the user-approved evaluation-only experiment with an additional $20 compute cap. The preceding $10 pilot authorization remains separate. The original measurements and checkpoints are read-only evidence.

## Question and constructions

What is the effect of omitting the exact SFT parent when reconstructing the one historical SFT-to-GRPO run?

| Arm | Effective weights | Source |
|---|---|---|
| `sft_parent` | Pinned Qwen plus the original SFT update | `outputs/sft_then_grpo/checkpoint-117` |
| `missing_sft` | Pinned Qwen plus the original final GRPO update | `outputs/sft_then_grpo` |
| `reconstructed` | Exact reconstructed SFT parent plus the same final GRPO update | Both preceding sources |

The missing-SFT arm is an intentionally incorrect loading control. It is not the independently trained GRPO-only model. The SFT reference is the actual parent, not the separately trained `sft_traces` adapter.

## Frozen selection and evaluation

- Protocol: `results/proposals/checkpoint-ablation/6b95572ffeae5d5f98176ffd/protocol.json`.
- Full protocol ID: `6b95572ffeae5d5f98176ffd6f755143f4bd68e347b8c76d5814a02f06e99ede`.
- Exactly 256 GSM8K test questions, uniformly sampled without replacement with Python seed 2026, then placed in original index order. The 32 IDs in the final development pilot are excluded. The full benchmark was used in the historical study; this is not a previously unseen test set.
- The protocol stores every selected ID, question, ground truth, subset/population digest, adapter/config hash, model/dataset revision, and implementation hash before remote execution.
- All three arms use the same pinned official tokenizer, BF16 full exports, GPU backend, thinking-enabled chat template, boxed-final-answer request, greedy decoding, 2,048-token output budget, and batch size 32.
- CPU reconstruction checks original adapter hashes, binds the corrected arm to the exact SFT export, and requires exact logits on a fixed probe after saving and reloading. This is a serialization check, not a reasoning benchmark.
- Save raw responses, token counts, termination reasons, strict scores, format failures and unfinished-thinking rates. Frozen subset checks preserve original IDs and reject changed questions/answers.
- A separate container reads each result after the producer commits it. Local collection checks the raw-record hash and summary, reproduces every score on CPU, and requires identical evaluation identities apart from model identity.

## Analysis fixed before execution

The primary effect is `reconstructed - missing_sft` accuracy. The secondary, exploratory effect is `reconstructed - sft_parent`. Report counts, paired differences, 10,000 paired bootstrap resamples (seed 42), exact two-sided McNemar tests, and per-arm Wilson intervals. There is one primary comparison; the secondary is not a second confirmatory claim.

Preserve null or inconclusive results. Do not choose another subset, token budget, or checkpoint based on the resulting accuracies. Test-item uncertainty does not include training-seed variability. An interval containing zero does not establish equivalence or absence of a meaningful effect.

Archived legacy scoring of the same new responses is a CPU diagnostic only. It does not recreate historical scores, because the prompts and runtime differ. A shuffled CSV with model labels removed and a separate key supports response review; a review template is not evidence that an independent review happened. Primary results are not silently replaced by adjudicated scores.

## Spending and failure handling

The exact plan and quote accompany the protocol. Runtime reservations total $10.4364324 at the dated Modal rates, including two physical CPU cores, 16 GiB RAM and startup allowances. Each of the three L40S evaluation calls has a 5,400-second timeout. CPU preparation, three imports and three persistence checks have separate limits. Another $2 is allocated to setup/builds and $6 to recovery contingency, leaving $1.5635676 unallocated below the $20 cap. Timeouts are not a provider billing guarantee.

The shared persistent ledger records a reservation before each call. An interrupted or failed paid task blocks automatic retry until its outcome and cost are reconciled. No SFT or GRPO training can be dispatched by this experiment. Do not spend the contingency automatically or expand the sample if results look promising.

Run from the repository root:

```sh
PYTHONPATH=. python scripts/16_checkpoint_ablation.py
PYTHONPATH=. python scripts/16_checkpoint_ablation.py --execute \
  --proposal results/proposals/checkpoint-ablation/6b95572ffeae5d5f98176ffd \
  --budget-usd 20
```

Default invocation only freezes a proposal on CPU. Execution requires the exact frozen proposal and cap. It rejects changed source, configuration, pilot evidence, pricing or protocol before starting Modal. Completed ledger entries are reused; failed entries are not relaunched automatically. Use `--collect-only` with the same proposal and cap to collect an already completed run without starting any model jobs.

## Interpretation boundary

The historical base revision and complete transitive environments are unknown. The study uses one explicitly pinned reconstruction and a new answer-output instruction. It measures a loading intervention on these artifacts under this evaluation protocol, not an exact replay of the original numerical table. Common BF16 merging is used for all three evaluation exports; this differs from an unmerged-adapter historical loader and does not isolate floating-point merge effects.

The historical GRPO adapter was trained with the legacy reward implementation. Reconstructing its parent does not repair that training history. The secondary comparison therefore describes that particular historical training run and cannot validate the current GRPO training recipe.

The experiment cannot establish general benefits or harms of GRPO, reasoning emergence, a model-size threshold, or a causal benefit of verified traces. It can support a narrowly scoped reproducibility case study. Publication novelty and acceptance remain separate questions.
