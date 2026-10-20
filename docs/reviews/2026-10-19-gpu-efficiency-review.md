# GPU efficiency review — October 19, 2026

This is the pre-pilot assessment. The subsequent [pilot report](../remediation/pilot-2026-10-19.md) supersedes its untested-runtime assumptions: image layer order was corrected, vLLM sleep was disabled after a real synchronization failure, and the run exposed scorer false negatives and a token-accounting defect. Refer to that report for measured GPU results and remaining gates.

The GPU paths have several concrete efficiency fixes, implemented locally. The code is ready for a bounded measurement pilot. GPU utilization, full-size compatibility, the best batch size, and the cheapest suitable GPU remain unmeasured. No GPU jobs, remote image builds, or full-size model downloads were launched for this review.

Review performed sequentially in the main context. This is code inspection plus CPU behavioral verification, not an independent review or a GPU benchmark.

## Findings and changes

| Finding | Implemented change | Evidence and limits |
|---|---|---|
| Model/benchmark downloads could happen during billed GPU time | A CPU runtime-preparation node warms the pinned model and benchmarks into the existing persistent volume. GPU images use offline Hugging Face mode; all dependent GPU work waits for preparation. | Mocked pinned-cache test and complete pilot callback-order test pass. Real remote cache warming has not run. See [Modal model-weight guidance](https://modal.com/docs/guide/model-weights). |
| SFT repeated tokenization, packing and completion validation after allocating a GPU | CPU preparation uses the real pinned TRL trainer with a tiny randomly initialized CPU model. Saved token IDs, completion masks and document boundaries are hashed and verified before GPU weight loading. The GPU trainer skips preparation. | Real 1,000-example verified arm prepared successfully and reloaded with a fresh pinned tokenizer. Tiny integration tests compare raw/preprocessed TRL inputs exactly and reject altered token files. No pretrained weights are needed for this step. |
| Packed SFT could silently fall back to unsuitable attention | Packed CUDA SFT now requires the supported FlashAttention path rather than proceeding with SDPA. | Guard tested on CPU. The actual FlashAttention wheel/kernel remains a GPU pilot gate. |
| Sampled evaluation serialized generation by question | One generation call now handles several questions and all their samples. Greedy evaluation also uses a larger bounded batch. Output tensors transfer to CPU once per batch. Inference mode and KV caching are explicit. | Actual tiny Qwen sampling is identical after interruption inside a canonical batch. This reduces dispatches; it does not establish a throughput speedup. |
| GPU volume commits happened after every evaluation batch | Predictions still use per-question local fsync. Remote commits occur after four batches or at the first batch boundary after 60 seconds, plus a final flush, including handled failures. | Cadence/flush and persisted-response tests pass. A process/container failure can require replay since the last successful remote commit. |
| GRPO over-reserved the model's nominal maximum context and logged completions routinely | vLLM context is bounded by the prepared maximum prompt length plus the completion budget, rounded up to 128. Completion logging is off by default; main training metrics log every ten steps. | Prompt lengths are recorded during CPU preparation. Real vLLM allocation/sleep/wake behavior needs GPU verification. |
| Parent metadata and evaluation identity checks repeatedly hashed complete weights | Metadata reads verify manifest integrity; the actual artifact loader still verifies weight hashes once before loading. | Existing artifact tampering and lineage tests remain passing. Export verification is retained. |
| Short pilot performed intermediate checkpoint writes and lacked useful measurements | Intermediate pilot checkpoints are disabled; final complete exports and adapters remain. The proposed pilot runs two SFT updates, six GRPO updates and 32 evaluation questions, exercising the configured primary inference batch. | Final exports, source snapshots and exposure accounting remain mandatory. Six GRPO updates provide a diagnostic trace, not a stable training or throughput estimate. |
| Quotes multiplied cold-start/export overhead by every projected update | Main quotation separates measured training/generation time from one-time loading/setup/export overhead. It reads performance files instead of deriving throughput from whole-call wall time. | No main quote or speed estimate can be generated without completed pilot evidence. Cross-batch, cross-length and cross-benchmark extrapolation remains uncertain. |

## CPU evidence

The verified SFT arm contains 1,000 source examples, 5,865,452 processed tokens and 5,769,915 supervised tokens, packed into 360 sequences of at most 16,384 tokens. This fills **99.44% of packed token capacity**. That is a packing measure, not measured GPU utilization. The token cache is approximately 70.4 MB. Exact values and cache location are in [`gpu-preflight.json`](../../results/recovery/gpu-preflight.json).

For 1,319 GSM8K questions, the configured dispatch counts are:

| Output limit | Samples per question | Questions per batch | Generation calls |
|---|---:|---:|---:|
| 2,048 | 1 | 32 | 42 |
| 4,096 | 1 | 16 | 83 |
| 8,192 | 1 | 8 | 165 |
| 2,048 | 8 | 4 | 330 |

The earlier greedy batch of eight required 165 calls at 2K; the earlier sampled loop required 1,319 calls at eight samples. Padding, unequal answer lengths and GPU memory bandwidth affect the real speedup. The limits of 32 live sequences and 65,536 potential generated tokens per batch bound output workload; they are not a VRAM guarantee and do not include prompt KV storage. Canonical batch membership, seed policy and effective batch size are part of the run identity. An interrupted batch is regenerated with the same seed, then already committed questions are skipped.

The full CPU suite has 38 passing tests, including real model/trainer tests and eleven additional subtests. Ruff passes. Local controlled/legacy DAG dry runs pass. The CUDA runtime, kernels, 16K training memory peaks and separate-container behavior remain untested.

## Training settings retained

The review retains BF16, LoRA, Liger, packed SFT, non-reentrant gradient checkpointing, the optimizer, learning rates and effective training batches. GRPO keeps colocated vLLM with sleep mode, a 0.3 vLLM memory fraction, eight generations, and one optimization iteration. This avoids casually changing the research protocol while addressing visible overhead. The pinned vLLM version is within the versions supported by [TRL 0.28's integration documentation](https://huggingface.co/docs/trl/v0.28.0/en/vllm_integration).

Gradient checkpointing trades compute for memory. The eight-bit paged optimizer, the small GRPO microbatch and the H100 allocation are plausible starting settings, not established performance optima. Model size alone cannot settle them at these sequence lengths.

## What the pilot must measure

Training emits a sibling `sft-performance.json` or `grpo-performance.json`, including model loading, trainer setup, training and export durations, update counts and trainer step metrics. Evaluation emits `performance.json` beside predictions, with generation time and actual generated tokens, including replayed work. Each file includes sampled device utilization, sampled device-wide memory and power, process-local Torch allocator peaks and GPU identity. Evaluation profiles cover the whole function invocation across its benchmarks; a resumed invocation records its own work rather than pretending previously generated tokens were generated again. Result collection includes the profile.

Utilization sampling can miss short peaks and is not achieved FLOP utilization. Device memory includes vLLM, whereas Torch allocator peaks cover the training process. The first update can include lazy initialization and compilation; compare later updates and separate fixed costs. Check that each six-update GRPO trace includes repeated generation and policy-weight synchronization, finite losses and the expected rollout accounting.

Use measured cost per completed update/token alongside correctness and memory headroom. If memory allows, the next controlled comparison is GRPO microbatch `1 × accumulation 8` versus `2 × accumulation 4`, keeping eight total trajectories and the same objective. Benchmark gradient checkpointing/optimizer alternatives only after establishing a working baseline; no alternate configuration is enabled by this review.

H100 is **not yet proven cost optimal** for this 1.7B workload. At the dated [Modal rates](https://modal.com/pricing), including the configured two physical CPU cores and 16 GiB RAM, H100 costs about 1.92 times L40S per second. It must therefore deliver roughly 1.92 times the end-to-end throughput to break even for equivalent successful work, assuming both fit. The proposed pilot measures H100 GRPO only; it cannot settle that comparison. A hardware comparison would require a separately priced run, before the full study chooses its final hardware.

## Updated proposal

The proposal now includes CPU runtime-cache warming, two SFT updates, six GRPO updates, 32 primary-protocol evaluation questions and a separate persistence check. Its timeout reservations total approximately **$2.55**; the suggested cap remains **$10**. This is an unapproved proposal, not a billing guarantee or an estimate of actual utilization. The quote was regenerated for the current configuration and implementation. Long-context and eight-sample memory/performance still need measurements before their full evaluation matrix; the primary smoke pilot does not validate them.
