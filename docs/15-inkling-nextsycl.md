# Native Inkling on one Arc Pro B70

## Long-context performance with patch 0056

One B70 generated the same 1,254 tokens at **52.31 tok/s** across a three-turn source-reading conversation. The original runtime averaged **23.20 tok/s**. The cold request read 99,978 tokens at **1,543 tok/s**, up from **1,023 tok/s**. All three answers, generation lengths and draft acceptance counts matched at fixed seeds. [Per-turn counters and answer hashes](../bench/results/inkling-native-long-attention.csv)

| Turn | Original decode tok/s | Patch 0056 decode tok/s | Original first token | Patch 0056 first token |
|---|---:|---:|---:|---:|
| Cold source | 24.24 | 55.76 | 98.00 s | 64.98 s |
| Configuration question | 23.29 | 52.82 | 3.96 s | 1.19 s |
| Runbook handoff | 22.17 | 48.98 | 3.96 s | 1.15 s |

The patch tiles full-attention prompt reads and groups several decode queries so they can share key and value reads. The GPU computes each split's merge coefficient once. The sums retain the original floating-point operation order. Cached conversations also restore just the portions of sliding attention rings that subsequent writes could have overwritten.

At each of 192 continuation positions, the runtime exported all 200,058 target logits. Five runs covered prompt lengths of roughly 7.3K, 40K and 100K tokens, with forward passes of one, four or seven rows. Each complete output file matched its control byte for byte. The three distinct text fixtures cover a narrower range of content than the five executions. The model, quantization, sampling distribution, draft head and configured context were the same throughout. Apply 0056 after 0055 and rebuild the executable and the Inkling silo. [Output hashes](../bench/results/inkling-native-long-logits.csv) · [Runtime patch](../patches/0056-nextsycl-inkling-long-attention.patch)

## Workloads and compaction with patch 0056

| Workload | Requests | Fresh prefill tok/s | Decode tok/s | Original decode tok/s |
|---|---:|---:|---:|---:|
| User's graded suite | 39 | 525.21 | 102.19 | 104.17 |
| Everyday tasks | 6 | 1,712.05 | 99.64 | 94.56 |
| Continuous conversation | 14 | 330.71 | 78.83 | 78.87 |
| Source handoff and follow-up at 40K | 2 | 1,763.21 | 79.13 | 45.08 |
| Two 98K compaction exercises | 6 | 1,653.50 | 64.69 | 29.57 |

Rates divide total tokens by the corresponding total engine time. Decode includes reasoning and visible output. The unmodified graded suite drew 57,870 tokens and passed 35 of 39, compared with 54,037 tokens and 34 of 39 in the original run. It samples different answers between runs. The other four workloads used fixed seeds. All 28 visible answers and generated lengths matched the original runtime. All six everyday tasks and both source handoff tasks passed their checks. Each compaction exercise generated two summaries and answered a question after a context reset. Both retained all five requested facts, including the current bay, booking and owner. [Per-request timings, grades and answer hashes](../bench/results/inkling-native-workloads.csv)

## Original native measurements

The AtomicChat Inkling-2-Mini-Uncensored AD-Q4_K checkpoint reached 104.2 generated tokens/s on the 39-task suite, 78.9 in a 14-turn chat, and 23.1 over three requests at 100K context. The native engine ran one stream with a 131,072-token session, low reasoning effort, temperature 1 and top-p 1.

| Workload | Requests | Prefill tok/s | Decode tok/s | Median first streamed token |
|---|---:|---:|---:|---:|
| 39-task suite | 39 | 535 | 104.2 | 0.38 s |
| Standalone work tasks | 6 | 1,635 | 94.6 | 0.67 s |
| Continuous conversation | 14 | 274 | 78.9 | 0.67 s |
| Cold source, 40,416 input tokens | 1 | 1,439 | 49.6 | 28.24 s |
| Cold source, 99,978 input tokens | 1 | 1,017 | 24.2 | 98.60 s |
| First follow-up, 99,978 tokens reused | 1 | 66 | 23.3 | 3.97 s |
| Second follow-up, 100,226 tokens reused | 1 | 37 | 22.1 | 4.01 s |

Rates use total fresh prompt or generated tokens divided by the corresponding native runtime seconds. Generation includes reasoning. The full 39-task suite produced 54,037 tokens and scored 34/39, matching the previous DSpark-on llama.cpp run's score. The suite is stochastic. The first *visible answer* took 2.73 seconds median in chat. The 100K requests generated 1,254 tokens in 54.25 seconds. [Per-request counters](../bench/results/inkling-native-b70.csv)

## Reproduce the runtime

Apply patch 0054 to nextsycl commit `e3f1fd891b7fda6f478a326d913d28893a4e9811`, then use nextsycl's oneAPI build for the B70. The launcher expects the built executable beside the LLM shared library and its Inkling silo subdirectory. Set `RUNTIME`, `MODELS` and `API_KEY_FILE` to local paths and `SYCL_IMAGE` to the compatible oneAPI Docker runtime image. Prepare the 32,768-row draft head with the procedure in section 5 of the llama.cpp notes. [Patch](../patches/0054-nextsycl-atomic-inkling-b70.patch) · [Launcher](../scripts/serve-inkling-nextsycl.sh) · [Draft-head preparation](14-inkling-uncensored-dspark.md) · [Upstream source](https://github.com/maxfridbe/nextsycl/tree/e3f1fd891b7fda6f478a326d913d28893a4e9811)

The GPU kernels load the original Q4_K gate/up and Q5_0 down experts. The 32K head supplies up to six DSpark proposals at a draft confidence threshold of 0.65. The target verifies each proposal with full-vocabulary sampling. The patch also adds bearer authentication to the HTTP endpoint. Verification included exact agreement with GGML on 155,520 sampled expert weights expanded to half precision, a full-vocabulary sampling distribution check, and identical generated text across three fixed-seed prompts with the full and reduced proposal heads.

The GGUF records 1,048,576 tokens as its context length. The measurements above used the 131,072-token launcher configuration. [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)


## Follow-up latency at 100K

Patch 0055 lets consecutive Inkling checkpoints share immutable stretches of KV data in host memory. When a session restores the same checkpoint it just saved, its full-attention keys and values are already on the GPU. Branches, cold restores and disk checkpoints keep the original file layout. Apply 0055 after 0054 and rebuild the executable. [Prefix checkpoint patch](../patches/0055-nextsycl-inkling-prefix-checkpoints.patch)

| Follow-up | Reused tokens | Fresh tokens | Previous first token | With shared checkpoints |
|---|---:|---:|---:|---:|
| First follow-up | 99,978 | 294 | 3.99 s | 1.36 s |
| Second follow-up | 100,272 | 142 | 3.83 s | 1.28 s |

This A/B used the same attention binary, weights, sampling settings and seeds on both sides. All three answers and generated token counts matched. A separate six-request branching test using the original attention kernels produced seven byte-identical checkpoint files. Each executable also read the other one's checkpoints and reproduced the same answers. [Request timings and answer hashes](../bench/results/inkling-native-prefix-cache.csv)
