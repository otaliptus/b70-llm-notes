# Native Inkling on one Arc Pro B70

## Recurrent projections with patch 0058

The twelve recurrent layers use the same two source tensors during each return through the loop. Computing their four messages together reduced twelve separate launches to one. With a 2,304-wide hidden state, this part took **128 to 10 microseconds** for one row and **123 to 33 microseconds** for seven rows. All 294 tested kernel outputs matched the separate projections bit for bit.

In the six everyday tasks, the recurrent-projection build averaged **102.97 decode tok/s**, compared with 99.64 on 0056. The two source-reading requests went from 79.13 to **81.28**, and the fourteen-turn conversation went from 78.83 to **81.63**. All 22 visible answers, generated lengths and draft acceptance counts matched. This build included the batched convolution rollback. The final 0058 patch also writes the target output projection directly into its destination and reads draft IDs and probabilities in one transfer. Its 153,644,544 scored target logits matched the 0054 control across four runs at 7.3K, 40K and 100K. Apply 0058 after 0057 and rebuild the executable and Inkling silo. [Workload counters](../bench/results/inkling-native-workloads.csv) · [Logit hashes](../bench/results/inkling-native-long-logits.csv) · [Runtime patch](../patches/0058-nextsycl-inkling-recurrent-projections.patch)

## Long-context performance with patch 0056

One B70 generated the same 1,254 tokens at **52.31 tok/s** across a three-turn source-reading conversation. The original runtime averaged **23.20 tok/s**. The cold request read 99,978 tokens at **1,543 tok/s**, up from **1,023 tok/s**. All three answers, generation lengths and draft acceptance counts matched at fixed seeds. [Per-turn counters and answer hashes](../bench/results/inkling-native-long-attention.csv)

| Turn | Original decode tok/s | Patch 0056 decode tok/s | Original first token | Patch 0056 first token |
|---|---:|---:|---:|---:|
| Cold source | 24.24 | 55.76 | 98.00 s | 64.98 s |
| Configuration question | 23.29 | 52.82 | 3.96 s | 1.19 s |
| Runbook handoff | 22.17 | 48.98 | 3.96 s | 1.15 s |

The patch tiles full-attention prompt reads and groups several decode queries so they can share key and value reads. The GPU computes each split's merge coefficient once. The sums retain the original floating-point operation order. Cached conversations also restore just the portions of sliding attention rings that subsequent writes could have overwritten.

At each of 192 continuation positions, the runtime exported all 200,058 target logits. Five runs covered prompt lengths of roughly 7.3K, 40K and 100K tokens, with forward passes of one, four or seven rows. Each complete output file matched its control byte for byte. The three distinct text fixtures cover a narrower range of content than the five executions. The model, quantization, sampling distribution, draft head and configured context were the same throughout. Apply 0056 after 0055 and rebuild the executable and the Inkling silo. [Output hashes](../bench/results/inkling-native-long-logits.csv) · [Runtime patch](../patches/0056-nextsycl-inkling-long-attention.patch)

## Follow-ups near the 131K launch limit

At 126,458 input tokens, patch 0056 generated **47.6 tok/s** on a cold source-reading request. The default launcher now allows 6 GiB of host RAM for prompt checkpoints. The 4 GiB setting could retain only a shorter checkpoint. Its two follow-ups reread over 56,000 tokens each. With 6 GiB the same turns read 262 and 158 fresh tokens, and time to first token fell from **47.25 to 1.19 seconds** and **52.12 to 1.14 seconds**. Decode on the two follow-ups measured 51.8 and 41.6 tok/s. The cold answer and first follow-up matched exactly. The last prompt contained the same tokens but produced two differently worded checklists (620 and 503 generated tokens). The 4 GiB run reconstructed more of its state by prefill. Both runs used the same weights, GPU context allocation and seed. Set `PROMPT_CACHE_MIB` to choose the host memory budget. [Per-turn measurements and output hashes](../bench/results/inkling-native-near-context-limit.csv)

With both engines given the same 6 GiB host cache, all three replies, generated lengths and proposal counts matched. Patch 0054 averaged **18.58 decode tok/s** and **914 cold prefill tok/s**. Patch 0056 averaged **45.90 decode tok/s** and **1,449 cold prefill tok/s** on those requests. The last checklist also matched its counterpart in this equal-budget comparison. [126K request counters](../bench/results/inkling-native-near-context-limit.csv)

## Switching conversation branches with patch 0057

A five-request test started with 39,323 source tokens, added a different appendix to each of two branches and revisited both. Under a 2,560 MiB host-cache budget the previous server evicted 11 checkpoints. Patch 0057 counts a shared allocation once while any resident branch references it. All eight final checkpoints fit in 2,234 MiB. The return to branch A read 15 fresh tokens in place of 43,563, reducing time to first token from **24.73 to 0.40 seconds**. The return to B read 18 in place of 43,980, going from **24.81 to 0.43 seconds**. All requests reached the 196-token cap during reasoning, so this fixture measures cache behavior. [Cache counters for all five requests](../bench/results/inkling-native-branch-cache.csv)

The same patch batches convolution history repair after the verifier rejects a proposed suffix. Three fixed 768-token generations matched in full and ran at 116.6, 120.9 and 124.0 tok/s with batched repair. Apply 0057 after 0056 and rebuild the executable and Inkling silo. The on-disk checkpoint format is compatible. [Runtime patch](../patches/0057-nextsycl-inkling-branch-cache.patch)

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
