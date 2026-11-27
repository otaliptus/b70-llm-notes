# Native Inkling on one Arc Pro B70

For the exact model and runtime files, the command for the 39-task grader, and the experiments that stayed out of the release, see the [November 23 experiment record](16-inkling-b70-experiment-log.md).

## Single-launch ESIMD FlashDecoding, coupled INT8 DSpark, and SYCL graphs with patch 0062

Patch 0062 stacks on 0061 across the `inkling2` silo and Rust engine while keeping every target score and fixed-seed output **byte-identical** to 0059/0061 (18/18 teacher-forced full-vocabulary score specs, 215/215 attention-bench cases, and 14/14 conversation hashes). It replaces two-pass long full-attention with per-`T` L1/SLM-aware single-launch ESIMD FlashDecoding, couples DSpark's temperature-1 proposals to the target's inverse-CDF draw over INT8 `fold` + INT8 Layer-1 KV + dynamic session-OOV rows, captures the decode/drafter/sampling passes in SYCL command graphs, and fuses Q8_0 loop controllers, router scans, and MoE combine+conv. Fixed-seed drafted decode rose to **96.3–96.8 tok/s** at 100K (+50 % over 0061, +60 % over 0059), **140.8 tok/s** at 39K (+31 %), **133.1 tok/s** at 7.3K (+20 %), **146.2 tok/s** steady-state short, and **112.9 tok/s** on the 14-turn chat replay; empirical max context on the 32 GB B70 is **375,808 tokens** with DSpark (**418,816** without). Apply 0062 after 0061 and rebuild the executable and silo. [Patch](../patches/0062-nextsycl-inkling-neon-decode.patch) · [Record and measurements](18-inkling-neon-decode.md)

## Faster decode attention with patch 0061

Patch 0061 changes only the inkling2 silo. Check steps of 1 to 8 rows now share one tile of keys and values instead of each row reading its own copy; small kernels issue their reads first; and pointers read from job tables are cast to global memory. Every output is byte-identical to 0059 (full-vocabulary scores at 900 to 100,000 tokens and 1 to 8 rows a pass, the windowed ring's wrap, the 14-turn chat hashes). Decode is **6 to 13 % faster** at fixed seed, and the 39-task suite went from **105.0 to 115.1 tok/s** client-side pooled decode over three interleaved runs of each build. Prompt reading is unchanged. Apply 0061 after 0059 and rebuild the silo. [Patch](../patches/0061-nextsycl-inkling-decode-attention.patch) · [Record and measurements](17-inkling-decode-attention.md) · [Kernel bench](../scripts/inkling-attn-bench.sh)

## Wider prefill with patch 0059

On a cold 99,978-token source prompt, the B70 read **1,770 tokens/s** with patch 0059. Patch 0058 read **1,546 tokens/s**. The first generated token arrived in **56.68 seconds**, down from 64.82. Across three turns, the reasoning, visible text and generated lengths matched for each response, totaling 1,254 generated tokens. Pooled decode measured **53.87 tok/s**. [Per-request measurements](../bench/results/inkling-native-workloads.csv)

A separate replay through the installed 0059 launcher and public client read the cold input at **1,779.55 tok/s**, generated that reply at **57.57 tok/s**, and delivered its first token in **56.35 seconds**. The three-turn decode rate was **53.87 tok/s**. All three reasoning hashes, visible hashes and generated lengths reproduced the first run. [Installed replay timings and hashes](../bench/results/inkling-native-100k-installed.jsonl)

The installed release selected an 8,192-token batch with a 131,072-token allocation. At 126,458 input tokens, cold prefill rose from 1,460 to 1,669 tok/s and time to first token fell from 86.81 to 75.95 seconds. Cold decode was 48.24 tok/s and the three-turn pooled rate was 46.62 tok/s. The two follow-ups returned their first tokens in 1.20 and 1.15 seconds. All three complete answers and reasoning digests matched 0058. An authenticated request through the normal launcher also passed. [Per-request measurements](../bench/results/inkling-native-workloads.csv)

Simply increasing the prompt batch from 2,048 to 8,192 takes the cold wait down to 51.3 seconds, but oneMKL then sees different matrix shapes. In three independent executions on prefixes of the same source, top-token agreement with the 2,048 reference fell to 81.2%, 88.0% and 91.7%. The 0059 implementation expands each expert's weights once per wide batch and applies them to the same groups of rows used by the 2,048-token reference. It also preserves the decode-width kernel for a short final group.

With wide staging and reference-sized products, **115,233,408 full-vocabulary target scores were byte-identical** across the three executions. Another 4,801,392 scores matched on 4,097-, 4,112- and 4,113-token inputs that straddle the decode-width boundary. A separate 98K logistics log and 51-token continuation had 100% top-token agreement and mean KL of 0.00000 at the reported precision. All 28 fixed-seed everyday, conversation, source and compaction replies matched 0058 in visible text, reasoning digest and generated length. Both twice-compacted handoffs retained five of five requested facts. [Quality measurements and file hashes](../bench/results/inkling-native-prefill-quality.csv)

The B70 launcher enables wide staging. At startup the runtime tries 8,192, 4,096, 2,048, 1,024 and 512 rows and selects the first size that fits its existing memory budget. The actual chunk appears in the load log. `WIDE_PREFILL=0` restores the previous selection policy and `PREFILL_CHUNK` pins a size for a controlled comparison. Wide-batch checkpoints use a distinct disk cache fingerprint. Apply 0059 after 0058 and rebuild the executable. [Runtime patch](../patches/0059-nextsycl-inkling-wide-prefill.patch) · [Launcher](../scripts/serve-inkling-nextsycl.sh)

### User benchmark and conversation on 0059

| Workload | Requests | Fresh prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Original graded suite | 39 | 531.35 | 105.59 |
| Everyday tasks | 6 | 1,933.10 | 103.32 |
| Continuous conversation | 14 | 333.57 | 81.91 |
| 40K source and follow-up | 2 | 2,072.13 | 81.28 |
| Two 98K compaction exercises | 6 | 1,878.21 | 66.16 |
| 100K source and two follow-ups | 3 | 1,722.69 | 53.87 |
| 126K source and two follow-ups | 3 | 1,633.28 | 46.62 |

The original grader passed **36 of 39** on the automatically configured release, with 58,422 generated tokens and a median of 0.38 seconds to the first streamed token. The failed cases were `r03_trace`, `k05_mem` and `c11_json`. Earlier runs scored 34/39 on 0054 and 35/39 on 0056. The grader samples fresh answers in each run. The fixed-seed workloads below the graded suite give paired comparisons. Throughput divides total fresh or generated tokens by corresponding engine time. Decode includes reasoning and visible output. [Per-request counters, grades and visible-answer hashes](../bench/results/inkling-native-workloads.csv)

A fresh replay of the conversation through the normal installed launcher produced the same 14 visible and reasoning hashes and generated lengths. This run measured 329.35 fresh prefill and 81.82 decode tok/s. [Streaming timings and both output hashes](../bench/results/inkling-native-chat-replay.jsonl)

The fourteen-turn conversation streamed its first token after a median of **0.57 seconds** and its first visible text after **2.56 seconds**. Its 72-word handoff carried forward the revised 22-visitor limit, the host swap at 3 pm and the final £30 budget. An earlier invitation rewrite omitted the revised capacity, so even this short conversation required checking the deliverable. [Conversation runner](../bench/inkling_context.py)

## Optional long-context drafting with patch 0060

Patch 0060 adds a B70 Q8_0 output projection and a configurable attention window for DSpark proposals. Set `DRAFT_WINDOW=2048` with the 0060 runtime to let its full-attention proposal layers look at roughly the latest 2,048 tokens once the input reaches 32,768 tokens. The window starts on a 64-token boundary. The target continues to attend to its complete context and verifies proposals against its full vocabulary. The default proposal window setting is zero. Windowed checkpoints omit old proposal keys and values that the drafter can no longer read and use a distinct disk-cache fingerprint. [Runtime patch](../patches/0060-nextsycl-inkling-long-proposals.patch)

| Three-turn source replay | 0059 decode tok/s | 0060 with proposal window, decode tok/s | 0059 cold decode | 0060 cold decode |
|---|---:|---:|---:|---:|
| 99,978 input tokens | 53.87 | 55.09 | 57.56 | 56.44 |
| 126,458 input tokens | 46.62 | 48.39 | 48.24 | 50.13 |

The six source replies matched 0059 in visible text, reasoning digest and generated length. Draft acceptance varied. Both 98K compaction exercises retained **5/5** final fields, and 6/6 compaction responses matched the complete reference reply. Pooled compaction decode was 64.65 tok/s with the window and 66.16 on 0059. Across two target-only runs with 7.3K and 100K prompts, **76,822,272 full-vocabulary scores** matched the original control byte for byte. The slower cold decode on the 100K case is included above. The checks cover these prompts and seeds. [Per-request measurements](../bench/results/inkling-native-workloads.csv) · [Full target score hashes](../bench/results/inkling-native-long-logits.csv)

The six everyday tasks, two 40K source requests and fourteen-turn conversation also reproduced all 22 reference answers, reasoning digests and generated lengths. The everyday tasks averaged **103.62** decode tok/s, the 40K pair **74.83** and the conversation **82.19**. The 40K pair reached 81.28 tok/s on 0059, so this particular proposal window slowed the intermediate-length source task. The installed launcher defaults to 0059. [Per-request counters](../bench/results/inkling-native-workloads.csv)

On the cold 126,458-token request, the largest saved host checkpoint shrank from 4.51 to 4.03 GiB. Total resident checkpoint storage fell from 4.84 to 4.39 GiB at the same 6 GiB budget. These measurements describe host RAM. The target KV cache on the GPU retains its original representation. [Checkpoint measurements](../bench/results/inkling-native-proposal-cache.csv)

With the host budget explicitly set to **4.25 GiB**, two builds used the same 2,048-token draft window, 2,048-token prompt batches, target and prompts. Keeping the full proposal checkpoint made the following two requests feed **56,145 and 56,303 fresh tokens**. Saving only the proposal rows still in use reduced those to **262 and 158**. Time to first token went from **48.70 and 48.87 seconds** to **1.18 and 1.14 seconds**. The cold answer and configuration follow-up matched in reasoning, visible text and generated length. The final checklist differed in wording and used 503 generated tokens with the cropped checkpoint and 620 with the full checkpoint. These 2,048-batch controls isolate checkpoint retention from the wide-prefill timing above. [Per-turn budget, residency and reuse](../bench/results/inkling-native-proposal-cache.csv)

For the optional build, apply 0060 after 0059 and rebuild the executable and the Inkling silo. Point `RUNTIME` at that build and pass `DRAFT_WINDOW=2048` to the launcher. The four-hour run used 0059 for the 39-task graded results above. [Launcher](../scripts/serve-inkling-nextsycl.sh)

## Recurrent projections with patch 0058

The twelve recurrent layers use the same two source tensors during each return through the loop. Computing their four messages together reduced twelve separate launches to one. With a 2,304-wide hidden state, this part took **128 to 10 microseconds** for one row and **123 to 33 microseconds** for seven rows. All 294 tested kernel outputs matched the separate projections bit for bit.

In the six everyday tasks, the recurrent-projection build averaged **102.97 decode tok/s**, compared with 99.64 on 0056. The two source-reading requests went from 79.13 to **81.28**, and the fourteen-turn conversation went from 78.83 to **81.63**. All 22 visible answers, generated lengths and draft acceptance counts matched. This build included the batched convolution rollback. The final 0058 patch also writes the target output projection directly into its destination and reads draft IDs and probabilities in one transfer. Its 153,644,544 scored target logits matched the 0054 control across four runs at 7.3K, 40K and 100K. Apply 0058 after 0057 and rebuild the executable and Inkling silo. [Workload counters](../bench/results/inkling-native-workloads.csv) · [Logit hashes](../bench/results/inkling-native-long-logits.csv) · [Runtime patch](../patches/0058-nextsycl-inkling-recurrent-projections.patch)

The installed 0058 executable and silo were then tested together. All 28 fixed-seed visible answers and generated lengths matched 0056. The six everyday tasks and both source-reading tasks passed their checks. Each twice-compacted handoff retained all five fields. The first exercise produced summaries of 29 and 33 words, the second 34 and 42. [Per-request measurements](../bench/results/inkling-native-workloads.csv)

| Installed 0058 workload | Requests | Fresh prompt tok/s | Decode tok/s |
|---|---:|---:|---:|
| Everyday tasks | 6 | 1,720.07 | 103.34 |
| Continuous conversation | 14 | 330.29 | 81.90 |
| 40K source and follow-up | 2 | 1,764.53 | 81.33 |
| Two 98K compaction exercises | 6 | 1,677.04 | 65.86 |

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

With a 768-token cap, all five requests completed naturally in both configurations. On the revisits, time to first generated token fell from **25.00 and 25.15 seconds to 0.70 and 0.68 seconds**. Time until the first visible answer text fell from **28.00 and 31.51 seconds to 4.52 and 6.26 seconds**. The first three replies matched in both reasoning and visible text. The fourth gave the same function name with different reasoning. The fifth gave a differently worded ten-word description. The earlier runtime reread the source from scratch on the revisits and the new runtime restored the shared checkpoint. [Request counters and both output hashes](../bench/results/inkling-native-branch-cache.csv)

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

## Replay the long conversations

The source-reading replay starts with a 99,978-token or 126,458-token request and sends two follow-ups containing the model's actual visible answers. Supply a separate fixture checkout at the pinned revision with only patch 0054 applied. The runner verifies the source text hash before sending it. Run these commands from the notes repository with the server already listening and `API_KEY_FILE` pointing to its key.

```sh
notes=$PWD
git clone https://github.com/maxfridbe/nextsycl.git /tmp/inkling-fixture
git -C /tmp/inkling-fixture checkout --detach e3f1fd891b7fda6f478a326d913d28893a4e9811
git -C /tmp/inkling-fixture apply "$notes/patches/0054-nextsycl-atomic-inkling-b70.patch"
python3 bench/inkling_context.py --source /tmp/inkling-fixture --case 100k >100k.jsonl
```

Restart the single-stream server with an empty prompt cache before each cold case. Use `--case 126k` for the near-limit run or `--case branches` for the five-turn branching workload. The published branching comparison used `PROMPT_CACHE_MIB=2560` and both near-limit controls used 6144. For the two independent compaction exercises, run `python3 bench/inkling_context.py --case compaction`. Each shift supplies a new log of about 98K tokens. Only the model's handoff carries over to the next shift. The final answer is graded against five fields.

To replay the fourteen-turn conversation, run `python3 bench/inkling_context.py --case chat`. It plans a community zine swap, revises the visitor limit and budget, changes hosts halfway through and requests a final handoff. Each turn includes the assistant's actual visible replies. Two follow-ups adapt their wording to the previous answer, and the client pauses between turns.

Per-turn JSON includes fresh and reused tokens, engine read and generation seconds, first generated and first visible text latency, and SHA-256 hashes of the reasoning and answer. The final rates printed to standard error divide total tokens by total engine time. The public runner reproduced all three visible answers, generated lengths and prompt counts from the 100K reference in a 0058 replay. [Replay script](../bench/inkling_context.py)

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

## Other single-B70 results as of November 23, 2026

Mike Garuccio's vLLM XPU plugin published a single-B70 DSpark run using native NVFP4/MXFP8 experts and optional CUTLASS Xe2 kernels. On November 23 its six short greedy coding and reasoning prompts reached a median **99.0 decode tok/s**. Three prompts with nominal 6K and 30K background logs reached **103.7** and **100.7 tok/s**. Each generated 256 tokens and the card used a 275 W cap. A separate 8K cold prefill reached **4,084 tok/s**. Those are faster prompt reads than ours on the published workloads. Our card reported a 230 W cap during this session. [Pinned project measurements](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/91cede41571055745fc7e0e5e98b0bc604526427#performance) · [Raw depth and prefill summaries](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/91cede41571055745fc7e0e5e98b0bc604526427/results/decode/cutlass-xe2-20261123)

We replayed the upstream benchmark's six short prompts and first three depth-30K prompts on installed 0059. Both runs requested greedy decoding with a 256-token cap and used the same filler, warm-up and prompt texts. For these rows, client decode throughput counts tokens after the first and divides by the interval between the first and last streamed token. At depth zero, the local median was **109.36 tok/s**, compared with **99.04** in the published vLLM run. At nominal 30K, the local median was **80.02**, compared with **100.74**. Our tokenizer counted 28,549–28,552 prompt tokens at that depth. All nine local requests generated the same requested 256 tokens. The local launcher used low reasoning effort and recorded first and last nonempty streamed text or reasoning chunks. The upstream client recorded first and last streamed log-probability entries. The engines, weights, power limits and chat templates can affect the comparison. [Per-case results and common prompt hashes](../bench/results/inkling-native-vllm-comparison.csv) · [Original benchmark script](https://github.com/mgaruccio/vllm-inkling2-xpu/blob/91cede41571055745fc7e0e5e98b0bc604526427/results/decode/decode_bench.py)

The arcint 1.8.0 llama.cpp OpenCL backend measured **45.3, 41.6 and 39.0 decode tok/s** with IQ4_NL at depths of 0, 4,096 and 32,768 tokens on a 220 W B70. Its matching 512-token prefill figures were 672, 641 and 561 tok/s. The author distinguishes the benchmark's sliding-window cache from that release's server cache. [Pinned arcint B70 table](https://github.com/marfrit/arcint/tree/b6e2b505eeb9f0436e0d42b885d9b96c364aa0bb#inkling-2-mini-180)

Strata's 0.2.19 release lists **159.2 decode tok/s** on one 32 GB B70. Its B70 figure has no associated prompt length or per-request trace in the release notes. A B70 owner running the released binary reported 25.2–29.5 tok/s with drafting off and 10.8–12.0 with DSpark on a 300-token prose task configured for 32K context. These measurements use a different engine and checkpoint pack. [Strata release](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) · [User's hardware, commands and timings](https://github.com/Niko1221/Strata/issues/6595#issuecomment-7089002127)

OpenDynalmx published a context sweep on a single 32 GB B70 using a vLLM XPU fork, the original Inkling Mini NVFP4/MXFP8 weights and no speculative decoding. Sustained single-request decode measured 68.3 tok/s near empty, 53.4 at 64K, 43.8 at 128K and 34.5 at 224K. At 128K the run measured 1,562 prompt tok/s. Each decode measurement lasted 30 seconds after cache warm-up. Our 126,458-token source and follow-ups use AtomicChat AD-Q4_K and DSpark, so prompts, weights and measurement windows differ. [Submitted run and full matrix](https://www.localmaxxing.com/en/inference-benchmarks/cmwsroyel007bld01mkdlnopk)

## Reproduce the runtime

For the measured release, start with upstream nextsycl commit `e3f1fd891b7fda6f478a326d913d28893a4e9811` and apply patches 0054 through 0059 in numerical order. The source fixtures above use their own checkout. From the notes repository, the following commands prepare the runtime sources and start the upstream B70 build. The upstream build script uses Podman and puts the resulting executable and libraries under `dist/`.

```bash
notes=$PWD
runtime_src=/tmp/inkling-runtime
git clone https://github.com/maxfridbe/nextsycl.git "$runtime_src"
git -C "$runtime_src" checkout --detach e3f1fd891b7fda6f478a326d913d28893a4e9811
for number in 0054 0055 0056 0057 0058 0059; do
    git -C "$runtime_src" apply "$notes"/patches/"$number"-nextsycl-*.patch || exit 1
done
AOT=bmg-g31 "$runtime_src/build.sh" all
```

For the faster decode attention, also apply `0061-nextsycl-inkling-decode-attention.patch` after 0059 before building. It changes only the inkling2 silo, and the patched tree is `853e9daff944d88c979a6f0b1f67c55d0a78504d`. [Build notes for 0061](17-inkling-decode-attention.md#build-and-identify-it)

For the full ESIMD FlashDecoding, coupled INT8 DSpark, and SYCL-graph decode stack, apply `0062-nextsycl-inkling-neon-decode.patch` after 0061 and rebuild both the host executable/library and the silo (`AOT=bmg-g31 "$runtime_src/build.sh" all`). The patched tree is `26a3e0b21ebecfd2777238eacf4c324e443be291`. [Build notes for 0062](18-inkling-neon-decode.md#build-and-identify-it)

The launcher expects the built executable beside the LLM shared library and its Inkling silo subdirectory. Set `RUNTIME`, `MODELS` and `API_KEY_FILE` to local paths and `SYCL_IMAGE` to the compatible oneAPI Docker runtime image. Prepare the 32,768-row draft head with the procedure in section 5 of the llama.cpp notes. [Patch](../patches/0054-nextsycl-atomic-inkling-b70.patch) · [Launcher](../scripts/serve-inkling-nextsycl.sh) · [Draft-head preparation](14-inkling-uncensored-dspark.md) · [Upstream source](https://github.com/maxfridbe/nextsycl/tree/e3f1fd891b7fda6f478a326d913d28893a4e9811)

The GPU kernels load the original Q4_K gate/up and Q5_0 down experts. The 32K head supplies up to six DSpark proposals at a draft confidence threshold of 0.65. The target verifies each proposal with full-vocabulary sampling. The patch also adds bearer authentication to the HTTP endpoint. Verification included exact agreement with GGML on 155,520 sampled expert weights expanded to half precision, a full-vocabulary sampling distribution check, and identical generated text across three fixed-seed prompts with the full and reduced proposal heads.

Both the GGUF and the original model configuration (`text_config.model_max_length`) specify 1,048,576 tokens. The measurements above used the 131,072-token launcher configuration. [Model configuration](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json) · [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)


## Follow-up latency at 100K

Patch 0055 lets consecutive Inkling checkpoints share immutable stretches of KV data in host memory. When a session restores the same checkpoint it just saved, its full-attention keys and values are already on the GPU. Branches, cold restores and disk checkpoints keep the original file layout. Apply 0055 after 0054 and rebuild the executable. [Prefix checkpoint patch](../patches/0055-nextsycl-inkling-prefix-checkpoints.patch)

| Follow-up | Reused tokens | Fresh tokens | Previous first token | With shared checkpoints |
|---|---:|---:|---:|---:|
| First follow-up | 99,978 | 294 | 3.99 s | 1.36 s |
| Second follow-up | 100,272 | 142 | 3.83 s | 1.28 s |

This A/B used the same attention binary, weights, sampling settings and seeds on both sides. All three answers and generated token counts matched. A separate six-request branching test using the original attention kernels produced seven byte-identical checkpoint files. Each executable also read the other one's checkpoints and reproduced the same answers. [Request timings and answer hashes](../bench/results/inkling-native-prefix-cache.csv)
