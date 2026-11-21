# Inkling 2 Mini on SYCL: NVFP4 experts, a per-head flash-attention mask, and graph fusions (28 -> 58 tok/s)

Dates 2026-11-20/21. Same machine and containers as [01](01-environment.md). The model is Inkling 2 Mini (Thinking
Machines, released 2026-11-20, Apache 2.0). It is a 26.2B mixture-of-experts model with 3.4B active parameters. It has
24 stored layers, and layers 6-17 run three times, so every token goes through 48 layer passes (5.4B parameters per
token). There are 128 experts with 8 used per token, sliding-window (1024 tokens) and global attention layers, a
relative-position bias added to the attention scores, and attention sinks. The released weights store the routed
experts of layers 0-19 in NVFP4, layers 20-23 in MXFP8, and everything else in BF16.

Unless a row says otherwise, speeds are tok/s as reported by llama-server or llama-bench on the B70. "Server decode"
means the median of 6 runs: 3 prompts x 2 rounds, `reasoning_effort` low, temperature 1.0, top_p 1.0, 800 output
tokens, context 131072, micro-batch 256, no drafter.

## 1. Result

| | before (patches 0019-0021) | after (0019-0029) |
|---|---|---|
| server decode (6-run median) | 28.4 | **57.7** (57.4-57.8) |
| llama-bench pp512 / tg64 | 343 / 28.3 | **889 / 59.6** |
| depth 32k, pp512 / tg32 | not measured | 388 / 49.2 (flash attention off: 287 / 36.4) |
| 103,347-token prompt: prefill | 142 (726 s) | **265** (390 s) |
| decode at that depth | 14.1 | **37.7** |
| KLD vs CPU reference, `-ub 1` / `-ub 512` | 0.0047 / 0.0054 | 0.0054 / 0.0052 |

The KLD reference is the Unsloth UD-Q8_K_XL file run on the CPU, over 4 x 512 tokens of mixed text. The change in KLD is
within the run-to-run movement we saw per fusion (about +-0.0009), so no regression was detected at this sample size. A
needle placed in the 103k-token prompt is found. On the 39-task suite at effort low, the new runtime scored 34, 34 and
32 out of 39 over three single runs, against one run of 36 before. Failures move between runs at temperature 1.0, and
two of the 32-run failures are known grader false negatives. That is within the spread seen for other models, but it is
not proof that quality is unchanged.

## 2. Build and model file

Source: Unsloth's llama.cpp release tarball b12572-mix (b12572 plus the DSpark drafter PR and Unsloth's looped-layer
support). The upstream Inkling PR on its own could not load the Unsloth GGUFs: it does not load the per-expert NVFP4
scale of the merged gate_up tensor (734 tensors expected, 714 found). Apply patches 0019-0029 in order and build with
`scripts/build-sycl-patched.sh`.

Model file: a requant of the Unsloth UD-Q8_K_XL file. The NVFP4 expert tensors and their F32 scales are copied byte for
byte; `token_embd`, `loop_*`, `attn_sink_proj` and `*_conv_proj` stay BF16; F32 stays F32; all other BF16 tensors
(attention projections, output head, and the layer 20-23 experts that shipped as MXFP8) become Q8_0. The result is
18.0 GB. Nothing is quantized to 4 bits by this step: the 4-bit experts are the model's own released weights.

```sh
llama-quantize --allow-requantize \
  --tensor-type "blk\.([0-9]|1[0-9])\.ffn_(gate_up|down)_exps=nvfp4" \
  --tensor-type "token_embd=bf16" --tensor-type "loop_=bf16" \
  --tensor-type "attn_sink_proj=bf16" --tensor-type "conv_proj=bf16" \
  Inkling-2-Mini-UD-Q8_K_XL.gguf Inkling-2-Mini-NVFP4-Q8_0.gguf Q8_0 16
```

Run `--dry-run` first to read the plan. Against the CPU reference, this file has KLD 0.0052 and the same top token
97.4 % of the time; UD-Q8_K_XL itself on SYCL is 0.0046, so the Q8_0 step adds nothing measurable at this sample size.

Serve flags: `-ngl 99 -c 131072 -ub 256 --flash-attn on --jinja`, no vision projector. Reasoning effort is set per
request with `"chat_template_kwargs": {"reasoning_effort": "low"}` (values none, minimal, low, medium, high, xhigh,
max, or a number 0.0-0.99; the template default is 0.9, which is high). Effort "none" still reasoned about 300 tokens.

One container trap: the SYCL image's working directory held old `libggml-*.so.0` files that the backend loader picks
up and aborts on, so every container command must `cd /` first.

## 3. The patches

**0019, SYCL SCALE accepts a strided f32 source.** The SITU activation scales the gate and up halves of the merged
gate_up result, which are views. Without this they ran on the CPU, adding 96 graph splits per token.

**0020, debug.** Graph-compatibility messages name the node; `INKLING_SHAPES=1` adds perf shapes to test-backend-ops.

**0021, SYCL get_rows uses row-sized work-groups and splits the launch.** The relative-position bias gather (rows of
24 floats, one per key/token pair) overflowed the 32-bit launch range at 32768 keys and aborted.

**0022, NVFP4 mat-vec decode.** The NVFP4 expert mat-vec was limited by arithmetic, not memory bandwidth: each 4-bit
value went through an emulated table lookup and each UE4M3 scale through `sycl::pow`. The patch decodes the values
with the same SWAR trick the MXFP4 path already used and builds the scale directly from its bits. It was checked
against the old version for every scale code. The 8-expert gate_up mat-vec went from 161 to 50 us and down from 96 to
28 us; tg64 went from 28.3 to 37.5 from this patch alone.

**0023, per-node profiler (debug).** `GGML_SYCL_OPPROF=1` prints a per-node wall-time table for 1-token graphs, using
a host sync after every node. `GGML_SYCL_OPPROF=2` with `GGML_SYCL_QUEUE_PROFILING=1` uses GPU barrier timestamps
instead, which is the cost each node really has inside the graph. Use mode 2; host syncs inflate small kernels.
`GGML_SYCL_OPPROF_DUMP=1` prints the decode graph.

**0024, flash attention with a mask per head.** Inkling's relative-position bias makes the attention mask different
for every head, and the SYCL flash-attention kernels rejected such a mask, so with flash attention on every attention
layer ran on the CPU (290 splits, tg 22.6). The VEC and TILE kernels now offset the mask by head; TILE runs without GQA
packing in that case, and the mask scan used to skip empty key blocks is skipped. Over 30 new FLASH_ATTN_EXT test
cases pass against the CPU, up to 98304 keys (0028).

**0025-0027 and 0029, Inkling node-group fusions** (`ggml-sycl/inkling-fuse.cpp`). Inkling's graph is full of small
ops. These patches recognise nine node groups and run each as one or two kernels. `GGML_SYCL_IFUSE=<mask>` selects them
(default all): 1 clip, 2 MoE router plus the f32 logits mat-vec, 4 SITU plus the expert scale, 8 expert combine, 16
mix, 32 control, 64 conv tail, 128 flash-attention relative-position mask, 256 conv input plus state store.
`GGML_SYCL_IFUSE_DEBUG=1/2/3` reports matches and rejections. The decode graph goes from 7,609 to about 2,250 kernel
launches. The model file `inkling2.cpp` also stops copying single-token row views that are already contiguous.
Expert combine and control only fuse for 8 tokens or fewer (`GGML_SYCL_IFUSE_COMBINE_MAX_TOK`), because the fused
control kernel is slower than the separate matrix multiplies in prompt batches.

Two rules each matcher follows, learned the hard way:

- **Check the output against every buffer the kernel reads.** ggml-alloc reuses an input's memory for a later output
  once the input's last reader has run. Inside a fused group, that "last reader" may be a node the fused kernel
  replaces, so the allocator can place the fused output on top of data the fused kernel is still reading. The expert
  combine output landed on the expert ids this way and hung the device at more than 8 tokens, until 0029 added the
  check. Each matcher also checks that no intermediate of the group is read outside it.
- **Avoid single-work-group kernels.** The first mix and rel-pos mask kernels ran in one work-group and were slower
  than the ops they replaced (mix 26.7 us, mask 39 us). Spreading them over the device took them to 12.4 and 11 us.

## 4. A long-context bug in the unfused path

Without the mask fusion, the relative-position path builds f32 tensors of keys x micro-batch x 24 heads. Past about
2^29 elements (about 87k keys at micro-batch 256, about 22k at micro-batch 1024) the result goes bad. With flash
attention on and the fusion off it gives NaN: a 103k-token prompt was answered with "!!!!". With flash attention off it
degrades: llama-perplexity at `-c 24576 -ub 1024` gives 4.099 with flash attention off, against 3.938 with flash
attention on and fused, and 3.934 with flash attention off at `-ub 256`. This bug predates our patches. With flash
attention on, the mask fusion (0026, and for prompt batches since 0029) never builds those tensors, so the served
configuration avoids it. Which kernel overflows (GET_ROWS, CONT, ADD, CPY or the softmax over more than 2 GiB) is not
pinned down.

## 5. Where the time goes now

GPU timestamps for one decode token, about 17 ms in total:

| part | time | note |
|---|---|---|
| Q8_0 attention projections | ~4.3 ms | ~400 GB/s |
| NVFP4 experts | ~4.3 ms | gate_up 61 us, down 37 us per layer pass, ~390 GB/s |
| mix (fused) | 1.8 ms | |
| control (fused) | 1.2 ms | |
| ~1,500 remaining small kernels | rest | ~4-5 us each |

Tried without gain: SYCL graphs (+2 %, within noise), the Level Zero v2 adapter and in-order command lists, the DMMV
path for Q8_0, and 2-8 rows per work-group in the MoE mat-vec. The remaining ideas are fewer small kernels (attention
norm, conv and residual adds) and mat-vecs closer to the B70's ~550 GB/s.

## 6. Things that did not help

**DSpark block drafter** (the model's own drafter, Q8_0). It works but is slower here. With probabilistic draft
sampling it gives 1.1 tok/s: about 2.5 s per step goes to host-side sampling over the 201k-token vocabulary. With
greedy draft sampling it gives 18.6-22.4 tok/s against 28.4 without a drafter (measured before the speed pass), at an
acceptance of 0.83 for one drafted token. The verify step grows too fast with the batch (the NVFP4 MoE mat-vec at 8
columns took 2,020 us against 160 us at 1, before 0022), and the drafter adds about 35 ms per step.

**Vulkan** (Unsloth's prebuilt b12572-mix, llama-bench): pp512 50.6 and tg64 19.2 with flash attention off or on, and
pp512 57.5 / tg64 17.4 at depth 32k. KLD was fine (0.0052 / 0.0054), but it is far slower than SYCL here.

**Requanting the experts from UD-Q8_K_XL** (measured before patch 0022, on the old SYCL runtime, flash attention off):
all experts Q8_0 (28.2 GB) gave pp512 477 / tg64 34.9 at KLD 0.0059; gate_up Q6_K with down Q8_0 (24.3 GB) gave
431 / 36.0 at KLD 0.0060. Both beat the unpatched NVFP4 file (343 / 28.3) only because the NVFP4 kernel was slow. After
0022 the NVFP4 file alone reaches tg64 37.5 with a smaller file and lower KLD, so the requants were dropped.
