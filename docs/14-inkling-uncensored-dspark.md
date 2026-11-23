# Inkling 2 Mini Uncensored on ggml-org master: DSpark drafting, a reduced-vocabulary draft head and SYCL graphs (58.9 -> 69 tok/s)

Dates 2026-11-21 to 11-23. Same machine and containers as [01](01-environment.md): one Intel Arc Pro B70 (32 GB),
llama.cpp SYCL in the oneAPI container. This continues [13](13-inkling-2-mini.md) (the original model on Unsloth's
b12572-mix tree) for the uncensored checkpoint, which needs ggml-org master. It is written so that someone else can
pick the work up from this repository alone: section 9 lists what is open, in order of expected value.

All speeds are tok/s reported by llama-server or llama-bench. **"Sweep"** means: llama-server, 3 fixed prompts (an
ISO-8601 duration parser with tests, a B-tree split in C, a bash "largest files" script) x 2 rounds,
`reasoning_effort` low, temperature 1.0, top_p 1.0, top_k 0, 800 output tokens, median of the 6 requests. Speculative
runs vary more between runs than plain decode (the accepted length depends on the sampled text).

## 1. Result

| setup (uncensored AD-Q4_K, ctx 131072, micro-batch 256, flash attention on) | sweep tok/s |
|---|---|
| master + Inkling patches 0032-0044 (the starting point, no drafter) | 58.9 |
| + small-batch MoE (0045), no drafter, SYCL graphs on (0051) | 61.2 |
| + DSpark drafter, n-max 1 / 2 / 3, full draft head (0046) | 63.1 / 61.8 / 59.4 |
| + reduced-vocabulary draft head (0048), n-max 1 / 2 / 3 | 64.3 / **68.7** / 63.7 |
| + SYCL graphs for the verify step (0051), n-max 2 (**default**) | **69.1** (61-74 per request) |
| same, temperature 0 | 75.0 |

Other measurements of the default:

| | |
|---|---|
| llama-bench tg64 at depth 512, no drafter, graphs on | 63.2 |
| 103,347-token prompt (needle at 7 % depth, found) | read at 294 tok/s (351 s), then decode 39.8 tok/s |
| prompt reading, real agent traffic (6-11k tokens of code and tool output) | ~550 tok/s |
| 39-task suite (effort low, one run at temperature 1.0) | 34/39 in 17 min (no drafter: 36/39 in 19 min; single runs of this model range 32-36) |

The original model (NVFP4-Q8_0, Unsloth tree + 0030-0031, full draft head, no graphs) goes 57.9 -> 64.5 with the same
drafter fixes at n-max 2.

## 2. Files

- Target: `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from AtomicChat/Inkling-2-Mini-Uncensored-GGUF (their largest quant:
  expert gate_up Q4_K, expert down Q5_0, attention / loop controllers / lm head Q8_0). Architecture `inkling2_loop`,
  which needs ggml-org master (merged 2026-11-20); Unsloth's b12572-mix cannot load it.
- Drafter: `dspark-Inkling-2-Mini-Q8_0.gguf` from unsloth/Inkling-2-Mini-GGUF (3-layer block drafter, block 8, recurrent
  "markov" head, target features from layer passes 5 / 23 / 41 and the final hidden). It was trained on the original
  model; on the uncensored one its first draft is accepted 0.62-0.71 of the time (0.66 on the original). Use the file as
  published: a copy with `markov_w1` trimmed to the 200,058 real vocabulary rows does not load.
- Reduced-vocabulary drafter (made locally, section 5): `dspark-Inkling-2-Mini-Q8_0-sub32k-unc.gguf`.

## 3. Build

Base: ggml-org master 19bc41d (2026-11-21). Cherry-pick upstream PR #32799 (`ignore_merges` for the inkling2
tokenizer; merged upstream, not copied here), then apply `patches/0032-0053` in order (they apply cleanly on 19bc41d;
checked with `git am`), and build with `scripts/build-sycl-patched.sh` (oneDNN on, `-DGGML_SYCL_GRAPH=ON`,
`-DGGML_SYCL_F16=ON`). For the original model on Unsloth's b12572-mix tree, apply 0019-0031 instead.

| patch | what |
|---|---|
| 0032-0042 | ports of 0019-0029 (SCALE strided source, get_rows launch split, NVFP4 SWAR decode, OPPROF per-op profiler, per-head flash-attention mask, `inkling-fuse.cpp` node-group fusions) to master |
| 0043 | Q5_0 MoE mat-vec through local memory (the expert down projection: 8-expert k=1152 94 -> 34 us; real decode 50.7 -> 63.2 tok/s with it) |
| 0044 | fusions for master's graph (control gates after the down mat-vec, Q8_0 weights in the fused kernels) |
| 0045 | 2-8 token MUL_MAT_ID batches (a speculative verify) take one fused MMVQ launch per token instead of the generic per-expert path: pp2 49 -> 92, pp4 70 -> 131 tok/s |
| 0046 | **DSpark for Inkling 2 on master** (port of Unsloth's drafter graph) plus three fixes that make it fast, section 4 |
| 0047 | experimental shared-expert kernel for small batches (`GGML_SYCL_MMID_DEDUP=1`, off; section 8) and `GGML_SYCL_MMID_STATS` |
| 0048 | **reduced-vocabulary draft head**, section 5 |
| 0049 | reuse the q8_1 quantization of a mat-mul input within one graph (`GGML_SYCL_Q8_1_CACHE`, on; small) |
| 0050 | Q5_0 MoE kernel tuning variants (`GGML_SYCL_Q5_0_MOE_ROWS`, `_YSLM`; defaults unchanged, none faster) |
| 0051 | **SYCL graphs for 2-8 token MoE batches**: the graph-compatibility check judged them by the generic path's host wait, so every verify graph ran without graphs |
| 0052 | confidence gate for the drafter (`LLAMA_DSPARK_GATE`, off), confidence log, SYCL graph cache keyed per shape (section 7) |
| 0053 | debug: `LLAMA_GRAPH_TIMING` (graph build and scheduler allocation time) |

## 4. Why Unsloth's DSpark drafter was slow, and the fixes (0046; 0031 for the Unsloth tree)

As shipped, a draft step took ~37 ms on the B70, so drafting made decoding slower (18-28 tok/s). The causes:

1. The markov-head chain ran all 7 block positions, whatever `--spec-draft-n-max` was.
2. Each chained position made two passes over the 492 MB lm head: the backbone logits, and the markov correction.
3. The next token in the chain was chosen by a "first maximum" built from `pool_1d`, which the SYCL backend runs in a
   single work-item scanning all 200k logits, once per position.
4. The host copied 8 x 200k logits back and picked the tokens on the CPU.

Fixes: the chain stops at n-max (`LLAMA_DSPARK_NCHAIN`, set by the driver); each position takes one lm-head pass over
(backbone row + correction), which is the same linear map (`LLAMA_DSPARK_FOLD=0` restores the old form; acceptance
was identical in an A/B at temperature 0); the next token is a plain argmax (the SYCL argmax reduction now breaks ties to
the lowest index, like the CPU). A draft step is now 2.5 / 3.7 / 4.8 ms at n-max 1 / 2 / 3.

Porting notes for master (all in 0046): the drafter's target features are the inputs of the layer passes in
*execution* order (a looped core counts once per pass), and the final feature is the hidden state after the output mix
and norm, so the inkling2 graph tags `t_layer_inp[e]` per pass; the DSpark drafter takes a per-token "embd state"
(llama-batch had to size it for dspark drafters, otherwise the server's start-up test decode asserts); only greedy
drafting is ported (the probabilistic path is refused, and was not better, section 8).

## 5. Reduced-vocabulary draft head (0048, `scripts/make-dspark-subvocab.py`)

Even with the fixes, every drafted position ran a full 201k-row lm-head pass, and the host still copied all logits.
Inkling's own answers on code and security tasks (2.0M tokens) use only 10,783 distinct tokens; the top 8,192 cover
99.86 %. So:

1. `scripts/token-freq.py URL KEY OUT.json FILES...` counts the token ids of text the target model produced (here:
   answers and reasoning from a code + security suite) through llama-server's `/tokenize`.
2. `scripts/make-dspark-subvocab.py DRAFT.gguf TARGET.gguf FREQ.json 32768 OUT.gguf` (run with
   `GGUF_PY=<llama.cpp>/gguf-py` and a Python with numpy) writes a drafter copy with three extra tensors: 32,768 rows of
   the **target's** lm head (Q8_0 bytes copied), the same rows of `markov_w1`, and their token ids. The file is tied to the
   target it was made from.
3. With those tensors present, the chain runs its lm-head pass over 80 MB instead of 492 MB, takes the argmax in the
   subset, and writes the full token id into the position's chain-state row, which the driver reads; the draft batch
   asks for logits only at the anchor, so the 6.4 MB logits copy and the CPU argmax disappear.

Draft step 2.43 -> 1.27 ms at n-max 1. Acceptance at temperature 0, n-max 2: 0.487 (subset) vs 0.494 (full head).
`LLAMA_DSPARK_SUB=0` uses the full head.

## 6. Serving

```
llama-server --model Inkling-2-Mini-Uncensored-AD-Q4_K.gguf --gpu-layers 99 --ctx-size 131072 --parallel 1 \
  --flash-attn on --batch-size 2048 --ubatch-size 256 --jinja --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --spec-type draft-dspark --model-draft dspark-Inkling-2-Mini-Q8_0-sub32k-unc.gguf --spec-draft-n-max 2 \
  --spec-draft-sampling greedy --gpu-layers-draft 99
# environment: GGML_SYCL_ENABLE_GRAPH=1, ONEAPI_DEVICE_SELECTOR=level_zero:0
```

Reasoning effort per request: `"chat_template_kwargs": {"reasoning_effort": "low"}` (none ... max). n-max 2 is the
best fixed length here (sweep, uncensored with the reduced-vocabulary head: n1 64.3, n2 68.7, n3 63.7; longer drafts
were measured on the original model with the full head: n4 55.3, n5 52.5, n7 37.8). It fits at ctx 131072. Micro-batch 512 does not fit at 131072 (section 8).

Correctness checks: greedy text with and without the drafter diverges at an early near-tie (the verify batch rounds
differently from single-token decode; Inkling's short-convolution state is rolled back by llama.cpp's built-in
recurrent-state rollback, which lists inkling2 / inkling2_loop); both texts are coherent and converge on the same
answer. Greedy text with and without SYCL graphs is byte-identical; so is perplexity at micro-batch 1 / 2 / 4 with the
q8_1 cache on and off. The 103k needle passes with the drafter and graphs.

## 7. Where a speculative step spends its time

At n-max 2 a step takes ~29.4 ms with graphs: draft 1.5 ms, feature injection into the drafter 0.3 ms, verifying a
3-token batch ~27 ms. On the GPU alone (llama-bench, depth 512) 1 token costs 17.4 ms, 2 cost 23.2, 3 cost 27.4, 4 cost
32.0: each extra token adds ~4.6 ms, which is its 8 experts' weights (the MoE mat-vec runs once per token). The
llama-server adds ~0.6 ms per verified position (logits copy and CPU sampling over 200k).

The drafter's own confidence predicts rejection well. Logged over 1,899 steps at n-max 4 and temperature 1.0, a
drafted position with top-1 probability >= 0.9 is accepted 89 % of the time, [0.8, 0.9) 55 %, [0.5, 0.6) 34 %,
[0.2, 0.3) 17 %. A cost model fitted to the measurements (it reproduces the fixed n-max 2 result, 68.6 vs 69.1) says
"draft up to 4 and stop at the first position below 0.8" would give ~78.6 tok/s (+14 %); a perfect oracle 96.6.
0052 builds that gate (top-1 probability computed on the GPU in the reduced-vocabulary chain, `LLAMA_DSPARK_GATE=0.8`).
It behaves as simulated (1.52 drafted and 2.17 emitted tokens per step) but gives **no net gain**: the verify batch
size now changes from step to step, and every change rebuilds the llama.cpp graph (~0.2 ms build + ~0.44 ms scheduler
allocation, ~2.5 rebuilds per step) and, with SYCL graphs on, re-records the SYCL graph (step time 40 ms instead of 29).
Temperature-0 A/B: gate 74.8 vs fixed 74.0 tok/s without graphs, 53.9 vs 75.0 with graphs (61.2 with the per-shape
graph cache key also in 0052). The gate stays off; see 9.

## 8. What did not help (kept for the record)

| tried | result |
|---|---|
| probabilistic drafting (Unsloth's host path) | acceptance 0.60 vs greedy 0.585 on the same prompt; not worth a GPU implementation |
| shared experts across the verify batch (0047, `GGML_SYCL_MMID_DEDUP=1`) | adjacent tokens share 3.3 of 8 experts (distinct/slots 0.79 at 2 tokens, 0.62 at 4), but a kernel that reads each shared expert once is bit-exact and not faster (-1.6 % pass time at 4 tokens, +3.5 % at 2): the MoE mat-vec is bound by dot products and activation loads per row, not by expert bytes |
| `--backend-sampling` (GPU sampling) | step 29.5 vs 27.8 ms |
| `--samplers temperature` instead of the default chain | no change |
| target top_k 64 (Strata's default) | acceptance unchanged (0.511 / 0.503 vs 0.551 / 0.495) |
| Q5_0 MoE kernel: 2/8/16 rows per work-group, activation in local memory (0050) | none faster than the default 4 rows |
| `mix()` (the residual mix) as one launch per token | slower: tg64 62.3 vs 63.2, and the 4-token perplexity pass 23.7 vs 12.1 s (one work-group per token leaves the GPU idle) |
| micro-batch 512 / 1024 | do not fit at ctx 131072: the relative-position bias is materialised as an f32 mask of n_kv x micro-batch x 24 heads (3.2 GB at 131k x 256). At ctx 65536, micro-batch 512 fits and reads prompts 11-12 % faster |
| Strata's own Inkling engine (strata-inkling 0.1.66, SQ-20 pack, real prompts through its serve protocol) | 25.2 tok/s without its DSpark, 19.6 with it |

Lessons: with SYCL graphs on, a kernel launch is cheap, so a fusion only pays when it removes work, not when it trades
parallelism for fewer launches. Microbenchmarks in `test-backend-ops perf` reuse the same 8 experts every run, so they
are served from L2 (a Q8_0 expert mat-vec showed ~900 GB/s, above the card's 608 GB/s); judge kernels in real decode.

## 9. Open, in order of expected value

1. **Flash attention with the relative-position bias computed on the fly.** The graph now writes and reads an f32 mask
   of n_kv x micro-batch x 24 for the full-attention layers. Beyond distance 1024 the bias is one value per query and
   head, so a kernel that takes the per-token bias table and positions could drop the mask: less memory traffic at depth,
   and room for micro-batch 1024 at 131k. This is where prompt reading is weakest (294 tok/s at 103k, ~550 at 10k on
   real traffic), and agent workloads are dominated by prompt reading.
2. **Unlock the confidence gate (section 7).** Needs ggml's scheduler to keep one allocated graph per batch size (it
   runs the splits of the graph it allocated last), and the SYCL graph cache to replay per shape. Estimated +6 % at
   temperature 0, up to +14 % at 1.0.
3. **Host time per step**: ~2 ms per step are not explained by draft + injection + GPU verify; time the target decode,
   sampling and rollback calls in llama-server before choosing a fix.
4. Small graph items: 97 extra CONT launches that appear only in multi-token graphs (~0.4 ms per verify); a PAD of a
   constant weight and a zero FILL in every layer pass (48 launches each per token).
5. The reduced-vocabulary head for the original model (its own lm-head rows) and SYCL graphs on the Unsloth tree
   (it lacks 0051).

## 10. Agent side

For long code review the useful numbers are confirmed findings per hour, misses and false positives, not short-prompt
tok/s. See [11](11-agent-context-management.md) section "Mid-task evidence ledger" for the change made to the agent so
that one long review turn keeps its evidence through compactions, and for the evaluation design used to compare servers.
