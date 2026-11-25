# Faster Inkling decode attention on the B70 (patch 0061)

Patch 0061 changes only the nextsycl inkling2 silo: the decode attention kernels, their merges and a few small
kernels. Every output is **byte-identical** to 0059. On one Arc Pro B70 with AtomicChat Inkling-2-Mini-Uncensored
AD-Q4_K it makes decode **6 to 13 % faster**, and the 39-task suite went from **105.0 to 115.1 tok/s** pooled over three
interleaved runs each. Prompt reading is unchanged. [Patch](../patches/0061-nextsycl-inkling-decode-attention.patch) ·
[Kernel bench](../scripts/inkling-attn-bench.sh) · [Release guide](15-inkling-nextsycl.md)

The ideas come from steveseguin's own one-card SYCL engine for the original Inkling-2-Mini
([b70-optimization-lab, `experiments/own-xpu-runtime/inkling2` at 1d3a7d0](https://github.com/steveseguin/b70-optimization-lab/tree/1d3a7d0445082aab48711c511a034301a268a38a/experiments/own-xpu-runtime/inkling2),
Unlicense). That engine runs Unsloth's UD-IQ4_XS file from a command line and cannot load the AD-Q4_K file, so its
ideas were rewritten for the nextsycl silo rather than its code copied. The code comments credit each one.

## Why decode attention

A GPU profile of one plain decode step on 0059 (`NS_INK2_PROFILE=gpu`, the difference between a 192-token and a
64-token run, divided per step) showed everything except attention flat with context:

| context | attention per step | attention merge per step |
|---|---:|---:|
| short prompt | 1.02 ms | 0.17 ms |
| 8K | 2.58 ms | 0.59 ms |
| 40K | 5.35 ms | 0.66 ms |
| 100K | 10.59 ms | 1.20 ms |

Inkling runs 48 layer applications per token (layers 6 to 17 loop three times). 40 of them are windowed (1,024 keys)
and 8 attend to the whole context. Each layer splits its keys into 64-key blocks, computes one partial result per
block, and merges the partials. When the drafter's proposals are checked, 2 to 8 rows go through at once. In 0059 each
row read its whole window again, so a windowed layer cost 36 µs for one row and 159 µs for seven.

For comparison, steveseguin's README lists 87.7 to 90 tok/s plain decode on a short prompt and 53 after 100K, with
greedy decoding on their own model file. 0059 measured 76.8 and 42.3 here (fixed seed, temperature 1). The files
differ (about 3.8 GB read per step here against about 3.45 GB there), so this is not a head-to-head. Their prompt
reading is 713 tok/s at 100K against about 1,800 here, so their prompt-side work does not transfer.

## What the patch does

All three changes keep the same lane for each key, the same dimension rotation, the same fma chains and the same
summation order. That is why the outputs match bit for bit.

1. **One tile for all rows of a check.** For 1 to 8 rows a pass, one work-group now handles one (key head, 64-key
   block) for all rows and loads the block's keys and values into local memory once, instead of once a row. Each
   (row, query head) is spread over 4 sub-groups for up to 2 rows and over 2 for up to 4 rows. 9 rows and more keep the
   old kernel. (Their idea: check steps read each key once for all their tokens.)
2. **Reads issued first.** The attention merges, the exact merge's inner loop and the tile fills now issue several
   loads together, with indexes past the end clamped, and then do the same operations in the same order. (Their rule
   for small latency-bound kernels.)
3. **Job-table pointers cast to global memory** in the dense Q8_0 products, the loop-controller kernels and
   `rms_xq_many`, with the job copied using fixed indexes. (Their `gptr` and `pick_job`.) The measured change was
   within the noise of kernels the patch does not touch, so no gain is claimed for this part.

`NS_INK2_ATTN_ROWS` selects the attention kernel inside the silo: `0` is the 0059 kernel, `1` is the shared tile with
one sub-group a (row, head), and `2` (the default when the variable is unset) adds the spread. The portable launcher
does not forward this variable. Setting it to an empty string selects `0`.

A fourth change was tried and dropped: computing the exact merge's weighted sums 8 or 4 dimensions a work-item instead
of 16 gave no gain (100K, 7 rows: 3,103 µs at 16, 3,128 at 8, 3,131 at 4).

## Exactness checks

- **Full-vocabulary scores.** `nextsycl llm generate MODEL --prompt-file PREFIX --score continuation.txt --score-max 192
  --dump FILE` with `NS_SCORE_ROWS` rows a pass writes the target's scores for all 200,058 vocabulary entries at each of
  192 teacher-forced positions. The 0061 files were compared byte for byte with 0059's on prefixes of about 900, 4,000,
  7,300, 39,000 and 100,000 tokens at 1 to 8 rows a pass (15 combinations). The 900- and 4,000-token prefixes put the
  scored positions across the window (1,024) and the long-attention (4,096) boundaries. A 9,465-token prefix at 1, 2 and
  5 rows a pass puts every scored row's window across the windowed ring's wrap at 9,215 slots. All 18 were identical.
- **Kernel outputs.** [`inkling-attn-bench.sh`](../scripts/inkling-attn-bench.sh) builds a small bench against two
  source trees, runs the silo's own entry points on synthetic data, and compares every output. With `BENCH_WIDE=1` it
  covers every row count from 1 to 8, positions around the window and long-path boundaries, positions across the ring's
  wrap for 1 to 16 rows, and the drafter's block attention. All outputs were byte-identical. The wrap cases were also
  run with `NEW_ENV=NS_INK2_ATTN_ROWS=0` and were identical too. A rerun of the published script on trees made from the
  published patches (0054 to 0059 against 0054 to 0059 plus 0061) gave 24 of 24 default cases and 215 of 215
  `BENCH_WIDE=1` cases byte-identical, with times within about 1 % of the table below.
- **Served conversation.** The 14-turn chat replay (`bench/inkling_context.py --case chat`, cold server through the
  launcher) reproduced all 14 published visible and reasoning hashes and generated lengths.

## Speed

Kernel bench, microseconds a call (attention plus merge), fastest of two alternating rounds:

| case | 0059 | 0061 |
|---|---:|---:|
| windowed layer, full window, 1 row | 35.6 | 20.7 |
| windowed layer, full window, 2 rows | 50.6 | 29.3 |
| windowed layer, full window, 4 rows | 96.5 | 44.0 |
| windowed layer, full window, 7 rows | 156.5 | 81.3 |
| full layer at 2K, 1 row | 53.3 | 31.1 |
| full layer at 100K, 1 row | 1,192 | 1,011 |
| full layer at 100K, 7 rows | about 3,100 | about 3,100 |
| drafter block attention, 4 rows at 5K | 72.2 | 33.7 |

The last full-layer row is unchanged: past 4,096 keys, 2 to 8 rows use the long-attention path from patch 0056, which
0061 does not touch.

Decode with `nextsycl llm generate`: 256 tokens, temperature 1, top-p 1, seed 1, low effort, full-vocabulary sampling.
Plain means no drafter. Drafted means the launcher's setting of 6 DSpark proposals at confidence 0.65 with the
32,768-row head. Each cell is the mean of 3 pairs run in alternating order. Every pair generated the same text and
accepted the same drafts, because the sampled token depends only on the seed and the position.

| prompt | plain 0059 | plain 0061 | drafted 0059 | drafted 0061 |
|---|---:|---:|---:|---:|
| short (56 tokens) | 76.7 | 82.6 (+7.7 %) | 101.4 | 107.4 (+5.9 %) |
| 7,300 tokens | 66.0 | 73.6 (+11.4 %) | 98.0 | 110.6 (+12.9 %) |
| 39,000 tokens | 55.6 | 62.0 (+11.5 %) | 98.2 | 107.4 (+9.4 %) |
| 100,000 tokens | 42.3 | 47.8 (+13.0 %) | 60.2 | 64.4 (+6.9 %) |

On a short prompt, the attention kernel went from 21.3 to 9.3 µs a call in the GPU profile.

Served workloads, both through the launcher:

- **14-turn chat replay:** decode 81.65 tok/s on 0059 and 91.16 on 0061 (+11.6 %). Fresh prefill was 331.5 and 329.9.
- **39-task suite** (`bench/run_bench.py`, low effort, temperature 1, top-p 1, top-k 0), three runs of each build,
  interleaved. The metric is client-side pooled decode: the sum of completion tokens divided by the sum of their
  decode times. It is not the engine-time figure in the release guide.
  - 0061: 110.0, 118.6 and 119.4, pooled **115.1 tok/s**, 105 of 117 tasks passed.
  - 0059: 106.4, 104.0 and 104.6, pooled **105.0 tok/s**, 101 of 117 passed.

  The ranges do not overlap. The pass counts differ by chance: the suite samples fresh answers on every run, and the
  two builds give identical scores.

## Draft settings re-checked

Cheaper check rows could move the best number of proposals, so the setting was swept again on 0061. Short, 7,300 and
39,000-token prompts were each run at seeds 11 to 13 with 384 tokens. Each setting has one run per cell, pooled over
the 9 runs:

| proposals @ confidence | pooled tok/s |
|---|---:|
| 6 @ 0.65 (launcher default) | 101.9 |
| 4 @ 0.65 | 103.9 |
| 6 @ 0.8 | 103.2 |
| 7 @ 0.65 | 103.2 |
| 6 @ 0.5 | 99.0 |
| 7 @ 0.5 | 99.3 |
| 7 @ 0.4 | 93.1 |

The generated text was the same in all 63 runs. No setting is separable from the default: they are within about 2 %
in no consistent order, from single runs. Thresholds of 0.5 and below are 3 to 9 % slower. The default stays.

## Build and identify it

Start from nextsycl `e3f1fd891b7fda6f478a326d913d28893a4e9811`, apply 0054 to 0059 and then 0061. The patched tree is
`853e9daff944d88c979a6f0b1f67c55d0a78504d` (0059 alone is `661e2787...`). The measured runtime used 0059's `nextsycl` and
`libnextsycl-llm.so` with only `silo/libnextsycl-inkling2.so` rebuilt from that tree, using the release's compile flags
(`icpx -fsycl -std=c++20 -O3 -fPIC -fp-model=precise -fsycl-default-sub-group-size=32
-fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen`, linked for `-device bmg-g31` with
`-cl-fp32-correctly-rounded-divide-sqrt`). A full upstream `build.sh all` from the same tree should build the same
sources, but that path was not run for 0061.

Silo rebuilds here are not byte-reproducible: a rebuild of the unchanged 0059 tree gave a different hash. So no silo
hash is given as a check. The score comparison and the kernel bench above are the check.

0061 also applies cleanly on top of the optional 0060 (both touch `mv.cpp`), but that combination was not built or
measured.

## Not done

- **Long full layers with 2 to 8 rows.** At 100K with 7 rows a full layer still takes about 3.1 ms on the 0056 long
  path. The ESIMD dot pass pads to 16 or 32 query rows, and the value pass and exact merge are about 0.5 ms more.
- **Choosing the number of proposals from expected tokens per millisecond.** steveseguin's engine picks the guess
  count from the drafter's probabilities and a table of step costs. That would live in the Rust engine and was not
  built.
- **Flash decode for full layers** (scores never stored, running maximum). It changes rounding, so it cannot be
  bit-identical to 0059.
- Also not ported: a transposed value cache (it would touch prompt attention), an int8 drafter table and drafter
  products on XMX (the drafter's fold and pick cost about 0.1 ms a step here), SYCL graph replay (host submission is not the limit here),
  and the embedding table in host memory (it would only raise the context ceiling).
