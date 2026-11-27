# Native Inkling decode on the B70: 64.4 to 96.7 tok/s at 100K, bit for bit (patch 0062)

Patch 0062 stacks on patch 0061 across both the `nextsycl` `inkling2` silo (`kernels/llm/inkling2/silo/`) and the Rust engine (`llm/inkling2/src/`). Every target forward-pass output and full-vocabulary score remains **byte-identical** to 0059 and 0061. On one Intel Arc Pro B70 (32 GB) serving AtomicChat `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` with the 32,768-row DSpark head, fixed-seed drafted decode after a 100K prompt rose from **60.2 tok/s** (0059) and **64.4 tok/s** (0061) to **96.3–96.8 tok/s** (+50 % over 0061, +60 % over 0059). At 39K it rose from **107.4 to 140.8 tok/s** (+31 %); at 7.3K from **110.6 to 133.1 tok/s** (+20 %); steady-state short decode reaches **146.2 tok/s**; and the 14-turn conversation replay rose from **81.7** (0059) and **91.2** (0061) to **112.9 tok/s** (+24 %) while reproducing all 14 published visible and reasoning hashes. Prompt reading (~1,820 tok/s at 100K) is unchanged. [Patch](../patches/0062-nextsycl-inkling-neon-decode.patch) · [Patch 0061 notes](17-inkling-decode-attention.md) · [Release guide](15-inkling-nextsycl.md)

## Why 0061 plateaued at 64.4 tok/s at 100K

After patch 0061 shared each 64-key window tile across 1–8 check rows, two large bottlenecks remained:

1. **Multi-row long full-attention (`p0 >= 4096`, `T = 1..8`).** In 0061, checking 2–8 rows on the 8 full-attention layer applications still ran the two-pass 0056 kernel (`dot_2pass` writing every key logit to global memory, followed by a separate value pass and exact merge) at ~3.1 ms per full-layer call for `T = 7` at 100K (~25 ms across 8 full layers). Even `T = 1` read uncached keys/values without hoisting queries into SLM across splits.
2. **Uncoupled greedy DSpark proposals, fixed 32K head OOV misses, and per-step kernel/launch overhead.** At temperature 1, greedy proposals diverged from the target's inverse-CDF draw; tokens specific to a session's code or log context outside the static 32,768-row head forced immediate verification rejection; DSpark's own full-attention layer (`win == 0`) read FP16 KV across 100K positions; and each 48-layer pass launched hundreds of eager kernels (including 64 tiny 6–8 work-group auxiliary projections, loop controllers, router scans, and an ungraphed 200K-vocab softcap + sampling tail).

## What patch 0062 changes

Every target-model optimization preserves the exact IEEE-754 lane assignments, dimension rotations, FMA chains, Q8_0 activation rounding, and split-reduction order of 0059/0061. And because target verification uses a deterministic stateless draw keyed strictly by `(seed, token_position)`, drafter changes alter only how many proposals are verified at once — never the accepted token sequence.

1. **Single-launch ESIMD FlashDecoding for long full attention (`attn_long.hpp`, `merge_long.hpp`).**
   - Replaces the two-pass global-logit path for `T = 1..8` (`win == 0, p0 >= 4096`) with single-launch ESIMD kernels that keep each 64-key split's scores in registers/SLM, evaluate exact `exp16_exact` matching SYCL `sycl::exp`, transpose `V` into SLM for 256-byte `slm_block_load` bursts, and write identical `[m, s, vo[128]]` split partials for `merge_exact`:
     - `T = 1` (`prefix >= 256`): 4-split `Q_SLM`-hoisted `fused_esimd_t1<4>` (`full 1 99900`: **1,011 -> 891.6 µs**).
     - `T = 2`: 2-split `Q_SLM`-hoisted SIMD16 `fused_esimd<8, 2, 3>` sized to stay inside Xe2's 128 KB L1 (`full 2 99900`: **1,037.2 µs**).
     - `T = 3..8`: 4-split `k_next`/`v_next`-pipelined `fused_esimd` and `fused_esimd_24` using native Xe2 zero-MOV region broadcasts (`replicate_vs_w_hs`), cutting `full 7 99900` from **~3,100 µs to 1,945 µs** (including exact merge).
   - `merge_long.hpp` loads split `(m, s)` headers via a single coalesced `float2` pass into SLM instead of a serial lane-0 loop.

2. **Coupled INT8 DSpark drafter, INT8 Layer-1 KV + `s8` XMX DPAS, and dynamic session-OOV head (`draft.cpp`, `draft_attn_esimd.hpp`, `dspark.rs`).**
   - **Coupled inverse-CDF sampling & sunk-cost proposal policy:** At `temperature > 0`, DSpark samples from the exact coupled uniform draw `u(seed, pos)` via a 3-pass CSR tile inverse-CDF kernel over per-32-block INT8 transposed `fold` weights (`ns_ink2s_fold_sample_i8_oov`), reporting hierarchical boundary-margin acceptance probability into an expected-`tok/ms` stopping rule calibrated with the sunk `ds_draft` cost.
   - **Symmetric INT8 KV cache + `s8` XMX DPAS on DSpark's full-attention layer:** Stores DSpark Layer 1 (`win == 0`) keys and values as per-head-row symmetric INT8 (`int8_t[Hkv, cap, 128]` + `float` scale, `2,112 B/tok` instead of `4,096 B/tok` FP16, including checkpoint save/restore/copy) and runs online flash-decoding directly on hardware `xmx::dpas<8, 8, int, int, s8, s8>`.
   - **Dynamic session-OOV proposal head (`sub32k + session OOV <= 8,192`):** Tracks out-of-vocabulary tokens encountered during prompt prefill (`feed_until` / `pass_out`), unpacks and quantizes their target output-head and `fold` rows on the GPU once after prefill, and includes them in `fold_pick_i8_oov` / `fold_sample_i8_oov` for ~2 µs/tok without invalidating the captured draft graph. At 100K this raised accepted proposals from `165/208` to `172/211` (+3.5 tok/s, matching a full 200K drafter head's acceptance without its +1.8 ms/step cost).

3. **SYCL command graphs (`ext_oneapi_graph`) and fused graphed target sampling (`dec.cpp`, `engine.rs`, `ffi.rs`).**
   - Captures and replays the 9 sliding-window segments (`GraphKey::Pass`), `ds_inject`, `ds_draft` (`GraphKey::Draft`), and the output head using device-resident `Ink2sPassParams` and `Ink2sDraftParams` buffers (`p0` at offset 0).
   - Records `ns_ink2s_softcap_argmax` and `ns_ink2s_softcap_sample_full_p0` directly inside `GraphKey::Pass`, eliminating the 11.2 MB/step `logits` DRAM roundtrip and cutting full-vocab target sampling to **7.8 µs/tok**.

4. **Dense/MoE/controller & launch fusions (`dec.cpp`, `mv.cpp`, `q8_head.hpp`, `engine.rs`).**
   - Losslessly repacks loop-controller matrices (`ctrl_down_q8_many`, `ctrl_apply_q8_many`) to Q8_0 with T-shared reads and pipelined 4-channel `char4` loads + 4-lane parallel `tanh`; fuses T-shared `router_union` (`float4` rank loop + subgroup `exclusive_scan_over_group`); fuses `moe_down_conv_res_union` with spill-free `m = 1, 2, 3+` specializations; adds XMX DPAS `q8_head` for `>= 16K`-row Q8_0 heads; caches `token_embd` in host RAM for decode; batches Q8_0 `sink`, `k_dyn`, and `v_dyn` into `mm_many` via exact `ns_ink2s_xq_many` (eliminating 64 tiny `mv` launches/step); and trims barriers/SLM in `mix_norm`, `mix_exact`, `rms_xq_many`, and `post_qkv`.

## Exactness checks

- **Full-vocabulary target scores (18 specs).** Compared byte for byte against 0061/0059 across 192 teacher-forced positions x 200,058 vocabulary logits per spec: `p900:{1,2,3,5}`, `p4000:{1,2,3,5,8}`, `p9400:{1,2,5}` (windowed ring wrap), `8k:1`, `40k:{2,3,4}`, and `100k:{1,7}`. **18 of 18 score files matched byte for byte.**
- **Attention & merge kernel bench.** [`inkling-attn-bench.sh`](../scripts/inkling-attn-bench.sh) with `BENCH_WIDE=1` between 0061 and 0062: **0 of 215 output files differed.**
- **Fixed-seed decode text.** Every fixed-seed generation across `short`, `7.3K`, `39K`, and `100K` prompts (seeds 1 and 11, plain and drafted) produced **byte-identical output tokens** (`cmp -s`).
- **Served 14-turn conversation.** Replaying `bench/inkling_context.py --case chat` through the launcher reproduced **all 14 published `visible_sha256` and `reasoning_sha256` digests and `generated_tokens` counts**.
- **Compiler hygiene.** Zero register spills across all decode, drafter, MoE, merge, and graph kernels when compiled AOT for `-device bmg-g31`.

## Speed and context ceiling

Attention + merge kernel bench (`inkling-attn-bench.sh`, microseconds per call, fastest of two alternating rounds):

| case | 0059 | 0061 | **0062** |
|---|---:|---:|---:|
| windowed layer, full window, 4 rows | 96.5 | 44.0 | **39.4** |
| windowed layer, full window, 7 rows | 156.5 | 81.3 | **71.5** |
| full layer at 39.3K, 1 row | — | 401.2 | **372.5** |
| full layer at 39.3K, 2 rows | — | — | **415.4** |
| full layer at 39.3K, 7 rows | — | — | **801.7 + merge** |
| full layer at 100K, 1 row | 1,192 | 1,011 | **891.6 + 85.2 merge** |
| full layer at 100K, 2 rows | — | — | **1,037.2 + merge** |
| full layer at 100K, 7 rows | ~3,100 | ~3,100 | **1,945.4 + 236.0 merge** |

Fixed-seed CLI decode (`nextsycl llm generate`, AD-Q4_K + 32K DSpark head, 6 proposals at confidence 0.65 unless noted, full-vocabulary sampling, seed 1 unless noted):

| workload | 0059 | 0061 | **0062** | vs 0061 | vs 0059 |
|---|---:|---:|---:|---:|---:|
| **100K drafted** (`256` tok, `seed=1`, `temp=1`) | 60.2 | 64.4 | **96.30–96.69** (`172/211`) | **+49.5–50.1 %** | **+60.0–60.6 %** |
| **100K drafted** (`256` tok, `seed=11`, `temp=1`) | — | — | **96.81** (`173/213`) | **+50.3 %** | — |
| **100K drafted greedy** (`256` tok, `temp=0`) | — | — | **103.1+** (`186/223`) | — | — |
| **39K drafted** (`256` tok, `temp=1`) | 98.2 | 107.4 | **140.61–140.81** (`187/216`) | **+31.1 %** | **+43.4 %** |
| **7.3K drafted** (`256` tok, `temp=1`) | 98.0 | 110.6 | **133.06** (`165/207`) | **+20.3 %** | **+35.8 %** |
| **Short-1024 drafted** (`temp=1`, steady-state) | — | — | **146.12–146.20** (`657/848`) | — | — |
| **Short-256 drafted** (`temp=1`, cold CLI) | 101.4 | 107.4 | **126.07** (`134/195`) | **+17.4 %** | **+24.3 %** |
| **Short-256 drafted** (`seed=11`, `temp=1`) | — | — | **131.17** (`141/202`) | — | — |
| **Short-256 greedy** (`temp=0`) | 121.7 | — | **142.29** (`159/218`) | — | **+16.9 %** |
| **Short-256 plain** (`drafts=0`, `temp=1`) | 76.7 | 82.6 | **88.37** | **+7.0 %** | **+15.2 %** |
| **100K plain** (`drafts=0`, `temp=1`) | 42.3 | 47.8 | **50.89** | **+6.5 %** | **+20.3 %** |
| **14-turn chat replay** (served via launcher) | 81.7 | 91.2 | **112.86** (14/14 hashes) | **+23.8 %** | **+38.1 %** |
| **100K prompt reading** | ~1,800 | ~1,800 | **1,813–1,824** | unchanged | unchanged |

### Empirical maximum context on one 32 GB Arc Pro B70

Measured by binary search in 1,024-token steps on the 32 GB card (`31.5 GiB` usable Level Zero VRAM, preserving the engine's strict `1.5 GiB` spare VRAM reserve so allocations never spill over PCIe):

- **With DSpark drafter enabled** (`34.0625 KiB/tok` combined KV after INT8 DSpark Layer-1 cache + host-cached `token_embd`): **`375,808` tokens (`367K`)** loads and decodes cleanly (`376,832` hits the `1.5 GiB` reserve).
- **Without drafter (`--no-mtp`)** (`32.00 KiB/tok` FP16 target KV): **`418,816` tokens (`409K`)** loads and decodes cleanly.

### Where a 0062 step spends its GPU time (`NS_INK2_PROFILE=gpu`, 192 minus 64 tokens)

- **100K drafted (`~9,700 µs/tok` GPU):** `attention silo` `3,703 µs` + `merge silo` `500 µs` (`43 %`), `experts gu` `1,883 µs` (98 % GDDR6 BW), `experts down+conv` `1,078 µs` (100 % BW), `mv many` `711 µs` (81 % BW), `mv silo` `602 µs`, `ctrl down` `195 µs`, `mix norm` `188 µs`, `router union` `188 µs`, `draft attention` `148 µs`, `rms xq` `125 µs`, `ctrl apply` `117 µs`, `post qkv` `94 µs`, `conv res` `55 µs`, `draft fold samp` `39 µs`, `xq` `31 µs`, `softcap+sample full` `7.8 µs`.
- **Short drafted (`~8,656 µs/tok` GPU):** `experts gu` `2,500 µs`, `experts down+conv` `1,398 µs`, `mv many` `1,266 µs`, `mv silo` `1,086 µs` (`72 %` weight-bandwidth bound), `mix norm` `352 µs`, `ctrl down` `289 µs`, `attention` `273 µs`, `router union` `266 µs`, `rms xq` `227 µs`, `ctrl apply` `172 µs`, `post qkv` `148 µs`, `merge` `109 µs`, `conv res` `86 µs`, `draft fold samp` `70 µs`, `xq` `55 µs`, `softcap+sample full` `7.8 µs`.

## Tested and dropped

- **3-barrier full-64-key `V_SLM` kernel for `T = 6..7`:** Kept SLM at 32,256 B (`0` spills, `0/215` bit diffs) and cut barriers `5 -> 3`, but regressed `full 7 99900` (`1,948.6 -> 2,158.2 µs`) due to SLM read-port stalls over the wider 64-key `P * V` inner loop.
- **Hoisting `K` across splits for `QP >= 16`:** Holding `32–64 KB` of `K` per work-group across splits evicted Xe2's 128 KB L1 (`1,954 -> 2,047–2,083 µs` at `T = 7`).
- **Softer coupled acceptance exponent (`p^0.85`) and aggressive `T = 6..7` cost curve:** Raised accepted proposals (`172 -> 176` at 100K, `657 -> 667` at short-1024) with identical outputs, but reduced wall-clock throughput by ~1 % because `T = 6..7` crosses into `fused_esimd_24` and captures extra SYCL graph shapes.
- **Monolithic 48-layer SYCL graph across full-attention layers in short CLI runs:** Added 40–125 ms of Level Zero graph finalization latency on short runs; graphing the 9 sliding-window segments plus drafter/output heads captured the launch savings without compile overhead.
- **Reusing SLM-quantized `nq` from `rms_q` for `sink/k_dyn/v_dyn`:** Differed by 1 int8 ULP from `ns_ink2s_xq` on global `normed`, flipping a router tie at `p4000` (`pos = 4082`). Unconditionally quantizing `normed[..aux_cnt]` via `ns_ink2s_xq_many` restored 18/18 exactness.
- **Sparse `c[i] == 0.0f` skip in `merge_long.hpp`:** L2-clipped `Q, K` keep split logit spans narrow (`~10..40`), so almost no splits underflow to zero (`523 -> 617 µs/tok`).

## Build and identify it

Start from `nextsycl` commit `e3f1fd891b7fda6f478a326d913d28893a4e9811`, apply `0054` through `0059`, then `0061`, and then `0062`. Because `0062` updates both the `inkling2` silo and `llm/inkling2/src/`, rebuild both `nextsycl` (`libnextsycl-llm.so` + `nextsycl`) and `silo/libnextsycl-inkling2.so` (`AOT=bmg-g31 ./build.sh all`). The resulting source tree hash is `26a3e0b21ebecfd2777238eacf4c324e443be291`.
