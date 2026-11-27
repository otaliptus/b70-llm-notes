# Native Inkling decode on the B70: 96.4 tok/s exact (147 tok/s short) and opt-in INT8 KV to 155.7 tok/s at 39K / 110–128 tok/s at 100K / 672K context (patches 0062–0063)

Patches 0062 and 0063 stack on patch 0061 across both the `nextsycl` `inkling2` silo (`kernels/llm/inkling2/silo/`) and the Rust engine (`llm/inkling2/src/`). By default (`NS_INK2_KV=f16`), every target forward-pass output and full-vocabulary score remains **byte-identical** to 0059 and 0061 while raising single-stream decode on one Intel Arc Pro B70 (32 GB) serving AtomicChat `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` + 32,768-row DSpark from **60.2 tok/s** (0059) and **64.4 tok/s** (0061) to **96.4–96.8 tok/s** at 100K (+50 % over 0061, +60 % over 0059), **141.0 tok/s** at 39K (+31 %), **133.8 tok/s** at 7.3K (+21 %), **147.2 tok/s** steady-state short decode, **90.6 tok/s** plain short decode (+10 %), and **113.5 tok/s** on the 14-turn conversation replay (+25 %) while reproducing all 14 published visible and reasoning hashes.

Patch 0063 also adds an opt-in symmetric INT8 full-attention target KV cache (`NS_INK2_KV=q8` or `--opt-kv q8`) with hybrid XMX `s8 x s8` prefix + SLM tail FlashDecoding, cutting 100K full-attention + merge GPU time 2.45x (`4,203 -> 1,719 µs/tok`), lifting 39K decode to **152.4–155.7 tok/s** (+45 % over 0061), 100K decode to **108.3–113.5 tok/s** at temperature 1 (**128.1 tok/s** greedy, **64.5 tok/s** plain), 100K prompt reading to **1,899 tok/s**, 14-turn chat decode to **115.7 tok/s**, and empirical single-B70 maximum context from **375,808 to 688,128 tokens (`672K`)** with DSpark enabled (**733,184 tokens / `716K`** without drafter). [Patch 0062](../patches/0062-nextsycl-inkling-neon-decode.patch) · [Patch 0063](../patches/0063-nextsycl-inkling-q8-kv-and-moe-sfu.patch) · [Patch 0061 notes](17-inkling-decode-attention.md) · [Release guide](15-inkling-nextsycl.md)

## Why 0061 plateaued at 64.4 tok/s at 100K

After patch 0061 shared each 64-key window tile across 1–8 check rows, two large bottlenecks remained:

1. **Multi-row long full-attention (`p0 >= 4096`, `T = 1..8`) and FP16 KV bandwidth.** In 0061, checking 2–8 rows on the 8 full-attention layer applications still ran the two-pass 0056 kernel (`dot_2pass` writing every key logit to global memory, followed by a separate value pass and exact merge) at ~3.1 ms per full-layer call for `T = 7` at 100K (~25 ms across 8 full layers). Even after single-launch FP16 FlashDecoding, 8 full layers x 100K x FP16 K+V stream `3.28 GB/step` across the 578 GB/s GDDR6 bus and consume `32 KiB/token` of VRAM.
2. **Uncoupled greedy DSpark proposals, fixed 32K head OOV misses, MoE SFU serialization, and per-step launch overhead.** At temperature 1, greedy proposals diverged from the target's inverse-CDF draw; tokens specific to a session's code or log context outside the static 32,768-row head forced immediate verification rejection; DSpark's own full-attention layer (`win == 0`) read FP16 KV across 100K positions; `moe_gu` / `moe_gu_union` evaluated `tanh` and `exp` via serial lane-indexed loops; and each pass launched hundreds of eager kernels.

## What patches 0062 and 0063 change

### Default bit-exact path (`NS_INK2_KV=f16`, patches 0062 + 0063)

Every default target-model optimization preserves the exact IEEE-754 lane assignments, dimension rotations, FMA chains, Q8_0 activation rounding, and split-reduction order of 0059/0061. Because target verification uses a deterministic stateless draw keyed strictly by `(seed, token_position)`, drafter changes alter only how many proposals are verified at once — never the accepted token sequence.

1. **Single-launch ESIMD FlashDecoding for long FP16 full attention (`attn_long.hpp`, `merge_long.hpp`).**
   - Replaces the two-pass global-logit path for `T = 1..8` (`win == 0, p0 >= 4096`) with single-launch ESIMD kernels that keep each 64-key split's scores in registers/SLM, evaluate exact `exp16_exact` matching SYCL `sycl::exp`, transpose `V` into SLM for 256-byte `slm_block_load` bursts, and write identical `[m, s, vo[128]]` split partials for `merge_exact`:
     - `T = 1` (`prefix >= 256`): 4-split `Q_SLM`-hoisted `fused_esimd_t1<4>` (`full 1 99900`: **1,011 -> 891.6 µs**).
     - `T = 2`: 2-split `Q_SLM`-hoisted SIMD16 `fused_esimd<8, 2, 3>` sized to stay inside Xe2's 128 KB L1 (`full 2 99900`: **1,037.2 µs**).
     - `T = 3..8`: 4-split `k_next`/`v_next`-pipelined `fused_esimd` and `fused_esimd_24` using native Xe2 zero-MOV region broadcasts (`replicate_vs_w_hs`), cutting `full 7 99900` from **~3,100 µs to 1,945 µs** (plus coalesced `float2` `ms_p` in `merge_long.hpp`).

2. **Coupled INT8 DSpark drafter, INT8 Layer-1 KV + `s8` XMX DPAS, dynamic session-OOV head, and fused `add_rms_xq` (`draft.cpp`, `draft_attn_esimd.hpp`, `dspark.rs`).**
   - **Coupled inverse-CDF sampling & sunk-cost proposal policy:** At `temperature > 0`, DSpark samples from the exact coupled uniform draw `u(seed, pos)` via a 3-pass CSR tile inverse-CDF kernel over per-32-block INT8 transposed `fold` weights (`ns_ink2s_fold_sample_i8_oov`), reporting hierarchical boundary-margin acceptance probability into an expected-`tok/ms` stopping rule calibrated with the sunk `ds_draft` cost.
   - **Symmetric INT8 KV cache + `s8` XMX DPAS on DSpark's full-attention layer:** Stores DSpark Layer 1 (`win == 0`) keys and values as per-head-row symmetric INT8 (`int8_t[Hkv, cap, 128]` + `float` scale, `2,112 B/tok` instead of `4,096 B/tok` FP16, including checkpoint save/restore/copy) and runs online flash-decoding directly on hardware `xmx::dpas<8, 8, int, int, s8, s8>`.
   - **Dynamic session-OOV proposal head (`sub32k + session OOV <= 8,192`) with bucketed draft graphs (`0 / 256 / 1,024 / 8,192`):** Tracks out-of-vocabulary tokens encountered during prompt prefill, unpacks and quantizes their target output-head and `fold` rows on the GPU once after prefill, and buckets `oov_cap` in `GraphKey::Draft` so growing OOV sets reuse at most three graph captures.
   - **Fused residual + RMSNorm + Q8_0 quantization (`ns_ink2s_add_rms_xq`):** Replaces all six two-kernel `add + rms_xq` transitions per `ds_draft` step with single-launch `add_rms_xq`.

3. **SYCL command graphs (`ext_oneapi_graph`) and fused graphed target sampling (`dec.cpp`, `engine.rs`, `ffi.rs`).**
   - Captures and replays the 9 sliding-window segments (`GraphKey::Pass`), `ds_inject`, `ds_draft` (`GraphKey::Draft`), and the output head using device-resident `Ink2sPassParams` and `Ink2sDraftParams` buffers (`p0` at offset 0).
   - Records `ns_ink2s_softcap_argmax` and `ns_ink2s_softcap_sample_full_p0` directly inside `GraphKey::Pass`, eliminating the 11.2 MB/step `logits` DRAM roundtrip (`7.8 µs/tok`).

4. **Parallel MoE SFU, dense/controller, and launch fusions (`dec.cpp`, `mv.cpp`, `q8_head.hpp`, `engine.rs`).**
   - **Parallel 4/8-lane register-`csel` `tanh` + `exp` SFU (`mv.cpp`):** Replaces runtime lane-indexed `sycl::tanh`/`sycl::exp` loops in `moe_gu` (`T = 1`) and `moe_gu_union` (`m = 1, 2, 3+`, plus batched `load_raw<12>` in `m >= 3` and skipping the trailing `group_barrier` on the final tile), saving **26.0 µs/tok** on `experts gu silo` while keeping exact per-lane IEEE-754 results.
   - Losslessly repacks loop-controller matrices (`ctrl_down_q8_many`, `ctrl_apply_q8_many`) to Q8_0 with T-shared reads and pipelined 4-channel `char4` loads + 4-lane parallel `tanh`; fuses T-shared `router_union` (`float4` rank loop + subgroup `exclusive_scan_over_group`); fuses `moe_down_conv_res_union`; adds XMX DPAS `q8_head` for `>= 16K`-row Q8_0 heads; caches `token_embd` in host RAM for decode; batches Q8_0 `sink`, `k_dyn`, and `v_dyn` into `mm_many` via exact `ns_ink2s_xq_many` (with an `m == 1` fast path and 32-bit indexing); and trims barriers/SLM in `mix_norm`, `mix_exact`, `rms_xq_many`, and `post_qkv`.

### Opt-in symmetric INT8 full-attention target KV (`NS_INK2_KV=q8` / `--opt-kv q8`, patch 0063)

1. **64-byte aligned zero-copy INT8 ring layout (`engine.rs`, `checkpoint.rs`, `lib.rs`).**
   - For the 8 full-attention layer applications (`!ly.swa`; the 40 sliding-window layers stay exact FP16 over their tiny 1,024-token rings), `kc` and `vc` each store `int8_t[Hkv, cap, 128]` (`align256`) followed by `float[Hkv, sc_stride]` scales (`sc_stride = align16(cap) + 16`), cutting full-attention KV from `32.00` to `16.50 KiB/tok`.
   - Every head's `int8` base is 128B-aligned (`es::load_2d` safe for arbitrary `cap`), every head's `float` scale base is 64B-aligned with `>= 16` zero-padded tail floats (`es::block_load<float, 16>` safe), `save_ring`/`restore_ring`/`copy_session` preserve zero-copy `Arc<Vec<u8>>` band sharing, and `cache_fingerprint()` tags `q8-kv` vs `f16-kv`.
2. **Fused `post_qkv` quantization & XMX `s8 x s8` prompt attention (`dec.cpp`, `pattn.cpp`, `pattn_esimd.hpp`).**
   - Decode passes quantize clipped `K` and RMS-normed `V` inline inside `ns_ink2s_post_qkv` (`kv_i8 = 1`) with zero extra kernel launches; prompt passes use `ns_ink2s_target_kv_store_i8` and XMX `s8 x s8` `ink_prompt::run_i8` (`p0 + T >= 4096`) / `attn_flash<true>` (`< 4096`), speeding up 100K prefill by **+4.7 %** (`1,813 -> 1,899 tok/s`).
3. **Hybrid XMX-prefix + SLM-tail online FlashDecoding (`target_attn_i8.hpp`, `attn.cpp`, `engine.rs`).**
   - At `p0 + T <= 4096`, `launch_attn_rows_i8` (`P = 4, 2, 1`, plus two-slice `8 + (T - 8)` for `T = 9..16` within 64 KB SLM) dequantizes 64-key INT8 splits once per work-group into SLM with zero `es::gather`.
   - At `p0 + T > 4096`, `ink_target_attn_i8::run_prefix` streams `[0, prefix_kb * 64)` across `NP = 46` balanced splits (`368` work-groups, `xmx::dpas<8, 8, int, int, s8, s8>` `Q*K^T` + `f16` `P*V`, **zero `relb`, zero `es::gather`, zero causal/tail masks**, 100 % `#pragma unroll`), while `launch_attn_rows_i8` handles the `16..18` relative-bias/causal tail splits (`46 + tail_splits <= 64` for all `p0 >= 0, 1 <= T <= 16, E = 1,024`), feeding single-pass `ns_ink2s_merge_gate_xq` directly (`~8 µs` vs `~236 µs` `merge_exact`).

## Exactness and quality checks

- **Default `NS_INK2_KV=f16` (100 % bit-exact):**
  - **Full-vocabulary target scores (18 specs):** `p900:{1,2,3,5}`, `p4000:{1,2,3,5,8}`, `p9400:{1,2,5}`, `8k:1`, `40k:{2,3,4}`, and `100k:{1,7}` — **18 of 18 score files matched 0061/0059 byte for byte.**
  - **Fixed-seed decode text:** Every 3-pair alternating A/B run (`21` pairs across `short`, `short-1024`, `7.3K`, `39K`, and `100K`, plain and drafted) produced **byte-identical output tokens** (`cmp -s`).
  - **Served 14-turn conversation:** Replaying `bench/inkling_context.py --case chat` through the launcher reproduced **all 14 published `visible_sha256` and `reasoning_sha256` digests and `generated_tokens` counts** at **113.50 tok/s**.
- **Opt-in `NS_INK2_KV=q8` / `--opt-kv q8`:**
  - **100K teacher-forced `--score` (`100k:7`, 192 positions x 200,058 logits):** Mean KL divergence **`0.0416` nats/tok**, top-1 token agreement **`90.10 %` (`173/192`)**, logit RMSE **`0.703`**; `8k:1` median KLD **`0.0484` nats/tok** (`80.7 %` top-1); `8k:1` vs `8k:12` (`T = 1` vs `T = 12`) median KLD **`0.0043` nats/tok** (`94.8 %` top-1); 100K needle/source-reading answers verbatim-accurate across seeds 1 and 11.
  - **Served 14-turn conversation (`--opt-kv q8`):** All **14 of 14 turns** completed cleanly with multi-turn `q8-kv` prefix checkpoint save/restore at **115.69 tok/s** decode and **355.81 tok/s** fresh prefill.
- **Compiler hygiene.** Zero register spills across all decode, drafter, MoE, merge, graph, and INT8 attention kernels when compiled AOT for `-device bmg-g31`.

## Speed and context ceiling

Fixed-seed CLI and served decode (`nextsycl llm generate` / `serve-inkling-nextsycl.sh`, AD-Q4_K + 32K DSpark head, 6 proposals at confidence 0.65 unless noted, full-vocabulary sampling, seed 1 unless noted):

| workload | 0059 | 0061 | 0062 | **0063 (`f16` exact)** | **0063 (`KV=q8` opt-in)** |
|---|---:|---:|---:|---:|---:|
| **100K drafted** (`256` tok, `seed=1`, `temp=1`) | 60.2 | 64.4 | 96.30–96.69 | **96.36–96.44** (`172/211`, **+50 %**) | **108.26–110.27** (`161/205`, **+71 % vs 0061**) |
| **100K drafted** (`256` tok, `seed=11`, `temp=1`) | — | — | 96.81 | **96.8+** (`173/213`) | **113.48** (`170/211`, **+76 % vs 0061**) |
| **100K drafted greedy** (`256` tok, `temp=0`) | — | — | 103.08 | **103.1+** (`186/223`) | **128.09** (`192/229`, **+24 % vs 0062**) |
| **39K drafted** (`256` tok, `temp=1`) | 98.2 | 107.4 | 140.61–140.81 | **140.92–141.06** (`187/216`, **+31 %**) | **152.35–155.65** (`190/216`, **+45 % vs 0061**) |
| **7.3K drafted** (`256` tok, `temp=1`) | 98.0 | 110.6 | 133.06 | **133.66–133.79** (`165/207`, **+21 %**) | **136.91** (`166/212`, **+24 % vs 0061**) |
| **Short-1024 drafted** (`temp=1`, steady-state) | — | — | 146.12–146.20 | **147.14–147.21** (`657/848`) | **142.05–143.88** (`640/829`) |
| **Short-256 drafted** (`temp=1`, cold CLI) | 101.4 | 107.4 | 126.07 | **126.39–126.47** (`134/195`, **+18 %**) | **134.42** (`148/204`, **+25 % vs 0061**) |
| **Short-256 plain** (`drafts=0`, `temp=1`) | 76.7 | 82.6 | 88.37 | **90.54–90.65** (**+9.7 % vs 0061**) | **90.46** |
| **100K plain** (`drafts=0`, `128` tok) | 42.3 | 47.8 | 50.89 | **53.15–53.18** (**+11.2 % vs 0061**) | **64.50** (**+34.9 % vs 0061**, **+52.5 % vs 0059**) |
| **14-turn chat replay** (served via launcher) | 81.7 | 91.2 | 112.86 | **113.50** (14/14 hashes) | **115.69** (14/14 turns) |
| **100K prompt reading** | ~1,800 | ~1,800 | 1,813–1,824 | **1,812–1,814** | **1,891–1,899** (**+4.7 %**) |

### Empirical maximum context on one 32 GB Arc Pro B70

Measured by binary search in 1,024-token steps on the 32 GB card (`31.5 GiB` usable Level Zero VRAM, preserving the engine's strict `1.5 GiB` spare VRAM reserve so allocations never spill over PCIe):

- **Default `NS_INK2_KV=f16`:** **`375,808` tokens (`367K`)** with DSpark enabled (`34.0625 KiB/tok` combined KV); **`418,816` tokens (`409K`)** with `--no-mtp` (`32.00 KiB/tok`).
- **Opt-in `NS_INK2_KV=q8` (`--opt-kv q8`):** **`688,128` tokens (`672K`)** with DSpark enabled (`18.5625 KiB/tok` combined KV, **1.83x**); **`733,184` tokens (`716K`)** with `--no-mtp` (`16.50 KiB/tok`, **1.75x**).

### Where a 100K step spends its GPU time (`NS_INK2_PROFILE=gpu`, 192 minus 64 tokens)

- **Default `f16` (`~9,700 µs/tok` GPU):** `attention silo` `3,703 µs` + `merge silo` `500 µs` (`43 %`), `experts gu` `1,857 µs`, `experts down+conv` `1,078 µs`, `mv many` `711 µs`, `mv silo` `602 µs`, `ctrl down` `195 µs`, `mix norm` `188 µs`, `router union` `188 µs`, `draft attention` `148 µs`, `rms xq` `125 µs`, `ctrl apply` `117 µs`, `post qkv` `94 µs`, `conv res` `55 µs`, `draft fold samp` `39 µs`, `xq` `31 µs`, `softcap+sample full` `7.8 µs`.
- **Opt-in `KV=q8` (`7,172 µs/tok` GPU — 2.45x faster attention + merge):** `experts gu` `1,828 µs`, `attention silo` `1,609 µs` + `merge silo` `109 µs` (`24 %`), `experts down+conv` `1,055 µs`, `mv many` `711 µs`, `mv silo` `594 µs`, `mix norm` `195 µs`, `ctrl down` `195 µs`, `router union` `172 µs`, `draft attention` `148 µs`, `rms xq` `125 µs`, `ctrl apply` `117 µs`, `post qkv` `102 µs`, `conv res` `55 µs`, `draft fold samp` `39 µs`, `xq` `31 µs`.

## Tested and dropped

- **Monolithic XMX `target_attn_i8` with `es::gather` tail (`sp = 63`):** Ran `24,576` serial `es::gather` calls on 8 work-groups while 24 Xe-cores idled (`2,297 µs/tok` attention at 100K and `130.3 tok/s` at short-1024). Splitting into pure XMX `run_prefix` (`NP = 46`) + SLM `launch_attn_rows_i8` (`16..18` tail splits) cut 100K attention to `1,609 µs/tok` and lifted 39K decode from `132.8` to `155.7 tok/s`.
- **Non-unrolled 64-byte `V` `load_2d` + `prefetch_2d` in `run_prefix` / `draft_attn_esimd`:** Indexing `accum[half_idx * 4 + sub]` across a non-unrolled outer loop forced indirect GRF addressing on 64 accumulator registers (`1,609 -> 2,156 µs/tok`).
- **`R = 2 / 4` rows/subgroup on medium `2,048..8,600`-row dense projections (`mv.cpp`) and `CW = 64` in `conv_res`:** Reduced work-group occupancy across 32 Xe-cores (`mv_many` `+41.7 µs/tok`, `mv` `+26.0 µs/tok`, `conv_res` `+5..10 µs/tok`).
- **3-barrier full-64-key `V_SLM` kernel (`T = 6..7`), hoisted `K` across splits (`QP >= 16`), softer coupled `p^0.85` cost curve, monolithic 48-layer short CLI graphs, SLM-`nq` auxiliary reuse, and sparse `merge_long` skip:** All measured and reverted cleanly (see commit history and earlier notes).

## Build and identify it

Start from `nextsycl` commit `e3f1fd891b7fda6f478a326d913d28893a4e9811`, apply `0054` through `0059`, then `0061`, `0062`, and `0063`, and rebuild both `nextsycl` (`libnextsycl-llm.so` + `nextsycl`) and `silo/libnextsycl-inkling2.so` (`AOT=bmg-g31 ./build.sh all`). After `0062` the source tree hash is `26a3e0b21ebecfd2777238eacf4c324e443be291`; after `0063` the source tree hash is `f1df1eda5d8013ebbb4e372019b9d8e49d72e04a`. Pass `--opt-kv q8` (or `KV=q8` in `scripts/serve-inkling-nextsycl.sh`) to enable symmetric INT8 full-attention KV.
