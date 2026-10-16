# Nemotron 3.5 Bolt 74B-A9B on the Arc Pro B70: 29 -> 50 tok/s

Bolt is a hybrid: 88 layers, Mamba-2 layers whose projections (`ssm_in` 4096x18560, `ssm_out` 8192x4096) ship as
**NVFP4** with separate f32 scale tensors, MoE layers with 240-496 experts top-9 plus a shared expert in **IQ2_XXS**,
latent/attention projections in Q8_0, 262k context. llama.cpp support landed in b11669 (first release with the
architecture) together with NVIDIA's **DFlash** speculative drafter, a 1 GB companion GGUF.

| step | decode tok/s | 39-task score | wall |
|---|---|---|---|
| official SYCL build in the oneAPI container | 29.7 | 34 | 42 min |
| official Vulkan build on the host | 41.9 | – | – |
| **Vulkan + DFlash drafter, `--spec-draft-n-max 3`** | **50 (median over 39 tasks)** | 33 | 25 min |
| same, Mamba projections requantized NVFP4 -> Q4_0 (opt-in) | 56 | 33 | 23 min |

Score differences of 1 are noise (see `02-benchmark.md`). Wall time dropped 1.7x at the same quality.

## 1. SYCL: flat
`llama-bench` with `SYCL_UR_USE_LEVEL_ZERO_V2=0`, `GGML_SYCL_FA_ONEDNN=0`, both, and no-mmap: 29.5-30.2 tok/s,
pp512 605-624. Nothing moves. (`scripts/sweep-sycl-env.sh`)

DFlash on SYCL crashes immediately:
```
ggml/src/ggml-sycl/outprod.cpp:12: GGML_ASSERT(ggml_is_contiguous(src0)) failed
```
The drafter hands `out_prod` a transposed f32 view. `patches/0001-ggml-sycl-out_prod-accept-strided-f32-src0.patch`
accepts a strided/transposed f32 `src0` by passing the right transpose flag and leading dimension to oneMKL GEMM
(the Q1_0 path still needs a contiguous source). With the patch the drafter runs, but decode is *slower* than no
drafter (14-27 tok/s vs 29): the SYCL backend's cost per small op is high enough that the extra draft/verify graphs
cost more than they save. So on SYCL, run Bolt without a drafter.

## 2. Vulkan: +45 % for free
The same model on the official Vulkan build on host Mesa 26.1.6: 41.9 tok/s, pp512 777. The `GGML_VK_PERF_LOGGER`
profile (`scripts/perf-logger-vulkan.sh`, ~25.6 ms per token) explains why the rest is hard:

| share of decode time | what |
|---|---|
| 34 % | `MUL_MAT_VEC nvfp4` for `ssm_in` (141 µs) and `ssm_out` (78 µs), 1,320 calls each. The B70 reports `fp4: 0`: no hardware FP4 path, the shader dequantizes pairs in software and reaches ~1.0 TFLOP/s = about half the card's memory bandwidth equivalent |
| 23 % | IQ2_XXS mat-vecs: shared expert (32 µs x 1,320), expert `MUL_MAT_ID_VEC` (28-44 µs, ~1.1-1.2 TFLOP/s) |
| 22 % | ~760 tiny ops per token: ADD/MUL/CONT/CPY/RELU/SQR/GLU/RMS_NORM at 3-9 µs each, i.e. launch-bound |
| 7 % | fused top-k router `TOPK_MOE_SIGMOID_NORM_BIAS` (46 µs x 1,320) |
| rest | Q8_0 projections, SSM scan/conv, flash attention (28 µs), lm head |

Everything tried as a *setting* was flat (41.7-42.1 tok/s): fa on/off, q8_0 KV, ubatch 1024/2048, 8 threads, every
`GGML_VK_DISABLE_*`/`FORCE_*` toggle. Only prefill reacts (coopmat off 777 -> 156, f16 off -> 411).
(`scripts/sweep-vulkan-env.sh`)

Mesa 26.2.4 in a sid container: 42.4 tok/s (same), pp512 669 (slower). Not worth a driver upgrade.

The Intel-specific rows-per-workgroup constants in `ggml-vulkan.cpp` (`rm_stdq`, `rm_kq`, MUL_MAT_ID rows) were
exposed as env knobs (`patches/0002-...`) and swept in a from-source build: the defaults are already best
(43.8 tok/s; `kq=4` falls to 39.4). (`scripts/sweep-vulkan-rm.sh`)

## 3. DFlash on Vulkan: +20 %, but shorten the draft
Upstream's default draft length (`--spec-draft-n-max 7`) gave 27-47 tok/s on the 4-task speed subset, *below* the
42 of no drafter on long thinking answers. Sweep (`scripts/sweep-dflash-draft-n.sh`):

| n_max | tok/s on the 4 tasks | acceptance | mean accepted length |
|---|---|---|---|
| 2 | 41-52 | – | – |
| **3** | **45-58** | 0.63-0.91 | 2.9-3.7 |
| 4 | 46-61 | 0.59-0.88 | 3.4-4.5 |
| 5 | 41-62 | 0.48-0.84 | 3.4-5.2 |
| 7 | 27-47 | – | – |

Acceptance falls off fast with length on reasoning text (code and tables accept 85-90 %, free-form thinking 60 %),
and every rejected draft token is a wasted verify step on a decode that is already bandwidth-bound. Length 3 is the
best trade; the full 39-task run at n_max 3 gave a 50.2 tok/s median and the same score. `serve-bolt-vulkan.sh`
defaults to it. The n-gram drafter (`ngram-mod`) gave nothing on this model (28 vs 29 on SYCL).

## 4. Requantizing the Mamba projections (opt-in)
Since the NVFP4 mat-vec is the biggest single cost and has no hardware path, `scripts/requant-ssm.py` rewrites the
80 `ssm_in`/`ssm_out` tensors to a type with a fast Vulkan kernel and copies everything else byte for byte
(80 tensors, ~75 s, file size unchanged for Q4_0, +0.5 GB for Q5_0):

| variant | pp512 | tg64 | with DFlash n=3 (39 tasks) | perplexity (24x2048 chunks of technical markdown) |
|---|---|---|---|---|
| original NVFP4 | 780 | 41.9 | 50.2 median, 33/39 | 8.90 ± 0.15 |
| ssm Q4_0 | 966 | 47.95 | 55.7 median, 33/39 | 10.09 ± 0.18 (+13 %) |
| ssm Q5_0 | 954 | 46.3 | – | 9.99 ± 0.18 (+12 %) |

Q5_0 is no better than Q4_0 in quality, which says the loss is not the bit width: NVFP4 uses 16-element blocks
with e4m3 scales, and re-blocking into 32-element Q4_0/Q5_0 blocks throws away half the scale resolution. The
bench score does not see the difference, perplexity does. The original stays the default; the Q4_0 file is kept as a
speed option. A Q6_K/Q8_0 requant would need `llama-quantize` for the type and is predicted slower (Q8_0 reads 2x
the bytes of NVFP4).

## Kernel work (later the same day)
`09-vulkan-kernels.md`: an IQ2_XXS integer-dot path, a dedicated NVFP4 shader, a router fix and a fused relu^2 took
plain decode to 51 tok/s and the DFlash configuration to a 64 tok/s median over the 39 tasks (34/39), with unchanged
perplexity. The list below is what was still open before that work.

## What was left at this point
Kernel work in llama.cpp's Vulkan backend, not settings: a better NVFP4 mat-vec (or a Q4_0-style dequant with the
16-element scales kept), fusing the ~760 per-token elementwise ops in the Mamba blocks, and the IQ2_XXS mat-vec /
mat_id, which runs at roughly a quarter of peak bandwidth. Intel's own upstream Vulkan PRs for mat-vec geometry
were stale drafts at the time; the one mergeable PR touched flash-attention prefill only.
