# Vulkan kernel work for Nemotron 3.5 Bolt on the B70: 42 -> 51 tok/s plain decode

After the settings-level work (`05-bolt-speedups.md`) the per-op profile said the remaining time was in three
kernels. This page is the kernel work: what was measured, what was changed in llama.cpp's Vulkan backend
(`patches/0002`..`0006`, against tag b11669), and what did not work. All numbers: Arc Pro B70, Mesa ANV.

| stage | llama-bench tg64 (tok/s) | pp512 |
|---|---|---|
| official b11669, host Mesa 26.1.6 | 41.9 | 777 |
| + IQ2_XXS integer-dot mat-vec | 45.5 | – |
| + dedicated NVFP4 mat-vec shader | 48.4 | 784 |
| + router (top-k MoE) fix, Q8_0 policy | 50.8 | 785 |
| + fused relu^2 (`patches/0007`) | **51.0** | 789 |
| same build (without 0007) in a Mesa 26.2.4 container | 53.4 | 678 |

End to end, serving with the DFlash drafter at draft length 3 and a 2048 thinking budget, the 39-task suite went from a
50.2 tok/s median (33/39, 25 min) to **64.2 tok/s (34/39, 26 min for 27 % more generated tokens)**; the 4-task speed
subset runs at 60-78 tok/s (was 45-58).

Quality is unchanged: batch-1 perplexity (the path that uses these kernels) 16.79 -> 16.84 on 3x2048 tokens, inside the
±0.93 error; `test-backend-ops` passes every MUL_MAT / MUL_MAT_ID / TOPK_MOE case against the CPU backend.

## Method: a fast loop and the right diagnostics
- `scripts/vk-dev.sh build|check|perf|bench`: incremental build of the patched tree in a container that has the shader
  toolchain, correctness against CPU, `test-backend-ops perf` on the model's exact decode shapes
  (`patches/0006` adds them under `BOLT_SHAPES=1`), and the real model through llama-bench.
- **Defeat the L2 in microbenchmarks.** Small weights (< ~25 MB) sit in the GPU's L2 between runs: a 4096x4096 Q8_0
  mat-vec showed an impossible 1.3 TB/s. Batch the weights (`bs=[3..20,1]`) so a run touches > 100 MB and divide.
- **Read the Intel shader compiler stats** (`INTEL_DEBUG=cs`, in a container is fine): instruction count, "sends"
  (memory messages, including shared-memory loads), spills, SIMD width. This is what turned guesswork into a plan:
  - NVFP4 float mat-vec: 8265 instructions / 225 sends vs Q4_0's 3840 / 225 for the same work, same memory ops,
    2.15x the instructions. Instruction-issue bound, not bandwidth bound.
  - The integer-dot NVFP4 variant built like MXFP4's: 920 sends, because every shared-memory table lookup is a send.
- **Find the memory floor with a fake decode**: replacing the NVFP4 decode with a 4-byte-to-float cast gave 235 µs
  for the batched ssm_in shape, exactly Q4_0's time (546 GB/s). Everything above that is arithmetic.
- The GPU's practical ceiling is ~600 GB/s (the Q8_0 lm-head mat-vec reaches 594) and roughly 600 G SIMD16
  instructions per second; a mat-vec at ~11 instructions per weight is already issue-bound at ~300 GB/s equivalent.

## IQ2_XXS: integer dot product (`patches/0004`, `0002`)
The IQ2_XXS codebook dequant (grid lookup, 7+1 sign bits, FMA per weight) is ALU-bound in fp32. The fix mirrors
CUDA's `vec_dot_iq2_xxs_q8_1`: quantize the activation vector to Q8_1 once, then per 32-weight sub-block do 4 grid
lookups, expand the sign bits into byte masks with one multiply (`(s & 0xF) * 0x00204081 & 0x01010101`), negate
bytes with `(x ^ m) + (m & 0x01010101)` (grid magnitudes are never zero, so no carry crosses a byte), and 8
`dotPacked4x8EXT`. Wired into the existing `mul_mat_vecq.comp` framework as `K_PER_ITER 32`, plus the host
pipelines/switches and an Intel policy that always uses it for IQ2_XXS.

| shape (per matrix, L2 defeated) | fp32 path | int8 path |
|---|---|---|
| shared expert 5376x4096 | 24.5 µs (232 GB/s) | 15.7 µs (361 GB/s) |
| expert MUL_MAT_ID 240x(1024x2048), top-9 | 25.2 µs | 17.3 µs |
| expert MUL_MAT_ID 240x(2048x1024), top-9 | 30.0 µs | 16.5 µs |

In the model the gain is smaller for the first consumer of an activation because the Q8_1 quantize is an extra
dispatch with barriers (~5-13 µs on this driver); the shared-expert up-projection is slightly slower (32 -> 36 µs)
while the down-projection (38 -> 31) and the experts (43 -> 31, 40 -> 30) win. Rows per workgroup: 4 (8: -2 %, 16: -15 %).

## NVFP4: a dedicated shader (`patches/0003`)
NVFP4 went through the generic `mul_mat_vec.comp` with per-pair `dequantize()` calls (8-bit loads, two
shared-memory lookups per pair, redundant scale decodes). `mul_mat_vec_nvfp4.comp` gives each thread one 16-element
sub-block per step: two 32-bit loads of nibbles, the scale byte from the block's 32-bit scale word, UE4M3 decoded
arithmetically, the e2m1 nibbles decoded four at a time with SIMD-within-a-register integer ops, then 16 FMAs.

| shape (per matrix) | before | after |
|---|---|---|
| ssm_in 18560x4096 | 138 µs (310 GB/s) | 100 µs (435 GB/s) |
| ssm_out 4096x8192 | 67 µs (283 GB/s) | 45 µs (430 GB/s) |

The floor for this access pattern is 78 / 35 µs; the kernel is still instruction-bound (about 8.5 instructions per
weight). Things that did **not** close the gap: arithmetic instead of table decode inside the generic shader (same
8265 instructions, same speed); removing the shared-table init barrier; a 2-block unroll with hoisted loads (same);
subgroup 32 instead of 16; larger workgroups; a 16-entry table held in subgroup lanes via `subgroupShuffle` (2.7x
*slower*: Intel compiles per-lane shuffles into something expensive); the integer-dot path with the MXFP4-style
shared-memory table (slower, 920 sends); the integer-dot path with SIMD-within-register decode (kernel 15 % faster
than the float shader, but the extra Q8_1 quantize dispatch per Mamba projection costs more in the model: 49.5 vs
50.8 tok/s; kept behind `GGML_VK_INTEL_NVFP4_MMVQ=1`).

## Fused top-k router (`patches/0005`)
`topk_moe.comp` loops over `n_experts`, which is a push constant for expert counts that are not a power of two
(Bolt has 240..496). A runtime trip count cannot be unrolled, so the per-thread `probs[]`/`selection_probs[]`
arrays were indexed dynamically and ended up in scratch memory: 45 µs per call for a 1-row routing decision, 1.8 ms
per token. Iterating over the compile-time `experts_per_thread` with an `expert < n_experts` guard (and static-index
loops for the two single-slot writes) brings it to 26 µs; 416/416 `TOPK_MOE` tests pass. Still mostly dispatch
latency; the kernel itself is microseconds.

## Fused relu^2 (`patches/0007`, detection in `0002`)
Nemotron-H's FFNs use relu^2, which llama.cpp builds as UNARY(RELU) followed by SQR: two dispatches per FFN. A
`relu_sqr` unary variant plus a two-node fusion (same pattern as the existing UNARY+MUL fusion) folds them.
It fuses the shared-expert path (one per layer); the routed-expert relu/sqr pair did not qualify under the generic
fusion checks, so about half the pairs remain. Gain: 50.8 -> 51.0 tok/s; 6/6 `RELU_SQR` tests pass.

## Policy and knobs (`patches/0002`)
- Intel Xe2: Q8_0 mat-vec at n=1 stays on the float path (already near peak; the quantize dispatch cost more than it
  saved: 21 -> 17 µs on the 1024x4096 latent projection).
- Experiment knobs, all default-off: `GGML_VK_INTEL_SG` (subgroup size 16/32), `GGML_VK_INTEL_DMMV_WG` (0 small /
  1 large workgroups), `GGML_VK_INTEL_RM_STDQ|RM_KQ|RM_ID|RM_IQ2|RM_NVFP4` (rows per workgroup),
  `GGML_VK_INTEL_NVFP4_MMVQ`, `GGML_VK_INTEL_Q8_MMVQ`. None of the geometry knobs beat the defaults on the B70.

## Where the remaining time is (host profile, 19.7 ms per token at 50.8 tok/s)
| share | what | note |
|---|---|---|
| 31 % | NVFP4 ssm_in / ssm_out | instruction-bound at 435 GB/s; floor would be another 2.1 ms |
| 25 % | IQ2_XXS shared expert + experts | int8 path; includes the Q8_1 quantize dispatches |
| 22 % | ~1000 small dispatches (ADD, MUL, CPY, CONT, RELU, SQR, norms, scan, concat) at 3-5 µs each | dispatch + barrier bound; needs graph-level fusion |
| 12 % | Q8_0 projections and the lm head | near peak |
| 5 % | router | dispatch latency |
| 1 % | attention | 8 layers only |

With the DFlash drafter the verify step runs 4 tokens per pass, which amortizes the dispatch-bound share; see the
end-to-end numbers at the bottom of `05-bolt-speedups.md`.

## Building it
`scripts/build-vulkan-host.sh` builds the patched tree in a Debian-trixie container (same glibc as the host) so the
binaries run on the host with its Mesa; `scripts/build-vulkan-sid-tests.sh` + `vk-dev.sh` is the development loop
in a Debian-sid container (newer Mesa and glslc). Apply `patches/0002`..`0006` to llama.cpp b11669 first
(`0006` only adds the perf shapes). The Mesa 26.2.4 container decodes 5 % faster but prefills 14 % slower than
host Mesa 26.1.6 with these kernels.
