# Strata on the B70 (Qwen3.8 Flash-Next and Swift 1.5)

Strata is the engine used for the two Qwen3.8-era MoE models. Notes specific to running it on this card:

- **Build**: the SYCL engine is built ahead-of-time for the Battlemage target (`bmg-g31`) inside the same oneAPI
  image used for llama.cpp, following the project's Intel build notes. The 0.1.44 build with the project's SYCL
  expert profile raised prompt reading on a 15.5k-token prompt from 250-525 tok/s to ~1,180 tok/s; decode 47-48 tok/s
  at 0-15k depth (was 43-46 at 66k on the previous version).
- **262k context**: `--kv-resident 32768` streams the KV cache from ~3 GiB of pinned RAM with `--vram-reserve-mib
  2048` and `--prefill 4096`. Measured 859 tok/s prompt reading at 25k and 848 at 74k, decode 52 short / 47 after a
  74k prompt, 2.2 GB VRAM spare. The project's docs warn that 262k *without* kv-resident livelocked a B70; keep it.
- **Thinking budget** is per request (`reasoning_budget_tokens`), which is what made the budget matrix cheap to run
  on these two models (no server restart). Both configs should carry 2048 as default, see `04-thinking-budget.md`.
- **Sampling**: Strata decodes greedily when the request has no sampling block, which loops. Put the Qwen
  recommendation (temp 1.0, top_p 0.95, top_k 20, presence 1.5) in the config and send it from the client.
- **Control vector**: Strata ships a per-layer control vector (`--cvec-mode project`, layers 4-44) that removes one
  direction at runtime. The `proj` benchmark scenario measured it: 36/39 vs 34 with it off, within noise, i.e. no
  measurable cost on these tasks (the vector's own metadata reports mean KL 0.19 vs the stock model). Requests can
  opt out per call with `"experimental_speed_projection": false`, which the bench does for the "stock" rows.
