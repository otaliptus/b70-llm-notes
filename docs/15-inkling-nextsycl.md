# Native Inkling on one Arc Pro B70

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
