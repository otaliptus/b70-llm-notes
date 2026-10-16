# Environment

Everything in this repo was measured on one workstation. Numbers elsewhere are specific to it.

| part | value |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB (Battlemage; Vulkan reports `BMG G31`, `KHR_coopmat`, int dot 1, **fp4: 0**, warp 32, 48 KB shared) |
| CPU / RAM | 16-core Zen 4 mobile part, 64 GB |
| OS | Debian 13 (trixie), kernel 7.2, Mesa 26.1.6 from trixie-backports, `libvulkan1` 1.4.309, `libze1` 1.20 |
| Firmware GPU power cap | 230 W max (requests above are clamped); resets at boot |
| Network | ~11 MB/s download ceiling, so model downloads are the slow part of every experiment |

## Two ways to run llama.cpp on this GPU

**SYCL (Intel oneAPI).** The official `llama-bXXXXX-bin-ubuntu-sycl-fp16-x64` release archives run unmodified inside any
container image that has the oneAPI runtime (`/opt/intel/oneapi/setvars.sh`) when the container gets `/dev/dri`.
No host oneAPI install. `scripts/serve-sycl-container.sh` does this with read-only mounts for the runtime, the
models and the API key, and a persistent JIT cache (first start of a new model/driver pair compiles kernels for
a minute; later starts are seconds). Container image used here: a oneAPI 2026.1 base image that also carries icpx,
cmake and ninja, so the same image builds patched llama.cpp trees (`scripts/build-sycl-patched.sh`). It has the oneDNN
runtime libraries but no headers, hence `-DGGML_SYCL_DNN=OFF` in the build.

**Vulkan (Mesa ANV).** The official `...-bin-ubuntu-vulkan-x64` archive runs directly on the host. No container.
Which backend is faster depends on the model (see `06-orca-27b-sycl.md` vs `05-bolt-speedups.md`): SYCL won
2.4x on prefill and 10-30 % on decode for Orca 27B, a dense Q4_K_S model, Vulkan won 45 % on decode for a hybrid
Mamba-2 / MoE model whose tensors are IQ2_XXS and NVFP4.

**Newer Mesa in a container.** A Debian sid container (`scripts/mesa-sid/`) gives Mesa 26.2.x and a current glslc
without touching the host. A llama.cpp built inside it links a newer glibc than trixie has, so the binary only runs in
the sid runtime container; the official release archive runs in both.

## Verified before use
Release archives were checked against the SHA-256 reported by the GitHub release API. Model files were checked
against the Hugging Face file digests. When a repo was re-uploaded (the Bolt GGUF repo was rewritten by its uploader
with different file names within a day of release), the digest check is what caught it.

## Gotchas that cost time
- An image whose `ENTRYPOINT` is `bash -lc` swallows the command of a plain `docker run image cmd`. Always pass
  `--entrypoint bash ... -lc '...'`.
- `pkill -f PATTERN` matches the shell that runs it when the pattern appears in that shell's own command line, and
  kills it (exit 144). Kill by PID, or `pgrep -f '[l]lama-server'`.
- A stopped systemd unit does not stop a container started by a non-exec'd script unless the script traps the
  signal; the serve script does, and an `ExecStop=docker stop NAME` drop-in is a belt-and-braces addition.
- Thinking budgets are a *server* flag in llama.cpp (`--reasoning-budget N --reasoning-budget-message "..."`), so a
  budget sweep restarts the server per budget. Strata takes it per request (`reasoning_budget_tokens`).
- llama.cpp's `--reasoning-budget` with `--reasoning-preserve` keeps the thinking in the stream so the client can
  count it; without the message flag the model is cut off mid-thought and answers badly.
- Only one model fits on a 32 GB card at these sizes. Every script stops the previous server first.
