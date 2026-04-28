# 03 — Build Instructions

> **Status: PENDING** — populated by CC2 (build infrastructure lane).

When complete: full reproducible build instructions for the sm_86 wheel against pinned torch / cuda / nvcc / xformers / Triton stack, including:

- Required toolchain versions
- `cmake -DTORCH_CUDA_ARCH_LIST="8.6"` flags
- ccache + sccache setup (for fast iteration)
- ABI compatibility checks
- Smoke-test command verifying the rebuilt vLLM still serves a non-DSA baseline model before applying the new kernels

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
