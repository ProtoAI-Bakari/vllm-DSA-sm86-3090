# 00 — The Problem

## Plain language

vLLM 0.19.x ran fine on RTX 3090 because every kernel had both a Hopper code path and an Ampere code path. When DeepSeek released DSV4-Flash and Z.ai released GLM-5.1 (both using DSA — DeepSeek Sparse Attention), the new attention kernels landed in vLLM 0.20 (PR #40760) but were written **Hopper-only**. They depend on hardware features the RTX 3090 doesn't have:

| Hopper feature | What it does | Ampere status |
|---|---|---|
| **TMA** (Tensor Memory Accelerator) | Asynchronous bulk tensor copy unit | Doesn't exist on sm_86 |
| **WGMMA** (warpgroup MMA) | Whole-warpgroup matrix multiply-accumulate | Doesn't exist on sm_86 |
| **cp.async.bulk** | Bulk async copies | Doesn't exist on sm_86 |
| **PDL** (Programmatic Dependent Launch) | Kernel chaining without host roundtrip | Doesn't exist on sm_86 |

The crash is not in Python code — it's in the compiled `.so` extension `vllm/_C.abi3.so`. So Python patches can hold up the request boundary (server starts, `/v1/models` returns 200) but the actual CUDA assertion fires inside the compiled kernel during forward pass.

## What we tried (and why it didn't work)

A 17-patch Python cascade got the server to boot and `/v1/models` 200 OK. After that, every cold POST to `/v1/completions` produced a deeper assertion. We patched 13 of those, but the final wall is `gpu_model_runner.py:4062` — a bare assertion inside a compiled extension we can't reach with Python.

We tried every alternate kernel backend in vLLM:

| Backend | Result on sm_86 |
|---|---|
| MARLIN | SIGFPE in `marlin_moe_wna16::marlin_mm()` for DSV4's [1,32] block-scaled FP8 |
| TRITON | Rejects `QuantKey(f8e4m3fn, GroupShape(row=128, col=128))` |
| TRITON_UNFUSED | Not in valid-options list |
| DEEP_GEMM | Hopper-only (TMA / PDL) |
| CUTLASS | Hardcoded `allow_vllm_cutlass=False` |
| FLASHINFER_TRTLLM / FLASHINFER_CUTLASS | sm_90+ device check |
| AITER | ROCm-only |

We also tried alternate quants (AWQ-INT4, GPTQ-Int4) — they don't exist for DSV4 yet, and even if they did, the `DeepseekV4FlashMLASparseBackend` model class still routes through the same Hopper-locked kernels.

## The wall, in one sentence

**To run DSV4 or GLM-5.1 on RTX 3090, the compiled `.so` has to be rebuilt with sm_86 substitution paths for every Hopper-only intrinsic in the DSA kernel chain.**

That's what this repo does.

## Why this matters

DSV4 and GLM-5.1 are two of the strongest publicly available SOTA MoE models (256 experts × 8 active per token each). They're designed for inference, not just training. Their DSA architecture is *substantially* more efficient than vanilla MLA at long contexts.

Locking them to Hopper means:
- ~$25K/GPU minimum entry cost (H100)
- Anyone with a multi-3090 or A100 fleet is excluded
- Open-source SOTA inference becomes a Hopper-only club

This repo is the path back to the open. If we ship sm_86 substitutions cleanly, future DSA-class models inherit the fix — the value floor of every Ampere fleet on Earth jumps overnight.

## Expected outcome

| Model | Topology | Expected (rebuilt sm_86) | Tier |
|---|---|---|---|
| DSV4-Flash-FP8 | PP=2 × TP=8 → TP=2 × EP=8 | conc=1: 10-20 t/s; aggregate: 30-80 t/s (BF16 emulation cost) | conc=1 below-Bronze; aggregate below-Bronze |
| GLM-5.1-IQ2XXS | TP=2 × EP=8 | conc=1: 25-35 t/s; aggregate: 100-300 t/s | conc=1 Bronze-Silver; aggregate Bronze-Silver |
| DSV4 stretch (post-optimization) | TP=2 × EP=8 | conc=1: 25-35 t/s; aggregate: 100-300 t/s | reachable in optimization loop |

For perspective: GLM-5.1 today on llama.cpp PP backend ceiling = 35.95 t/s aggregate. We're targeting **~10× that** through proper TP×EP routing on rebuilt vLLM.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
