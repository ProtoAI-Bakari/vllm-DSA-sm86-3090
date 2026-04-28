# 01 — Architecture

## DSA kernel chain (the things we have to port)

DeepSeek Sparse Attention is not one kernel — it's a fused pipeline. Each stage feeds the next, and each stage was written for Hopper.

```
input prompt
   │
   ▼
┌──────────────────────────┐
│  Lightning Indexer       │  ← top-k expert + token selection
│  - sparse_attn_indexer   │     (the FIRST kernel that asserted yesterday)
│  - compressor            │     (compress key tokens for sparse path)
└─────────┬────────────────┘
          │ top-k indices
          ▼
┌──────────────────────────┐
│  MLA-Sparse Attention    │  ← multi-head latent attention, sparse path
│  - mla_sparse            │     (FlashMLA fwd/bwd)
│  - flashmla_sparse_torch │     (decode path with sparse KV)
└─────────┬────────────────┘
          │ attended values
          ▼
┌──────────────────────────┐
│  Sliding Window Attn     │  ← SWA cache + dispatch
│  - swa kernels           │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  MoE Expert Dispatch     │  ← 256 experts × 8 active per token
│  - MoEPrepareAndFinalize │     under DSA's indexer-driven routing
│  - DEEP_GEMM (Hopper)    │     (the [128,128] FP8 block GEMM wall)
└─────────┬────────────────┘
          │
          ▼
       output
```

## Hopper → Ampere substitution table

Each kernel in the chain uses some combination of Hopper intrinsics. Here's the substitution plan:

| Hopper intrinsic | Used in | Ampere replacement | Risk |
|---|---|---|---|
| `cp.async.bulk.tensor.*` (TMA loads) | Lightning Indexer, MLA-Sparse, Compressor | `cp.async.cg` (sm_80 cached-global async copy) per-tile loop | LOW — well-understood pattern |
| `wgmma.mma_async.*` (warpgroup MMA) | MLA-Sparse, MoE GEMM | `mma.sync.aligned.m16n8k16` (warp-level, sm_80) | MED — accumulator dtype matters; bf16 acc safer than fp16 |
| `cp.async.bulk` | various bulk loads | `cp.async` loop with manual sync | LOW — slower but functional |
| `griddepcontrol.launch_dependents` (PDL) | kernel chaining | `cudaStreamSynchronize` host roundtrip | LOW — costs latency, no correctness risk |
| FP8 [128,128] block-scaled GEMM (DEEP_GEMM Hopper) | MoE expert path | MARLIN W8A8 (already exists for sm_86) with appropriate scale fold-in | MED — numerics match needs validation |
| TMA descriptors (`CUtensorMap`) | data layout for TMA | manual stride/offset arrays | LOW |

## Why these substitutions work

**Algorithm is hardware-agnostic.** Sparse attention's online-softmax + indexer top-k is the same math regardless of how data moves into shared memory. The bottleneck on Hopper is `(memory bandwidth × compute throughput)` — TMA + WGMMA win on Hopper because the chip has them, not because the algorithm requires them. On Ampere, the same algorithm runs ~2× slower with `cp.async.cg` + WMMA, but it runs **correctly**.

**Numerics preservation.** The substitutions don't change any *numerical* operation — only *data movement* + *MMA shape*. The accumulator dtype must match Hopper's choice (typically bf16 acc, fp16 inputs) or the perplexity will drift. The CC6 gate catches drift.

## Why this isn't a model-quantization project

A natural question: "why not just quantize DSV4 to AWQ-INT4 and use the existing Marlin path?" Answer: even an AWQ-quantized DSV4 still loads through the `DeepseekV4FlashMLASparseBackend` model class, which routes through `flashmla_sparse_fwd` — same Hopper-locked kernels. **The architecture is the wall, not the precision.** This is why CC2's lane is "rebuild the .so" not "find a smaller quant."

## The 5-tier test pyramid

| Tier | Hardware | What it proves | Wall per run |
|---|---|---|---|
| L1 unit | 1× 3090 | kernel produces numerically-correct tensor in isolation | seconds |
| L2 forward slice | 1 node (2× 3090) TP=2 | kernel fits in vLLM's call chain w/ surrounding stack | 1-2 min |
| L3 small-model load | 1-2 nodes | full vLLM request path on tiny test fixture | 5 min |
| L4 full integration | All 8 nodes, drained+nuked | model loads, generates real tokens, hits next blocker if any | 10-15 min |
| L5 perf bench | All 8 nodes | conc 1/2/4/8/16 sweep, NCCL fabric profile | 30-60 min |

L4 is the **only level that finds the next deeper sm_90 wall.** Each accepted kernel merge → CC6 fires L4 → either passes (move to next kernel) or reveals next assertion (CC1 logs it, CC3/CC4 picks it up).

## Topology decision: TP=2 × EP=8

Both target models (DSV4 + GLM-5.1) have 256 experts × 8 active per token → **EP=8 is exact fit** (one expert-shard per node-pair). TP=2 within each Mac Pro node (2× 3090 over NVLink) covers the dense layers. EP=8 across the 8-node 100 GbE cluster covers the MoE.

This beats TP=16-over-100GbE by 5-10× on decode latency because TP=16 every-step all-reduce is fabric-bound. EP=8 only synchronizes at expert-dispatch boundaries.

PP=2 + TP=8 = boot fallback only (the topology that survived yesterday's 17-patch cascade because the MoE-DSA dispatch lane wasn't yet patched). Once the rebuild ships, TP=2 × EP=8 is canonical.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
