# Sparse Attention Indexer — Hopper Kernel Anatomy
**Story CC3-#2** | DSA-v4 + GLM-5.1 token-gen unblock trace
// --ProtoAI-Bakari--

## 0. TL;DR — sm_86 port surface

| # | Symbol | Source | Hopper-bound? | sm_86 plan |
|---|---|---|---|---|
| 1 | `torch.ops.vllm.sparse_attn_indexer` | `sparse_attn_indexer.py:84-371` | NO | unchanged |
| 2 | `fp8_fp4_mqa_logits` | DeepGEMM Hopper | YES | MARLIN-INT4 sm_86 / Triton |
| 3 | `fp8_fp4_paged_mqa_logits` | DeepGEMM Hopper | YES | MARLIN paged sm_86 / Triton paged-MQA |
| 4 | `flash_mla_with_kvcache` | FlashMLA `csrc/sm90/decode/sparse_fp8/` | YES | new `csrc/sm86/decode/sparse_fp8/` cp.async.cg + mma.sync.aligned.m16n8k16 |
| 5 | `flash_mla_sparse_fwd` | FlashMLA `csrc/sm90/prefill/sparse/` | YES | new `csrc/sm86/prefill/sparse/` same substitutions |
| 6 | `ops.indexer_k_quant_and_cache` | `vllm/_C` | NO (verify) | likely unchanged |
| 7 | `ops.cp_gather_indexer_k_quant_cache` | `vllm/_C` | NO (verify) | likely unchanged |
| 8 | `torch.ops._C.persistent_topk` | `csrc/persistent_topk.cuh` | NO (sm_80+ already, uses `ld.global.cg`) | unchanged ✅ |
| 9 | `top_k_per_row_prefill/decode` | `csrc/topk.cu` | NO (radix HW-agnostic) | unchanged ✅ |

**Net port surface: items 2-5.** Items 6-9 stay. Item 1 is Python plumbing.
**Rationale → unblocks DSA-v4 + GLM-5.1 token gen:** items 2-5 are the four compiled callsites that 500 today on `/v1/completions` past the patch-#16 stub. Producing sm_86 binaries for them = first coherent token. Items 8-9 already work; reusing them halves the rebuild scope.

## 1. Entry-point chain

```
DeepseekV4MLAAttention.forward [vllm/model_executor/models/deepseek_v2.py]
  └── SparseAttnIndexer.forward_cuda [sparse_attn_indexer.py:463]
        └── torch.ops.vllm.sparse_attn_indexer [registered :395]
              └── sparse_attn_indexer() [:84]
                    ├── ops.indexer_k_quant_and_cache         [vllm/_C]
                    ├── ops.cp_gather_indexer_k_quant_cache   [vllm/_C]
                    ├── fp8_fp4_mqa_logits                    [DeepGEMM Hopper] ⚠️
                    ├── fp8_fp4_paged_mqa_logits              [DeepGEMM Hopper] ⚠️
                    ├── torch.ops._C.persistent_topk          [Ampere ✅]
                    └── torch.ops._C.top_k_per_row_*          [Ampere ✅]

FlashMLASparseImpl.forward_mqa [flashmla_sparse.py:1014]
  ├── _bf16_flash_mla_kernel [:983] → flash_mla_sparse_fwd ⚠️
  └── _fp8_flash_mla_kernel  [:944] → flash_mla_with_kvcache ⚠️
```

## 2. Hopper hard-gate verbatim

`vllm/v1/attention/backends/mla/flashmla_sparse.py:131-133`:
```python
@classmethod
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    return capability.major in [9, 10]
```

`DeepseekV4FlashMLASparseBackend` (line 150) inherits without override → DSA-v4 hard-locked Hopper/Blackwell.

**Resolution:** sibling `DeepseekV4FlashMLASparseAmpereBackend` overrides to include `8` + wires `get_impl_cls()` to new `FlashMLASparseAmpereImpl` calling sm_86 ops.

## 3. FlashMLA cmake matrix

`cmake/external_projects/flashmla.cmake:55-65`:
```cmake
SUPPORT_ARCHS = "9.0a" (CUDA≥12.3) + "10.0f"|"10.0a" (CUDA≥12.8/12.9)
intersection with CUDA_ARCHS=8.6 → empty
→ add_custom_target(_flashmla_C)            # empty stub
→ add_custom_target(_flashmla_extension_C)  # empty stub
```

Empty stub = Python shim falls through → `RuntimeError`. Matches observed 500 at `/v1/completions`.

**Upstream pin:** `https://github.com/vllm-project/FlashMLA` @ `a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b`.

```
FlashMLA/csrc/
├── torch_api.cpp
├── smxx/decode/{get_decoding_sched_meta, combine}/        # arch-agnostic
├── sm90/decode/dense/instantiations/{fp16,bf16}.cu
├── sm90/decode/sparse_fp8/instantiations/                 # ← Story 4 ⚠️
│   ├── model1_persistent_h64.cu / h128.cu
│   └── v32_persistent_h64.cu / h128.cu
├── sm90/prefill/sparse/                                    # ← Story 5 ⚠️
│   ├── fwd.cu
│   └── instantiations/phase1_k{512,576}{,_topklen}.cu
├── sm100/...
├── extension/sm90/dense_fp8/...
├── kerutils/include/                                       # arch-agnostic
└── cutlass/                                                # bundled CUTLASS
```

Stories 4-5 populate `csrc/sm86/decode/sparse_fp8/` + `csrc/sm86/prefill/sparse/`.

## 4. Hopper intrinsics → sm_86 substitutions

### 4.1 TMA → `cp.async.cg.shared.global` (sm_80+)

**Hopper:** `cp.async.bulk.tensor.{1d,2d}.shared.global` — async bulk tensor copy via TMA descriptor.

**sm_86:**
```cuda
#pragma unroll
for (int i = 0; i < TILE_BYTES; i += 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem + i), "l"(gmem + i));
}
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");
__syncthreads();
```

Helper: `csrc/sm86/include/cp_async_tile.cuh`. Risk LOW. Perf hit ~30-60%.

### 4.2 WGMMA → `mma.sync.aligned.m16n8k16` (sm_80+)

**Hopper:** `wgmma.mma_async.m64nXk16.{f16,bf16,f32}.{f8e4m3,f8e5m2}` — warpgroup async MMA.

**sm_86:**
```cuda
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
  : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
    "r"(B[0]), "r"(B[1]),
    "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
);
```

**FP8 inputs:** sm_86 has no native FP8 MMA. Two paths:
1. **bf16 emulation** — dequant FP8→bf16 in smem before MMA (2× mem-bw)
2. **MARLIN-FP8** — sm_80+ FP8 via dequant-on-load + INT8 MMA (faster GEMM portion)

**Numerics:** bf16 acc preferred (fp16 acc loses precision in long-seq softmax exponentials).
**Tile shape change:** Hopper m64×n×k16 → sm_86 m16×n8×k16 = 4× more issues per equivalent block. Profile in Story 8.
**Owner:** Story 5. Risk MED.

### 4.3 cp.async.bulk → `cp.async.{ca,cg}` (sm_80)

Same loop pattern as §4.1 for non-tensor data (scales, indices, sched meta). Header: `cp_async_bulk.cuh`. Risk LOW.

### 4.4 PDL → host-side `cudaStreamSynchronize`

**Hopper:** `griddepcontrol.launch_dependents.acquire/release` — GPU-triggered kernel chaining.

**sm_86:**
```cuda
indexer_topk<<<...>>>(stream);
sparse_fwd<<<...>>>(stream);
// Stream-order implicit sync. Cost ~10-30 μs per chain.
```

CUDA-graph-incompatible. Either fuse into single kernel OR accept non-graph mode for indexer leg.

## 5. KV-cache layout (DSA-v4)

From `flashmla_sparse.py:81-89`:
```
DSA-v4 fp8_ds_mla cache (584 B/token):
[0..447]   = 448 × float8_e4m3 (NoPE, quantized)
[448..575] = 64  × bfloat16    (RoPE, unquantized)
[576..583] = 7   × ue8m0 + 1B pad (per-64 NoPE block scale)
```

ue8m0 → fp32: `2^(byte - 127)` (bias-127 exponent, DeepGEMM convention). Per-64-element-block multiplier on dequant. Helper: `ue8m0_dequant.cuh`.

`get_kv_cache_shape` v4 → `(num_blocks, block_size, 584)` (`flashmla_sparse.py:167`). `block_size=256` (`:152`).

## 6. Compiled artifact map

| Module | Sources | Python entry | Hopper-only? |
|---|---|---|---|
| `vllm._flashmla_C` | `FlashMLA_SOURCES` (sm_90/100 sparse) | `flash_mla_with_kvcache`, `flash_mla_sparse_fwd`, `get_mla_metadata` | YES → Stories 4-5 produce sm_86 |
| `vllm._flashmla_extension_C` | `FlashMLA_Extension_SOURCES` (sm_90 dense FP8) | dense FP8 (verify hot-path use Story 3) | YES |
| `vllm._C` | vLLM csrc inc. `persistent_topk.cuh`, `topk.cu`, `cache_kernels*.cu`, `concat_mla_q.cuh`, `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | `persistent_topk`, `top_k_per_row_*`, `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`, `concat_mla_q` | NO — already sm_86 (verify Story 3) |

## 7. Hour-1 LRU shim wraps `sparse_attn_indexer.py:84`

Patch #16 stub replaces `torch.ops.vllm.sparse_attn_indexer` with dummy-topk return. LRU shim wraps over patch #16:

- **Cache key:** `(num_tokens_pow2_bucket, total_seq_lens, has_decode, has_prefill)`
- **Cache value:** prior `topk_indices_buffer` slice (cloned to avoid vLLM mutation)
- **Decorator:** `@functools.lru_cache(maxsize=64)` on a wrapper
- **Goal:** latency reduction at decode (avoid repeat allocator pressure / NaN amplification in dummy path)
- **Correctness:** neutral (dummy is wrong regardless); real correctness fix is Stories 4-5

## 8. Numerics gates (Stories 7 + CC6)

1. PyTorch reference ↔ sm_86 kernel: cosine ≥0.97 logits, top-1 ≥98% indices match
2. CC6 1500-prompt fixed-seed: top-1 ≥98% vs llama.cpp PP baseline, perplexity ≤2% MMLU subset
3. conc=4 over 50 prompts: zero NaN, zero index OOB (debug-build PTX assertion)
4. CUDA-graph compat (PDL substitution kills capture → fuse or non-graph mode for indexer leg)

## 9. References

**Local vLLM source:** `/Users/z/AGENTIC/sources/vllm-mainline/`
- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `cmake/external_projects/flashmla.cmake`
- `csrc/persistent_topk.cuh` (Ampere-clean)
- `csrc/concat_mla_q.cuh`
- `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`

**Cmake-fetched (NOT local):**
- FlashMLA upstream @ `a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b`
- DeepGEMM upstream — `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`

**Corpus:** `~/AGENT/comms/DSA_ENGINEERING_JOURNEY.md`, `~/AGENT/comms/LEAD_CORPUS_LOAD_20260428.md` (§2 Wall, §4 17-patch cascade), `~/AGENT/comms/PROJECT_3WEEK_MASTER_PLAN_20260428.md`.

## 10. Story-2 → Story-3 handoff

Story 3 needs FlashMLA upstream local. Two unblock paths:
- CC1 mirrors `vllm-project/FlashMLA@a6ec2ba` to `~/AGENTIC/sources/FlashMLA-upstream/` (~30 MB)
- Self-clone via WC3: `git clone https://github.com/vllm-project/FlashMLA && git checkout a6ec2ba7`

Story 3 reads TMA descriptor layouts from `csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu` (DSA-v4 path) → produces stride/offset arrays for sm_86 manual cp.async.cg loop.

— END Story 2.
// --ProtoAI-Bakari--
