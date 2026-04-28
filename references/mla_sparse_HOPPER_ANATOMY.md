# FlashMLA-Sparse — Hopper Anatomy + sm_86 Substitution Map

**Lane:** CC4 lightning (Story 5)
**Source:**
- `vllm/v1/attention/backends/mla/flashmla_sparse.py` (Python backend wrapper)
- `vllm/v1/attention/ops/flashmla.py` (Python torch-ops shim)
- `cmake/external_projects/flashmla.cmake` (build glue)
- FlashMLA upstream repo `https://github.com/vllm-project/FlashMLA` pinned SHA `a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b` (NOT vendored into vllm-mainline checkout — fetched at cmake time)

**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC4]**

---

## TL;DR — sm_86 risk class: HIGH

FlashMLA-Sparse is the **opposite of the compressor** (Story 2): it is **pure C++/CUDA + CUTLASS sm_90 templates**, NOT Triton. There is no Triton fallback, no software-emulation path. The cmake gates the entire `_flashmla_C` extension to `[9.0a, 10.0a, 10.0f]` only (lines 55-65 of `flashmla.cmake`); on sm_86 the build emits empty stub targets (lines 180-185) and `_is_flashmla_available()` returns False at runtime, so every call raises.

To unblock DSV4 + GLM-5.1 on sm_86, **we must author a sibling sm_86 backend**, not patch FlashMLA. The Story 6 deliverable is `csrc_patches/mla_sparse_sm86.cu` — a fresh kernel that:

- Reads the same paged KV cache layout the compressor writes (584B/token for DSV4 = 448 fp8 NoPE + 128 bf16 RoPE + 8 ue8m0 scale per token).
- Performs sparse online-softmax attention over `topk_tokens` selected K rows (indices come from CC3's sparse_attn_indexer).
- Uses sm_86-legal intrinsics: `cp.async.cg` (per-tile loop, sm_80+) instead of TMA bulk, `mma.sync.aligned.m16n8k16` (sm_80 PTX) instead of WGMMA, host-side stream sync instead of PDL.
- Targets **decode path first** (single Q row per head, much smaller working set than prefill). Prefill is Story-X stretch.

Risk band: HIGH. CUTLASS-template-equivalent on sm_86 has more registers contention, lower occupancy, and no async pipeline parity with sm_90. Realistic expectation: 30-50% of Hopper throughput per Anatomy §3.5 estimates and AO1's 2× emulation cost.

---

## 1. Build gate — why FlashMLA is invisible on sm_86

`cmake/external_projects/flashmla.cmake`:

```
55:  set(SUPPORT_ARCHS)
56:  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
57:      list(APPEND SUPPORT_ARCHS "9.0a")     # Hopper
58:  endif()
...
62:      list(APPEND SUPPORT_ARCHS "10.0f")    # Blackwell family-feature
64:      list(APPEND SUPPORT_ARCHS "10.0a")    # Blackwell architecture-specific
68:  cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")
```

If our `CUDA_ARCHS=8.6` (sm_86) intersects empty with `[9.0a, 10.0a, 10.0f]`, `FLASH_MLA_ARCHS` is empty, `if(FLASH_MLA_ARCHS)` is False, and lines 180-185 fire:

```
add_custom_target(_flashmla_C)              # empty stub
add_custom_target(_flashmla_extension_C)    # empty stub
```

The wheel installs `vllm._flashmla_C` as a Python module that imports OK but has zero ops. Then `flashmla.py:33-48` `_is_flashmla_available()` returns False, and `flashmla.py:81-83` `_raise_flashmla_unavailable` is bound to all six callable names. Every call raises `RuntimeError`.

**Path A (CC2 / CC4 joint):** add an `sm_86` SUPPORT_ARCH bucket pointing at our patches, with a parallel source list that compiles `csrc_patches/mla_sparse_sm86.cu` into the *same* `_flashmla_C` extension. Then the Python torch-ops shim resolves to our kernel transparently when `current_platform.compute_capability == (8, 6)`.

**Path B (cleaner long-term):** new extension `_flashmla_sm86_C`, registered separately, with `flashmla_sparse.py` dispatching by compute capability. Slightly more wiring, doesn't touch upstream cmake.

CC4 will draft Path B (less coupling). CC2 owns the cmake side; we coordinate via bridge `topic=cmake_negotiate`.

---

## 2. Hopper kernel inventory (FlashMLA upstream)

The 4 sparse-decode + 4 sparse-prefill instantiations enumerated in `flashmla.cmake:85-96`:

### 2.1 sm90 sparse decode (FP8) — what DSV4 + GLM-5.1 actually call at decode

| File | Heads | Schedule | Cache layout |
|---|---|---|---|
| `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu` | 64 | persistent | DSV4 (584B / token, compress_ratio=128) |
| `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu` | 128 | persistent | DSV4 (584B / token, compress_ratio=128) |
| `csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu` | 64 | persistent | V3.2 (656B / token, compress_ratio=4) |
| `csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu` | 128 | persistent | V3.2 (656B / token, compress_ratio=4) |

**Hopper-only constructs (per CUTLASS 3.x sm90 templates):**
- `cp.async.bulk.tensor.4d` for FP8 NoPE tile load (TMA)
- `wgmma.mma_async.sync.aligned.m64n*k32.fp8.bf16` for QK^T and softmax-weighted V combine
- `griddepcontrol.launch_dependents` (PDL) for chaining decode → combine
- `tma.descriptor` setup for cluster-wide async copies
- Persistent kernel grid sizing (one CTA per SM, looping over Q tokens)

### 2.2 sm90 sparse prefill (BF16) — used only when num_heads_per_rank ≥ 32

| File | Cache K | Description |
|---|---|---|
| `csrc/sm90/prefill/sparse/fwd.cu` | (entry) | dispatch + scheduling |
| `csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu` | 512 | DSV4 NoPE |
| `csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu` | 512 | DSV4 NoPE w/ variable topk len |
| `csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu` | 576 | V3.2 NoPE+RoPE |
| `csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu` | 576 | V3.2 NoPE+RoPE w/ topk len |

These are the **BF16 prefill** kernels — NOPE is dequantized FP8→BF16 *before* prefill (see `flashmla_sparse.py:60-62`: "use the BF16 prefill kernel for prefill (upconverting the FP8 cache to BF16 then calling the prefill kernel)"). Hopper-isms are similar: TMA + WGMMA in BF16 instead of FP8.

### 2.3 sm100 (Blackwell) — out of scope

`csrc/sm100/...` enumerated for parity but irrelevant to sm_86. We will not port these; if we ever target Blackwell at L0/L1 OSS tier, we can pull upstream.

---

## 3. KV cache layout (DSV4) — the contract CC4 must honor

From `flashmla_sparse.py:81-89` (verbatim):

```
For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   First 448 bytes:  "quantized NoPE" — 448 float8_e4m3 values
-   Next 128 bytes:   "RoPE" — 64 bfloat16 values (NOT quantized)
-   Last 8 bytes:     Scale factors — 7 ue8m0 values + 1B pad
                      First ue8m0 scales the first 64 fp8 values (block 0 of NoPE),
                      second scales the next 64, ... up through 448 / 64 = 7 blocks.
```

Compressor (Story 2-3) writes this layout. Sparse-attn must read it identically. **Inter-lane contract:** any change to NoPE width / RoPE width / scale layout breaks both lanes.

For sm_86 port, we read the same layout. Dequant FP8 → BF16 happens in registers per K row before the QK^T mma:

```
ue8m0 byte b → scale = 2^(b - 127)        // ue8m0 stores biased exponent
fp8 byte e → e4m3 normal: (-1)^sign * 2^(exp-7) * (1 + mant/8)   // table lookup or direct bit-decode
bf16 K row[i] = scale[block(i)] * fp32(e4m3[i])
```

Per-block dequant is cheap because every 64 consecutive fp8 values share one scale byte.

---

## 4. Topk sparsity — CC3's contract

`flashmla_sparse.py:298`:
```
self.topk_tokens = vllm_config.model_config.hf_config.index_topk
```

DSV4 default = 2048 topk K rows per query token. Indexer (CC3 lane) writes a `[num_q_tokens, topk]` int32 tensor of indices (or -1 for invalid). FlashMLA-Sparse decode reads these indices, gathers the corresponding K rows from the paged cache, and runs online-softmax over those 2048 rows.

For our sm_86 port, the indices are produced by:
- Hopper: CC3's stubbed `torch.ops.vllm.sparse_attn_indexer` (currently a Python stub returning dummy topk in the 17-patch cascade)
- sm_86 port: CC3's authored `csrc_patches/sparse_attn_indexer_sm86.cu`

Indices are in **logical** (compressed) space. Conversion to physical block-table slots happens in `triton_convert_req_index_to_global_index` (sparse_utils.py) — this Triton helper is arch-agnostic.

---

## 5. sm_86 substitution map — Story 6 plan

### 5.1 Decode path (priority — DSV4 conc=1 path)

Replace `model1_persistent_h{64,128}.cu` with `mla_sparse_sm86_decode.cu`:

| Hopper construct | sm_86 substitution | File / function |
|---|---|---|
| `cp.async.bulk.tensor.4d` (TMA NoPE load) | `cp.async.cg.shared.global` per-tile (sm_80+) loop, 16-byte chunks | `mla_sparse_sm86_decode.cu` |
| `wgmma.mma_async.fp8.bf16` (QK^T) | `mma.sync.aligned.m16n8k16` over BF16-dequanted K (sm_80+) | same |
| `wgmma.mma_async.fp8.bf16` (PV combine) | same `mma.sync.aligned.m16n8k16` | same |
| `griddepcontrol.launch_dependents` (PDL) | host `cudaStreamSynchronize` between decode and combine | torch_api.cpp shim |
| Persistent kernel (one CTA / SM) | non-persistent: one CTA per (batch, head) tile, smaller grid OK on sm_86 | same |
| Cluster (multi-CTA cooperative) | n/a (Hopper-only); sm_86 single-CTA | same |
| 256-byte aligned descriptors | 16-byte aligned; pure offset arithmetic | same |
| Phase scheduling via `combine.cu` | reuse upstream `combine.cu` if it's arch-agnostic; verify in story 5b | scripts/check_combine_arch.sh |

### 5.2 Prefill path (deferred to Story-X stretch)

Prefill on Ampere is *much* slower than Hopper because:
- WGMMA throughput is ~2× of Ampere `mma.sync` per SM
- TMA+pipelining dominates prefill (one kernel feeds all the way through)
- 32-head BF16 prefill kernel is the heavier of the two paths

For Week-1 first-token, route prefill through **mixed-batch FP8 decode mode** (`flashmla_sparse.py:60-66`, "we use #1 [mixed batch] when the number of heads per rank is low (i.e. TP)"). DSV4 with TP=2 has h_q=128/2 = 64 per rank; if we pad to 128 we may accidentally trigger the BF16 prefill path. **CC8 must benchmark with `MIN_HEADS_FOR_BF16_PREFILL` set high enough to force mixed-batch mode** during sm_86 bring-up. Add to CC8's flight-deck profile knobs.

### 5.3 Cache reader (FP8 → BF16 dequant)

Helper `csrc_patches/fp8_kv_dequant_sm86.cu` already exists (CC3's lane, fb0cc82). CC4's mla_sparse_sm86 will *call* it, not re-implement. **Inter-lane reuse confirmed.**

### 5.4 Persistent vs non-persistent

Hopper sparse-decode uses persistent kernels (one CTA per SM, looping across batches+heads) for fewer launches. On sm_86 with 84 SMs (RTX 3090), we have plenty of grid headroom for one CTA per (batch, head) tile and don't need persistent looping. **Drop persistent design** — simpler kernel, same throughput at our batch sizes.

---

## 6. Numerics gate plan (Story 7)

Reference: pure-PyTorch sparse online-softmax over the same indices and K rows, on a 1024-prompt fixture. Acceptance:

- cosine ≥ 0.97 vs reference
- max-abs-diff ≤ 1e-2 (RoPE bf16 precision floor; tighter than fp8 noise)
- no NaN, no Inf
- top-1 token equality ≥ 98% in chained mode (full DSV4 forward)

CC6 owns 1500-prompt set; CC4 reuses CC3's `tests/unit/conftest.py` + `gen_fixtures.py` infrastructure (already exists per CC3's commit log on lane-cc3-sparse-attn-cuda).

---

## 7. Open questions (escalate to CC0)

1. **Persistent vs non-persistent on sm_86:** confirm with CC8 perf bench. Non-persistent is simpler, persistent might recover 10-15% throughput — defer to Story 10 (v2 tuned kernel).
2. **`combine.cu` arch-portability:** confirm by reading `csrc/smxx/decode/combine/combine.cu` — header path `csrc/smxx` suggests "any sm" but verify no Hopper-isms. Add a short script `scripts/check_combine_arch.sh` to grep for `wgmma|tma|cp\.async\.bulk|griddepcontrol`.
3. **Build integration:** decide Path A (intrusive cmake) vs Path B (sibling extension). CC2 to weigh in.
4. **MIN_HEADS_FOR_BF16_PREFILL knob:** does CC8's flight-deck profile expose it? If not, we need to add it to keep prefill on the mixed-batch FP8 decode path during sm_86 bring-up.
5. **C128A metadata path:** lines 372-410 of `flashmla_sparse.py` build extra topk metadata for `compress_ratio == 128` (DSV4-specific). The downstream consumer is the FP8 decode kernel via `tile_scheduler_metadata`. **CC4 must mirror this structure or our sm_86 kernel will mis-schedule.** Read `_build_c128a_topk_metadata_kernel` (Triton, in cache_utils.py per line 381 reference) before Story 6 hot-path coding.

---

## 8. Wall estimate for Story 6

- Decode kernel author + smem layout: 6-10h
- BF16 dequant integration (reuse CC3's fp8_kv_dequant_sm86): 0.5h
- Topk gather + online-softmax: 3-4h
- C128A metadata mirror: 2-3h
- Compile + launch shim (pybind / torch ops): 1-2h
- L1 numerics debug: 2-4h (kernel will fail first try; numerics gate iterations expected)

Total: **15-23h focused work** for first compiling-and-numerically-correct decode kernel. Performance tuning (Story 10 v2) is separate.

---

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC4]

// --ProtoAI-Bakari--
