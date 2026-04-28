# DeepSeek Compressor — Hopper Anatomy + sm_86 Substitution Map

**Lane:** CC4 lightning (Story 2)
**Source:** vLLM mainline `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` + `vllm/model_executor/layers/deepseek_compressor.py` + `vllm/v1/attention/backends/mla/compressor_utils.py`
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC4]**

## TL;DR — sm_86 risk class: LOW

The compressor is **already Triton, not custom Hopper CUDA.** The only architectural blocker on sm_86 is the `tl.float8e4nv` casts at three call sites (cache write path). Replace them with bf16 (or sm_89-style fp8e4nv emulation via `tl.uint8` byte-pack) and the kernels compile + run on Ampere with no sm_90 intrinsic dependency. There is **no TMA, no WGMMA, no `cp.async.bulk`, no `griddepcontrol`** in the compressor path — only `launch_pdl=False` flags at the wrapper level, which is the disabled state already.

Risk band: LOW (Triton fp8e4nv lowering is the entire port).

---

## 1. Module map

```
deepseek_compressor.py    (Python wrapper / nn.Module)
  ├── DeepseekCompressor.forward                              # orchestrator
  │     ├── cublas_gemm_bf16_bf16_fp32   (input projection)   # arch-agnostic
  │     ├── _save_partial_states_kernel  (Triton)             # arch-agnostic
  │     └── self._fused_kernel  →  one of:
  │           - _fused_kv_compress_norm_rope_insert_sparse_attn   # head=512, FP8 nope + bf16 rope
  │           - _fused_kv_compress_norm_rope_insert_indexer_attn  # head=128, all FP8
  │           - _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn  # head=128, MXFP4 nibble pack
  ├── CompressorStateCache (nn.Module + AttentionLayerBase)   # KV-spec wrapper
  ├── CompressorBackend (AttentionBackend)                    # plumbing
  └── CompressorMetadataBuilder                               # token_to_req_indices builder

compressor_utils.py
  └── _compressed_slot_mapping_kernel                          # Triton, arch-agnostic
      get_compressed_slot_mapping                              # Python entry

fused_compress_quant_cache.py   ← THE HOT PATH
  ├── _fused_kv_compress_norm_rope_insert_sparse_attn         (head=512 / DSV4 attn path)
  ├── _fused_kv_compress_norm_rope_insert_indexer_attn        (head=128 / indexer path)
  └── _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn  (head=128 / MXFP4 path)
```

---

## 2. Hopper-ism inventory (verbatim line refs)

| # | File | Line(s) | Construct | Hopper-only? | sm_86 substitution |
|---|---|---|---|---|---|
| 1 | `fused_compress_quant_cache.py` | 172 | `x_fp8 = x_clamped.to(tl.float8e4nv)` (sparse_attn cache write) | **YES** — Triton emits `cvt.rn.satfinite.e4m3x2.f32` PTX requiring sm_89+ (and sm_90 for some load ops). On sm_86, Triton compiler raises `PTX ISA does not support .e4m3` | Bf16 store + manual UE8M0 absmax (already computed) → reader path mirror. OR `tl.uint8` bitcast of an emulated `f32 → e4m3` lookup table done in registers (slower). Decision: bf16 path for Story 3. |
| 2 | `fused_compress_quant_cache.py` | 384 | `x_fp8 = x_clamped.to(tl.float8e4nv)` (indexer cache write) | YES (same) | Same: bf16 path; or per-block fp8 lookup in registers. |
| 3 | `fused_compress_quant_cache.py` | (mxfp4 path) | `_e2m1_nibble` (imported from `fused_indexer_q.py`) | NO — software nibble pack, arch-agnostic | leave as-is |
| 4 | `deepseek_compressor.py` | 332 | `launch_pdl=False` on `_save_partial_states_kernel` | NO — flag disabled. Hopper-only kernel arg path but the value **is already** False. | leave as-is |
| 5 | `deepseek_compressor.py` | 381 | `launch_pdl=False` on `self._fused_kernel` | NO — same | leave as-is |
| 6 | `deepseek_compressor.py` | 169 | `alignment=576` (FlashMLA cache spec) | NO directly — number is sized for FlashMLA Hopper TMA bulk-load alignment (576 = 448 fp8 + 128 bf16 token stride). On sm_86 the `cp.async.cg` per-tile path doesn't *require* 576 alignment but doesn't break with it either. | leave 576; downstream FlashMLA-Sparse port (Story 6) handles its own alignment story |
| 7 | `compressor_utils.py` | full file | `_compressed_slot_mapping_kernel` | NO — pure index math + paged block_table loads | leave as-is |
| 8 | `_save_partial_states_kernel` | full body (`deepseek_compressor.py:386-438`) | Triton, fp32 throughout | NO | leave as-is |

**No TMA. No WGMMA. No `cp.async.bulk`. No `griddepcontrol`. No `mma.sync` intrinsics in this file.** The compressor is "fast enough" with Triton's normal codegen because the inner reduce dim is small (HEAD_SIZE = 128 or 512).

---

## 3. Kernel walkthrough — `_fused_kv_compress_norm_rope_insert_sparse_attn`

### 3.1 Entry contract
- One Triton program per token (`pid_b = tl.program_id(0)`, grid = `(num_actual,)`).
- Early-exit on `slot_id < 0` (PAD sentinel, line 77-78).
- Early-exit on non-boundary positions: `(position + 1) % COMPRESS_RATIO != 0` (line 81-82). Compressor only writes once per `COMPRESS_RATIO` tokens — most calls early-exit.

### 3.2 State-cache gather (lines 86-127)
For each token, gather `(1 + OVERLAP) * COMPRESS_RATIO` rows from the per-request state cache via the shared block_table → block_number lookup. This is a small 2D load (typically 4×512 fp32 = 8 KB). Done via Triton's `tl.load(2D, mask=...)`. Arch-agnostic, no Hopper intrinsic.

### 3.3 Compress: softmax + weighted sum (lines 115-129)
```python
score = tl.softmax(score, dim=0)                     # row reduction over compress_ratio dim
compressed_kv = tl.sum(kv * score, axis=0)           # weighted sum
```
Pure Triton fp32. **Performance note:** if Hopper backend lowers this to a WGMMA via `tl.dot` autotune, it does; on sm_86 it falls back to register-blocked accumulation. Triton handles the lowering — no code change needed.

### 3.4 RMSNorm (lines 131-135)
Inline fp32. Trivial.

### 3.5 FP8 UE8M0 quant — **THE Hopper blocker** (lines 152-187)
```python
quant_input = normed.to(tl.bfloat16).to(tl.float32)         # round-trip for parity w/ ref
quant_2d    = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))   # 7×64 tile
abs_2d      = tl.abs(quant_2d)
block_absmax = tl.max(abs_2d, axis=1)                                  # [7] fp32
block_absmax = tl.maximum(block_absmax, 1e-4)

raw_scales  = block_absmax * INV_FP8_MAX                                # = absmax/448
exponents   = tl.ceil(tl.log2(raw_scales))                              # UE8M0 exponent
inv_scales  = tl.exp2(-exponents)
inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
x_scaled    = quant_2d * inv_scales_col
x_clamped   = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
x_fp8       = x_clamped.to(tl.float8e4nv)                  # ← sm_86 REJECTS THIS LINE
x_uint8     = x_fp8.to(tl.uint8, bitcast=True)             # storage byte
x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))
nope_mask   = block < NOPE_HEAD_DIM                                     # 448
tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)
```
The UE8M0 absmax block-scaling **is already in Triton-fp32 and arch-agnostic**. Only the `to(tl.float8e4nv)` cast fails on sm_86.

**Substitution candidates (Story 3):**

| Candidate | Cost | Numerics | Implementation |
|---|---|---|---|
| **A. bf16 cache (drop fp8 entirely)** | 2× memory bandwidth, 2× cache footprint | bit-exact bf16 | Replace `tl.float8e4nv` with `tl.bfloat16`, store 2 bytes/elem instead of 1. NOPE_HEAD_DIM=448 → 896 B/token instead of 448 B/token. Cache spec needs new alignment (1024 or keep 576 + double NOPE_BYTES). **MED-RISK** — touches every reader. |
| **B. Software fp8e4nv quantize in registers (sm_86 emulation)** | ~5-10% kernel slowdown | bit-exact UE8M0 + e4m3 | Use `tl.uint8` lookup: clamp to ±448, scale, then bit-pack via integer ops in registers. Triton emits sm_86-legal PTX. **Reader path unchanged.** **LOW-RISK numerics, MED-RISK perf.** |
| **C. fp16 cache** | 2× footprint vs fp8, 1× vs bf16 | dynamic range narrower than bf16 | Risky for normalized values. Skip. |

**Decision (Story 3):** start with **B** (emulation) because reader path stays untouched. Fall back to **A** if perf ≤ 50% of Hopper baseline; fork CC8 to bench under TP=2×EP=8.

### 3.6 RoPE (lines 189-214)
GPT-J style register-rotated, `tl.split` + `tl.interleave`. Pure register math, arch-agnostic.

### 3.7 Cache write summary
Lines 177 (FP8 nope), 182-187 (UE8M0 scale bytes), 211-214 (bf16 rope) are stores into the paged cache. All three are normal `tl.store` with mask — no `cp.async.bulk` / TMA.

---

## 4. Kernel walkthrough — `_fused_kv_compress_norm_rope_insert_indexer_attn` (head=128)

Same shape as 3, but:
- HEAD_SIZE = 128, single QUANT_BLOCK = 128 (so `N_QUANT_BLOCKS = 1`)
- All 128 elements stored as FP8 (no bf16 rope split — indexer path uses FP8 throughout)
- `tl.float8e4nv` cast at line 384 — **same sm_86 blocker, same substitution**.
- `num_warps = 1` (vs 4 for sparse_attn path) — smaller working set

Port: copy decision from §3.5; smaller QUANT_BLOCK means substitution B has near-zero overhead because the absmax reduce is over a single tile.

---

## 5. Kernel walkthrough — `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` (head=128, MXFP4)

Uses `_e2m1_nibble` (imported from `fused_indexer_q.py`) — **already a software nibble pack, no fp8e4nv intrinsic.** This kernel is sm_86-clean as-is. Verify via grep in Story 3:
```
grep -nE 'float8e4nv|e4m3|wgmma|cp\.async\.bulk' fused_compress_quant_cache.py
```
should return only the 2 sparse_attn / indexer hits, not the mxfp4 path.

---

## 6. PDL surface

`launch_pdl` is a Triton `@triton.jit` kernel-launch arg. On Hopper, when True, the kernel emits PDL prologue/epilogue via `griddepcontrol.launch_dependents` PTX instructions to chain into the next kernel without host roundtrip. **The Python wrapper sets `launch_pdl=False` at both call sites** (deepseek_compressor.py:332, 381) — so even the Hopper build does not currently use PDL here. Comment at line 310-314 explains: a prior bug surfaced read-after-write race when launch_pdl=True, so it was disabled. **No work needed for sm_86.** Confirm in Story 3 that Triton's sm_86 backend silently ignores `launch_pdl=False` (it should — flag is no-op when False on any arch).

---

## 7. Surface-area summary for Story 3 (Compressor sm_86 port)

**Files to author:**
- `csrc_patches/compressor_sm86.cu` — **misnamed in backlog**; the port is actually a Triton kernel patch, not .cu. Will produce `csrc_patches/compressor_sm86.py` instead, which monkey-patches `vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache._fused_kv_compress_norm_rope_insert_sparse_attn` + `..._indexer_attn` to use sm_86-safe fp8e4nv emulation. (CC4 will note this in the commit + bridge milestone for CC0 visibility.)
- `tests/unit/test_compressor.py` — bit-exact comparison vs Hopper reference at fp32 precision (round-trip parity gate); cosine ≥ 0.999 over 1024 random tokens.

**Substitution code skeleton (preview):**
```python
# In csrc_patches/compressor_sm86.py
def _fp8e4nv_emulate(x_clamped_fp32):
    """sm_86 emulation of tl.float8e4nv cast.
    Bit-exact for normal range [-448, 448]; flushes denormals to 0.
    """
    # In kernel: replace `x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)`
    # with the manual e4m3 bit-layout: sign(1) | exp(4) | mantissa(3).
    # Done with bit ops on tl.uint32 reinterpretation of the fp32 value.
    ...
```

**Acceptance gate (Story 4):**
- Bit-exact match vs reference (max abs diff = 0 after round-trip dequant) on 1024 random fp32 tokens
- Triton compile success on sm_86 (i.e. no `PTX ISA` errors)
- Wall ≤ 1.3× original Hopper runtime on cuda4 (CC9 L4 verifies)

---

## 8. Open questions (escalate to CC0)

1. **Cache reader path:** does any other vLLM kernel read the FP8 cache via TMA bulk-load? If yes, those readers also need sm_86 paths (cp.async.cg per-tile). CC1's inventory should answer; pre-flagging here.
2. **CompressorBackend.get_supported_head_sizes returns [512, 1024]** — but the module also takes head_dim=128 for indexer paths. Confirm the 128 path goes through a different backend registration or is implicitly handled via the indexer attention layer (likely `IndexerCompressorBackend` — verify in CC3's lane).
3. **Parity vs FlashMLA Hopper reference:** FlashMLA reads the FP8 cache during sparse attention. Its own sm_86 port (Story 6) must agree with our quantized format byte-for-byte. **Inter-lane contract:** keep the `(NOPE_HEAD_DIM=448) FP8 + (ROPE_HEAD_DIM=64) bf16 + 7 UE8M0 scale bytes` layout intact in the sm_86 port — substitution B preserves it; substitution A breaks it (would force CC3+CC4+CC5 reader patches). Strong preference for B.

---

## 9. Estimated wall for Story 3

- Substitution B (fp8e4nv emulation in registers): 1.5-2.5h author + 0.5h cuda4 compile-test (CC9 L4 slot)
- Bit-exact verification: 0.5h author of `test_compressor.py`
- Total: 2.5-3.5h elapsed, <2h focused work — if substitution B is bit-exact (likely; e4m3 has a closed-form integer representation).

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC4]

// --ProtoAI-Bakari--
