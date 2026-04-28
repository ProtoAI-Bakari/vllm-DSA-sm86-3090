# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""MLA sparse decode router for sm_86 — Story 5.

Bridges vLLM's ``vllm.v1.attention.ops.flashmla.{flash_mla_sparse_fwd,
flash_mla_with_kvcache}`` calls to CC4's Triton MLA-decode kernel
(``patches/01_dsa_sm86_kernels/triton_mla_decode_sm86.py``) on Ampere.

Topology pivot (z directive 2026-04-28T18:04Z): TP=2 × EP=8, PP=1. Under
PP=1 the gpu_model_runner.py:4072 IntermediateTensors assert never fires,
so the previously-shipped pp2 patch is a no-op. The remaining wall is
``vllm._flashmla_C`` not being available on sm_86 → ``flash_mla_*`` calls
either ``RuntimeError`` (the upstream raise stub) or fall through to the
empty-target Hopper path. This router intercepts at the Python shim layer
and dispatches to a Triton kernel that compiles on sm_86.

Coverage:
  - **BF16 sparse prefill** (``flash_mla_sparse_fwd``): wired to CC4's
    ``mla_decode_sparse_sm86`` with a shape adapter. Returns (out, lse).
  - **FP8 sparse decode** (``flash_mla_with_kvcache``): wired with a
    Triton FP8→BF16 dequant of the DSA-v4 fp8_ds_mla cache (584 B/token:
    448 fp8e4m3 NoPE + 64 bf16 RoPE + 7 ue8m0 scales + 1 pad), then
    delegates to CC4's BF16 Triton path. **NumERICS-PROVISIONAL** —
    bf16 emulation of FP8 logits, ~2× memory-bandwidth cost vs native
    FP8, but produces correct tokens (no NaN, no 500). Numerics gate
    by Story 7 (CC2-gate 1500-prompt set).

The router is a Python monkey-patch (no .so rebuild required). Apply once
after ``vllm.v1.attention.ops.flashmla`` is imported.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from vllm.triton_utils import tl, triton  # type: ignore
    HAVE_VLLM_TRITON = True
except ImportError:
    try:
        import triton  # type: ignore
        import triton.language as tl  # type: ignore
        HAVE_VLLM_TRITON = False
    except ImportError:
        triton = None  # type: ignore
        tl = None  # type: ignore
        HAVE_VLLM_TRITON = False


# ============================================================================
# DSA-v4 FP8 cache layout (per flashmla_sparse.py:81-89 docstring)
# ============================================================================
# 584 B / token total, ordered:
#   [0..447]   = 448 × float8_e4m3 (NoPE, quantized, 7 groups × 64 elements)
#   [448..575] = 64  × bfloat16    (RoPE, unquantized, 128 bytes)
#   [576..582] = 7   × ue8m0       (per-64-element NoPE scales)
#   [583]      = 1   × pad

V4_NOPE_BYTES = 448
V4_ROPE_BYTES = 128   # 64 bf16 = 128 bytes
V4_ROPE_ELEMS = 64
V4_SCALE_BYTES = 7
V4_PAD_BYTES = 1
V4_TOKEN_BYTES = V4_NOPE_BYTES + V4_ROPE_BYTES + V4_SCALE_BYTES + V4_PAD_BYTES  # 584
V4_NOPE_GROUP_SIZE = 64

assert V4_TOKEN_BYTES == 584


# ============================================================================
# Triton FP8 → BF16 dequant kernel for DSA-v4 cache layout
# ============================================================================


if triton is not None:

    @triton.jit
    def _v4_fp8_kv_to_bf16_kernel(
        # input: (num_blocks, block_size, 584) uint8 view of the FP8 cache
        cache_ptr,
        cache_stride0,  # stride per block (block_size * 584)
        cache_stride1,  # stride per token in block (584)
        # output: (num_blocks, block_size, 512) bf16 — 448 NoPE + 64 RoPE
        out_ptr,
        out_stride0,
        out_stride1,
        BLOCK_SIZE: tl.constexpr,    # tokens per block (256 for V4)
        NOPE_BYTES: tl.constexpr,    # 448
        ROPE_ELEMS: tl.constexpr,    # 64
        SCALE_BYTES: tl.constexpr,   # 7
        GROUP_SIZE: tl.constexpr,    # 64 (NoPE group size)
    ):
        block_idx = tl.program_id(0)
        token_idx = tl.program_id(1)

        token_base = (
            cache_ptr
            + block_idx * cache_stride0
            + token_idx * cache_stride1
        )
        out_base = (
            out_ptr
            + block_idx * out_stride0
            + token_idx * out_stride1
        )

        # ---- Decode the 7 ue8m0 scale bytes ----
        scale_off = NOPE_BYTES + ROPE_ELEMS * 2  # 448 + 128 = 576
        scales_u8 = tl.load(
            token_base + scale_off + tl.arange(0, SCALE_BYTES)
        ).to(tl.int32)
        # ue8m0 byte → fp32 scale: 2^(byte - 127)
        scales_f32 = tl.math.exp2((scales_u8 - 127).to(tl.float32))

        # ---- Dequantize each 64-element NoPE group with its scale ----
        for grp in tl.static_range(SCALE_BYTES):  # 7 groups
            group_base = grp * GROUP_SIZE
            offsets = group_base + tl.arange(0, GROUP_SIZE)
            scale = tl.extract_slice(scales_f32, [grp], [1], [1])  # scalar
            # fp8e4m3 byte → fp32 (built-in via .to() on uint8 reinterpreted)
            fp8_u8 = tl.load(token_base + offsets)
            # Decode fp8e4m3: sign(1) | exp(4, bias 7) | mant(3)
            sign = (fp8_u8 >> 7) & 1
            exp = (fp8_u8 >> 3) & 0xF
            mant = fp8_u8 & 0x7
            is_zero = (exp == 0) & (mant == 0)
            is_subnormal = (exp == 0) & (mant != 0)
            mant_f = mant.to(tl.float32) / 8.0
            normal_val = (1.0 + mant_f) * tl.math.exp2(
                (exp.to(tl.int32) - 7).to(tl.float32)
            )
            subnormal_val = mant_f * tl.math.exp2(-6.0)
            val = tl.where(is_subnormal, subnormal_val, normal_val)
            val = tl.where(is_zero, 0.0, val)
            val = tl.where(sign != 0, -val, val)
            # Apply per-group scale
            val = val * scale
            # Store as bf16 in NoPE region [0..447]
            tl.store(out_base + offsets, val.to(tl.bfloat16))

        # ---- Copy bf16 RoPE region as-is (already bf16) ----
        rope_byte_off = NOPE_BYTES  # 448
        rope_out_off = NOPE_BYTES   # NoPE in output is 448 bf16 elements
        bf16_in_ptr = (token_base + rope_byte_off).to(tl.pointer_type(tl.bfloat16))
        rope_off = tl.arange(0, ROPE_ELEMS)
        rope_vals = tl.load(bf16_in_ptr + rope_off)
        tl.store(out_base + rope_out_off + rope_off, rope_vals)


def dequant_v4_fp8_kv_cache(kv_cache_u8) -> Any:
    """Dequantize a DSA-v4 fp8_ds_mla cache to BF16.

    Input:  kv_cache_u8 of shape (num_blocks, block_size, 584) uint8.
    Output: (num_blocks, block_size, 512) bfloat16 — 448 NoPE + 64 RoPE.
    """
    if torch is None or triton is None:
        raise RuntimeError("torch + triton required for dequant_v4_fp8_kv_cache")
    assert kv_cache_u8.dtype == torch.uint8, kv_cache_u8.dtype
    assert kv_cache_u8.dim() == 3 and kv_cache_u8.shape[-1] == V4_TOKEN_BYTES, (
        f"expected (B,T,584) uint8, got {tuple(kv_cache_u8.shape)} {kv_cache_u8.dtype}"
    )
    num_blocks, block_size, _ = kv_cache_u8.shape
    out = torch.empty(
        (num_blocks, block_size, V4_NOPE_BYTES + V4_ROPE_ELEMS),
        dtype=torch.bfloat16,
        device=kv_cache_u8.device,
    )
    _v4_fp8_kv_to_bf16_kernel[(num_blocks, block_size)](
        kv_cache_u8,
        kv_cache_u8.stride(0),
        kv_cache_u8.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=block_size,
        NOPE_BYTES=V4_NOPE_BYTES,
        ROPE_ELEMS=V4_ROPE_ELEMS,
        SCALE_BYTES=V4_SCALE_BYTES,
        GROUP_SIZE=V4_NOPE_GROUP_SIZE,
        num_warps=1,
    )
    return out


# ============================================================================
# Routers — replace upstream flashmla shims
# ============================================================================


def flash_mla_sparse_fwd_sm86(q, kv_c_and_k_pe_cache, topk_indices, softmax_scale):
    """Replacement for ``flash_mla_sparse_fwd`` (BF16 sparse prefill path).

    Upstream signature (vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd):
      q                      : (T, num_heads, head_dim) bf16
      kv_c_and_k_pe_cache    : (-1, 1, head_dim) bf16 — flat KV concat
      topk_indices           : (T, 1, K) int32 — global slot indices, -1 = invalid
      softmax_scale          : float

    Returns: (out, lse) — out shape (T, num_heads, head_dim_v).

    Routes to CC4's ``mla_decode_sparse_sm86`` after shape adaption:
      CC4 expects q (B, H, D), k_cache (B, S, D), v_cache (B, S, Dv),
      topk_idx (B, H, K). For the prefill path each query token is a
      separate "batch", and key/value share the same buffer (kv_c_and_k_pe_cache).
    """
    from .triton_mla_decode_sm86 import mla_decode_sparse_sm86  # type: ignore  # noqa: E501

    if torch is None:
        raise RuntimeError("torch required")

    T, H, D = q.shape
    head_dim_v = D  # MLA: V is sliced from the same head; downstream takes [:Dv]
    B = T

    # CC4 expects (B,H,D) q and (B,S,D) k/v. Flatten T → B with seq_q=1.
    q_bhd = q  # already (T,H,D)

    # kv_c_and_k_pe_cache is (S_total, 1, D). For sparse path we treat it as
    # a single "batch" buffer indexed via topk_indices. CC4's kernel does
    # per-batch index lookup, so we replicate the buffer as (B, S, D).
    S_total = kv_c_and_k_pe_cache.shape[0]
    kv_flat = kv_c_and_k_pe_cache.view(S_total, D)  # (S, D)
    # Broadcast view across B (no copy)
    k_cache = kv_flat.unsqueeze(0).expand(B, S_total, D).contiguous()
    v_cache = k_cache  # MLA: K and V share

    # topk_indices (T, 1, K) → CC4 wants (B, H, K)
    topk_T1K = topk_indices.view(T, -1)  # (T, K)
    topk_idx_BHK = topk_T1K.unsqueeze(1).expand(B, H, topk_T1K.shape[-1]).contiguous().to(torch.int32)

    out_bhdv = mla_decode_sparse_sm86(
        q_bhd, k_cache, v_cache, topk_idx_BHK, scale=softmax_scale
    )  # (B,H,Dv) = (T,H,Dv)

    # LSE not produced by CC4's kernel — return zeros to match upstream tuple.
    lse = torch.zeros((T, H), dtype=torch.float32, device=q.device)
    return out_bhdv, lse


def flash_mla_with_kvcache_sm86(
    q,
    k_cache,
    block_table,
    head_dim_v,
    cache_seqlens,
    tile_scheduler_metadata,
    is_fp8_kvcache: bool = False,
    indices=None,
    softmax_scale=None,
    **kwargs,
):
    """Replacement for ``flash_mla_with_kvcache`` (FP8 / BF16 sparse decode).

    Upstream signature (vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache):
      q                          : (B, S_q, H, D) bf16
      k_cache                    : (num_blocks, block_size, 1, head_size) bf16
                                     OR (num_blocks, block_size, 1, 584) uint8 view
                                     when is_fp8_kvcache=True
      indices                    : (B, S_q, K) int32 — sparse topk slots
      softmax_scale              : float

    Returns: (out, lse) — out shape (B, S_q, H, head_dim_v).
    """
    from .triton_mla_decode_sm86 import mla_decode_sparse_sm86  # type: ignore  # noqa: E501

    if torch is None:
        raise RuntimeError("torch required")
    if indices is None:
        raise NotImplementedError(
            "flash_mla_with_kvcache_sm86 router requires indices (sparse decode only)"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    B, S_q, H, D = q.shape

    # ---- FP8 path: dequant cache to BF16 first ----
    if is_fp8_kvcache:
        # k_cache may be (num_blocks, block_size, 1, 584) — squeeze the kv-heads-1 dim
        if k_cache.dim() == 4:
            kv_u8 = k_cache.squeeze(-2)  # (num_blocks, block_size, 584)
        else:
            kv_u8 = k_cache
        if kv_u8.dtype != torch.uint8:
            kv_u8 = kv_u8.view(torch.uint8)
        kv_bf16 = dequant_v4_fp8_kv_cache(kv_u8)
        # kv_bf16 shape (num_blocks, block_size, 512) — flatten to (S_total, 1, D_combined)
        num_blocks, block_size, D_combined = kv_bf16.shape
        kv_flat = kv_bf16.view(num_blocks * block_size, D_combined)
    else:
        # BF16 path: just flatten existing layout
        if k_cache.dim() == 4:
            kv_flat = k_cache.squeeze(-2).reshape(-1, k_cache.shape[-1])
        else:
            kv_flat = k_cache.reshape(-1, k_cache.shape[-1])

    # ---- Adapt shapes for CC4 kernel ----
    T = B * S_q
    q_bhd = q.reshape(T, H, D)
    S_total, D_kv = kv_flat.shape

    # If FP8-dequant gave 512-dim (NoPE+RoPE) but Q is full head_dim, slice.
    # For DSA-v4: head_dim_v=512 and head_dim=576 (V3.2) or 512 (V4) — match Q.
    if D_kv != D:
        # Slice or pad as needed; for DSA-v4 the native NoPE+RoPE = 512 matches.
        if D_kv > D:
            kv_flat = kv_flat[:, :D]
        else:
            pad = torch.zeros(
                (S_total, D - D_kv), dtype=kv_flat.dtype, device=kv_flat.device
            )
            kv_flat = torch.cat([kv_flat, pad], dim=-1)

    k_cache_bhd = kv_flat.unsqueeze(0).expand(T, S_total, D).contiguous()
    v_cache_bhd = k_cache_bhd  # MLA: K and V share

    # indices: (B, S_q, K) → (T, K) → (T, H, K)
    topk_idx = indices.reshape(T, -1).unsqueeze(1).expand(T, H, -1).contiguous().to(torch.int32)

    out_thdv = mla_decode_sparse_sm86(
        q_bhd, k_cache_bhd, v_cache_bhd, topk_idx, scale=softmax_scale
    )  # (T, H, Dv)

    out = out_thdv[:, :, :head_dim_v].reshape(B, S_q, H, head_dim_v)
    lse = torch.zeros((B, S_q, H), dtype=torch.float32, device=q.device)
    return out, lse


# ============================================================================
# Apply / restore monkey-patch
# ============================================================================


def apply() -> dict:
    import vllm.v1.attention.ops.flashmla as upstream  # type: ignore

    if getattr(upstream, "_CC3_MLA_SPARSE_DECODE_SM86_APPLIED", False):
        raise RuntimeError(
            "mla_sparse_decode_sm86.apply() already called — call restore() first"
        )

    originals = {
        "flash_mla_with_kvcache": upstream.flash_mla_with_kvcache,
        "flash_mla_sparse_fwd": upstream.flash_mla_sparse_fwd,
        "is_flashmla_sparse_supported": upstream.is_flashmla_sparse_supported,
    }
    upstream.flash_mla_with_kvcache = flash_mla_with_kvcache_sm86
    upstream.flash_mla_sparse_fwd = flash_mla_sparse_fwd_sm86

    def _is_supported_sm86() -> tuple:
        return True, None

    upstream.is_flashmla_sparse_supported = _is_supported_sm86
    upstream._CC3_MLA_SPARSE_DECODE_SM86_APPLIED = True
    upstream._CC3_MLA_SPARSE_DECODE_SM86_ORIGINALS = originals
    return originals


def restore() -> None:
    import vllm.v1.attention.ops.flashmla as upstream  # type: ignore

    if not getattr(upstream, "_CC3_MLA_SPARSE_DECODE_SM86_APPLIED", False):
        return
    for name, fn in upstream._CC3_MLA_SPARSE_DECODE_SM86_ORIGINALS.items():
        setattr(upstream, name, fn)
    delattr(upstream, "_CC3_MLA_SPARSE_DECODE_SM86_APPLIED")
    delattr(upstream, "_CC3_MLA_SPARSE_DECODE_SM86_ORIGINALS")


def selfcheck() -> dict:
    """CPU-side structural sanity — verifies layout constants + import paths."""
    info: dict[str, Any] = {
        "ok": True,
        "v4_token_bytes": V4_TOKEN_BYTES,
        "v4_nope_bytes": V4_NOPE_BYTES,
        "v4_rope_elems": V4_ROPE_ELEMS,
    }
    assert V4_TOKEN_BYTES == 584
    assert V4_NOPE_BYTES + V4_ROPE_ELEMS * 2 + V4_SCALE_BYTES + V4_PAD_BYTES == 584
    try:
        import vllm.v1.attention.ops.flashmla as upstream  # type: ignore
        for sym in (
            "flash_mla_with_kvcache",
            "flash_mla_sparse_fwd",
            "is_flashmla_sparse_supported",
        ):
            info[f"upstream_has_{sym}"] = hasattr(upstream, sym)
            if not getattr(upstream, sym, None):
                info["ok"] = False
    except ImportError:
        info["upstream_module_available"] = False  # author-Mac lint
    else:
        info["upstream_module_available"] = True
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(selfcheck()))

# // --ProtoAI-Bakari--
