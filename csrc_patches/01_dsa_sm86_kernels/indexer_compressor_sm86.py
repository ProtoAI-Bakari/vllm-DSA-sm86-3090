# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""Indexer-path compressor sm_86 port — Story 6 ("Hopper compressor port").

Companion to CC4's ``compressor_sm86.py`` (which patched the *sparse_attn*
variant of the upstream ``_fused_kv_compress_norm_rope_insert_*`` kernel
family). This file patches the **indexer-path** variant
(``_fused_kv_compress_norm_rope_insert_indexer_attn``) which shares the
same float8e4nv-cast wall but with a different signature:

  - HEAD_SIZE = 128 (vs 512 for sparse_attn)
  - QUANT_BLOCK == TRITON_BLOCK_SIZE (single quant block, flat reduction)
  - SCALE_DIM = 4 bytes (one float32 scale per token)
  - Uses ``tl.exp2`` to encode the scale as float32 directly

Same software fp8e4nv encoder as the sparse_attn port (reused from CC4's
``compressor_sm86._fp32_to_e4m3_uint8``).

Apply:
  ``import this_module; this_module.apply()``
after vLLM imports ``deepseek_v4_ops.fused_compress_quant_cache``. The MXFP4
indexer variant (``_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn``)
does NOT use float8e4nv and is left unpatched.
"""

from __future__ import annotations

from typing import Any

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


def _import_fp32_to_e4m3_uint8():
    try:
        from .compressor_sm86 import _fp32_to_e4m3_uint8  # type: ignore
        return _fp32_to_e4m3_uint8
    except ImportError:
        import importlib.util
        import pathlib
        here = pathlib.Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "_compressor_sm86_local",
            here / "compressor_sm86.py",
        )
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._fp32_to_e4m3_uint8


if triton is not None:

    _fp32_to_e4m3_uint8 = _import_fp32_to_e4m3_uint8()

    @triton.jit
    def _fused_kv_compress_norm_rope_insert_indexer_attn_sm86(
        # ── state cache (compressor internal state) ──
        state_cache_ptr,
        state_cache_stride0,
        state_cache_stride1,
        # ── metadata ──
        token_to_req_indices_ptr,
        positions_ptr,
        slot_mapping_ptr,
        block_table_ptr,
        block_table_stride,
        block_size,
        # ── RMSNorm ──
        rms_norm_weight_ptr,
        rms_norm_eps,
        # ── RoPE ──
        cos_sin_cache_ptr,
        cos_sin_stride,
        # ── KV cache output ──
        k_cache_ptr,
        kv_slot_mapping_ptr,
        kv_cache_block_size,
        # ── constexprs ──
        HEAD_SIZE: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
        STATE_WIDTH: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        OVERLAP: tl.constexpr,
        ROPE_HEAD_DIM: tl.constexpr,
        FP8_MAX: tl.constexpr,
        QUANT_BLOCK: tl.constexpr,
        TOKEN_STRIDE: tl.constexpr,
        SCALE_DIM: tl.constexpr,
        KV_BLOCK_STRIDE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)

        slot_id = tl.load(slot_mapping_ptr + token_idx)
        if slot_id < 0:
            return

        position = tl.load(positions_ptr + token_idx)
        if (position + 1) % COMPRESS_RATIO != 0:
            return

        req_idx = tl.load(token_to_req_indices_ptr + token_idx)

        # Gather state cache entries.
        start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
        tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
        pos = start + tokens
        mask_pos = pos >= 0

        block_indices = pos // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask_pos,
            other=0,
        )
        block_offsets = pos % block_size
        head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

        block = tl.arange(0, TRITON_BLOCK_SIZE)
        mask = block < HEAD_SIZE
        block_numbers_i64 = block_numbers.to(tl.int64)

        row_base = (
            state_cache_ptr
            + block_numbers_i64 * state_cache_stride0
            + block_offsets * state_cache_stride1
            + head_offset
        )
        combined_mask = mask_pos[:, None] & mask[None, :]

        score = tl.load(
            row_base[:, None] + STATE_WIDTH + block[None, :],
            mask=combined_mask,
            other=float("-inf"),
        )
        score = tl.softmax(score, dim=0)
        kv = tl.load(
            row_base[:, None] + block[None, :],
            mask=combined_mask,
            other=0.0,
        )
        compressed_kv = tl.sum(kv * score, axis=0)

        # RMSNorm.
        rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
        variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
        rrms = tl.rsqrt(variance + rms_norm_eps)
        normed = compressed_kv * rrms * rms_w

        # KV cache pointers.
        kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
        if kv_slot_idx < 0:
            return
        kv_block_idx = kv_slot_idx // kv_cache_block_size
        kv_pos_in_block = kv_slot_idx % kv_cache_block_size

        cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
        fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
        scale_ptr = (
            cache_block_ptr
            + kv_cache_block_size * TOKEN_STRIDE
            + kv_pos_in_block * SCALE_DIM
        )

        NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
        HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

        NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
        NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

        normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
        even, odd = tl.split(normed_2d)

        pair_idx = tl.arange(0, NUM_PAIRS)
        rope_pair_local = pair_idx - NOPE_PAIRS
        is_rope_pair = rope_pair_local >= 0
        cs_idx = tl.maximum(rope_pair_local, 0)

        compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
        cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        result = tl.interleave(new_even, new_odd)

        # FP8 quant: indexer path is single quant block.
        tl.static_assert(
            TRITON_BLOCK_SIZE == QUANT_BLOCK,
            "Indexer compressor expects QUANT_BLOCK == TRITON_BLOCK_SIZE",
        )
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

        result_bf16 = result.to(tl.bfloat16).to(tl.float32)
        absmax = tl.max(tl.abs(result_bf16), axis=0)
        absmax = tl.maximum(absmax, 1e-4)
        raw_scale = absmax * INV_FP8_MAX
        exponent = tl.ceil(tl.log2(raw_scale))
        inv_scale = tl.exp2(-exponent)

        x_scaled = result_bf16 * inv_scale
        x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)

        # ─── sm_86 substitution: software fp8e4nv encoding ────────────────
        # Original (sm_90+):
        #   x_fp8 = x_clamped.to(tl.float8e4nv)
        #   x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
        # Replacement (sm_86):
        x_uint8 = _fp32_to_e4m3_uint8(x_clamped)
        # ──────────────────────────────────────────────────────────────────

        tl.store(fp8_ptr + block, x_uint8, mask=mask)

        # Single float32 scale (unchanged from upstream).
        scale_val = tl.exp2(exponent)
        tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)


# ============================================================================
# Apply / restore monkey-patch
# ============================================================================


def apply() -> dict:
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # type: ignore  # noqa: E501

    if getattr(upstream, "_CC3_INDEXER_COMPRESSOR_SM86_APPLIED", False):
        raise RuntimeError(
            "indexer_compressor_sm86.apply() already called — "
            "call restore() first or guard at call site"
        )

    originals = {
        "_fused_kv_compress_norm_rope_insert_indexer_attn":
            upstream._fused_kv_compress_norm_rope_insert_indexer_attn,
    }
    upstream._fused_kv_compress_norm_rope_insert_indexer_attn = (
        _fused_kv_compress_norm_rope_insert_indexer_attn_sm86
    )
    upstream._CC3_INDEXER_COMPRESSOR_SM86_APPLIED = True
    upstream._CC3_INDEXER_COMPRESSOR_SM86_ORIGINALS = originals
    return originals


def restore() -> None:
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # type: ignore  # noqa: E501

    if not getattr(upstream, "_CC3_INDEXER_COMPRESSOR_SM86_APPLIED", False):
        return
    for name, fn in upstream._CC3_INDEXER_COMPRESSOR_SM86_ORIGINALS.items():
        setattr(upstream, name, fn)
    delattr(upstream, "_CC3_INDEXER_COMPRESSOR_SM86_APPLIED")
    delattr(upstream, "_CC3_INDEXER_COMPRESSOR_SM86_ORIGINALS")


def selfcheck() -> dict:
    info: dict[str, Any] = {"ok": True}
    try:
        encoder = _import_fp32_to_e4m3_uint8()
        info["encoder_imported"] = True
        info["encoder_name"] = getattr(encoder, "__name__", "unknown")
    except Exception as exc:
        info["ok"] = False
        info["encoder_imported"] = False
        info["encoder_error"] = repr(exc)
    try:
        import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # type: ignore  # noqa: E501
        for sym in (
            "_fused_kv_compress_norm_rope_insert_indexer_attn",
            "_fused_kv_compress_norm_rope_insert_sparse_attn",
        ):
            info[f"upstream_has_{sym}"] = hasattr(upstream, sym)
            if not getattr(upstream, sym, None):
                info["ok"] = False
    except ImportError:
        info["upstream_module_available"] = False
    else:
        info["upstream_module_available"] = True
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(selfcheck()))

# // --ProtoAI-Bakari--
