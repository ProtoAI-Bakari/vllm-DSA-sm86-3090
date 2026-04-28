# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""Lightning Indexer Q-side fp8e4nv emulation for sm_86 — Story 4.

Story-4 of the CC3 lane (now CC1 kernel-master post-rename), parallel to CC4's
compressor_sm86 port. Replaces the three ``tl.float8e4nv`` casts in
``vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q._fused_indexer_q_rope_quant_kernel``
(lines 154/159/163 of the upstream file) with the same software fp32→e4m3
encoder CC4 ships in ``compressor_sm86._fp32_to_e4m3_uint8``.

Why this is the same pattern as CC4 compressor:
  - Both kernels do scaled-fp32 → e4m3 quantization for the FP8 KV / Q cache.
  - Both fail at compile time on sm_86 with
      "PTX ISA does not support .e4m3 on sm_86"
    when emitting ``cvt.rn.satfinite.e4m3.f32``.
  - Both can be repaired by emitting the bit pattern in scalar arithmetic.

Byte layout written to the paged FP8 Q cache is bit-equivalent for all
*normal* e4m3 values (exp 1..15, mantissa any). Subnormals (exp 0,
mantissa non-zero) flush to ±0 — documented and gated by Story 7 numerics.

The MXFP4 indexer-Q path (``_fused_indexer_q_rope_mxfp4_kernel``) does NOT use
``tl.float8e4nv`` so it is left unpatched.

Apply: ``import this_module; this_module.apply()`` after vLLM imports
``deepseek_v4_ops.fused_indexer_q``. Idempotent guard prevents double-wrap.
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


# Reuse the same encoder CC4 ships. Lazy-imported so apply()-time order doesn't
# force a hard dependency on CC4 lane being installed first.
def _import_fp32_to_e4m3_uint8():
    try:
        from .compressor_sm86 import _fp32_to_e4m3_uint8  # type: ignore
        return _fp32_to_e4m3_uint8
    except ImportError:
        # Fallback: import the file by path if the package layout differs.
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
    def _get_cos_sin(
        cos_sin_cache_ptr,
        cos_sin_cache_stride,
        pos,
        HALF_ROT_DIM: tl.constexpr,
    ):
        block = tl.arange(0, HALF_ROT_DIM)
        cos = tl.load(
            cos_sin_cache_ptr + pos * cos_sin_cache_stride + block
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM
        ).to(tl.float32)
        return cos, sin

    @triton.jit
    def _fused_indexer_q_rope_quant_kernel_sm86(
        pos_ptr,
        # Index Q RoPE
        index_q_ptr,
        index_q_stride0,
        index_q_stride1,
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        INDEX_Q_HALF_ROT_DIM: tl.constexpr,
        # Index Q Quantize — FP8 cache reinterpreted as uint8 for sm_86 store path
        index_q_fp8_ptr,
        index_q_fp8_stride0,
        index_q_fp8_stride1,
        INDEX_Q_HEAD_DIM: tl.constexpr,
        # Index weights
        index_weights_ptr,
        index_weights_stride,
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out_ptr,
        index_weights_out_stride,
    ):
        INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
        INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
        tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

        tok_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            index_q_cos_sin_ptr,
            index_q_cos_sin_stride,
            pos,
            INDEX_Q_HALF_ROT_DIM,
        )
        half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
        base_ptr = (
            index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
        )

        # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
        rot_base = base_ptr + INDEX_Q_NOPE_DIM
        x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
        x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
        r_even = x_even * cos - x_odd * sin
        r_odd = x_odd * cos + x_even * sin

        # Match upstream numerics: bf16 round-trip before absmax computation.
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
        if INDEX_Q_NOPE_DIM > 0:
            nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
            x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
        index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
        index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

        # ─── sm_86 substitution: software fp8e4nv encoding ────────────────
        # Original (sm_90+): tl.store(..., x.to(tl.float8e4nv))
        # Replacement: encode fp32 → uint8 then store via uint8-view of fp8 ptr.
        # The caller passes index_q_fp8 reinterpreted as uint8 (.view(torch.uint8))
        # so the byte layout written here matches the upstream e4m3 storage.
        fp8_base_ptr = (
            index_q_fp8_ptr
            + tok_idx * index_q_fp8_stride0
            + head_idx * index_q_fp8_stride1
        )

        if INDEX_Q_NOPE_DIM > 0:
            nope_q_scaled = tl.div_rn(x_nope, index_q_scale)
            nope_u8 = _fp32_to_e4m3_uint8(nope_q_scaled)
            tl.store(fp8_base_ptr + nope_offset, nope_u8)

        fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
        even_scaled = tl.div_rn(r_even, index_q_scale)
        odd_scaled = tl.div_rn(r_odd, index_q_scale)
        even_u8 = _fp32_to_e4m3_uint8(even_scaled)
        odd_u8 = _fp32_to_e4m3_uint8(odd_scaled)
        tl.store(fp8_rot_base + half_offset * 2, even_u8)
        tl.store(fp8_rot_base + half_offset * 2 + 1, odd_u8)
        # ──────────────────────────────────────────────────────────────────

        # FP8 weight-fold contract — unchanged from upstream.
        index_weights = tl.load(
            index_weights_ptr + tok_idx * index_weights_stride + head_idx
        ).to(tl.float32)
        index_weights *= index_q_scale
        index_weights *= index_weights_softmax_scale
        index_weights *= index_weights_head_scale
        tl.store(
            index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
            index_weights,
        )


# ============================================================================
# Python-side launch wrapper — replaces the FP8 path of
# fused_indexer_q.fused_indexer_q_rope_quant. The MXFP4 path is unchanged
# (no float8e4nv casts), so we delegate it to the original.
# ============================================================================


def _fused_indexer_q_rope_quant_sm86(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    import torch

    if use_fp4:
        # Delegate to the original (MXFP4 path needs no patch).
        from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (  # type: ignore
            fused_indexer_q_rope_quant as _orig,
        )
        return _orig(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            use_fp4=True,
        )

    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    # Allocate fp8_e4m3 then reinterpret as uint8 for the kernel's byte writes.
    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    index_q_fp8_u8 = index_q_fp8.view(torch.uint8)

    _fused_indexer_q_rope_quant_kernel_sm86[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_fp8_u8,
        index_q_fp8_u8.stride(0),
        index_q_fp8_u8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,
    )
    return index_q_fp8, index_weights_out


# ============================================================================
# Apply / restore monkey-patch
# ============================================================================


def apply() -> dict:
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q as upstream  # type: ignore

    if getattr(upstream, "_CC3_LIGHTNING_INDEXER_SM86_APPLIED", False):
        raise RuntimeError(
            "lightning_indexer_sm86.apply() already called — "
            "call restore() first or guard at call site"
        )

    originals = {
        "fused_indexer_q_rope_quant": upstream.fused_indexer_q_rope_quant,
        "_fused_indexer_q_rope_quant_kernel": upstream._fused_indexer_q_rope_quant_kernel,
    }
    upstream.fused_indexer_q_rope_quant = _fused_indexer_q_rope_quant_sm86
    if triton is not None:
        upstream._fused_indexer_q_rope_quant_kernel = (
            _fused_indexer_q_rope_quant_kernel_sm86
        )
    upstream._CC3_LIGHTNING_INDEXER_SM86_APPLIED = True
    upstream._CC3_LIGHTNING_INDEXER_SM86_ORIGINALS = originals
    return originals


def restore() -> None:
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q as upstream  # type: ignore

    if not getattr(upstream, "_CC3_LIGHTNING_INDEXER_SM86_APPLIED", False):
        return
    for name, fn in upstream._CC3_LIGHTNING_INDEXER_SM86_ORIGINALS.items():
        setattr(upstream, name, fn)
    delattr(upstream, "_CC3_LIGHTNING_INDEXER_SM86_APPLIED")
    delattr(upstream, "_CC3_LIGHTNING_INDEXER_SM86_ORIGINALS")


def selfcheck() -> dict:
    """Structural sanity — verifies the substitution points exist in upstream
    and that the encoder import path resolves. Skipped if vllm not available
    (so author Mac can lint without full vllm install)."""
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
        import vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q as upstream  # type: ignore
        for sym in (
            "fused_indexer_q_rope_quant",
            "_fused_indexer_q_rope_quant_kernel",
            "_fused_indexer_q_rope_mxfp4_kernel",
        ):
            info[f"upstream_has_{sym}"] = hasattr(upstream, sym)
            if not getattr(upstream, sym, None):
                info["ok"] = False
    except ImportError:
        info["upstream_module_available"] = False  # author-Mac lint path
    else:
        info["upstream_module_available"] = True
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(selfcheck()))

# // --ProtoAI-Bakari--
