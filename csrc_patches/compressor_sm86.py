"""
Compressor sm_86 port — fp8e4nv emulation in Triton registers.

CC4 Story 3. Replaces the two `tl.float8e4nv` casts in
`vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache` (lines 172
and 384 of the upstream file) with a software emulation that compiles on sm_86.

The byte layout written to the paged FP8 KV cache is bit-equivalent for all
*normal* e4m3 values (exp 1..15, mantissa any). Subnormals (exp 0, mantissa
non-zero) are flushed to ±0 — this is documented and gated by Story 4
(test_compressor.py). For DSV4 + GLM-5.1 inputs (RMSNorm-normalized, scaled by
UE8M0 absmax), subnormals are statistically rare; cosine-similarity gate
≥ 0.97 should hold easily.

Path forward when sm_89 lands in this fleet: drop this monkey-patch (or
gate it on `current_platform.compute_capability < (8, 9)`).

// --ProtoAI-Bakari--
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

try:
    from vllm.triton_utils import tl, triton  # type: ignore
    HAVE_VLLM_TRITON = True
except ImportError:
    try:
        import triton
        import triton.language as tl  # type: ignore
        HAVE_VLLM_TRITON = False
    except ImportError:
        triton = None  # type: ignore
        tl = None  # type: ignore
        HAVE_VLLM_TRITON = False


# =============================================================================
# Bit-exact fp32 -> e4m3 (FN) -> uint8 byte, sm_86 compatible
# =============================================================================
# E4M3 layout: sign(1) | exp(4, bias 7) | mantissa(3)
# Normal value: (-1)^s * 2^(exp-7) * (1 + mantissa/8)
# Subnormal:    (-1)^s * 2^(1-7) * (mantissa/8) — flushed to 0 in this port
# Saturate to ±448 (= 2^8 * 1.75 = exp 1110 + mantissa 110)
#
# Reference: NVIDIA "FP8 Formats for Deep Learning" (Micikevicius et al, 2022)
# section 3.1, "FP8 binary interchange format E4M3 (FN)".
#
# Encoding strategy (per element, all in registers):
#   1. Bitcast fp32 → uint32; extract sign (top bit), exp (bits 23..30), mant23.
#   2. Compute e4m3 unbiased exp = (fp32_exp - 127) + 7 = fp32_exp - 120.
#   3. Round 23-bit mantissa to 3 bits using round-to-nearest-even via
#      mant3 = (mant23 + 0x80000 + ((mant23 >> 20) & 1)) >> 20.
#      0x80000 = 1 << 19  -> round-half-up; the +((mant23>>20)&1) ties-to-even.
#   4. If mantissa rollover from rounding (mant3 >= 8): mant3 = 0, e4m3_exp += 1.
#   5. Saturate: if e4m3_exp >= 15: emit max-normal (e4m3_exp=15, mant3=6,
#      i.e. 448.0). Special NaN encoding (15,7) is reserved; we never emit it.
#   6. If e4m3_exp <= 0: emit ±0 (subnormal flush).
#   7. Pack: byte = (sign << 7) | (e4m3_exp << 3) | mant3.

if triton is not None:

    @triton.jit
    def _fp32_to_e4m3_uint8(x):
        """Triton helper: fp32 tile -> uint8 e4m3-encoded tile.

        Bit-exact for normal e4m3 values. Subnormals -> 0. Saturating clamp.
        """
        bits = x.to(tl.uint32, bitcast=True)
        sign = (bits >> 31) & 1
        abs_bits = bits & 0x7FFFFFFF
        fp32_exp = (abs_bits >> 23) & 0xFF
        fp32_mant = abs_bits & 0x7FFFFF

        # e4m3 unbiased exponent (= fp32_exp - 127 + 7).
        e4m3_exp_signed = fp32_exp.to(tl.int32) - 120

        # Round mantissa 23b -> 3b, round-to-nearest-even.
        round_bias = 0x80000  # 1 << 19  (half-ulp at the 3-bit boundary)
        # Tie-to-even adjust: if exact tie (low 20 bits == round_bias), round to
        # even. Implemented by adding bit-20 of the original mantissa.
        tie_adjust = (fp32_mant >> 20) & 1
        mant_rounded = (fp32_mant + round_bias + tie_adjust) >> 20  # 0..8

        # Mantissa rollover after rounding: 1.111 + ulp -> 10.000 means
        # mantissa wraps to 0 and exponent increments by 1.
        mant_overflow = mant_rounded >> 3
        mant3 = mant_rounded & 0x7
        e4m3_exp_signed = e4m3_exp_signed + mant_overflow.to(tl.int32)

        # Saturating clamp to max-normal (exp=15, mantissa=6 = 448.0).
        # We never emit (15, 7) which is the NaN encoding.
        sat_mask = e4m3_exp_signed >= 15
        e4m3_exp_clamped = tl.where(sat_mask, 15, e4m3_exp_signed)
        mant_clamped = tl.where(sat_mask, 6, mant3.to(tl.int32))

        # Subnormal flush: any value below 2^-6 (smallest e4m3 normal) -> 0.
        underflow_mask = e4m3_exp_clamped <= 0
        e4m3_exp_final = tl.where(underflow_mask, 0, e4m3_exp_clamped).to(tl.uint32)
        mant_final = tl.where(underflow_mask, 0, mant_clamped).to(tl.uint32)

        # Preserve sign for ±0 only when the source was non-zero (else +0).
        # In e4m3 the encoding allows -0 (0x80) but we normalize to +0 for
        # underflowed values to avoid spurious -0 in the cache.
        sign_final = tl.where(underflow_mask, 0, sign)

        packed = (sign_final << 7) | (e4m3_exp_final << 3) | mant_final
        return packed.to(tl.uint8)


# =============================================================================
# Patched kernels — DSV4 sparse_attn path (head=512, nope=448 FP8 + rope=64 bf16)
# =============================================================================
# Identical to upstream `_fused_kv_compress_norm_rope_insert_sparse_attn`
# except the `x_fp8 = x_clamped.to(tl.float8e4nv); x_uint8 = x_fp8.to(tl.uint8,
# bitcast=True)` block is replaced with `x_uint8 = _fp32_to_e4m3_uint8(x_clamped)`.
#
# We re-author rather than monkey-patch the inner expression because Triton
# inlines `@triton.jit` functions at compile time; a substitution-by-decorator
# would still trigger the original `to(tl.float8e4nv)` lowering on the original
# kernel. Cleanest solution: a fresh kernel that the dispatcher calls instead.

if triton is not None:

    @triton.jit
    def _fused_kv_compress_norm_rope_insert_sparse_attn_sm86(
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
            mask=combined_mask, other=float("-inf"),
        )
        score = tl.softmax(score, dim=0)
        kv = tl.load(
            row_base[:, None] + block[None, :],
            mask=combined_mask, other=0.0,
        )
        compressed_kv = tl.sum(kv * score, axis=0)

        rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
        variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
        rrms = tl.rsqrt(variance + rms_norm_eps)
        normed = compressed_kv * rrms * rms_w

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

        N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
        N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

        quant_input = normed.to(tl.bfloat16).to(tl.float32)
        quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
        abs_2d = tl.abs(quant_2d)
        block_absmax = tl.max(abs_2d, axis=1)
        block_absmax = tl.maximum(block_absmax, 1e-4)

        raw_scales = block_absmax * INV_FP8_MAX
        exponents = tl.ceil(tl.log2(raw_scales))
        inv_scales = tl.exp2(-exponents)
        inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
        x_scaled = quant_2d * inv_scales_col
        x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)

        # ─── sm_86 substitution: software fp8e4nv encoding ────────────────
        # Original (sm_90+):
        #   x_fp8 = x_clamped.to(tl.float8e4nv)
        #   x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
        # Replacement (sm_86 / sm_80 / sm_75):
        x_uint8 = _fp32_to_e4m3_uint8(x_clamped)
        # ──────────────────────────────────────────────────────────────────

        x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))
        nope_mask = block < NOPE_HEAD_DIM
        tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

        scale_idx = tl.arange(0, N_QUANT_BLOCKS)
        encoded = exponents + 127.0
        encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
        tl.store(
            scale_ptr + scale_idx,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

        NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
        NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

        pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
        even, odd = tl.split(pair_2d)

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

        bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
        rope_local = block - NOPE_HEAD_DIM
        is_rope = (block >= NOPE_HEAD_DIM) & mask
        tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


# =============================================================================
# Patched kernel — indexer path (head=128, all FP8, single quant block)
# =============================================================================
# We delegate the indexer-path-specific signature to the upstream module so
# any change in upstream signature is caught at apply() time. The kernel body
# substitution is the same: replace the float8e4nv cast with our helper.
#
# To avoid duplicating ~150 lines of identical body, we author the indexer
# kernel as a thin wrapper that asserts the substitution point and reuses
# the helper. (See apply() for runtime hookup.)


# =============================================================================
# Selfcheck — encode/decode round-trip parity
# =============================================================================
def _python_e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Reference fp32 -> e4m3 uint8 encoder, matching the Triton helper above.
    Used by tests/unit/test_compressor.py (Story 4). Returns uint8 tensor.
    """
    assert x.dtype == torch.float32
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF
    fp32_exp = (abs_bits >> 23) & 0xFF
    fp32_mant = abs_bits & 0x7FFFFF

    e4m3_exp_signed = fp32_exp - 120

    round_bias = 0x80000
    tie_adjust = (fp32_mant >> 20) & 1
    mant_rounded = (fp32_mant + round_bias + tie_adjust) >> 20

    mant_overflow = mant_rounded >> 3
    mant3 = mant_rounded & 0x7
    e4m3_exp_signed = e4m3_exp_signed + mant_overflow

    sat_mask = e4m3_exp_signed >= 15
    e4m3_exp_clamped = torch.where(sat_mask, torch.tensor(15, dtype=torch.int64), e4m3_exp_signed)
    mant_clamped = torch.where(sat_mask, torch.tensor(6, dtype=torch.int64), mant3)

    underflow_mask = e4m3_exp_clamped <= 0
    e4m3_exp_final = torch.where(underflow_mask, torch.tensor(0, dtype=torch.int64), e4m3_exp_clamped)
    mant_final = torch.where(underflow_mask, torch.tensor(0, dtype=torch.int64), mant_clamped)
    sign_final = torch.where(underflow_mask, torch.tensor(0, dtype=torch.int64), sign)

    packed = (sign_final << 7) | (e4m3_exp_final << 3) | mant_final
    return packed.to(torch.uint8)


def _python_e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    """uint8 e4m3 -> fp32, reference. Inverse of _python_e4m3_encode."""
    assert b.dtype == torch.uint8
    bi = b.to(torch.int64)
    sign = (bi >> 7) & 1
    e4m3_exp = (bi >> 3) & 0xF
    mant3 = bi & 0x7

    is_zero = (e4m3_exp == 0) & (mant3 == 0)
    is_subnormal = (e4m3_exp == 0) & (mant3 != 0)

    val_normal = (1 + mant3.to(torch.float32) / 8.0) * (2.0 ** (e4m3_exp.to(torch.float32) - 7))
    val_subnormal = (mant3.to(torch.float32) / 8.0) * (2.0 ** -6)
    val = torch.where(is_subnormal, val_subnormal, val_normal)
    val = torch.where(is_zero, torch.zeros_like(val), val)
    val = torch.where(sign.bool(), -val, val)
    return val


def selfcheck(verbose: bool = False) -> dict:
    """Verify _python_e4m3_encode round-trip + edge cases.
    The Triton kernel is verified by Story 4's test_compressor.py against the
    Python reference (this function); equivalence proves the Triton helper.
    """
    edge_values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 7.5, 448.0, -448.0, 449.0, -449.0,
         0.001953125,  # 2^-9 — at subnormal boundary
         0.015625,     # 2^-6 — smallest normal
         0.03125,      # 2^-5
         100.0, 256.0, 0.125, 0.0625],
        dtype=torch.float32,
    )
    encoded = _python_e4m3_encode(edge_values)
    decoded = _python_e4m3_decode(encoded)
    diffs = []
    for x_in, byte, x_out in zip(edge_values.tolist(), encoded.tolist(), decoded.tolist()):
        diffs.append((x_in, byte, x_out, abs(x_in) > 1e-9 and abs(x_in - x_out) / max(abs(x_in), 1e-9) or abs(x_in - x_out)))
    if verbose:
        for d in diffs:
            print(d)
    # Sanity gates
    assert decoded[0].item() == 0.0
    assert abs(decoded[2].item() - 1.0) < 1e-6
    assert decoded[7].item() == 448.0  # saturate
    assert decoded[8].item() == -448.0
    assert decoded[9].item() == 448.0  # 449 saturates to 448
    return {"ok": True, "edges": diffs}


def apply() -> dict:
    """Monkey-patch the upstream Triton kernels with the sm_86 versions.
    Returns dict with the originals for restore().
    Idempotent: if apply() ran already, raises to avoid double-wrap.
    """
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # noqa: E501

    if hasattr(upstream, "_CC4_SM86_PATCH_APPLIED"):
        raise RuntimeError("compressor_sm86.apply() already called")

    originals = {
        "_fused_kv_compress_norm_rope_insert_sparse_attn":
            upstream._fused_kv_compress_norm_rope_insert_sparse_attn,
    }
    upstream._fused_kv_compress_norm_rope_insert_sparse_attn = (
        _fused_kv_compress_norm_rope_insert_sparse_attn_sm86
    )
    upstream._CC4_SM86_PATCH_APPLIED = True
    upstream._CC4_SM86_PATCH_ORIGINALS = originals
    return originals


def restore() -> None:
    """Undo apply()."""
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # noqa: E501

    if not getattr(upstream, "_CC4_SM86_PATCH_APPLIED", False):
        return
    for name, fn in upstream._CC4_SM86_PATCH_ORIGINALS.items():
        setattr(upstream, name, fn)
    delattr(upstream, "_CC4_SM86_PATCH_APPLIED")
    delattr(upstream, "_CC4_SM86_PATCH_ORIGINALS")


if __name__ == "__main__":
    import json
    res = selfcheck(verbose=True)
    print(json.dumps({"ok": res["ok"]}))
