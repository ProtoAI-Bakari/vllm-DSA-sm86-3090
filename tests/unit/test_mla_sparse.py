"""
CC4 Story 7: L1 numerics gate for mla_sparse sm_86 decode kernel.

Reference: pure-PyTorch sparse online-softmax over the same indices and K rows.
Gate: cosine >= 0.97, max-abs-diff <= 1e-2, no NaN, no Inf.

Backlog says `test_mla_sparse.cu`; actual filename is `.py` because the L1
gate is a pure-Python reference comparison (no CUDA driver needed beyond the
extension import). The full integration test (lighting bundle, Story 11) is
the .cu equivalent for CC9's L4 driver.

Run: pytest tests/unit/test_mla_sparse.py -v -s
     OR python tests/unit/test_mla_sparse.py

// --ProtoAI-Bakari--
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Re-use CC4 compressor patch's e4m3 helpers for cache encoding.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "csrc_patches"))
import compressor_sm86 as cmp  # noqa: E402

CUDA = torch.cuda.is_available()
HAS_OP = False
if CUDA:
    try:
        torch.ops.load_library  # ensure ops API exists
        if hasattr(torch.ops, "_dsa_sm86") and hasattr(
            torch.ops._dsa_sm86, "mla_sparse_decode_sm86"
        ):
            HAS_OP = True
    except Exception:
        HAS_OP = False


# =============================================================================
# Test fixture: small DSV4-like geometry
# =============================================================================
HEAD_DIM   = 512
NOPE_DIM   = 448
ROPE_DIM   = 64
QUANT_BLK  = 64
TOKEN_B    = 584   # bytes/token
N_QUANT    = NOPE_DIM // QUANT_BLK  # 7


def _encode_kv_token(k_row_fp32: torch.Tensor) -> torch.Tensor:
    """Pack one D=512 fp32 K row into the 584-byte DSV4 paged-cache layout.

    Layout (matches `flashmla_sparse.py:81-89`):
      [0..448)   : NoPE 448 fp8e4m3 bytes
      [448..576) : RoPE 64 bf16 values (128 bytes)
      [576..584) : 7 ue8m0 scale bytes + 1 pad
    """
    assert k_row_fp32.shape == (HEAD_DIM,)
    out = torch.zeros(TOKEN_B, dtype=torch.uint8)

    nope = k_row_fp32[:NOPE_DIM].clone()
    rope = k_row_fp32[NOPE_DIM:].clone()  # 64

    # Per-block UE8M0 absmax + e4m3 quant (matches compressor pipeline).
    nope2d = nope.reshape(N_QUANT, QUANT_BLK)
    absmax = nope2d.abs().max(dim=1).values
    absmax = torch.maximum(absmax, torch.tensor(1e-4))
    raw = absmax / 448.0
    exponents = torch.ceil(torch.log2(raw))           # ue8m0 unbiased
    inv_scales = torch.exp2(-exponents)
    nope_scaled = nope2d * inv_scales.unsqueeze(1)
    nope_clamped = torch.clamp(nope_scaled, -448.0, 448.0)
    nope_bytes = cmp._python_e4m3_encode(nope_clamped.reshape(-1).contiguous())

    out[:NOPE_DIM] = nope_bytes

    # RoPE: bf16 raw store
    rope_bf16 = rope.to(torch.bfloat16)
    rope_bytes = rope_bf16.view(torch.uint8)  # little-endian on x86; matches CUDA layout
    out[NOPE_DIM:NOPE_DIM + ROPE_DIM * 2] = rope_bytes

    # Scale bytes: ue8m0 = exponent + 127, clamped
    enc_scales = (exponents + 127.0).clamp_(0, 255).to(torch.uint8)
    out[576:576 + N_QUANT] = enc_scales
    out[576 + N_QUANT] = 0  # pad

    return out


def _decode_kv_token(token_bytes: torch.Tensor) -> torch.Tensor:
    """Inverse of `_encode_kv_token` — used by the reference sparse-attn path."""
    out = torch.zeros(HEAD_DIM, dtype=torch.float32)
    nope_bytes = token_bytes[:NOPE_DIM]
    decoded_unscaled = cmp._python_e4m3_decode(nope_bytes)  # fp32, in [-448, 448]
    scale_bytes = token_bytes[576:576 + N_QUANT]
    exponents = scale_bytes.to(torch.float32) - 127.0
    scales = torch.exp2(exponents)
    nope_unscaled = decoded_unscaled.reshape(N_QUANT, QUANT_BLK)
    nope = nope_unscaled * scales.unsqueeze(1)
    out[:NOPE_DIM] = nope.reshape(-1)

    rope_bytes = token_bytes[NOPE_DIM:NOPE_DIM + ROPE_DIM * 2]
    rope_bf16 = rope_bytes.view(torch.bfloat16)
    out[NOPE_DIM:] = rope_bf16.to(torch.float32)
    return out


# =============================================================================
# Reference sparse online-softmax decode
# =============================================================================
def _reference_sparse_decode(
    q: torch.Tensor,            # [B, H, D] fp32
    kv_cache: torch.Tensor,     # [num_blocks, block_size, 584] uint8
    block_table: torch.Tensor,  # [B, max_blocks] int32
    topk_idx: torch.Tensor,     # [B, H, K] int32
    block_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    B, H, D = q.shape
    K = topk_idx.shape[-1]
    out = torch.zeros((B, H, D), dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            qrow = q[b, h]
            m_i = -math.inf
            l_i = 0.0
            acc = torch.zeros(D, dtype=torch.float32)
            any_valid = False
            for k in range(K):
                logical = int(topk_idx[b, h, k].item())
                if logical < 0:
                    continue
                blk_idx = logical // block_size
                slot    = logical % block_size
                phys = int(block_table[b, blk_idx].item())
                if phys < 0:
                    continue
                token = kv_cache[phys, slot]  # [584] uint8
                k_row = _decode_kv_token(token)
                qk = (qrow * k_row).sum().item() * softmax_scale
                m_new = max(m_i, qk)
                alpha = math.exp(m_i - m_new) if m_i != -math.inf else 0.0
                p = math.exp(qk - m_new)
                acc = acc * alpha + p * k_row
                l_i = l_i * alpha + p
                m_i = m_new
                any_valid = True
            if any_valid:
                out[b, h] = acc / l_i
    return out


# =============================================================================
# Tests
# =============================================================================
def _build_fixture(B=2, H=4, K=16, num_blocks=8, block_size=64, seed=42):
    """Build a small fixture: random Q + paged KV cache with random K rows."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, HEAD_DIM, dtype=torch.float32) * 0.5

    # Build paged cache: num_blocks * block_size random K rows encoded.
    kv_cache = torch.zeros((num_blocks, block_size, TOKEN_B), dtype=torch.uint8)
    for nb in range(num_blocks):
        for s in range(block_size):
            k_row = torch.randn(HEAD_DIM, dtype=torch.float32) * 0.5
            kv_cache[nb, s] = _encode_kv_token(k_row)

    # Block table: each request gets sequential blocks 0..max_blocks-1.
    max_blocks = num_blocks
    block_table = torch.arange(num_blocks, dtype=torch.int32).repeat(B, 1)

    # Topk indices: random valid logical indices in [0, num_blocks * block_size).
    max_logical = num_blocks * block_size
    topk_idx = torch.randint(0, max_logical, (B, H, K), dtype=torch.int32)
    # Sprinkle a few -1 invalids
    topk_idx[0, 0, 0] = -1
    topk_idx[1, 2, 5] = -1

    return q, kv_cache, block_table, topk_idx


def test_reference_self_consistency():
    """Reference path returns finite, non-NaN outputs on the fixture."""
    q, kv, bt, ti = _build_fixture()
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = _reference_sparse_decode(q, kv, bt, ti, block_size=64, softmax_scale=softmax_scale)
    assert torch.isfinite(out).all(), "reference produced NaN/Inf"
    # Non-trivial output (at least some non-zero)
    assert out.abs().sum().item() > 0


def test_encode_decode_roundtrip_kv_layout():
    """The 584B layout encode/decode round-trips with cosine >= 0.97."""
    torch.manual_seed(0)
    k_row = torch.randn(HEAD_DIM, dtype=torch.float32) * 0.7
    enc = _encode_kv_token(k_row)
    dec = _decode_kv_token(enc)
    cos = torch.nn.functional.cosine_similarity(k_row, dec, dim=0).item()
    assert cos >= 0.97, f"KV layout round-trip cosine {cos:.6f} below 0.97"


@pytest.mark.skipif(not (CUDA and HAS_OP), reason="needs CUDA + _dsa_sm86 ext built")
def test_kernel_matches_reference_cosine():
    """Story 6 kernel output cosine >= 0.97 vs reference on fixture."""
    q, kv, bt, ti = _build_fixture()
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    q_cuda  = q.to(torch.bfloat16).cuda().contiguous()
    kv_cuda = kv.cuda().contiguous()
    bt_cuda = bt.cuda().contiguous()
    ti_cuda = ti.cuda().contiguous()

    out_kernel = torch.ops._dsa_sm86.mla_sparse_decode_sm86(
        q_cuda, kv_cuda, bt_cuda, ti_cuda, 64, softmax_scale,
    ).float().cpu()

    out_ref = _reference_sparse_decode(q, kv, bt, ti, 64, softmax_scale)

    a = out_kernel.reshape(-1)
    b = out_ref.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()

    assert torch.isfinite(out_kernel).all(), "kernel produced NaN/Inf"
    assert cos >= 0.97, f"kernel-vs-reference cosine {cos:.6f} below 0.97"
    assert max_abs <= 1e-2, f"kernel-vs-reference max-abs-diff {max_abs:.6e} above 1e-2"


def test_topk_all_invalid_returns_zero():
    """If every topk index is -1, output must be zero."""
    q, kv, bt, _ = _build_fixture()
    B, H, _ = q.shape
    K = 16
    ti = torch.full((B, H, K), -1, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = _reference_sparse_decode(q, kv, bt, ti, 64, softmax_scale)
    assert torch.allclose(out, torch.zeros_like(out)), "all-invalid topk should produce zero output"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x", "-s"]))
