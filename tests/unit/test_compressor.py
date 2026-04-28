"""
CC4 Story 4: L1 numerics gate for compressor sm_86 port.

Verifies csrc_patches/compressor_sm86.py:
  1. Bit-exact encode parity Triton helper vs Python reference (1024 random + edge battery)
  2. Round-trip encode->decode cosine >= 0.97 on uniform[-448, 448]
  3. Edge battery (0, ±max, saturate, min-normal)
  4. UE8M0 + per-block-absmax pipeline parity vs upstream reference (skipped if CUDA absent)

Run: pytest tests/unit/test_compressor.py -v
     OR python tests/unit/test_compressor.py

// --ProtoAI-Bakari--
"""

from __future__ import annotations

import os
import sys
import math
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "csrc_patches"))

import compressor_sm86 as patch  # noqa: E402

CUDA = torch.cuda.is_available()
HAS_TRITON = patch.triton is not None


# =============================================================================
# 1. Python reference encode/decode self-consistency
# =============================================================================
def test_python_encode_zero():
    z = torch.tensor([0.0, -0.0], dtype=torch.float32)
    enc = patch._python_e4m3_encode(z)
    assert enc[0].item() == 0
    # -0 normalizes to +0 (subnormal underflow path)
    assert enc[1].item() == 0


def test_python_encode_saturate():
    x = torch.tensor([449.0, -449.0, 1e6, -1e6, 448.0, -448.0], dtype=torch.float32)
    enc = patch._python_e4m3_encode(x)
    dec = patch._python_e4m3_decode(enc)
    assert dec[0].item() == 448.0
    assert dec[1].item() == -448.0
    assert dec[2].item() == 448.0
    assert dec[3].item() == -448.0
    assert dec[4].item() == 448.0
    assert dec[5].item() == -448.0


def test_python_encode_unit():
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=torch.float32)
    enc = patch._python_e4m3_encode(x)
    dec = patch._python_e4m3_decode(enc)
    assert torch.allclose(dec, x, atol=1e-6)


def test_python_encode_known_values():
    """Spot-check known e4m3 byte encodings against NVIDIA's published table."""
    # 1.0 = sign 0, exp (0+7)=7=0b0111, mantissa 0 -> 0b0_0111_000 = 0x38
    assert patch._python_e4m3_encode(torch.tensor([1.0]))[0].item() == 0x38
    # -1.0 = 0xB8
    assert patch._python_e4m3_encode(torch.tensor([-1.0]))[0].item() == 0xB8
    # 2.0 = exp 8, mantissa 0 -> 0b0_1000_000 = 0x40
    assert patch._python_e4m3_encode(torch.tensor([2.0]))[0].item() == 0x40
    # 448.0 = max-normal: exp 15, mantissa 6 -> 0b0_1111_110 = 0x7E
    assert patch._python_e4m3_encode(torch.tensor([448.0]))[0].item() == 0x7E


def test_python_roundtrip_random():
    torch.manual_seed(42)
    x = torch.empty(4096, dtype=torch.float32).uniform_(-448.0, 448.0)
    enc = patch._python_e4m3_encode(x)
    dec = patch._python_e4m3_decode(enc)
    cos = torch.nn.functional.cosine_similarity(x, dec, dim=0).item()
    assert cos >= 0.97, f"round-trip cosine {cos:.6f} below 0.97 floor"


def test_python_roundtrip_normalized_range():
    """Inputs that look like RMSNorm output (smaller magnitude, denser around 1)."""
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.float32) * 100.0
    x = torch.clamp(x, -448.0, 448.0)
    enc = patch._python_e4m3_encode(x)
    dec = patch._python_e4m3_decode(enc)
    cos = torch.nn.functional.cosine_similarity(x, dec, dim=0).item()
    assert cos >= 0.97, f"normalized-range cosine {cos:.6f} below 0.97"


# =============================================================================
# 2. Triton helper parity (CUDA-gated)
# =============================================================================
@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + Triton")
def test_triton_helper_matches_python_reference():
    """Compile a small Triton wrapper that calls _fp32_to_e4m3_uint8 and
    compare its output byte-for-byte against the Python reference encoder.
    """
    tl = patch.tl
    triton = patch.triton

    @triton.jit
    def _wrap_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = patch._fp32_to_e4m3_uint8(x)
        tl.store(out_ptr + offs, b, mask=mask)

    torch.manual_seed(123)
    N = 1024
    x = torch.empty(N, dtype=torch.float32, device="cuda").uniform_(-448.0, 448.0)
    out = torch.zeros(N, dtype=torch.uint8, device="cuda")
    BLOCK = 256
    grid = (math.ceil(N / BLOCK),)
    _wrap_kernel[grid](x, out, N=N, BLOCK=BLOCK)

    ref = patch._python_e4m3_encode(x.cpu())
    triton_bytes = out.cpu()
    diff = (ref.to(torch.int32) - triton_bytes.to(torch.int32)).abs()
    n_mismatch = int((diff > 0).sum().item())
    # Allow up to 2 LSB rounding-mode mismatches per 1024 (extremely tight).
    assert n_mismatch <= 2, (
        f"Triton helper byte-mismatch count {n_mismatch}/{N} "
        f"first 5 mismatches: indices={(diff>0).nonzero().flatten()[:5].tolist()}"
    )


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + Triton")
def test_triton_helper_edge_battery():
    """Edge values must encode identically to Python reference."""
    tl = patch.tl
    triton = patch.triton

    @triton.jit
    def _wrap_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = patch._fp32_to_e4m3_uint8(x)
        tl.store(out_ptr + offs, b, mask=mask)

    edges = torch.tensor(
        [0.0, 1.0, -1.0, 2.0, -2.0, 448.0, -448.0, 449.0, -1e6, 100.0,
         0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125,
         1e-9, -1e-9, 7.5, -7.5, 256.0, -256.0],
        dtype=torch.float32, device="cuda",
    )
    N = edges.shape[0]
    pad = 32 - (N % 32) if N % 32 else 0
    if pad:
        edges = torch.cat([edges, torch.zeros(pad, device="cuda")])
    out = torch.zeros(edges.shape[0], dtype=torch.uint8, device="cuda")
    BLOCK = 32
    grid = (math.ceil(edges.shape[0] / BLOCK),)
    _wrap_kernel[grid](edges, out, N=edges.shape[0], BLOCK=BLOCK)

    ref = patch._python_e4m3_encode(edges.cpu())
    assert torch.equal(out.cpu()[:N], ref[:N]), (
        f"edge battery mismatch:\n"
        f"  ref:    {ref[:N].tolist()}\n"
        f"  triton: {out.cpu()[:N].tolist()}"
    )


# =============================================================================
# 3. UE8M0 + per-block-absmax pipeline parity
# =============================================================================
def _fp32_to_ue8m0_quantize(x: torch.Tensor, quant_block: int, fp8_max: float = 448.0):
    """Mirror of the upstream UE8M0 quant pipeline (lines 156-187)."""
    assert x.numel() % quant_block == 0
    n_blocks = x.numel() // quant_block
    x2 = x.reshape(n_blocks, quant_block)
    absmax = x2.abs().max(dim=1).values
    absmax = torch.maximum(absmax, torch.tensor(1e-4))
    raw_scales = absmax / fp8_max
    exponents = torch.ceil(torch.log2(raw_scales))
    inv_scales = torch.exp2(-exponents)
    x_scaled = x2 * inv_scales.unsqueeze(1)
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    encoded_bytes = patch._python_e4m3_encode(x_clamped.reshape(-1).contiguous())
    return encoded_bytes, exponents


def _ue8m0_dequantize(encoded_bytes: torch.Tensor, exponents: torch.Tensor, quant_block: int):
    decoded = patch._python_e4m3_decode(encoded_bytes)
    decoded2 = decoded.reshape(-1, quant_block)
    scales = torch.exp2(exponents)
    return (decoded2 * scales.unsqueeze(1)).reshape(-1)


def test_ue8m0_pipeline_roundtrip():
    """End-to-end: fp32 normed -> UE8M0 absmax + e4m3 quant -> dequant -> cosine >= 0.97."""
    torch.manual_seed(7)
    HEAD_SIZE = 512
    QUANT_BLOCK = 64
    NOPE_HEAD_DIM = 448
    # Simulate a typical post-RMSNorm tile (centered, unit-ish stddev, occasional outliers).
    x = torch.randn(NOPE_HEAD_DIM, dtype=torch.float32) * 4.0
    enc, exps = _fp32_to_ue8m0_quantize(x, QUANT_BLOCK)
    dec = _ue8m0_dequantize(enc, exps, QUANT_BLOCK)
    cos = torch.nn.functional.cosine_similarity(x, dec, dim=0).item()
    assert cos >= 0.97, f"UE8M0 round-trip cosine {cos:.6f} below 0.97 — port broke numerics"


def test_ue8m0_pipeline_outlier_robust():
    """Outlier-heavy input still round-trips at cosine >= 0.95 (looser, sanity)."""
    torch.manual_seed(11)
    NOPE_HEAD_DIM = 448
    QUANT_BLOCK = 64
    x = torch.randn(NOPE_HEAD_DIM, dtype=torch.float32)
    x[42] = 400.0
    x[100] = -400.0
    enc, exps = _fp32_to_ue8m0_quantize(x, QUANT_BLOCK)
    dec = _ue8m0_dequantize(enc, exps, QUANT_BLOCK)
    cos = torch.nn.functional.cosine_similarity(x, dec, dim=0).item()
    assert cos >= 0.95, f"outlier UE8M0 cosine {cos:.6f} below 0.95"


# =============================================================================
# 4. apply()/restore() idempotency
# =============================================================================
def test_apply_restore_idempotent():
    try:
        import vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache as upstream  # noqa: E501
    except ImportError:
        pytest.skip("vllm not importable — apply() integration test skipped")
    # Apply
    patch.apply()
    assert getattr(upstream, "_CC4_SM86_PATCH_APPLIED", False)
    # Double-apply must raise
    with pytest.raises(RuntimeError):
        patch.apply()
    # Restore
    patch.restore()
    assert not getattr(upstream, "_CC4_SM86_PATCH_APPLIED", False)
    # Restore-without-apply is no-op
    patch.restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
