"""
Triton MLA-decode kernel for sm_86 (RTX 3090 Ampere).
EPIC 3.0 hour-1 shim — replaces fp8e4nv-rejecting MLA-Sparse decode path.

The native vLLM DSA MLA-Sparse decode kernel uses fp8e4nv tile loads + WGMMA
on Hopper. Triton on sm_86 rejects fp8e4nv at compile time ("PTX ISA does not
support .e4m3 on sm_86"). This shim:

  - Accepts BF16 / FP16 Q, K_c, V_c (latent compressed K / V).
  - Reads sparse top-k indices produced by the (stubbed) sparse_attn_indexer.
  - Early-exits with zero-output if topk_count == 0 (no valid sparse rows).
  - Does online-softmax over the selected K rows, then weighted V combine.
  - Emits a numerics-check fingerprint (sum, max-abs, NaN count) on the output
    tile so CC6's gate can grep without a full L1 unit-test rebuild.

Inputs (per request, decode = 1 query token per head):
  q       : (B, H, D)       BF16   query latent
  k_cache : (B, S, D)       BF16   compressed K cache (latent dim D)
  v_cache : (B, S, Dv)      BF16   compressed V cache
  topk_idx: (B, H, K)       int32  sparse indices into the S axis (-1 = invalid)
  out     : (B, H, Dv)      BF16   output (overwritten)

Tile shape: BLOCK_K = 64 (k-rows per inner reduce). One program per (batch, head).

This is a CORRECTNESS prototype, not perf-tuned. Acceptance gate (CC6):
  - top-1 ≥ 98% vs llama.cpp PP reference
  - cosine ≥ 0.97 on output tensors
  - no NaN / no Inf

// --ProtoAI-Bakari--
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


NUMERICS_FP_PATH = os.environ.get(
    "CC4_MLA_NUMERICS_FP",
    "/tmp/cc4_mla_decode_numerics.jsonl",
)


if HAVE_TRITON:

    @triton.jit
    def _mla_decode_sparse_kernel(
        q_ptr, k_ptr, v_ptr, idx_ptr, out_ptr,
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kd,
        stride_vb, stride_vs, stride_vd,
        stride_ib, stride_ih, stride_ik,
        stride_ob, stride_oh, stride_od,
        H: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        K_MAX: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_D)
        dv_off = tl.arange(0, BLOCK_DV)
        d_mask = d_off < D
        dv_mask = dv_off < DV

        q_row = tl.load(
            q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

        any_valid = 0
        for ki in range(0, K_MAX):
            idx = tl.load(
                idx_ptr + pid_b * stride_ib + pid_h * stride_ih + ki * stride_ik
            )
            valid = idx >= 0
            any_valid = any_valid | tl.where(valid, 1, 0)
            safe_idx = tl.where(valid, idx, 0)

            k_row = tl.load(
                k_ptr + pid_b * stride_kb + safe_idx * stride_ks + d_off * stride_kd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)
            v_row = tl.load(
                v_ptr + pid_b * stride_vb + safe_idx * stride_vs + dv_off * stride_vd,
                mask=dv_mask, other=0.0,
            ).to(tl.float32)

            qk = tl.sum(q_row * k_row, axis=0) * scale
            qk = tl.where(valid, qk, -float("inf"))

            m_new = tl.maximum(m_i, qk)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)
            acc = acc * alpha + p * v_row
            l_i = l_i * alpha + p
            m_i = m_new

        # Early-exit: if no valid index ever fired, write zeros.
        out = tl.where(any_valid > 0, acc / l_i, tl.zeros([BLOCK_DV], dtype=tl.float32))

        tl.store(
            out_ptr + pid_b * stride_ob + pid_h * stride_oh + dv_off * stride_od,
            out.to(out_ptr.dtype.element_ty),
            mask=dv_mask,
        )


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _emit_numerics_fingerprint(tag: str, out: torch.Tensor) -> None:
    if os.environ.get("CC4_MLA_NUMERICS_DISABLE") == "1":
        return
    with torch.no_grad():
        flat = out.detach().float().reshape(-1)
        nan_count = int(torch.isnan(flat).sum().item())
        inf_count = int(torch.isinf(flat).sum().item())
        finite = flat[torch.isfinite(flat)]
        s = float(finite.sum().item()) if finite.numel() else 0.0
        m = float(finite.abs().max().item()) if finite.numel() else 0.0
    line = (
        f'{{"tag":"{tag}","shape":{list(out.shape)},'
        f'"sum":{s:.6e},"max_abs":{m:.6e},'
        f'"nan":{nan_count},"inf":{inf_count}}}\n'
    )
    try:
        with open(NUMERICS_FP_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass


def mla_decode_sparse_sm86(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """sm_86 MLA-Sparse decode shim. See module docstring."""
    assert q.dim() == 3, f"q expected (B,H,D) got {q.shape}"
    assert k_cache.dim() == 3, f"k_cache expected (B,S,D) got {k_cache.shape}"
    assert v_cache.dim() == 3, f"v_cache expected (B,S,Dv) got {v_cache.shape}"
    assert topk_idx.dim() == 3, f"topk_idx expected (B,H,K) got {topk_idx.shape}"
    assert q.dtype in (torch.bfloat16, torch.float16), (
        f"q dtype {q.dtype} unsupported on sm_86 — fp8e4nv rejected; use bf16/fp16"
    )
    assert q.is_cuda, "q must be CUDA"

    B, H, D = q.shape
    _, S, Dk = k_cache.shape
    _, Sv, DV = v_cache.shape
    K_MAX = topk_idx.shape[-1]
    assert D == Dk, f"q D={D} != k_cache D={Dk}"
    assert S == Sv, f"k_cache S={S} != v_cache S={Sv}"
    assert topk_idx.shape[:2] == (B, H), f"topk_idx leading dims {topk_idx.shape[:2]} != (B,H)=({B},{H})"
    assert topk_idx.dtype == torch.int32, f"topk_idx dtype {topk_idx.dtype} must be int32"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if out is None:
        out = torch.empty((B, H, DV), dtype=q.dtype, device=q.device)
    else:
        assert out.shape == (B, H, DV) and out.dtype == q.dtype

    if not HAVE_TRITON:
        _mla_decode_reference(q, k_cache, v_cache, topk_idx, scale, out)
        _emit_numerics_fingerprint("ref-fallback", out)
        return out

    BLOCK_D = _next_pow2(D)
    BLOCK_DV = _next_pow2(DV)
    grid = (B, H)

    _mla_decode_sparse_kernel[grid](
        q, k_cache, v_cache, topk_idx, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=H, D=D, DV=DV, K_MAX=K_MAX,
        BLOCK_D=BLOCK_D, BLOCK_DV=BLOCK_DV,
    )

    _emit_numerics_fingerprint("triton-sm86", out)
    return out


def _mla_decode_reference(q, k_cache, v_cache, topk_idx, scale, out) -> None:
    """Pure-PyTorch reference path. Used for CPU/no-Triton + numerics gate."""
    B, H, D = q.shape
    _, _, DV = v_cache.shape
    K = topk_idx.shape[-1]
    for b in range(B):
        for h in range(H):
            idx = topk_idx[b, h]
            valid_mask = idx >= 0
            if not bool(valid_mask.any()):
                out[b, h].zero_()
                continue
            sel = idx.clamp(min=0).to(torch.long)
            k_sel = k_cache[b, sel].float()
            v_sel = v_cache[b, sel].float()
            qrow = q[b, h].float()
            scores = (qrow.unsqueeze(0) * k_sel).sum(-1) * scale
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out[b, h] = (w.unsqueeze(-1) * v_sel).sum(0).to(out.dtype)


def numerics_selfcheck(
    seed: int = 0,
    B: int = 2, H: int = 4, S: int = 32, D: int = 64, DV: int = 64, K: int = 8,
    cosine_floor: float = 0.97,
    device: str = "cuda",
) -> dict:
    """Tiny self-check: Triton kernel vs reference. Returns dict, raises on fail."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, S, DV, dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, S, (B, H, K), dtype=torch.int32, device=device)
    idx[0, 0, 0] = -1

    out_triton = mla_decode_sparse_sm86(q, k, v, idx)
    out_ref = torch.empty_like(out_triton)
    _mla_decode_reference(q, k, v, idx, 1.0 / math.sqrt(D), out_ref)

    a = out_triton.float().reshape(-1)
    b = out_ref.float().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    max_abs_diff = float((a - b).abs().max().item())

    result = {
        "cosine": cos,
        "max_abs_diff": max_abs_diff,
        "nan_triton": int(torch.isnan(out_triton).sum().item()),
        "nan_ref": int(torch.isnan(out_ref).sum().item()),
        "passed": cos >= cosine_floor,
    }
    if not result["passed"]:
        raise AssertionError(f"numerics_selfcheck failed: {result}")
    return result


if __name__ == "__main__":
    import json
    import sys
    if not torch.cuda.is_available():
        print("CUDA unavailable — skipping selfcheck", file=sys.stderr)
        sys.exit(0)
    res = numerics_selfcheck()
    print(json.dumps(res))
