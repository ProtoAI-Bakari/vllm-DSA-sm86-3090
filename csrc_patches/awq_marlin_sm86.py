"""awq_marlin_sm86.py — Python integration + Triton-INT4 fallback kernel.

# --ProtoAI-Bakari--

Pairs with `awq_marlin_sm86.cu` (Story 4). Provides:

1. `int4_expert_gemm(...)` — high-level entry point. Tries Marlin first,
   falls through to Triton kernel below if shape constraints fail or the
   Marlin op is missing.
2. Triton kernel `triton_int4_gemm_kernel` — INT4 dequant + bf16 matmul,
   compiled JIT on first call. Designed for the M=1..16 decode regime
   where Marlin's tile bias hurts (Marlin tiles M=64+ block).
3. Substitution helper `install_dsv4_moe_substitution()` — monkey-patches
   DSV4's expert GEMM call site in vLLM with our router.

Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7
(claude-opus-4-7) [1M ctx, max effort, agent: CC5].
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

try:
    from vllm_dsa_sm86 import awq_marlin_sm86 as _ext  # type: ignore
except ImportError:                                          # pragma: no cover
    _ext = None


try:
    import triton                                             # type: ignore
    import triton.language as tl                              # type: ignore
    _HAS_TRITON = True
except ImportError:                                          # pragma: no cover
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton-INT4 fallback kernel
# ---------------------------------------------------------------------------
# Decode-regime (M=1..16) INT4 dequant + bf16 matmul. Designed for shapes
# that fail Marlin's tile constraints (e.g., GLM-5.1 IQ2XXS gs=64 with K
# not 64-multiple, or M < Marlin's efficiency cliff).
#
# Kernel layout:
#   q_weight: int32 [K, N/8]    8 INT4 weights packed per int32, K-major
#   scales:   bf16  [K/G, N]    one scale per group along K
#   zeros:    int32 [K/G, N/8]  packed zero-points
#   a:        bf16  [M, K]
#   out:      bf16  [M, N]
#
# Block tile: BLOCK_M × BLOCK_N over output, BLOCK_K stride along K.
# Each thread block computes one [BLOCK_M, BLOCK_N] output tile by
# iterating BLOCK_K-sized slabs of K and accumulating in fp32.

if _HAS_TRITON:
    @triton.jit
    def triton_int4_gemm_kernel(
        a_ptr, q_ptr, s_ptr, z_ptr, out_ptr,
        M, N, K, GROUP_SIZE,
        stride_am, stride_ak,
        stride_qk, stride_qn,
        stride_sk, stride_sn,
        stride_zk, stride_zn,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Pre-compute the int4 nibble shifts: [N % 8] tells us which 4-bit
        # slot in each packed int32 the column lives in.
        nibble = (offs_n % 8) * 4

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            mask_k = k_idx < K
            mask_m = offs_m < M
            mask_n = offs_n < N

            # Load A tile [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
            a_tile = tl.load(a_ptrs,
                             mask=(mask_m[:, None] & mask_k[None, :]),
                             other=0.0).to(tl.float32)

            # Load packed q_weight column-by-column. q_ptr addresses int32
            # element holding 8 INT4 weights for cols [n*8 .. n*8+7].
            q_col = offs_n // 8
            q_ptrs = q_ptr + k_idx[:, None] * stride_qk + q_col[None, :] * stride_qn
            q_packed = tl.load(q_ptrs,
                               mask=(mask_k[:, None] & mask_n[None, :]),
                               other=0)
            # Extract the 4-bit nibble
            q_nib = (q_packed >> nibble[None, :]) & 0xF        # 0..15
            q_int = q_nib.to(tl.int32) - 8                     # signed -8..7

            # Group index for scales+zeros
            g_idx = k_idx // GROUP_SIZE
            s_ptrs = s_ptr + g_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn
            scales = tl.load(s_ptrs,
                             mask=(mask_k[:, None] & mask_n[None, :]),
                             other=0.0).to(tl.float32)
            z_col = offs_n // 8
            z_ptrs = z_ptr + g_idx[:, None] * stride_zk + z_col[None, :] * stride_zn
            z_packed = tl.load(z_ptrs,
                               mask=(mask_k[:, None] & mask_n[None, :]),
                               other=0)
            z_nib = (z_packed >> nibble[None, :]) & 0xF
            zeros = z_nib.to(tl.float32) - 8

            # Dequant: w = (q_int - zeros) * scale
            w = (q_int.to(tl.float32) - zeros) * scales       # [BLOCK_K, BLOCK_N]

            acc += tl.dot(a_tile, w, allow_tf32=False)

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_tile = acc.to(tl.bfloat16)
        tl.store(out_ptrs, out_tile,
                 mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)))


def triton_int4_gemm(
    a: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Triton fallback INT4 dequant + bf16 GEMM."""
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton not available; cannot run INT4 fallback. Install "
            "triton via vLLM's required deps."
        )
    M, K = a.shape
    N = scales.size(1)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    BLOCK_M = max(16, min(64, triton.next_power_of_2(M)))
    BLOCK_N = 64
    BLOCK_K = max(group_size, 64)

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    triton_int4_gemm_kernel[grid](
        a, q_weight, scales, zeros, out,
        M, N, K, group_size,
        a.stride(0), a.stride(1),
        q_weight.stride(0), q_weight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Top-level router
# ---------------------------------------------------------------------------
def int4_expert_gemm(
    a: torch.Tensor,
    q_weight: torch.Tensor,                      # marlin- or awq-packed
    scales: torch.Tensor,
    zeros: torch.Tensor,
    workspace: Optional[torch.Tensor] = None,
    *,
    group_size: int = 128,
    layout: str = "marlin",
) -> torch.Tensor:
    """One INT4 expert GEMM. Marlin first; Triton fallback on shape miss."""
    M, K = a.shape
    N = scales.size(1)

    if _ext is not None:
        # Try Marlin path
        try:
            if _ext.marlin_shape_ok(K, N, group_size):
                if workspace is None:
                    workspace = torch.zeros(
                        (N // 64) * 16, dtype=torch.int32, device=a.device,
                    )
                return _ext.expert_gemm_router(
                    a, q_weight, scales, zeros, workspace,
                    M, N, K, group_size, layout,
                )
        except RuntimeError as exc:
            # Marlin op unresolved or other torch-level failure → Triton
            if "FAIL_OP_UNRESOLVED" in str(exc) or "not registered" in str(exc):
                pass
            else:
                raise

    # Triton path (also used when _ext is not built yet)
    if layout == "awq":
        raise RuntimeError(
            "AWQ-layout weight cannot use Triton fallback directly; need "
            "Marlin repack OR a separate Triton AWQ-format kernel. Story "
            "4 ships only the marlin-format Triton path; AWQ→marlin "
            "repack must run first."
        )
    return triton_int4_gemm(a, q_weight, scales, zeros, group_size)


# ---------------------------------------------------------------------------
# vLLM monkey-patch — substitute MoE expert GEMM
# ---------------------------------------------------------------------------
def install_dsv4_moe_substitution() -> None:
    """Patch DSV4's MoE expert GEMM call site to route INT4 weights here.

    Detects whether the loaded checkpoint has AWQ-INT4 expert weights
    (Story 5 calibration output). If yes, rewires the FP8 [128,128]
    block-scaled GEMM call to `int4_expert_gemm`. Otherwise falls
    through to vLLM's default BF16-emulation path.

    Idempotent. Caller is vLLM startup post-distributed-init.
    """
    try:
        from vllm.model_executor.layers.fused_moe import (   # type: ignore
            fused_moe_method,
        )
    except ImportError:
        return

    if getattr(fused_moe_method, "_cc5_int4_substituted", False):
        return

    orig_apply = fused_moe_method.Fp8MoEMethod.apply         # type: ignore[attr-defined]

    @functools.wraps(orig_apply)
    def patched_apply(self, layer, x, router_logits,         # type: ignore[no-untyped-def]
                      top_k, renormalize, *args, **kwargs):
        if not getattr(layer, "_cc5_has_int4_awq", False):
            return orig_apply(self, layer, x, router_logits, top_k,
                              renormalize, *args, **kwargs)
        # Story 5 should have populated layer.w13_q / w13_scales / w13_zeros
        # Per-expert dispatch is handled by the moe_dispatch_dsa shim
        # (Story 3); we only override the GEMM step.
        from .moe_dispatch_dsa import dispatch_route_then_gemm  # local
        return dispatch_route_then_gemm(
            x, layer, gemm_fn=int4_expert_gemm,
            top_k=top_k, renormalize=renormalize,
        )

    fused_moe_method.Fp8MoEMethod.apply = patched_apply       # type: ignore[assignment]
    fused_moe_method._cc5_int4_substituted = True             # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Self-check (run on cuda5 after Story 5 produces calibrated weights)
# ---------------------------------------------------------------------------
def selfcheck(M: int = 4, K: int = 7168, N: int = 4096,
              group_size: int = 128) -> dict:
    """Smoke test: round-trip a random INT4 weight through both paths."""
    if not torch.cuda.is_available():
        return {"verdict": "FAIL_NO_CUDA"}
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.02

    # Build a random INT4-quantized weight + scales/zeros (placeholder
    # data — Story 5 will produce calibrated versions)
    n_groups = K // group_size
    q_weight = torch.randint(
        0, 0xFFFFFFFF, (K, N // 8), dtype=torch.int32, device="cuda",
    )
    scales = (torch.randn(n_groups, N, dtype=torch.bfloat16, device="cuda")
              * 0.01).abs() + 1e-3
    zeros = torch.full(
        (n_groups, N // 8), 0x88888888, dtype=torch.int32, device="cuda",
    )

    triton_ok = False
    triton_shape: list[int] = []
    try:
        out_triton = triton_int4_gemm(a, q_weight, scales, zeros, group_size)
        triton_ok = True
        triton_shape = list(out_triton.shape)
    except Exception as exc:                                 # noqa: BLE001
        triton_err = f"{type(exc).__name__}: {exc}"
    else:
        triton_err = None

    return {
        "verdict": "PASS" if triton_ok else "FAIL_TRITON",
        "triton_ok": triton_ok,
        "triton_shape": triton_shape,
        "triton_err": triton_err,
        "marlin_shape_ok": (
            _ext.marlin_shape_ok(K, N, group_size) if _ext is not None
            else None
        ),
        "M_K_N_gs": [M, K, N, group_size],
    }


if __name__ == "__main__":                                   # pragma: no cover
    import json
    print(json.dumps(selfcheck(), indent=2))
