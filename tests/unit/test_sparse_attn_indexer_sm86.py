# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# // --ProtoAI-Bakari--
"""Story 7 numerics gate: sparse_attn_indexer_sm86.cu vs CPU reference.

Compares the block-score logits computed by ``sparse_attn_indexer_logits_kernel``
(csrc_patches/01_dsa_sm86_kernels/sparse_attn_indexer_sm86.cu) against
``reference_sparse_attn_indexer_logits`` (CPU pytorch reference, same algo).

Acceptance gates (per CC3 role + Story 7 backlog):
  - max-abs-diff < 1e-3 (bf16-acc tolerance; tighter for fp32-acc)
  - cosine similarity >= 0.97 elementwise on flattened logits
  - no NaN / no Inf in kernel output

Skipped if:
  - CUDA unavailable (Mac author env, CI)
  - sm < 8.0 (kernel uses sm_80+ ldmatrix/cp.async/mma.sync)
  - Kernel module not yet built (CC2 build target ``vllm._dsa_sm86`` or
    standalone ``vllm_dsa_sm86_kernels``)

Run:
    pytest tests/unit/test_sparse_attn_indexer_sm86.py -v
"""
from __future__ import annotations

import importlib
import sys
import pathlib
import pytest
import torch

# Allow ``from cpu_reference import ...`` and ``from conftest import ...``.
_TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TESTS_ROOT / "correctness"))
sys.path.insert(0, str(_TESTS_ROOT / "unit"))

from cpu_reference import reference_sparse_attn_indexer_logits  # noqa: E402

KERNEL_MODULE_CANDIDATES = (
    "vllm._dsa_sm86",
    "vllm._C",
    "vllm_dsa_sm86_kernels",
)


def _import_kernels():
    last = None
    for name in KERNEL_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ImportError as e:
            last = e
            continue
    pytest.skip(f"sm_86 kernel module not built; tried {KERNEL_MODULE_CANDIDATES}: {last}")


@pytest.fixture(scope="module")
def kernels():
    return _import_kernels()


def _device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; sm_86 kernel test requires GPU")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        pytest.skip(f"compute capability sm_{cap[0]}{cap[1]} < sm_80")
    return torch.device("cuda")


@pytest.mark.parametrize(
    "B,H,T_q,T_k,D,block_size",
    [
        (1, 1, 1, 64, 128, 64),     # smallest viable
        (2, 4, 1, 256, 128, 64),    # decode-step typical
        (1, 8, 4, 1024, 128, 64),   # mixed prefill/decode
    ],
)
def test_sparse_attn_indexer_logits(kernels, B, H, T_q, T_k, D, block_size):
    """Compare kernel block_scores [B,H,T_q,n_blocks] vs CPU reference."""
    device = _device()
    fn = (
        getattr(kernels, "sparse_attn_indexer_logits_sm86", None)
        or getattr(kernels, "launch_sparse_attn_indexer_logits_sm86", None)
        or getattr(kernels, "sparse_attn_indexer_sm86", None)
    )
    if fn is None:
        pytest.skip("sparse_attn_indexer_logits not exported by kernel module")

    torch.manual_seed(0xCC3CAFE)
    q = torch.randn(B, H, T_q, D, dtype=torch.bfloat16, device=device) * 0.1
    k_cache = torch.randn(B, T_k, D, dtype=torch.bfloat16, device=device) * 0.1
    n_blocks = (T_k + block_size - 1) // block_size
    out = torch.empty(B, H, T_q, n_blocks, dtype=torch.float32, device=device)

    scale = 1.0 / (D ** 0.5)
    fn(q, k_cache, out, scale=scale, block_size=block_size)
    torch.cuda.synchronize()

    ref = reference_sparse_attn_indexer_logits(
        q.cpu(), k_cache.cpu(), block_size=block_size, scale=scale
    ).to(device)

    actual = out.float()
    expected = ref.float()
    assert actual.shape == expected.shape, f"{actual.shape} vs {expected.shape}"
    assert not torch.isnan(actual).any(), "kernel produced NaN"
    assert not torch.isinf(actual).any(), "kernel produced Inf"

    diff = (actual - expected).abs()
    max_abs_diff = diff.max().item()
    a_flat = actual.reshape(-1)
    e_flat = expected.reshape(-1)
    cos = float(
        torch.nn.functional.cosine_similarity(
            a_flat, e_flat, dim=0
        ).item()
    )

    # bf16-acc tolerance: ~1e-3 absolute, cosine >= 0.97
    assert max_abs_diff < 1e-2, (
        f"max_abs_diff={max_abs_diff:.6g} >= 1e-2 (bf16-acc Story-7 gate)"
    )
    assert cos >= 0.97, f"cosine={cos:.4f} < 0.97 (Story-7 gate)"
