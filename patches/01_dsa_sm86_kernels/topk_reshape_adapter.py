# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""Topk reshape adapter — bridges LRU shim output to CC4 mla_decode input.

Contract:
  in:  topk_buf [T, K] int32   (vLLM sparse_attn_indexer output, T = B*seq_q)
  out: topk_idx [B, H, K] int32 (CC4 mla_decode_sparse_sm86 input)

DSA is MQA — sparse topk is shared across all H heads — so we expand the head
axis without copying scores per head. Decode path squeezes the seq_q=1 dim.
"""
from __future__ import annotations

import torch


def reshape_topk_for_mla_decode(
    topk_buf: torch.Tensor,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    """Adapt [T,K] int32 -> [B,H,K] int32 (decode) or [B,H,Sq,K] (prefill, Sq>1)."""
    assert topk_buf.dim() == 2, f"topk_buf must be 2D, got {topk_buf.shape}"
    assert topk_buf.dtype == torch.int32, f"topk_buf dtype must be int32, got {topk_buf.dtype}"
    T, K = topk_buf.shape
    assert T % batch_size == 0, f"T={T} not divisible by B={batch_size}"
    seq_q = T // batch_size
    expanded = (
        topk_buf.view(batch_size, seq_q, K)
        .unsqueeze(1)
        .expand(batch_size, num_heads, seq_q, K)
    )
    if seq_q == 1:
        expanded = expanded.squeeze(2)
    return expanded.contiguous().to(torch.int32)


def selfcheck() -> dict:
    B, H, K = 2, 4, 8
    T = B  # decode case
    buf = torch.arange(T * K, dtype=torch.int32).view(T, K)
    out = reshape_topk_for_mla_decode(buf, B, H)
    assert out.shape == (B, H, K), out.shape
    assert out.dtype == torch.int32
    for b in range(B):
        for h in range(H):
            assert (out[b, h] == buf[b]).all(), f"head broadcast mismatch b={b} h={h}"
    return {"ok": True, "shape_decode": list(out.shape)}


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck()))

# // --ProtoAI-Bakari--
