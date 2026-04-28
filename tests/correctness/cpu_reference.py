#!/usr/bin/env python3
# --ProtoAI-Bakari--
# Story 11: Hopper-fallback CPU pytorch reference implementations of the DSA-class
# kernels we are porting to sm_86. These are NOT performant — they are bit-correct
# (or close enough — rtol=1e-3 / atol=1e-4 for bf16) ground truth for L1 unit tests
# when no Hopper GPU is available to capture the canonical output.
#
# Used by tests/unit/test_*_sm86.py via the `reference_*` callables below.
#
# Each reference takes the same args+dtypes as its sm_86 counterpart and returns
# CPU torch tensors. CUDA tensors should be .cpu()'d before comparison.

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def reference_sparse_attn_indexer(
    q: torch.Tensor,             # [B, H_q, T_q, D]   queries (bf16/fp16/fp32)
    k_cache: torch.Tensor,       # [B, H_kv, T_k, D]  keys
    top_k: int,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Lightning-Indexer top-k selection: per query, score every K-block by sum of
    dot(q, k) over the block, then return the top-k block indices.
    Returns: [B, H_q, T_q, top_k] int64 indices into K-blocks.

    Mirrors the sm_86 kernel CC3 will port. Algorithm-faithful, not bandwidth-faithful.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.float()
    k = k_cache.float()
    B, H_q, T_q, D = q.shape
    _, H_kv, T_k, _ = k.shape
    assert H_q % H_kv == 0, "GQA group ratio must divide cleanly"
    group = H_q // H_kv
    # broadcast K across group
    k_g = k.repeat_interleave(group, dim=1)              # [B, H_q, T_k, D]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k_g) * scale   # [B, H_q, T_q, T_k]
    n_blocks = (T_k + block_size - 1) // block_size
    pad = n_blocks * block_size - T_k
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    block_scores = scores.view(B, H_q, T_q, n_blocks, block_size).sum(dim=-1)
    top_k = min(top_k, n_blocks)
    _, idx = torch.topk(block_scores, k=top_k, dim=-1)   # [B, H_q, T_q, top_k]
    return idx.to(torch.int64)


def reference_mla_decode(
    q: torch.Tensor,             # [B, H_q, 1, D_qk] decode-step query (T_q=1)
    kv_cache: torch.Tensor,      # [B, T_k, 2, D_kv] paged KV (k=...,0,:; v=...,1,:)
    block_table: torch.Tensor,   # [B, max_blocks] int32 paging
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """
    MLA-decode: dense attention over selected KV blocks for the single decode step.
    Returns: [B, H_q, 1, D_v] attended values.

    Mirrors CC4 lightning bundle's mla_decode entry point. Algorithm-faithful.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.float()
    B, H, _, D = q.shape
    out = torch.zeros((B, H, 1, kv_cache.shape[-1]), dtype=torch.float32)
    for bi in range(B):
        blocks = block_table[bi]
        valid = blocks[blocks >= 0]
        if len(valid) == 0:
            continue
        ks = []
        vs = []
        for blk in valid.tolist():
            base = blk * block_size
            ks.append(kv_cache[bi, base:base+block_size, 0, :])
            vs.append(kv_cache[bi, base:base+block_size, 1, :])
        K = torch.cat(ks, dim=0).float()                 # [T_kept, D]
        V = torch.cat(vs, dim=0).float()                 # [T_kept, D]
        scores = (q[bi] @ K.t()) * scale                 # [H, 1, T_kept]
        attn = F.softmax(scores, dim=-1)
        out[bi] = attn @ V                               # [H, 1, D]
    return out


def reference_compressor(
    x: torch.Tensor,             # [B, T, D] hidden states
    weight: torch.Tensor,        # [D_out, D]
    block_size: int = 64,
) -> torch.Tensor:
    """
    Block-mean compressor stage from the lightning bundle. Mean-pool every
    block_size tokens then linear-project to D_out.
    Returns: [B, T // block_size, D_out]
    """
    B, T, D = x.shape
    pad = (block_size - T % block_size) % block_size
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    n_blocks = (T + pad) // block_size
    pooled = x.view(B, n_blocks, block_size, D).mean(dim=2)   # [B, n_blocks, D]
    return F.linear(pooled.float(), weight.float())


def reference_swa(
    q: torch.Tensor,             # [B, H, T, D]
    k: torch.Tensor,             # [B, H_kv, T, D]
    v: torch.Tensor,             # [B, H_kv, T, D]
    window: int,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Sliding-Window Attention reference (each query attends to the previous
    `window` keys, plus itself). Returns: [B, H, T, D_v].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.float(); k = k.float(); v = v.float()
    B, H, T, D = q.shape
    H_kv = k.shape[1]
    assert H % H_kv == 0
    g = H // H_kv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    # mask: position s allowed iff (t - window) <= s <= t
    t_idx = torch.arange(T)[:, None]
    s_idx = torch.arange(T)[None, :]
    mask = (s_idx <= t_idx) & (s_idx >= t_idx - window)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def reference_marlin_int4_gemm(
    x_bf16: torch.Tensor,          # [M, K]
    w_int4_packed: torch.Tensor,   # [K//8, N] int32 (8 int4 vals packed per int32)
    scales: torch.Tensor,          # [K // group_size, N]
    group_size: int = 128,
) -> torch.Tensor:
    """
    Reference for MARLIN W4A16 GEMM (CC5 INT4-AWQ-Marlin sm_86). Dequantizes int4
    weights to bf16 then does plain matmul. Slow but bit-correct enough for the gate.
    Returns: [M, N] bf16.
    """
    K_packed, N = w_int4_packed.shape
    K = K_packed * 8
    # unpack int4: each int32 holds 8 nibbles, signed range [-8, 7]
    w = torch.empty((K, N), dtype=torch.float32)
    p = w_int4_packed.to(torch.int64)                  # [K_packed, N]
    for i in range(8):
        nib = ((p >> (4 * i)) & 0xF).to(torch.int32)
        nib = torch.where(nib >= 8, nib - 16, nib).float()
        w[i::8] = nib
    n_groups = K // group_size
    w = w.view(n_groups, group_size, N) * scales.float().unsqueeze(1)
    w = w.view(K, N)
    return (x_bf16.float() @ w).to(x_bf16.dtype)


def reference_sparse_attn_indexer_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """CPU reference for sparse_attn_indexer_sm86.cu BLOCK-SCORE LOGITS output.
    Per (batch, head, query, key_block) = sum over (k_in_block, d) of q[d]*k[blk,k,d]*scale.
    Returns [B, H, T_q, n_blocks] fp32 — matches the kernel's block_scores buffer.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.float()
    k = k_cache.float()
    B, H, T_q, D = q.shape
    _, T_k, _ = k.shape
    n_blocks = (T_k + block_size - 1) // block_size
    pad = n_blocks * block_size - T_k
    if pad:
        k = F.pad(k, (0, 0, 0, pad), value=0.0)
    k_blk = k.view(B, n_blocks, block_size, D)
    block_scores = torch.einsum("bhqd,bnkd->bhqn", q, k_blk) * scale
    return block_scores.float()


def reference_paged_attn_hd512(
    q: torch.Tensor,             # [B, H_q, 1, 512]
    kv_cache: torch.Tensor,      # [num_blocks, block_size, 2, H_kv, 512]
    block_table: torch.Tensor,   # [B, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,      # [B] int32
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Paged-KV attention reference for head_dim=512 (DSV4-Flash-FP8 / GLM-5.1).
    Tracks vLLM PR #38835 paged_attn signature targeting CC3's
    paged_attn_hd512_sm86 kernel. Algorithm-faithful, not bandwidth-faithful —
    L1 numerics ground truth only.
    Returns: [B, H_q, 1, 512] attended values.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.float()
    B, H_q, _, D = q.shape
    H_kv = kv_cache.shape[3]
    assert D == 512, f"reference_paged_attn_hd512 expects head_dim=512, got {D}"
    assert H_q % H_kv == 0, "GQA group ratio must divide cleanly"
    group = H_q // H_kv
    out = torch.zeros((B, H_q, 1, D), dtype=torch.float32)
    for bi in range(B):
        seq_len = int(seq_lens[bi].item())
        if seq_len <= 0:
            continue
        n_blocks_needed = (seq_len + block_size - 1) // block_size
        block_ids = block_table[bi, :n_blocks_needed].tolist()
        ks_pieces, vs_pieces = [], []
        for blk_idx, blk in enumerate(block_ids):
            if blk < 0:
                continue
            base = blk_idx * block_size
            this_block_used = min(block_size, seq_len - base)
            ks_pieces.append(kv_cache[blk, :this_block_used, 0, :, :].float())
            vs_pieces.append(kv_cache[blk, :this_block_used, 1, :, :].float())
        if not ks_pieces:
            continue
        K = torch.cat(ks_pieces, dim=0)        # [seq_len, H_kv, D]
        V = torch.cat(vs_pieces, dim=0)
        K = K.repeat_interleave(group, dim=1)
        V = V.repeat_interleave(group, dim=1)
        K_h = K.permute(1, 2, 0)                # [H_q, D, seq_len]
        scores = torch.matmul(q[bi], K_h) * scale
        attn = torch.softmax(scores, dim=-1)
        V_h = V.permute(1, 0, 2)                # [H_q, seq_len, D]
        out[bi] = torch.matmul(attn, V_h)
    return out


REFERENCES = {
    "sparse_attn_indexer": reference_sparse_attn_indexer,
    "sparse_attn_indexer_logits": reference_sparse_attn_indexer_logits,
    "mla_decode": reference_mla_decode,
    "compressor": reference_compressor,
    "swa": reference_swa,
    "marlin_int4_gemm": reference_marlin_int4_gemm,
    "paged_attn_hd512": reference_paged_attn_hd512,
}


def main():
    import sys, json
    print(json.dumps({"available_references": sorted(REFERENCES)}, indent=2))


if __name__ == "__main__":
    main()
