# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""LRU sparse-indexer cache shim — Hour-1 BF16 probe path.

Wraps the patch #16 dummy stub for `torch.ops.vllm.sparse_attn_indexer` with a
small per-process LRU memoization layer. Goal: reduce decode-step latency and
allocator pressure when the stub is hot, so /v1/completions can return a token
(non-500) on Ampere PP=2+TP=8.

Correctness is *neutral* — the dummy already returns wrong topk indices.
The real correctness path is the sm_86 kernel port (Stories 4-5 of CC3 lane).

Apply: source this file in the vLLM venv after patch #16 is active. It
re-registers the custom op via direct_register_custom_op (overrides #16's
stub function with a memoized wrapper around it).
"""

from __future__ import annotations

import functools
import os
from collections import OrderedDict
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tunables (env-overridable for benchmarking)
# ---------------------------------------------------------------------------

LRU_CAPACITY = int(os.environ.get("VLLM_DSA_LRU_CAPACITY", "64"))
LRU_NTOKEN_BUCKET_LOG2 = int(os.environ.get("VLLM_DSA_LRU_NTOKEN_BUCKET_LOG2", "3"))
LRU_DISABLE = os.environ.get("VLLM_DSA_LRU_DISABLE", "0") == "1"
LRU_DEBUG = os.environ.get("VLLM_DSA_LRU_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# Stub (drop-in for patch #16): dummy topk that returns -1 buffer untouched.
# ---------------------------------------------------------------------------


def _dummy_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    n = hidden_states.shape[0]
    topk_indices_buffer[:n] = -1
    return topk_indices_buffer


# ---------------------------------------------------------------------------
# Cache key bucketing
# ---------------------------------------------------------------------------


def _bucket_pow2(n: int, log2_step: int) -> int:
    if n <= 0:
        return 0
    step = 1 << log2_step
    return ((n + step - 1) // step) * step


# ---------------------------------------------------------------------------
# OrderedDict-backed LRU
# ---------------------------------------------------------------------------


class _IndexerLRU:
    __slots__ = ("_cache", "_capacity", "_hits", "_misses", "_evictions")

    def __init__(self, capacity: int) -> None:
        self._cache: "OrderedDict[tuple[int, int, bool, bool, int, int], torch.Tensor]" = (
            OrderedDict()
        )
        self._capacity = capacity
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: tuple, target: torch.Tensor) -> bool:
        cached = self._cache.get(key)
        if cached is None:
            self._misses += 1
            return False
        self._cache.move_to_end(key)
        n = min(cached.shape[0], target.shape[0])
        target[:n] = cached[:n]
        self._hits += 1
        return True

    def put(self, key: tuple, payload: torch.Tensor, n: int) -> None:
        # Clone the slice — vLLM mutates topk_indices_buffer between calls.
        self._cache[key] = payload[:n].detach().clone()
        self._cache.move_to_end(key)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
            self._evictions += 1

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        rate = (self._hits / total) if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "capacity": self._capacity,
            "size": len(self._cache),
            "hit_rate": rate,
        }


_LRU = _IndexerLRU(LRU_CAPACITY)


def _make_key(
    hidden_states: torch.Tensor,
    total_seq_lens: int,
    topk_tokens: int,
    head_dim: int,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool,
) -> tuple:
    n_bucket = _bucket_pow2(hidden_states.shape[0], LRU_NTOKEN_BUCKET_LOG2)
    seq_bucket = _bucket_pow2(total_seq_lens, LRU_NTOKEN_BUCKET_LOG2)
    return (
        n_bucket,
        seq_bucket,
        bool(skip_k_cache_insert),
        bool(use_fp4_cache),
        int(topk_tokens),
        int(head_dim),
    )


# ---------------------------------------------------------------------------
# Memoized wrapper — drop-in replacement for the patch #16 stub.
# ---------------------------------------------------------------------------


def sparse_attn_indexer_lru(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    if LRU_DISABLE:
        return _dummy_sparse_attn_indexer(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )

    key = _make_key(
        hidden_states,
        total_seq_lens,
        topk_tokens,
        head_dim,
        skip_k_cache_insert,
        use_fp4_cache,
    )
    n = hidden_states.shape[0]

    if _LRU.get(key, topk_indices_buffer):
        if LRU_DEBUG:
            stats = _LRU.stats()
            logger.debug_once(
                "DSA LRU HIT key=%s n=%d hit_rate=%.3f size=%d",
                key,
                n,
                stats["hit_rate"],
                stats["size"],
            )
        return topk_indices_buffer

    out = _dummy_sparse_attn_indexer(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_quant,
        q_scale,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert,
        use_fp4_cache,
    )
    _LRU.put(key, out, n)
    if LRU_DEBUG:
        stats = _LRU.stats()
        logger.debug_once(
            "DSA LRU MISS key=%s n=%d misses=%d size=%d",
            key,
            n,
            stats["misses"],
            stats["size"],
        )
    return out


def sparse_attn_indexer_lru_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


# ---------------------------------------------------------------------------
# Re-register the custom op (overrides patch #16's registration).
# ---------------------------------------------------------------------------


def install_lru_shim() -> None:
    try:
        direct_register_custom_op(
            op_name="sparse_attn_indexer",
            op_func=sparse_attn_indexer_lru,
            mutates_args=["topk_indices_buffer"],
            fake_impl=sparse_attn_indexer_lru_fake,
            dispatch_key=current_platform.dispatch_key,
        )
        logger.info(
            "DSA LRU shim installed cap=%d ntok_bucket_log2=%d disable=%s debug=%s",
            LRU_CAPACITY,
            LRU_NTOKEN_BUCKET_LOG2,
            LRU_DISABLE,
            LRU_DEBUG,
        )
    except Exception as exc:
        logger.warning("DSA LRU shim re-register skipped: %s", exc)


def lru_stats() -> dict[str, Any]:
    return _LRU.stats()


def lru_clear() -> None:
    _LRU._cache.clear()
    _LRU._hits = 0
    _LRU._misses = 0
    _LRU._evictions = 0


install_lru_shim()

# // --ProtoAI-Bakari--
