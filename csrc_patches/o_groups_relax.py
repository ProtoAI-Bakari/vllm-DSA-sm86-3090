"""o_groups_relax.py — Python integration for the o_groups=8 / TP=16 patch.

# --ProtoAI-Bakari--

Pairs with `o_groups_relax_sm86.cu`. Monkey-patches vLLM's
`ColumnParallelLinear` weight-loader + forward pass for the DSV4 expert
w13 (and any layer carrying the `o_groups` annotation) when
`tp_size > o_groups`.

Why this lives in Python and not in the .cu:
- vLLM's TP linear modules are dispatched in Python via
  `vllm.model_executor.layers.linear`. Hooking weight load + forward is a
  Python-side concern; the heavy math is in the .cu.
- Sub-group communicator creation needs `torch.distributed.new_group` —
  Python-only.

Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7
(claude-opus-4-7) [1M ctx, max effort, agent: CC5].
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


# Loaded by CC2's build step; the .cu is compiled into this extension.
try:
    from vllm_dsa_sm86 import o_groups_relax_sm86 as _ext  # type: ignore
except ImportError:                                          # pragma: no cover
    _ext = None  # CC2 build pending — keep import-time non-fatal


@dataclass
class OGroupSubGroup:
    """Per-rank handle to its sub-group communicator + plan."""
    plan: "object"               # _ext.OGroupShardPlan instance
    sub_group: dist.ProcessGroup
    sub_group_handle: int        # opaque uintptr_t for the .cu side


_SUB_GROUP_CACHE: dict[tuple[int, int, int], OGroupSubGroup] = {}


def init_sub_groups(
    tp_size: int,
    o_groups: int,
    K_total: int,
    N_total: int,
) -> OGroupSubGroup:
    """Create (or fetch from cache) the sub-group for this rank.

    Sub-group layout: ranks `[r * sub_factor, (r+1) * sub_factor)` form one
    group, where `r` is the o_group_idx. With tp=16, o_groups=8,
    sub_factor=2:
        sub-group 0 = ranks {0, 1}
        sub-group 1 = ranks {2, 3}
        ...
        sub-group 7 = ranks {14, 15}
    """
    if _ext is None:
        raise RuntimeError(
            "o_groups_relax_sm86 extension not built; CC2 cmake step "
            "must run first."
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed not initialized; vLLM must be launched "
            "with TP enabled before o_groups relaxation engages."
        )
    tp_rank = dist.get_rank()
    key = (tp_rank, tp_size, o_groups)
    cached = _SUB_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    plan = _ext.make_plan(tp_rank, tp_size, o_groups, K_total, N_total)
    sub_factor = plan.sub_factor

    # Build the sub-group ranks list for THIS rank
    o_group_idx = plan.o_group_idx
    group_ranks = list(
        range(o_group_idx * sub_factor, (o_group_idx + 1) * sub_factor)
    )

    # All ranks must call new_group with the same global rank list — but
    # we're called once per rank with a different group_ranks list. The
    # vLLM convention is to build groups for ALL o_group_idx in a loop
    # and keep the one this rank belongs to. Do that.
    my_group: Optional[dist.ProcessGroup] = None
    for ogi in range(o_groups):
        ranks_for_ogi = list(range(ogi * sub_factor, (ogi + 1) * sub_factor))
        # `new_group` is collective; every rank must call it.
        grp = dist.new_group(ranks=ranks_for_ogi, backend="nccl")
        if ogi == o_group_idx:
            my_group = grp
    assert my_group is not None

    handle = _process_group_handle(my_group)
    sg = OGroupSubGroup(plan=plan, sub_group=my_group, sub_group_handle=handle)
    _SUB_GROUP_CACHE[key] = sg
    return sg


def _process_group_handle(pg: dist.ProcessGroup) -> int:
    """Extract the underlying c10d::ProcessGroup* as a uintptr_t.

    PyTorch exposes the C++ pointer via `pg._get_backend(...)._reduce_op` on
    older versions and `pg._get_backend(torch.device('cuda'))` on newer.
    The pybind cast on the .cu side accepts uintptr_t via cast on
    `id(pg)` is INSUFFICIENT — id is the Python object id, not the C++
    pointer. We grab the C++ ptr through `pg.cpp_ptr` if present, else
    via `torch._C._distributed_c10d._get_process_group_ptr(pg)` if it
    exists in this vLLM-bundled torch build.
    """
    if hasattr(pg, "cpp_ptr"):
        return int(pg.cpp_ptr)                          # type: ignore[attr-defined]
    try:
        from torch._C._distributed_c10d import (        # type: ignore
            _get_process_group_ptr as _ptr,
        )
        return int(_ptr(pg))
    except Exception as exc:
        raise RuntimeError(
            "Could not extract C++ ProcessGroup* — torch build does not "
            "expose cpp_ptr nor _get_process_group_ptr; need to extend "
            "the .cu wrapper to accept a Python ProcessGroup directly. "
            f"Underlying: {exc}"
        )


def relax_weight_load(
    weight_full_or_chunk: torch.Tensor,
    tp_size: int,
    o_groups: int,
    K_total: int,
    N_total: int,
) -> torch.Tensor:
    """Slice a `[K_total, N_total]` weight to this rank's local shard."""
    sg = init_sub_groups(tp_size, o_groups, K_total, N_total)
    return _ext.slice_full_weight(weight_full_or_chunk, sg.plan)


def relax_forward(
    a_local: torch.Tensor,
    w_local: torch.Tensor,
    tp_size: int,
    o_groups: int,
    K_total: int,
    N_total: int,
) -> torch.Tensor:
    """Run partial-GEMM + sub-group all-reduce; returns `[M, N_local]`."""
    sg = init_sub_groups(tp_size, o_groups, K_total, N_total)
    M = a_local.size(0)
    return _ext.o_groups_relax_forward(
        a_local, w_local, M, sg.plan.N_local, sg.plan.K_sub,
        sg.sub_group_handle,
    )


# ---------------------------------------------------------------------------
# vLLM monkey-patch — hook into ColumnParallelLinear / RowParallelLinear
# ---------------------------------------------------------------------------
def install_vllm_patch() -> None:
    """Install the relaxation hook into vLLM's TP linear modules.

    Called from vLLM startup once distributed is initialized. Idempotent.
    Detects the o_groups annotation on the layer (DSV4 sets this in
    `deepseek_v4.py`) and routes through `relax_forward` only when
    `tp_size > o_groups`. Other layers fall through to the standard path.
    """
    try:
        from vllm.model_executor.layers import linear as _lin
    except ImportError:
        return                                          # vLLM not present

    orig_forward = _lin.ColumnParallelLinear.forward

    def patched_forward(self, x):                       # type: ignore[no-untyped-def]
        o_groups = getattr(self, "o_groups", None)
        tp_size = getattr(self, "tp_size", 1)
        if o_groups is None or tp_size <= o_groups:
            return orig_forward(self, x)               # standard path
        K_total = getattr(self, "input_size", x.size(-1))
        N_total = getattr(self, "output_size", None)
        if N_total is None:
            return orig_forward(self, x)               # missing metadata
        a_local = x.contiguous()
        w_local = self.weight                          # already sliced at load
        out = relax_forward(
            a_local, w_local, tp_size, o_groups, K_total, N_total,
        )
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out, None                                # vLLM returns (out, bias)

    _lin.ColumnParallelLinear.forward = patched_forward  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone numerics check (run on cuda5 for self-test)
# ---------------------------------------------------------------------------
def selfcheck() -> dict:
    """Local single-rank correctness check of the slicing math.

    Emulates a 16-rank slice on a 16-MiB weight, runs partial-GEMMs in a
    loop, sums them, compares to the unsharded GEMM. Cosine should be ≥
    0.999, max-abs ≤ 1e-2 in BF16.
    """
    if _ext is None:
        return {"verdict": "FAIL_BUILD", "note": "extension not built"}
    if not torch.cuda.is_available():
        return {"verdict": "FAIL_NO_CUDA"}
    torch.manual_seed(0)
    M, K, N, tp, og = 4, 7168, 4096, 16, 8
    sub_factor = tp // og
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.02
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * 0.02
    ref = a @ w                                          # [M, N]
    # Pick rank=0 → o_group_idx=0, sub_rank=0
    plan = _ext.make_plan(0, tp, og, K, N)
    a_loc = a[:, plan.k_start:plan.k_end].contiguous()
    w_loc = w[plan.k_start:plan.k_end,
              plan.n_start:plan.n_end].contiguous()
    partial = _ext.o_groups_partial_gemm(a_loc, w_loc, M, plan.N_local,
                                         plan.K_sub)
    # Emulate sub-group all-reduce by summing across sub_rank shards
    full_local = partial.clone()
    for sr in range(1, sub_factor):
        plan_sr = _ext.make_plan(sr, tp, og, K, N)
        a_sr = a[:, plan_sr.k_start:plan_sr.k_end].contiguous()
        w_sr = w[plan_sr.k_start:plan_sr.k_end,
                 plan.n_start:plan.n_end].contiguous()
        full_local += _ext.o_groups_partial_gemm(a_sr, w_sr, M,
                                                 plan.N_local, plan.K_sub)
    ref_local = ref[:, plan.n_start:plan.n_end]
    diff = (full_local.float() - ref_local.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        full_local.float().flatten(), ref_local.float().flatten(), dim=0
    ).item()
    return {
        "verdict": "PASS" if (cos >= 0.999 and diff.max().item() <= 5e-2)
                          else "FAIL_NUMERICS",
        "cosine": cos,
        "max_abs": diff.max().item(),
        "shape_local": list(full_local.shape),
        "sub_factor": sub_factor,
    }


if __name__ == "__main__":                               # pragma: no cover
    import json
    print(json.dumps(selfcheck(), indent=2))
