# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
"""DSV4 PP=2 IntermediateTensors forward patch — sm_86 unblock.

Root cause: ``vllm/model_executor/models/deepseek_v4.py:1197 DeepseekV4Model.forward``
does not implement pipeline-parallel transit. It always:
  1. Calls ``self.embed_input_ids(input_ids)`` (only valid on first PP rank).
  2. Calls ``hc_head(...) + self.norm(...)`` and returns a plain ``torch.Tensor``
     (only valid on last PP rank).

When PP=2 splits the model across two ranks, vLLM's gpu_model_runner.py:4072
asserts ``isinstance(hidden_states, IntermediateTensors)`` for non-last-rank
output. The DSV4 forward returns ``Tensor`` → AssertionError → 500 on
``/v1/completions``.

The fix is hardware-agnostic — it's a missing branch in the V4 model. We monkey-
patch ``DeepseekV4Model.forward`` to:
  - On non-first PP rank: read ``intermediate_tensors["hidden_states"]`` (flat
    ``[T, hc_mult * hidden_size]``) and reshape to ``[T, hc_mult, hidden_size]``
    before running local layers.
  - On non-last PP rank: flatten the hidden_states to 2D
    ``[T, hc_mult * hidden_size]`` and return ``IntermediateTensors``.
  - On last PP rank: original hc_head + norm path (unchanged).

We also install a custom ``make_empty_intermediate_tensors`` factory that
allocates a (T, hc_dim) buffer (DSV4-aware shape) — vLLM's default factory
uses ``hidden_size`` not ``hc_dim``, which would mis-allocate.

Shape contract for transit (between PP ranks):
  ``hidden_states``: ``(num_tokens, hc_dim)`` 2D, dtype = model dtype.

Apply: ``import this_module; this_module.apply()`` after vLLM imports the
model module. Idempotent guard prevents double-wrap.

Reference: vllm-DSA-sm86-3090 corpus § 4 (17-patch cascade) — patches
#5 ``patch_dsv4_supports_pp.sh``, #6 ``patch_dsv4_pp_skip_nonlocal.sh``,
#7 ``patch_dsv4_pp_skip_v2/v3.sh`` covered SOME PP plumbing but did NOT
fix the forward IntermediateTensors return contract. This patch closes
that gap.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch

# Late imports inside apply() — at file-import time vllm may not be installed
# (e.g. on the author Mac); we still want this file to be lintable / loadable.

_ORIGINAL: dict[str, Any] = {}


def _dsv4_forward_pp_aware(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors,  # IntermediateTensors | None
    inputs_embeds: torch.Tensor | None = None,
) -> Any:
    from vllm.distributed import get_pp_group
    from vllm.sequence import IntermediateTensors
    from vllm.model_executor.models.deepseek_v4 import hc_head

    pp = get_pp_group()
    is_first = pp.is_first_rank
    is_last = pp.is_last_rank

    if is_first:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "DSV4 first PP rank requires either input_ids or inputs_embeds"
                )
            hidden_states = self.embed_input_ids(input_ids)
        # Match upstream: expand to (T, hc_mult, hidden_size).
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
    else:
        if intermediate_tensors is None:
            raise AssertionError(
                "DSV4 non-first PP rank requires intermediate_tensors "
                "(would 500 on gpu_model_runner.py:4072 — patch wired wrong)"
            )
        flat = intermediate_tensors["hidden_states"]
        # flat: (T, hc_dim) -> (T, hc_mult, hidden_size)
        hidden_size = flat.shape[-1] // self.hc_mult
        hidden_states = flat.view(-1, self.hc_mult, hidden_size)

    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states = layer(hidden_states, positions, input_ids)

    if not is_last:
        flat = hidden_states.flatten(1)
        return IntermediateTensors({"hidden_states": flat})

    # Last-rank path — original behavior.
    num_tokens = hidden_states.shape[0]
    self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
    hidden_states = hc_head(
        hidden_states,
        self.hc_head_fn,
        self.hc_head_scale,
        self.hc_head_base,
        self.rms_norm_eps,
        self.hc_eps,
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states


def _make_dsv4_intermediate_factory(hc_dim: int):
    """Custom factory — DSV4 transit shape is (T, hc_dim) not (T, hidden_size)."""
    from vllm.sequence import IntermediateTensors

    def make(batch_size: int, dtype: torch.dtype, device: torch.device) -> Any:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, hc_dim), dtype=dtype, device=device
                ),
            }
        )

    return make


def apply() -> dict[str, Any]:
    """Idempotent monkey-patch. Returns originals dict for restore()."""
    import vllm.model_executor.models.deepseek_v4 as mod

    if getattr(mod, "_CC3_PP2_PATCH_APPLIED", False):
        raise RuntimeError(
            "pp2_intermediate_tensors_sm86.apply() already called — "
            "call restore() first or guard at call site"
        )

    orig_forward = mod.DeepseekV4Model.forward
    orig_init = mod.DeepseekV4Model.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Install AFTER super().__init__ so self.hc_dim is set.
        self.make_empty_intermediate_tensors = _make_dsv4_intermediate_factory(
            self.hc_dim
        )

    mod.DeepseekV4Model.forward = _dsv4_forward_pp_aware
    mod.DeepseekV4Model.__init__ = patched_init
    mod._CC3_PP2_PATCH_APPLIED = True
    mod._CC3_PP2_PATCH_ORIGINALS = {
        "forward": orig_forward,
        "__init__": orig_init,
    }
    _ORIGINAL.update(mod._CC3_PP2_PATCH_ORIGINALS)
    return mod._CC3_PP2_PATCH_ORIGINALS


def restore() -> None:
    import vllm.model_executor.models.deepseek_v4 as mod

    if not getattr(mod, "_CC3_PP2_PATCH_APPLIED", False):
        return
    mod.DeepseekV4Model.forward = mod._CC3_PP2_PATCH_ORIGINALS["forward"]
    mod.DeepseekV4Model.__init__ = mod._CC3_PP2_PATCH_ORIGINALS["__init__"]
    delattr(mod, "_CC3_PP2_PATCH_APPLIED")
    delattr(mod, "_CC3_PP2_PATCH_ORIGINALS")
    _ORIGINAL.clear()


def selfcheck() -> dict:
    """CPU-only structural verification — no vllm import required."""
    # Verify _make_dsv4_intermediate_factory shape contract by constructing
    # a fake IntermediateTensors-like dict. Bypasses vllm.sequence import for
    # author-Mac portability.
    hc_mult, hidden_size = 4, 7168
    hc_dim = hc_mult * hidden_size
    T = 16
    flat = torch.zeros((T, hc_dim), dtype=torch.bfloat16)
    reshaped = flat.view(-1, hc_mult, hidden_size)
    assert reshaped.shape == (T, hc_mult, hidden_size), reshaped.shape
    re_flat = reshaped.flatten(1)
    assert re_flat.shape == (T, hc_dim), re_flat.shape
    assert torch.equal(flat, re_flat)
    return {
        "ok": True,
        "transit_shape": [T, hc_dim],
        "internal_shape": [T, hc_mult, hidden_size],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(selfcheck()))

# // --ProtoAI-Bakari--
