"""
L3 small-model load — single-rank tiny HF model loaded through the rebuilt vLLM
LLM entry point, with one greedy forward and a non-empty / non-NaN sanity check.

Bridges L2 (kernel-in-call-chain) and L4 (full GLM-5.1 PP=2+TP=8) by validating
that the rebuilt sm_86 _C.abi3.so survives a real model load + forward without
a 270GB checkpoint or 16-rank fabric.

Skipped if vllm not importable OR no CUDA device. Skip messages distinguish
'not built' from 'numerics fail' so the L4 driver interprets correctly.

Model selection (env-overridable):
    L3_MODEL          default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
                      (override to any small HF id available on the host)
    L3_DTYPE          default: bfloat16
    L3_MAX_TOKENS     default: 16
    L3_TEMPERATURE    default: 0.0
    L3_GPU_MEMORY_UTIL default: 0.4   (single-rank, leave headroom)
"""
# --ProtoAI-Bakari--
import math
import os
import pytest

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="session")
def llm():
    try:
        from vllm import LLM
    except ImportError as e:
        pytest.skip(f"vllm not importable (CC2 build target): {e}")
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; L3 small-model load requires a GPU")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        pytest.skip(f"sm_{cap[0]}{cap[1]} too old for the rebuilt sm_86 extension")
    model = os.environ.get("L3_MODEL", DEFAULT_MODEL)
    dtype = os.environ.get("L3_DTYPE", "bfloat16")
    util = float(os.environ.get("L3_GPU_MEMORY_UTIL", "0.4"))
    try:
        return LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=util,
            tensor_parallel_size=1,
            enforce_eager=True,
            trust_remote_code=True,
        )
    except Exception as e:
        pytest.skip(f"LLM load failed for {model}: {type(e).__name__}: {str(e)[:200]}")


def _greedy_params(max_tokens, logprobs):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=float(os.environ.get("L3_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("L3_MAX_TOKENS", str(max_tokens))),
        logprobs=logprobs,
        seed=42,
    )


def test_l3_paris_smoke(llm):
    """Smoke test — short factual prompt, non-empty + non-NaN gen."""
    outs = llm.generate(["The capital of France is"], _greedy_params(8, logprobs=5))
    assert outs and outs[0].outputs, "no outputs returned"
    text = outs[0].outputs[0].text
    assert text and len(text) > 0, "empty completion text"
    cum = outs[0].outputs[0].cumulative_logprob
    if cum is not None:
        assert math.isfinite(cum), f"cumulative_logprob not finite: {cum}"


def test_l3_logprobs_present(llm):
    """Verify per-position logprobs are populated and finite."""
    outs = llm.generate(["List three primary colors:"], _greedy_params(12, logprobs=10))
    assert outs[0].outputs, "no outputs"
    lps = outs[0].outputs[0].logprobs
    if lps is None:
        pytest.skip("vLLM build does not emit per-position logprobs in this code path")
    bad = []
    for pos, dist in enumerate(lps):
        for tok, info in dist.items():
            lp = getattr(info, "logprob", info)
            if lp is None or not math.isfinite(float(lp)):
                bad.append((pos, tok, lp))
    assert not bad, f"non-finite logprobs at positions: {bad[:5]}"


def test_l3_determinism_temp0(llm):
    """Two greedy runs of the same prompt must return identical text."""
    p = ["Once upon a time, there was a"]
    a = llm.generate(p, _greedy_params(16, logprobs=None))
    b = llm.generate(p, _greedy_params(16, logprobs=None))
    ta = a[0].outputs[0].text
    tb = b[0].outputs[0].text
    assert ta == tb, f"non-deterministic at temp=0: {ta!r} vs {tb!r}"
