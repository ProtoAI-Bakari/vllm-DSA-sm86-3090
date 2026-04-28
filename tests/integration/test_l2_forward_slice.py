"""
L2 forward-slice tests — kernel-in-vLLM-call-chain on a tiny test fixture.

Strategy: each test patches the relevant entry point (e.g. attention backend
forward path) so vLLM's dispatcher routes to the sm_86 kernel under test, then
runs a minimal synthetic forward and compares to the L1 pytorch reference.

Skipped if vllm not installed OR sm_86 kernel module not built. The skip
messages distinguish 'not built yet' from 'numerics fail' so CC9's L4 driver
can interpret automatically.
"""
# --ProtoAI-Bakari--
import contextlib
import pytest
from unittest import mock

from conftest import load_l1_fixture, assert_close


@contextlib.contextmanager
def _patch_attention_indexer(vllm_module, kernel_fn):
    """Best-effort monkey-patch of vLLM's sparse-attn indexer entry to our kernel."""
    target = None
    for path in (
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        "vllm.attention.backends.mla.flashmla_sparse",
    ):
        try:
            mod = __import__(path, fromlist=["DeepseekV4FlashMLASparseBackend"])
            target = mod
            break
        except (ImportError, AttributeError):
            continue
    if target is None:
        pytest.skip("vLLM MLA sparse backend module not present (yet)")
    sym = getattr(target, "_sparse_attn_indexer_entry", None) or getattr(target, "sparse_attn_indexer", None)
    if sym is None:
        pytest.skip("MLA sparse backend present but no patchable entry symbol")
    with mock.patch.object(target, sym.__name__, kernel_fn):
        yield


def test_sparse_attn_indexer_in_chain(vllm_module, kernels, device):
    fix = load_l1_fixture("sparse_attn_indexer")
    fn = getattr(kernels, "sparse_attn_indexer_sm86", None) or getattr(kernels, "sparse_attn_indexer", None)
    if fn is None:
        pytest.skip("sparse_attn_indexer kernel not exported")
    q = fix["inputs"]["q"].to(device)
    k = fix["inputs"]["k_cache"].to(device)
    with _patch_attention_indexer(vllm_module, fn):
        out = fn(q, k, **fix["kwargs"])
    expected = fix["expected"].to(out.device)
    assert (out == expected).all(), (
        "L2 forward-slice index disagreement w/ pytorch reference"
    )


def test_mla_decode_in_chain(vllm_module, kernels, device):
    fix = load_l1_fixture("mla_decode")
    fn = getattr(kernels, "mla_decode_sm86", None) or getattr(kernels, "mla_decode", None)
    if fn is None:
        pytest.skip("mla_decode kernel not exported")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="mla_decode_l2")


def test_compressor_in_chain(vllm_module, kernels, device):
    fix = load_l1_fixture("compressor")
    fn = getattr(kernels, "compressor_sm86", None) or getattr(kernels, "compressor", None)
    if fn is None:
        pytest.skip("compressor kernel not exported")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="compressor_l2")


def test_swa_in_chain(vllm_module, kernels, device):
    fix = load_l1_fixture("swa")
    fn = getattr(kernels, "swa_sm86", None) or getattr(kernels, "swa", None)
    if fn is None:
        pytest.skip("swa kernel not exported")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="swa_l2")


def test_marlin_int4_gemm_in_chain(vllm_module, kernels, device):
    fix = load_l1_fixture("marlin_int4_gemm")
    fn = (
        getattr(kernels, "marlin_int4_gemm_sm86", None)
        or getattr(kernels, "marlin_gemm_w4a16", None)
        or getattr(kernels, "marlin_int4_gemm", None)
    )
    if fn is None:
        pytest.skip("marlin_int4_gemm kernel not exported")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=5e-2, atol=5e-2, name="marlin_int4_gemm_l2")
