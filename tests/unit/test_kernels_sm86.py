"""
L1 unit tests for sm_86 kernel ports.

Each test:
  1. loads a pre-generated fixture (input tensors + expected output from
     CPU pytorch reference).
  2. runs the sm_86 kernel under test on a CUDA device.
  3. compares output to expected within rtol/atol per kernel.

Skipped if CUDA unavailable or sm < 8.0.
The kernels are imported from `vllm._dsa_sm86` (the rebuilt extension CC2 produces).
If the import fails, the test is skipped with the build error so CC9's L4 driver
can distinguish 'not built yet' from 'numerics fail'.
"""
import importlib
import pytest

from conftest import load_fixture, assert_close

# --ProtoAI-Bakari--

KERNEL_MODULE_CANDIDATES = (
    "vllm._dsa_sm86",                 # post-CC2 build target
    "vllm._C",                        # fallback if CC2 lands the kernels in main _C
    "vllm_dsa_sm86_kernels",          # standalone test-build pkg
)


def _import_kernels():
    last_err = None
    for mod_name in KERNEL_MODULE_CANDIDATES:
        try:
            return importlib.import_module(mod_name)
        except ImportError as e:
            last_err = e
            continue
    pytest.skip(f"sm_86 kernel module not built; tried {KERNEL_MODULE_CANDIDATES}: {last_err}")


@pytest.fixture(scope="module")
def kernels():
    return _import_kernels()


# ---------- sparse_attn_indexer (CC3) ----------
def test_sparse_attn_indexer(kernels, device):
    fix = load_fixture("sparse_attn_indexer")
    q = fix["inputs"]["q"].to(device)
    k = fix["inputs"]["k_cache"].to(device)
    fn = getattr(kernels, "sparse_attn_indexer_sm86", None) or getattr(kernels, "sparse_attn_indexer", None)
    if fn is None:
        pytest.skip("sparse_attn_indexer not exported")
    out = fn(q, k, **fix["kwargs"])
    # indexer returns int64 indices — exact match required (no rtol)
    expected = fix["expected"].to(out.device)
    assert (out == expected).all(), (
        f"index disagreement: actual={out.flatten()[:32]} expected={expected.flatten()[:32]}"
    )


# ---------- mla_decode (CC4 lightning bundle) ----------
def test_mla_decode(kernels, device):
    fix = load_fixture("mla_decode")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    fn = getattr(kernels, "mla_decode_sm86", None) or getattr(kernels, "mla_decode", None)
    if fn is None:
        pytest.skip("mla_decode not exported")
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="mla_decode")


# ---------- compressor (CC4 lightning bundle) ----------
def test_compressor(kernels, device):
    fix = load_fixture("compressor")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    fn = getattr(kernels, "compressor_sm86", None) or getattr(kernels, "compressor", None)
    if fn is None:
        pytest.skip("compressor not exported")
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="compressor")


# ---------- swa (CC4 lightning bundle) ----------
def test_swa(kernels, device):
    fix = load_fixture("swa")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    fn = getattr(kernels, "swa_sm86", None) or getattr(kernels, "swa", None)
    if fn is None:
        pytest.skip("swa not exported")
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=2e-2, atol=2e-2, name="swa")


# ---------- marlin_int4_gemm (CC5) ----------
def test_marlin_int4_gemm(kernels, device):
    fix = load_fixture("marlin_int4_gemm")
    inp = {k: v.to(device) for k, v in fix["inputs"].items()}
    fn = (
        getattr(kernels, "marlin_int4_gemm_sm86", None)
        or getattr(kernels, "marlin_gemm_w4a16", None)
        or getattr(kernels, "marlin_int4_gemm", None)
    )
    if fn is None:
        pytest.skip("marlin_int4_gemm not exported")
    out = fn(**inp, **fix["kwargs"])
    assert_close(out, fix["expected"], rtol=5e-2, atol=5e-2, name="marlin_int4_gemm")
