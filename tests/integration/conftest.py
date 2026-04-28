# --ProtoAI-Bakari--
# L2 integration fixtures: kernel-in-vLLM-call-chain forward slice tests.
import importlib
import os
import pathlib
import pytest

try:
    import torch
except ImportError:                                   # pragma: no cover
    torch = None

L1_FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "unit" / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _seed_everything():
    if torch is not None:
        torch.manual_seed(0xC0FFEE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0xC0FFEE)


@pytest.fixture(scope="session")
def device():
    if torch is None:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; sm_86 forward-slice requires a GPU")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        pytest.skip(f"sm_{cap[0]}{cap[1]} too old for sm_86 forward-slice")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def vllm_module():
    """Try to import vllm; skip if not installed in env (CC2 build target)."""
    try:
        return importlib.import_module("vllm")
    except ImportError as e:
        pytest.skip(f"vllm not importable: {e}")


@pytest.fixture(scope="session")
def kernels():
    for name in ("vllm._dsa_sm86", "vllm._C", "vllm_dsa_sm86_kernels"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    pytest.skip("sm_86 kernel module not built; run scripts/build_vllm_dsa_sm86.sh")


def load_l1_fixture(name):
    """Reuse the L1 fixture (input tensors + expected) for L2 forward-slice tests."""
    path = L1_FIXTURES / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"L1 fixture {path} missing — run tests/unit/gen_fixtures.py")
    return torch.load(path, weights_only=False)


def assert_close(actual, expected, rtol=2e-2, atol=2e-2, name="forward_slice"):
    a = actual.detach().to(torch.float32).cpu()
    e = expected.detach().to(torch.float32).cpu()
    assert a.shape == e.shape, f"{name}: shape mismatch actual={a.shape} expected={e.shape}"
    diff = (a - e).abs()
    if not torch.allclose(a, e, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name}: forward-slice output diverges from pytorch reference "
            f"(rtol={rtol} atol={atol}); max_abs_diff={diff.max().item():.6g}"
        )
