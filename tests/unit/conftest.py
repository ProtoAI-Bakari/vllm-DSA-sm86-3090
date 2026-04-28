# --ProtoAI-Bakari--
# Pytest fixtures for L1 unit tests on sm_86 kernels.
# Tests must be deterministic + reproducible across nodes.
import os
import pathlib
import pytest

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _seed_everything():
    if torch is not None:
        torch.manual_seed(0xDEADBEEF)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0xDEADBEEF)


@pytest.fixture(scope="session")
def device():
    if torch is None:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; sm_86 kernel tests require a GPU")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        pytest.skip(f"sm_{cap[0]}{cap[1]} too old for sm_86 kernel tests")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def fixtures_dir():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


def load_fixture(name):
    """Load a torch fixture pickle from tests/unit/fixtures/<name>.pt."""
    path = FIXTURES_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"fixture {path} missing — generate via tests/unit/gen_fixtures.py")
    return torch.load(path, weights_only=False)


def assert_close(actual, expected, rtol=1e-3, atol=1e-4, name="tensor"):
    """Wraps torch.allclose with a useful failure message."""
    a = actual.detach().to(torch.float32).cpu()
    e = expected.detach().to(torch.float32).cpu()
    assert a.shape == e.shape, f"{name}: shape mismatch actual={a.shape} expected={e.shape}"
    diff = (a - e).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (e.abs() + 1e-12)).max().item()
    if not torch.allclose(a, e, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name}: not close (rtol={rtol} atol={atol}); "
            f"max_abs_diff={max_abs:.6g} max_rel_diff={max_rel:.6g}"
        )
