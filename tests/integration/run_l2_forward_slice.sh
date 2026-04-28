#!/bin/bash
# --ProtoAI-Bakari--
# Story 8: L2 forward-slice harness — runs each sm_86 kernel inside vLLM's call
# chain on a tiny synthetic fixture and compares to the L1 pytorch reference.
#
# Run all kernels:
#     ./run_l2_forward_slice.sh
# Run one:
#     ./run_l2_forward_slice.sh sparse_attn_indexer
# Verbose (per-test diff dump):
#     VERBOSE=1 ./run_l2_forward_slice.sh
#
# Exit: 0 ok, non-zero on test failure or build/import skip.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

PYTEST_ARGS=(-v --tb=short)
[[ -n "${VERBOSE:-}" ]] && PYTEST_ARGS+=(-s --log-cli-level=DEBUG)
[[ -n "${1:-}" ]] && PYTEST_ARGS+=(-k "$1")

# Ensure L1 fixtures exist (L2 reuses them as ground truth)
if [[ ! -d "$REPO/tests/unit/fixtures" ]] || [[ -z "$(ls -A "$REPO/tests/unit/fixtures" 2>/dev/null || true)" ]]; then
  echo "[run_l2_forward_slice] L1 fixtures missing → generating from CPU reference"
  python3 "$REPO/tests/unit/gen_fixtures.py"
fi

export PYTHONPATH="${REPO}:${HERE}:${PYTHONPATH:-}"
echo "[run_l2_forward_slice] pytest $HERE/test_l2_forward_slice.py ${PYTEST_ARGS[*]}"
exec python3 -m pytest "$HERE/test_l2_forward_slice.py" "${PYTEST_ARGS[@]}"
