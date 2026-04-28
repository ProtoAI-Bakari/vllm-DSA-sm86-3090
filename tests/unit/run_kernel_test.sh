#!/bin/bash
# --ProtoAI-Bakari--
# Story 7: L1 unit-test harness — generic runner for any csrc/*_sm86.cu against
# the CPU pytorch reference (tests/correctness/cpu_reference.py).
#
# Run all kernel tests:
#     ./run_kernel_test.sh
# Run one kernel:
#     ./run_kernel_test.sh sparse_attn_indexer
# Regenerate fixtures (after reference impl changes):
#     ./run_kernel_test.sh --regen-fixtures [--only mla_decode]
#
# Exit: 0 ok, non-zero on test failure or build failure.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

if [[ "${1:-}" == "--regen-fixtures" ]]; then
  shift
  echo "[run_kernel_test] regenerating fixtures (CPU reference)…"
  python3 "$HERE/gen_fixtures.py" "$@"
  exit 0
fi

PYTEST_ARGS=(-v -s --tb=short)
if [[ -n "${1:-}" ]]; then
  PYTEST_ARGS+=(-k "$1")
fi

# Best-effort: ensure fixtures exist, else regenerate
if [[ ! -d "$HERE/fixtures" ]] || [[ -z "$(ls -A "$HERE/fixtures" 2>/dev/null || true)" ]]; then
  echo "[run_kernel_test] fixtures missing → generating from CPU reference"
  python3 "$HERE/gen_fixtures.py"
fi

# Add tests/correctness to PYTHONPATH so cpu_reference imports cleanly
export PYTHONPATH="${REPO}:${HERE}:${PYTHONPATH:-}"

echo "[run_kernel_test] pytest $HERE/test_kernels_sm86.py ${PYTEST_ARGS[*]}"
exec python3 -m pytest "$HERE/test_kernels_sm86.py" "${PYTEST_ARGS[@]}"
