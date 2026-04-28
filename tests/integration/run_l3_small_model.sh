#!/bin/bash
# --ProtoAI-Bakari--
# Story 13 (W2): L3 small-model load harness.
#
# Loads a tiny HF model through the rebuilt vLLM (sm_86 _C.abi3.so) on a single
# rank, runs greedy generations, asserts non-empty/non-NaN/deterministic.
#
# Run:
#     ./run_l3_small_model.sh                       # uses TinyLlama 1.1B default
#     L3_MODEL=Qwen/Qwen2.5-0.5B-Instruct ./run_l3_small_model.sh
#     L3_GPU_MEMORY_UTIL=0.5 ./run_l3_small_model.sh
#
# Exit: 0 ok, non-zero on failure.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

PYTEST_ARGS=(-v --tb=short)
[[ -n "${VERBOSE:-}" ]] && PYTEST_ARGS+=(-s --log-cli-level=INFO)
[[ -n "${1:-}" ]] && PYTEST_ARGS+=(-k "$1")

export PYTHONPATH="${REPO}:${HERE}:${PYTHONPATH:-}"
echo "[run_l3_small_model] L3_MODEL=${L3_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
echo "[run_l3_small_model] pytest $HERE/test_l3_small_model.py ${PYTEST_ARGS[*]}"
exec python3 -m pytest "$HERE/test_l3_small_model.py" "${PYTEST_ARGS[@]}"
