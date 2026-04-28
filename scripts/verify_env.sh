#!/usr/bin/env bash
# verify_env.sh — Verify a build host matches scripts/build_env.lock.
#
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC2]
#
# Use:
#   bash scripts/verify_env.sh              # checks current host against build_env.lock
#   bash scripts/verify_env.sh --strict     # exit 1 on any mismatch (default warns)
#   bash scripts/verify_env.sh --json       # machine-readable
#   bash scripts/verify_env.sh --venv DIR   # also verify venv has correct torch/vllm/ray/triton
#
# Returns exit 0 = all good, 1 = mismatch under --strict, 2 = lock file unreadable.

set -uo pipefail

LOCK="${LOCK:-$(dirname "$0")/build_env.lock}"
STRICT=0
JSON=0
VENV_CHECK=""
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        --json)   JSON=1 ;;
        --venv)   shift; VENV_CHECK="$1" ;;
        --venv=*) VENV_CHECK="${arg#--venv=}" ;;
        --help|-h)
            grep -E '^# ' "$0" | head -20
            exit 0
            ;;
    esac
done

[[ -r "$LOCK" ]] || { echo "[verify_env] FATAL: lock file unreadable: $LOCK" >&2; exit 2; }

# Parse lock into env-style assoc (compatible with bash 4+; we test inline).
declare -A LOCKVALS=()
while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" =~ ^# ]] && continue
    LOCKVALS["$k"]="$v"
done < "$LOCK"

results=()
fail=0

check() {
    local label="$1" want="$2" got="$3" mode="${4:-eq}"
    local ok=0
    case "$mode" in
        eq)   [[ "$got" == "$want" ]] && ok=1 ;;
        ge)   # version compare
              if [[ "$(printf '%s\n%s\n' "$want" "$got" | sort -V | head -1)" == "$want" ]]; then ok=1; fi ;;
        contains) [[ "$got" == *"$want"* ]] && ok=1 ;;
    esac
    if [[ "$ok" == "1" ]]; then
        results+=("OK    $label  want=$want  got=$got")
    else
        results+=("MISMATCH  $label  want=$want  got=$got  mode=$mode")
        fail=1
    fi
}

# === toolchain checks ===
if [[ -n "${LOCKVALS[cuda_home]:-}" ]]; then
    if [[ -d "${LOCKVALS[cuda_home]}" ]]; then
        results+=("OK    cuda_home  path=${LOCKVALS[cuda_home]}")
    else
        results+=("MISMATCH  cuda_home  path=${LOCKVALS[cuda_home]} not present")
        fail=1
    fi
fi
NVCC_BIN="${LOCKVALS[cuda_home]:-/usr/local/cuda}/bin/nvcc"
if [[ -x "$NVCC_BIN" ]]; then
    NVCC_VER="$($NVCC_BIN --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    check "nvcc_version" "${LOCKVALS[cuda_version]:-?}" "${NVCC_VER:-none}" eq
else
    results+=("MISMATCH  nvcc not found at $NVCC_BIN")
    fail=1
fi
if command -v cmake >/dev/null 2>&1; then
    CMAKE_VER="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
    check "cmake_min" "${LOCKVALS[cmake_min]:-3.0.0}" "${CMAKE_VER:-0}" ge
else
    results+=("MISMATCH  cmake not found"); fail=1
fi
if command -v ninja >/dev/null 2>&1; then
    NINJA_VER="$(ninja --version 2>/dev/null)"
    check "ninja_min" "${LOCKVALS[ninja_min]:-1.10.0}" "${NINJA_VER:-0}" ge
else
    results+=("MISMATCH  ninja not found"); fail=1
fi
if command -v gcc >/dev/null 2>&1; then
    GCC_VER="$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion 2>/dev/null)"
    check "gcc_min" "${LOCKVALS[gcc_min]:-9.0.0}" "${GCC_VER:-0}" ge
fi

# === venv checks (optional --venv) ===
if [[ -n "$VENV_CHECK" ]]; then
    if [[ -x "$VENV_CHECK/bin/python" ]]; then
        py="$VENV_CHECK/bin/python"
        VLLM_V="$($py -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo none)"
        TORCH_V="$($py -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
        RAY_V="$($py -c 'import ray; print(ray.__version__)' 2>/dev/null || echo none)"
        TRITON_V="$($py -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"
        check "venv.vllm"   "${LOCKVALS[vllm_version]:-?}"   "$VLLM_V"   eq
        check "venv.torch"  "${LOCKVALS[torch_version]:-?}"  "$TORCH_V"  eq
        check "venv.ray"    "${LOCKVALS[ray_version]:-?}"    "$RAY_V"    eq
        check "venv.triton_min" "${LOCKVALS[triton_version_floor]:-3.0.0}" "$TRITON_V" ge
    else
        results+=("MISMATCH  venv $VENV_CHECK has no python"); fail=1
    fi
fi

# === report ===
if [[ "$JSON" == "1" ]]; then
    printf '{"results":['
    sep=""
    for r in "${results[@]}"; do
        st="${r%% *}"
        body="${r#* }"
        body="${body# }"
        body="${body//\"/\\\"}"
        printf '%s{"status":"%s","detail":"%s"}' "$sep" "$st" "$body"
        sep=","
    done
    printf '],"fail":%s}\n' "$fail"
else
    for r in "${results[@]}"; do echo "[verify_env] $r"; done
    if [[ "$fail" == "1" ]]; then
        echo "[verify_env] OVERALL: MISMATCH"
    else
        echo "[verify_env] OVERALL: OK"
    fi
fi

if [[ "$STRICT" == "1" && "$fail" == "1" ]]; then
    exit 1
fi
exit 0
