#!/usr/bin/env bash
# verify_env.sh — Validate host + venv vs build_env.lock.
# Usage: verify_env.sh [--strict] [--json] [--venv DIR]
set -uo pipefail
LOCK="$(dirname "$0")/build_env.lock"
STRICT=0; JSON=0; VENV_CHECK=""
for a in "$@"; do case "$a" in --strict) STRICT=1;; --json) JSON=1;; --venv) shift; VENV_CHECK="$1";; --venv=*) VENV_CHECK="${a#--venv=}";; esac; done
[[ -r "$LOCK" ]] || { echo "lock missing: $LOCK" >&2; exit 2; }
declare -A V; while IFS='=' read -r k v; do [[ -z "$k" || "$k" =~ ^# ]] && continue; V["$k"]="$v"; done < "$LOCK"
fail=0; out=()
chk(){ local lbl="$1" w="$2" g="$3" m="${4:-eq}"; local ok=0
  case "$m" in eq) [[ "$g" == "$w" ]] && ok=1;; ge) [[ "$(printf '%s\n%s\n' "$w" "$g" | sort -V | head -1)" == "$w" ]] && ok=1;; esac
  if [[ $ok == 1 ]]; then out+=("OK $lbl want=$w got=$g"); else out+=("MISMATCH $lbl want=$w got=$g"); fail=1; fi; }
[[ -d "${V[cuda_home]:-}" ]] || { out+=("MISMATCH cuda_home ${V[cuda_home]} not present"); fail=1; }
NV="${V[cuda_home]}/bin/nvcc"; [[ -x "$NV" ]] && chk "nvcc" "${V[cuda_version]}" "$($NV --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')" eq || { out+=("MISMATCH nvcc not found"); fail=1; }
command -v cmake >/dev/null && chk "cmake_min" "${V[cmake_min]}" "$(cmake --version | head -1 | awk '{print $3}')" ge
command -v ninja >/dev/null && chk "ninja_min" "${V[ninja_min]}" "$(ninja --version)" ge
if [[ -n "$VENV_CHECK" && -x "$VENV_CHECK/bin/python" ]]; then
  py="$VENV_CHECK/bin/python"
  chk "venv.vllm" "${V[vllm_version]}" "$($py -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo none)" eq
  chk "venv.torch" "${V[torch_version]}" "$($py -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)" eq
  chk "venv.ray" "${V[ray_version]}" "$($py -c 'import ray; print(ray.__version__)' 2>/dev/null || echo none)" eq
fi
if [[ $JSON == 1 ]]; then printf '{"fail":%s,"results":%s}\n' "$fail" "$(printf '%s\n' "${out[@]}" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().splitlines()))')"
else for r in "${out[@]}"; do echo "[verify_env] $r"; done; echo "[verify_env] OVERALL: $([[ $fail == 1 ]] && echo MISMATCH || echo OK)"; fi
[[ $STRICT == 1 && $fail == 1 ]] && exit 1; exit 0
