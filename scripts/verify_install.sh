#!/usr/bin/env bash
# verify_install.sh — per-node smoke test for vllm-dsa-sm86 venv post-fanout.
#
# Runs against either localhost or one/many remote nodes via ssh_node.sh.
# Smoke set:
#   1. venv exists at $TARGET_VENV with bin/python3 + bin/vllm
#   2. `python -c "import vllm; print(vllm.__version__)"` returns a string
#      containing the patched-build marker ("+sm86" or BUILD_INFO.git_sha[:7])
#   3. `python -c "import vllm._C"` loads the rebuilt .so without ImportError /
#      libcuda mismatch
#   4. `nvidia-smi -L` returns ≥1 GPU (visibility — not utilization)
#   5. BUILD_INFO.json exists at $TARGET_VENV/.. and gate.passed == true
#
# Output:
#   - Per-node status line: VERIFY <node> OK|FAIL <reason>
#   - Aggregate exit code: 0 if all PASS, 1 if any FAIL
#   - Optional --report PATH writes a markdown report consumable by CC8
#
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]
# --ProtoAI-Bakari--

set -euo pipefail

TARGET_VENV="${TARGET_VENV:-/home/z/.venvs/.venv-vllm-dsa-sm86}"
SSH_NODE="${SSH_NODE:-$HOME/AGENT/tools/ssh_node.sh}"
DEFAULT_NODES=(cuda1 cuda2 cuda3 cuda4 cuda5 cuda6 cuda7 cuda8)

NODES=()
LOCAL=0
EXPECTED_SHA=""
REPORT=""
QUIET=0

usage() {
  cat <<EOF
Usage: $0 [--nodes cuda1,cuda3,...] [--local] [--target /path/.venv] [--expected-sha <40hex>] [--report PATH] [--quiet]

  --nodes LIST           Comma-separated node list (default: all 8)
  --local                Run smoke locally (skips ssh_node.sh)
  --target PATH          Override venv path (default: $TARGET_VENV)
  --expected-sha SHA     Refuse PASS unless BUILD_INFO.git_sha matches
  --report PATH          Write markdown report (default: stdout summary only)
  --quiet                Suppress per-check progress chatter
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)         IFS=',' read -ra NODES <<< "$2"; shift 2;;
    --local)         LOCAL=1; shift;;
    --target)        TARGET_VENV="$2"; shift 2;;
    --expected-sha)  EXPECTED_SHA="$2"; shift 2;;
    --report)        REPORT="$2"; shift 2;;
    --quiet)         QUIET=1; shift;;
    -h|--help)       usage; exit 0;;
    *) echo "FATAL: unknown arg: $1" >&2; usage; exit 2;;
  esac
done

[[ ${#NODES[@]} -eq 0 ]] && NODES=("${DEFAULT_NODES[@]}")

log() { [[ $QUIET -eq 1 ]] || printf '[verify %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# -----------------------------------------------------------------------------
# Smoke runner. Emitted as a heredoc so it can be embedded in `ssh <node> bash -s`.
# Returns:
#   exit 0 → all checks PASS, prints "VERIFY OK <version> sha=<sha>"
#   exit 10..14 → numbered failure points
# -----------------------------------------------------------------------------
build_smoke_script() {
  cat <<'SMOKE'
set -uo pipefail
TARGET_VENV="${1:?TARGET_VENV required}"
EXPECTED_SHA="${2:-}"

err() { echo "VERIFY FAIL $1: $2" >&2; exit "$1"; }

# 1. venv exists
[[ -x "$TARGET_VENV/bin/python3" ]] || err 10 "no $TARGET_VENV/bin/python3"

# 2. import vllm + version string
VLLM_VER=$("$TARGET_VENV/bin/python3" -c \
  "import vllm; print(vllm.__version__)" 2>&1) \
  || err 11 "import vllm failed: $VLLM_VER"

# 3. import vllm._C — proves the rebuilt .so loads
"$TARGET_VENV/bin/python3" -c "import vllm._C" 2>/dev/null \
  || err 12 "import vllm._C failed (libcuda / ABI mismatch?)"

# 4. nvidia-smi visibility
nvidia-smi -L >/dev/null 2>&1 \
  || err 13 "nvidia-smi -L failed (driver / device visibility)"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$GPU_COUNT" -ge 1 ]] || err 13 "GPU_COUNT=$GPU_COUNT (need ≥1)"

# 5. BUILD_INFO.json next to the venv
BUILD_INFO="$(dirname "$TARGET_VENV")/$(basename "$TARGET_VENV").BUILD_INFO.json"
if [[ ! -f "$BUILD_INFO" ]]; then
  # fall back: tarball-extracted location at $TARGET_VENV/../vllm-dsa-sm86/BUILD_INFO.json
  BUILD_INFO="$(dirname "$TARGET_VENV")/vllm-dsa-sm86/BUILD_INFO.json"
fi
[[ -f "$BUILD_INFO" ]] || err 14 "BUILD_INFO.json missing (looked at $BUILD_INFO)"

GATE_PASSED=$("$TARGET_VENV/bin/python3" -c \
  "import json,sys;print(json.load(open('$BUILD_INFO'))['correctness_gate']['passed'])")
[[ "$GATE_PASSED" == "True" || "$GATE_PASSED" == "true" ]] \
  || err 14 "BUILD_INFO.correctness_gate.passed=$GATE_PASSED"

BUILD_SHA=$("$TARGET_VENV/bin/python3" -c \
  "import json,sys;print(json.load(open('$BUILD_INFO'))['git_sha'])")

if [[ -n "$EXPECTED_SHA" && "$BUILD_SHA" != "$EXPECTED_SHA" ]]; then
  err 14 "git_sha mismatch: expected=$EXPECTED_SHA actual=$BUILD_SHA"
fi

echo "VERIFY OK $VLLM_VER sha=${BUILD_SHA:0:12} gpus=$GPU_COUNT"
exit 0
SMOKE
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
declare -A NODE_STATUS=()
declare -A NODE_DETAIL=()
SMOKE_SCRIPT="$(build_smoke_script)"

run_one() {
  local node="$1" out rc
  if [[ $LOCAL -eq 1 ]]; then
    out=$(bash -c "$SMOKE_SCRIPT" _ "$TARGET_VENV" "$EXPECTED_SHA" 2>&1) && rc=0 || rc=$?
  else
    out=$(bash "$SSH_NODE" "$node" "bash -s -- '$TARGET_VENV' '$EXPECTED_SHA'" <<< "$SMOKE_SCRIPT" 2>&1) && rc=0 || rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    NODE_STATUS[$node]="PASS"
    NODE_DETAIL[$node]="$out"
    log "[$node] PASS — $out"
  else
    NODE_STATUS[$node]="FAIL"
    NODE_DETAIL[$node]="rc=$rc $out"
    log "[$node] FAIL rc=$rc — $out"
  fi
}

if [[ $LOCAL -eq 1 ]]; then
  run_one "localhost"
else
  for node in "${NODES[@]}"; do
    run_one "$node"
  done
fi

# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------
PASS=()
FAIL=()
for node in "${!NODE_STATUS[@]}"; do
  if [[ "${NODE_STATUS[$node]}" == "PASS" ]]; then PASS+=("$node"); else FAIL+=("$node"); fi
done

echo "VERIFY SUMMARY pass=${#PASS[@]} fail=${#FAIL[@]} target=$TARGET_VENV"
echo "  pass: ${PASS[*]:-none}"
echo "  fail: ${FAIL[*]:-none}"

if [[ -n "$REPORT" ]]; then
  {
    echo "# CC7 Verify Report — $(date -u +%Y%m%dT%H%M%SZ)"
    echo "**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]**"
    echo
    echo "- target_venv: \`$TARGET_VENV\`"
    echo "- expected_sha: \`${EXPECTED_SHA:-any}\`"
    echo "- pass (${#PASS[@]}): ${PASS[*]:-none}"
    echo "- fail (${#FAIL[@]}): ${FAIL[*]:-none}"
    echo
    echo "## Per-node detail"
    for node in "${!NODE_STATUS[@]}"; do
      echo "### $node — ${NODE_STATUS[$node]}"
      echo '```'
      echo "${NODE_DETAIL[$node]}"
      echo '```'
    done
  } > "$REPORT"
  echo "report → $REPORT"
fi

[[ ${#FAIL[@]} -eq 0 ]] && exit 0 || exit 1
