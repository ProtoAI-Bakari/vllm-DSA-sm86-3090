#!/usr/bin/env bash
# install_cluster.sh — fanout vllm-dsa-sm86 tarball to all 8 ZCCX nodes.
#
# Flow:
#   1. Resolve tarball path (--tarball or auto-pick newest in /repo/INSTALLERS/)
#   2. Approval gate: post fanout request, wait for APPROVED row
#   3. Parity check: scripts/parity_pre_fanout.sh (refuses if cluster SHA-drifted)
#   4. Per-node:
#        a. scp tarball + INSTALL.sh to /tmp/
#        b. ssh into target, run INSTALL.sh --target /home/z/.venvs/.venv-vllm-dsa-sm86 \
#                                          --prev-snapshot --cluster-vetted
#        c. capture exit code + log path
#   5. Summary report → /Users/z/AGENT/comms/CC7_FANOUT_REPORT_<datetime>.md
#   6. If any node failed → automatic rollback via scripts/rollback_install.sh
#
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]
# --ProtoAI-Bakari--

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
INSTALLERS_DIR="${INSTALLERS_DIR:-/repo/INSTALLERS}"
TARGET_VENV="${TARGET_VENV:-/home/z/.venvs/.venv-vllm-dsa-sm86}"
NODES=(cuda1 cuda2 cuda3 cuda4 cuda5 cuda6 cuda7 cuda8)
APPROVAL_GATE="${APPROVAL_GATE:-$HOME/AGENT/tools/approval_gate.py}"
SSH_NODE="${SSH_NODE:-$HOME/AGENT/tools/ssh_node.sh}"
PARITY_SH="$SCRIPTS_DIR/parity_pre_fanout.sh"
ROLLBACK_SH="$SCRIPTS_DIR/rollback_install.sh"
REPORT_DIR="${REPORT_DIR:-$HOME/AGENT/comms}"

TARBALL=""
DRY_RUN=0
SKIP_PARITY=0
SKIP_APPROVAL=0
APPROVAL_TIMEOUT=1800   # 30 min

usage() {
  cat <<EOF
Usage: $0 [--tarball PATH] [--dry-run] [--skip-parity] [--skip-approval] [--approval-timeout SEC]

Options:
  --tarball PATH        Explicit tarball (default: newest /repo/INSTALLERS/vllm-dsa-sm86_*.tar.gz)
  --dry-run             Print actions, do nothing destructive
  --skip-parity         Skip parity_pre_fanout.sh (NOT recommended; CC0 approval only)
  --skip-approval       Bypass approval gate (NOT recommended; CC0 approval only)
  --approval-timeout N  Seconds to wait for approval (default 1800)

Approval-gate hook (block_unapproved_cluster_op_hook.py) will reject ssh-fanout
without an APPROVED row in approval_queue.db unless this script passes
'--cluster-vetted' to INSTALL.sh on the target — which it does only after the
gate returns APPROVED.

Per-node ssh routes via $SSH_NODE; CC7's parity gate normally restricts CC7 to
cuda7. This script must be invoked from CC0 or CC9 (full cluster auth).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tarball)            TARBALL="$2"; shift 2;;
    --dry-run)            DRY_RUN=1; shift;;
    --skip-parity)        SKIP_PARITY=1; shift;;
    --skip-approval)      SKIP_APPROVAL=1; shift;;
    --approval-timeout)   APPROVAL_TIMEOUT="$2"; shift 2;;
    -h|--help)            usage; exit 0;;
    *) echo "FATAL: unknown arg: $1" >&2; usage; exit 2;;
  esac
done

log() { printf '[install_cluster %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
fatal() { log "FATAL: $*"; exit 1; }
run() { if [[ $DRY_RUN -eq 1 ]]; then log "DRY: $*"; else log "EXEC: $*"; "$@"; fi; }

# ---------------------------------------------------------------------------
# 1. Resolve tarball
# ---------------------------------------------------------------------------
if [[ -z "$TARBALL" ]]; then
  TARBALL=$(ls -1t "$INSTALLERS_DIR"/vllm-dsa-sm86_*.tar.gz 2>/dev/null | head -1 || true)
  [[ -n "$TARBALL" ]] || fatal "no tarball given and no vllm-dsa-sm86_*.tar.gz found in $INSTALLERS_DIR"
fi
[[ -f "$TARBALL" ]] || fatal "tarball does not exist: $TARBALL"

TARBALL_BASENAME="$(basename "$TARBALL")"
TARBALL_SHA256=$(sha256sum "$TARBALL" | awk '{print $1}')
log "tarball=$TARBALL"
log "sha256=$TARBALL_SHA256"

# Pull BUILD_INFO out of the tarball without full extract — refuses to fanout
# if correctness gate did not pass.
TMP_BI=$(mktemp)
trap 'rm -f "$TMP_BI"' EXIT
tar -xzOf "$TARBALL" "vllm-dsa-sm86/BUILD_INFO.json" > "$TMP_BI" \
  || fatal "tarball does not contain BUILD_INFO.json — refusing to fanout"

if command -v jq >/dev/null 2>&1; then
  GATE_PASSED=$(jq -r '.correctness_gate.passed' "$TMP_BI")
  BUILD_SHA=$(jq -r '.git_sha' "$TMP_BI")
  PKG_VERSION=$(jq -r '.version' "$TMP_BI")
else
  GATE_PASSED=$(python3 -c "import json,sys;print(json.load(open('$TMP_BI'))['correctness_gate']['passed'])")
  BUILD_SHA=$(python3 -c "import json,sys;print(json.load(open('$TMP_BI'))['git_sha'])")
  PKG_VERSION=$(python3 -c "import json,sys;print(json.load(open('$TMP_BI'))['version'])")
fi

[[ "$GATE_PASSED" == "True" || "$GATE_PASSED" == "true" ]] \
  || fatal "BUILD_INFO.correctness_gate.passed=$GATE_PASSED — refusing to fanout"

log "version=$PKG_VERSION  build_sha=$BUILD_SHA  gate=PASS"

# ---------------------------------------------------------------------------
# 2. Approval gate
# ---------------------------------------------------------------------------
if [[ $SKIP_APPROVAL -eq 0 ]]; then
  log "posting fanout request to approval_gate"
  REQ_ID=$(python3 "$APPROVAL_GATE" request \
    --agent claude-cc7 \
    --type fanout \
    --node ALL \
    --reason "vllm-dsa-sm86 $PKG_VERSION sha=$BUILD_SHA tarball=$TARBALL_BASENAME" \
    --extra "{\"tarball_sha256\":\"$TARBALL_SHA256\",\"target_venv\":\"$TARGET_VENV\",\"nodes\":\"${NODES[*]}\"}" \
    | awk '/^request_id=/{print $1}' | sed 's/request_id=//')
  [[ -n "$REQ_ID" ]] || fatal "could not post approval request"
  log "approval request_id=$REQ_ID — waiting up to ${APPROVAL_TIMEOUT}s"

  if ! python3 "$APPROVAL_GATE" wait "$REQ_ID" --timeout "$APPROVAL_TIMEOUT"; then
    fatal "approval gate did not return APPROVED within ${APPROVAL_TIMEOUT}s"
  fi
  STATUS=$(python3 "$APPROVAL_GATE" status "$REQ_ID")
  [[ "$STATUS" == "APPROVED" ]] || fatal "approval gate returned $STATUS, not APPROVED"
  log "approval=APPROVED (id=$REQ_ID)"
else
  log "WARN: --skip-approval set; bypassing approval_gate"
  REQ_ID="skip"
fi

# ---------------------------------------------------------------------------
# 3. Parity pre-fanout
# ---------------------------------------------------------------------------
if [[ $SKIP_PARITY -eq 0 ]]; then
  if [[ -x "$PARITY_SH" ]]; then
    log "running parity_pre_fanout.sh"
    run bash "$PARITY_SH" || fatal "parity check failed — aborting fanout"
  else
    log "WARN: $PARITY_SH not present yet (Story 5 pending); skipping parity check"
  fi
else
  log "WARN: --skip-parity set; bypassing parity check"
fi

# ---------------------------------------------------------------------------
# 4. Per-node fanout
# ---------------------------------------------------------------------------
declare -a PASS=() FAIL=()
declare -A NODE_LOG=()

REMOTE_TMP_DIR="/tmp/vllm-dsa-sm86-fanout-$$"
INSTALL_SCRIPT_LOCAL="${INSTALL_SCRIPT_LOCAL:-}"   # auto-extracted below if empty

# Extract INSTALL.sh once into a stable temp so we can scp it alongside the tarball
WORKDIR=$(mktemp -d)
trap 'rm -f "$TMP_BI"; rm -rf "$WORKDIR"' EXIT
tar -xzf "$TARBALL" -C "$WORKDIR" "vllm-dsa-sm86/INSTALL.sh" \
  || fatal "tarball missing INSTALL.sh"
INSTALL_SCRIPT_LOCAL="$WORKDIR/vllm-dsa-sm86/INSTALL.sh"
chmod +x "$INSTALL_SCRIPT_LOCAL"

for node in "${NODES[@]}"; do
  log "=== $node : start ==="

  # 4a. preflight reachability via ssh_node.sh
  if ! run bash "$SSH_NODE" "$node" true; then
    log "[$node] UNREACHABLE — skipping"
    FAIL+=("$node:unreachable"); continue
  fi

  # 4b. mkdir remote tmp
  run bash "$SSH_NODE" "$node" "mkdir -p $REMOTE_TMP_DIR" || {
    FAIL+=("$node:mkdir"); continue
  }

  # 4c. scp tarball + INSTALL.sh
  if [[ $DRY_RUN -eq 0 ]]; then
    REMOTE_HOST=$(bash "$SSH_NODE" "$node" --print-host 2>/dev/null || echo "$node")
    scp -o StrictHostKeyChecking=no "$TARBALL" "$INSTALL_SCRIPT_LOCAL" \
        "z@${REMOTE_HOST}:${REMOTE_TMP_DIR}/" \
      || { FAIL+=("$node:scp"); continue; }
  else
    log "DRY: scp $TARBALL $INSTALL_SCRIPT_LOCAL → $node:$REMOTE_TMP_DIR/"
  fi

  # 4d. extract + run INSTALL.sh on target
  REMOTE_LOG="$REMOTE_TMP_DIR/install.log"
  CMD="set -e; cd $REMOTE_TMP_DIR && \
       tar -xzf $TARBALL_BASENAME && \
       cd vllm-dsa-sm86 && \
       bash INSTALL.sh --target $TARGET_VENV --prev-snapshot --cluster-vetted \
         2>&1 | tee $REMOTE_LOG"
  if run bash "$SSH_NODE" "$node" "$CMD"; then
    log "[$node] INSTALL OK"
    PASS+=("$node")
    NODE_LOG[$node]="$REMOTE_LOG"
  else
    log "[$node] INSTALL FAILED"
    FAIL+=("$node:install")
    NODE_LOG[$node]="$REMOTE_LOG"
  fi
done

# ---------------------------------------------------------------------------
# 5. Summary report
# ---------------------------------------------------------------------------
mkdir -p "$REPORT_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="$REPORT_DIR/CC7_FANOUT_REPORT_${TS}.md"

{
  echo "# CC7 Fanout Report — $TS"
  echo "**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]**"
  echo
  echo "- tarball: \`$TARBALL_BASENAME\`"
  echo "- sha256: \`$TARBALL_SHA256\`"
  echo "- version: \`$PKG_VERSION\`"
  echo "- build_sha: \`$BUILD_SHA\`"
  echo "- target_venv: \`$TARGET_VENV\`"
  echo "- approval_id: \`$REQ_ID\`"
  echo "- nodes_pass (${#PASS[@]}): ${PASS[*]:-none}"
  echo "- nodes_fail (${#FAIL[@]}): ${FAIL[*]:-none}"
  echo
  echo "## Per-node logs"
  for n in "${!NODE_LOG[@]}"; do
    echo "- \`$n\`: \`${NODE_LOG[$n]}\`"
  done
} > "$REPORT"
log "report → $REPORT"

# ---------------------------------------------------------------------------
# 6. Rollback on any failure
# ---------------------------------------------------------------------------
if [[ ${#FAIL[@]} -gt 0 ]]; then
  log "fanout had ${#FAIL[@]} failure(s); invoking rollback"
  if [[ -x "$ROLLBACK_SH" ]]; then
    run bash "$ROLLBACK_SH" --target "$TARGET_VENV" --nodes "${NODES[*]}" || \
      log "WARN: rollback script returned non-zero — manual intervention required"
  else
    log "WARN: $ROLLBACK_SH not present yet (Story 4 pending); skipping rollback"
  fi
  exit 1
fi

log "ALL ${#PASS[@]} NODES OK"
exit 0
