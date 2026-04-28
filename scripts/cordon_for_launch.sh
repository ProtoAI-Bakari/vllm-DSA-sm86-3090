#!/usr/bin/env bash
# --ProtoAI-Bakari--
# cordon_for_launch.sh — Story #10. Cordon unwanted nodes and verify GID-index normalization
# before sky launch. Mandatory pre-flight per role CLAUDE.md.
#
# Usage:
#   scripts/cordon_for_launch.sh --keep cuda1,cuda2,cuda3,cuda4,cuda5,cuda6,cuda7,cuda8 [--dry-run]
#   scripts/cordon_for_launch.sh --uncordon-all   (post-bench restore)

set -euo pipefail

KEEP=""
UNCORDON_ALL=0
DRY_RUN=0
EXPECTED_GID_INDEX=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep) KEEP="$2"; shift 2 ;;
    --uncordon-all) UNCORDON_ALL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --expected-gid-index) EXPECTED_GID_INDEX="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[cordon %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
run() { if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: $*"; else "$@"; fi; }

if [[ "$UNCORDON_ALL" == 1 ]]; then
  log "uncordoning all nodes"
  ALL_NODES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
  for N in $ALL_NODES; do
    run kubectl uncordon "$N" || log "uncordon $N failed (continuing)"
  done
  exit 0
fi

[[ -z "$KEEP" ]] && { echo "missing --keep cuda1,cuda2,..." >&2; exit 2; }
IFS=',' read -ra KEEP_ARR <<< "$KEEP"

# Cordon every node not in --keep
ALL_NODES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
for N in $ALL_NODES; do
  KEEP_THIS=0
  for K in "${KEEP_ARR[@]}"; do
    [[ "$N" == "$K" ]] && { KEEP_THIS=1; break; }
  done
  if [[ "$KEEP_THIS" == 0 ]]; then
    log "cordoning $N (not in --keep)"
    run kubectl cordon "$N" || log "cordon $N failed"
  else
    log "keeping $N (uncordoning if cordoned)"
    run kubectl uncordon "$N" 2>/dev/null || true
  fi
done

# Verify GID-index normalization on kept nodes (RoCEv2 prereq)
log "verifying RoCEv2 GID index = $EXPECTED_GID_INDEX on kept nodes"
GID_FAIL=0
for K in "${KEEP_ARR[@]}"; do
  GID_LINE=$(bash "${HOME}/AGENT/tools/ssh_node.sh" "$K" \
    "ibv_devinfo -v 2>/dev/null | grep -A1 GID | grep RoCE | head -n 1 || true" 2>/dev/null || echo "")
  GID_IDX=$(echo "$GID_LINE" | grep -oE 'GID\[[0-9]+\]' | grep -oE '[0-9]+' | head -n 1 || echo "")
  if [[ -z "$GID_IDX" ]]; then
    log "WARN: $K has no RoCE GID line (ibv_devinfo)"
    GID_FAIL=$((GID_FAIL + 1))
  elif [[ "$GID_IDX" != "$EXPECTED_GID_INDEX" ]]; then
    log "FAIL: $K GID index=$GID_IDX, expected $EXPECTED_GID_INDEX"
    GID_FAIL=$((GID_FAIL + 1))
  else
    log "ok: $K GID index=$EXPECTED_GID_INDEX"
  fi
done

if [[ "$GID_FAIL" -gt 0 ]]; then
  log "ABORT: $GID_FAIL kept node(s) failed GID index check. Sky launch will silently fall back to TCP."
  exit 4
fi

log "cordon + GID index check PASS. Cluster ready for sky launch."
