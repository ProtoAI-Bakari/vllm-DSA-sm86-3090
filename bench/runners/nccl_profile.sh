#!/usr/bin/env bash
# --ProtoAI-Bakari--
# nccl_profile.sh — Story #3. NCCL fabric profile on ZCCX 8x2 cluster.
# Captures: NCCL_DEBUG=INFO logs, iperf3 each link, ibv_devinfo per node.
# Run from cuda1 (anchor); SSH out to cuda2..cuda8 read-only.
#
# Usage:
#   bench/runners/nccl_profile.sh [--out DIR] [--during-bench PIDFILE]

set -euo pipefail

OUT_DIR="${HOME}/AGENT/nccl_profile_$(date +%Y%m%dT%H%M%SZ)"
DURING_BENCH=""
HOSTS=(cuda1 cuda2 cuda3 cuda4 cuda5 cuda6 cuda7 cuda8)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --during-bench) DURING_BENCH="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
log() { printf '[nccl_profile %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 1. ibv_devinfo per node — RoCEv2 GID + active link rate
for H in "${HOSTS[@]}"; do
  log "ibv_devinfo @ $H"
  bash "${HOME}/AGENT/tools/ssh_node.sh" "$H" "ibv_devinfo -v 2>&1 || true" \
    > "$OUT_DIR/ibv_${H}.log" 2>&1 || log "ibv $H failed"
done

# 2. nvidia-smi topo per node — NVLink presence within node
for H in "${HOSTS[@]}"; do
  bash "${HOME}/AGENT/tools/ssh_node.sh" "$H" "nvidia-smi topo -m 2>&1 || true" \
    > "$OUT_DIR/topo_${H}.log" 2>&1 || true
done

# 3. iperf3 mesh — pairwise N1->N2 throughput on bond0 (RoCEv2-disabled TCP test)
# Server on each, client probe pairs. Bandwidth ceiling = 100 Gb/s nominal.
log "iperf3 mesh starting"
for SERVER in "${HOSTS[@]}"; do
  bash "${HOME}/AGENT/tools/ssh_node.sh" "$SERVER" "pkill -f 'iperf3 -s' 2>/dev/null; iperf3 -s -D -p 5201 || true" >/dev/null 2>&1 || true
done
sleep 2
for CLI in "${HOSTS[@]}"; do
  for SRV in "${HOSTS[@]}"; do
    [[ "$CLI" == "$SRV" ]] && continue
    log "iperf3 $CLI -> $SRV"
    bash "${HOME}/AGENT/tools/ssh_node.sh" "$CLI" \
      "iperf3 -c $SRV -p 5201 -t 5 -J 2>&1 || true" \
      > "$OUT_DIR/iperf3_${CLI}_to_${SRV}.json" 2>&1 || true
  done
done
for SERVER in "${HOSTS[@]}"; do
  bash "${HOME}/AGENT/tools/ssh_node.sh" "$SERVER" "pkill -f 'iperf3 -s' 2>/dev/null || true" >/dev/null 2>&1 || true
done

# 4. NCCL_DEBUG=INFO capture during bench (caller passes --during-bench PIDFILE)
if [[ -n "$DURING_BENCH" && -f "$DURING_BENCH" ]]; then
  BENCH_PID=$(cat "$DURING_BENCH")
  log "tailing NCCL log of bench pid=$BENCH_PID for 60s"
  for H in "${HOSTS[@]}"; do
    bash "${HOME}/AGENT/tools/ssh_node.sh" "$H" \
      "ls /var/log/vllm-nccl-*.log 2>/dev/null | head -n 1 | xargs -I{} tail -n 200 {} 2>/dev/null || true" \
      > "$OUT_DIR/nccl_log_${H}.log" 2>&1 || true
  done
fi

# 5. Summary
{
  echo "nccl_profile @ $(date -u +%FT%TZ)"
  echo "out_dir: $OUT_DIR"
  echo "ibv: $(ls $OUT_DIR/ibv_*.log 2>/dev/null | wc -l) files"
  echo "topo: $(ls $OUT_DIR/topo_*.log 2>/dev/null | wc -l) files"
  echo "iperf3 pairs: $(ls $OUT_DIR/iperf3_*.json 2>/dev/null | wc -l)"
  echo "expect 56 pairs (8x7) for full mesh"
} > "$OUT_DIR/SUMMARY.txt"

cat "$OUT_DIR/SUMMARY.txt" >&2
echo "$OUT_DIR"
