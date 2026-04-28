#!/usr/bin/env bash
# --ProtoAI-Bakari--
# gpu_util_capture.sh — Story #4. nvidia-smi 100ms samples per GPU during bench.
# Output: parquet-like CSV (timestamp, host, gpu_idx, util_gpu, util_mem, mem_used_mib, power_w, sm_clock_mhz)
#
# Usage:
#   bench/runners/gpu_util_capture.sh --duration 120 --interval-ms 100 [--out PATH]

set -euo pipefail

DURATION=120
INTERVAL_MS=100
OUT=""
HOSTS=(cuda1 cuda2 cuda3 cuda4 cuda5 cuda6 cuda7 cuda8)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --interval-ms) INTERVAL_MS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

TS=$(date +%Y%m%dT%H%M%SZ)
[[ -z "$OUT" ]] && OUT="${HOME}/AGENT/gpu_util_${TS}.csv"
mkdir -p "$(dirname "$OUT")"

# nvidia-smi --query-gpu can sample at 100ms via -lms 100. Use dmon for per-GPU per-line.
echo "ts,host,gpu_idx,util_gpu,util_mem,mem_used_mib,power_w,sm_clock_mhz" > "$OUT"

PIDS=()
for H in "${HOSTS[@]}"; do
  (
    bash "${HOME}/AGENT/tools/ssh_node.sh" "$H" \
      "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
       --format=csv,noheader,nounits -lms ${INTERVAL_MS} -c $((DURATION * 1000 / INTERVAL_MS))" 2>/dev/null \
      | awk -v h="$H" -F, '{ gsub(/^[ \t]+|[ \t]+$/, "", $1); printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1, h, $2, $3, $4, $5, $6, $7 }' \
      >> "$OUT"
  ) &
  PIDS+=($!)
done

for PID in "${PIDS[@]}"; do
  wait "$PID" || true
done

# Summary: per-host mean util_gpu
python3 - "$OUT" <<'PY'
import csv, sys, statistics
from collections import defaultdict
path = sys.argv[1]
by_host_gpu = defaultdict(list)
with open(path) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            by_host_gpu[(row["host"], row["gpu_idx"])].append(float(row["util_gpu"]))
        except (ValueError, KeyError):
            continue
print("host\tgpu\tmean_util\tp95_util\tn_samples")
for (h, g), vals in sorted(by_host_gpu.items()):
    if not vals: continue
    vals_sorted = sorted(vals)
    p95 = vals_sorted[int(0.95 * len(vals_sorted))]
    print(f"{h}\t{g}\t{statistics.mean(vals):.1f}\t{p95:.1f}\t{len(vals)}")
PY

echo "$OUT"
