#!/usr/bin/env bash
# --ProtoAI-Bakari--
# kv_metrics.sh — Story #5. Poll vLLM /metrics for KV cache stats.
# Reports: kv_cache_usage_perc, num_running, num_waiting, num_swapped, gpu_cache_usage.
#
# Usage:
#   bench/runners/kv_metrics.sh --endpoint http://cuda1:8000 --duration 120 [--interval-s 1] [--out PATH]

set -euo pipefail

ENDPOINT="http://cuda1:8000"
DURATION=120
INTERVAL_S=1
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --interval-s) INTERVAL_S="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

TS=$(date +%Y%m%dT%H%M%SZ)
[[ -z "$OUT" ]] && OUT="${HOME}/AGENT/kv_metrics_${TS}.jsonl"
mkdir -p "$(dirname "$OUT")"

# Pre-flight
HC=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$ENDPOINT/metrics" || echo 000)
[[ "$HC" == "200" ]] || { echo "ABORT: $ENDPOINT/metrics returned $HC" >&2; exit 3; }

END=$(($(date +%s) + DURATION))
KEYS="vllm:gpu_cache_usage_perc vllm:num_requests_running vllm:num_requests_waiting vllm:num_requests_swapped vllm:cpu_cache_usage_perc vllm:prompt_tokens_total vllm:generation_tokens_total"

while [[ $(date +%s) -lt $END ]]; do
  TS_NOW=$(date -u +%FT%TZ)
  RAW=$(curl -s --max-time 5 "$ENDPOINT/metrics" || echo "")
  python3 - "$TS_NOW" "$KEYS" <<PY >> "$OUT"
import json, re, sys
ts = sys.argv[1]
keys = sys.argv[2].split()
import urllib.request
try:
    raw = urllib.request.urlopen("$ENDPOINT/metrics", timeout=5).read().decode()
except Exception as e:
    print(json.dumps({"ts": ts, "err": str(e)}))
    sys.exit(0)
out = {"ts": ts}
for line in raw.splitlines():
    if line.startswith("#") or not line.strip(): continue
    parts = line.split()
    if len(parts) < 2: continue
    name = parts[0].split("{")[0]
    try:
        val = float(parts[-1])
    except ValueError:
        continue
    if name in keys:
        out.setdefault(name, []).append(val)
for k, v in list(out.items()):
    if isinstance(v, list):
        out[k] = sum(v) / len(v) if v else 0.0
print(json.dumps(out))
PY
  sleep "$INTERVAL_S"
done

# Final summary
python3 - "$OUT" <<'PY'
import json, sys, statistics
path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
metric_keys = set()
for r in rows: metric_keys.update(k for k in r if k.startswith("vllm:"))
print("metric\tmean\tp95\tmax\tn")
for k in sorted(metric_keys):
    vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
    if not vals: continue
    s = sorted(vals)
    p95 = s[int(0.95 * len(s))]
    print(f"{k}\t{statistics.mean(vals):.3f}\t{p95:.3f}\t{max(vals):.3f}\t{len(vals)}")
PY

echo "$OUT"
