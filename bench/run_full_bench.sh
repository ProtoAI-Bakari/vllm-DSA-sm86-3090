#!/usr/bin/env bash
# --ProtoAI-Bakari--
# run_full_bench.sh — pipeline wrapper composing Story 1-7+9 stages.
# Pre-authored while CC7 fanout pending; first run when endpoint is live.
#
# Stages:
#   0. Pre-flight cordon + GID-index check (Story 10)
#   1. /v1/models 200 + /health 200 sentinel
#   2. Per-GPU util capture (Story 4) — background for full run duration
#   3. KV metrics poll (Story 5) — background
#   4. NCCL fabric profile (Story 3) — once at start (mesh iperf3 + ibv)
#   5. Concurrency sweep (Story 2) — synchronous, drives both bg captures
#   6. Latency percentile (Story 6) — at conc=N target
#   7. Tier classify (Story 7) — over conc-sweep summary
#   8. Optimization loop proposal (Story 9) — over tier output
#   9. Write PATH_A_FINAL_RESULTS_<datetime>.md (Story 8)
#
# Usage:
#   bench/run_full_bench.sh \
#     --endpoint http://cuda1:8000 \
#     --model glm51-iq2xxs \
#     --profile bench/profiles/glm51_tp2_ep8.yaml \
#     --concs 1,2,4,8,16 \
#     --requests 64

set -euo pipefail

ENDPOINT=""
MODEL=""
PROFILE=""
CONCS="1,2,4,8,16,32"
REQUESTS=64
MAX_TOKENS=256
SKIP_CORDON=0
LATENCY_CONC=8
OUT_BASE="${HOME}/AGENT/path_a_run_$(date +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --concs) CONCS="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --latency-conc) LATENCY_CONC="$2"; shift 2 ;;
    --skip-cordon) SKIP_CORDON=1; shift ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$ENDPOINT" || -z "$MODEL" ]] && { echo "missing --endpoint or --model" >&2; exit 2; }
mkdir -p "$OUT_BASE"
log() { printf '[run_full_bench %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Stage 0: Cordon + GID-index check (skippable when not on k8s deploy)
if [[ "$SKIP_CORDON" == 0 ]]; then
  log "stage 0: cordon + GID-index check"
  bash "$ROOT/scripts/cordon_for_launch.sh" --keep cuda1,cuda2,cuda3,cuda4,cuda5,cuda6,cuda7,cuda8 \
    > "$OUT_BASE/00_cordon.log" 2>&1 || { log "cordon failed; abort"; exit 4; }
fi

# Stage 1: endpoint sentinel — must be HEALTHY before anything else.
log "stage 1: endpoint sentinel"
HC_M=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$ENDPOINT/v1/models" || echo 000)
HC_H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$ENDPOINT/health" || echo 000)
[[ "$HC_M" == "200" && "$HC_H" == "200" ]] || { log "ABORT: /v1/models=$HC_M /health=$HC_H"; exit 3; }
log "endpoint OK: /v1/models=$HC_M /health=$HC_H"

# Stage 2: GPU util capture (bg, duration sized for full sweep)
SWEEP_DUR_EST=$(python3 -c "n=len('$CONCS'.split(',')); print(n * 90 + 60)")
log "stage 2: gpu_util_capture bg (~${SWEEP_DUR_EST}s)"
bash "$ROOT/bench/runners/gpu_util_capture.sh" --duration "$SWEEP_DUR_EST" --interval-ms 100 \
  --out "$OUT_BASE/02_gpu_util.csv" > "$OUT_BASE/02_gpu_util.stderr.log" 2>&1 &
GPU_PID=$!

# Stage 3: KV metrics (bg)
log "stage 3: kv_metrics bg"
bash "$ROOT/bench/runners/kv_metrics.sh" --endpoint "$ENDPOINT" --duration "$SWEEP_DUR_EST" \
  --out "$OUT_BASE/03_kv_metrics.jsonl" > "$OUT_BASE/03_kv_metrics.stderr.log" 2>&1 &
KV_PID=$!

# Stage 4: NCCL fabric profile (one-shot at start)
log "stage 4: nccl_profile (one-shot)"
bash "$ROOT/bench/runners/nccl_profile.sh" --out "$OUT_BASE/04_nccl_profile" \
  > "$OUT_BASE/04_nccl_profile.log" 2>&1 || log "nccl_profile partial (continuing)"

# Stage 5: Concurrency sweep (foreground)
log "stage 5: conc_sweep ($CONCS)"
SWEEP_OUT="$OUT_BASE/05_conc_sweep.jsonl"
bash "$ROOT/bench/runners/conc_sweep.sh" --endpoint "$ENDPOINT" --model "$MODEL" \
  --concs "$CONCS" --prompts "$REQUESTS" --max-tokens "$MAX_TOKENS" \
  --out "$SWEEP_OUT" 2>&1 | tee "$OUT_BASE/05_conc_sweep.log"

# Stage 6: Latency percentile at target conc
log "stage 6: latency_percentile @ conc=$LATENCY_CONC"
python3 "$ROOT/bench/runners/latency_percentile.py" \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --concurrency "$LATENCY_CONC" --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" \
  --out "$OUT_BASE/06_latency.json" 2>&1 | tee "$OUT_BASE/06_latency.log"

# Wait on bg captures
wait "$GPU_PID" 2>/dev/null || true
wait "$KV_PID"  2>/dev/null || true

# Stage 7: Tier classify
log "stage 7: tier_classify"
python3 "$ROOT/bench/runners/tier_classify.py" \
  --conc-sweep "$SWEEP_OUT" \
  --out "$OUT_BASE/07_tier.json" 2>&1 | tee "$OUT_BASE/07_tier.log"

# Stage 8: Optimization loop
log "stage 8: optimize_loop"
python3 "$ROOT/bench/optimize_loop.py" \
  --tier-classify "$OUT_BASE/07_tier.json" \
  --gpu-util "$OUT_BASE/02_gpu_util.csv" \
  --kv "$OUT_BASE/03_kv_metrics.jsonl" \
  --out "$OUT_BASE/08_optimize.json" 2>&1 | tee "$OUT_BASE/08_optimize.log"

# Stage 9: Final results doc (Story 8 frame)
DT=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="${HOME}/AGENT/comms/PATH_A_FINAL_RESULTS_${DT}.md"
python3 - "$OUT_BASE" "$MODEL" "$ENDPOINT" "$PROFILE" "$RESULTS" <<'PY'
import json, os, sys
out_base, model, endpoint, profile, results_path = sys.argv[1:6]

def jload(p):
    try: return json.load(open(p))
    except Exception: return None

tier   = jload(os.path.join(out_base, "07_tier.json")) or {}
opt    = jload(os.path.join(out_base, "08_optimize.json")) or {}
lat    = jload(os.path.join(out_base, "06_latency.json")) or {}
lat_s  = (lat or {}).get("summary", {}) if isinstance(lat, dict) else {}

agg    = tier.get("aggregate", {})
conc1  = tier.get("conc1", {})

with open(results_path, "w") as f:
    f.write(f"# PATH A — Final Results ({model})\n\n")
    f.write(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC8]**\n\n")
    f.write(f"- endpoint: `{endpoint}`\n")
    f.write(f"- profile: `{profile}`\n")
    f.write(f"- run_dir: `{out_base}`\n\n")
    f.write("## Tier classification\n\n")
    f.write(f"- aggregate: **{agg.get('tier','?')}** ({agg.get('value','?')} t/s; gap to next: {agg.get('gap_to_next','?')})\n")
    f.write(f"- conc=1: **{conc1.get('tier','?')}** ({conc1.get('value','?')} t/s; gap to next: {conc1.get('gap_to_next','?')})\n\n")
    f.write("## Per-conc throughput\n\n| conc | agg_tps | per-req mean tps | n_ok | wall_s |\n|---|---|---|---|---|\n")
    for c, v in sorted((tier.get("by_conc") or {}).items(), key=lambda x: int(x[0])):
        f.write(f"| {c} | {v.get('agg_tps',0):.1f} | {v.get('per_request_tps_mean',0):.2f} | {v.get('n_ok','?')} | {v.get('wall_s',0):.2f} |\n")
    f.write("\n## Latency (streaming) @ conc=N\n\n")
    for k in ("ttft_s", "tpot_s", "itl_s", "pp_tps_per_request", "tg_tps_per_request"):
        b = lat_s.get(k, {})
        f.write(f"- **{k}**: p50={b.get('p50')}, p95={b.get('p95')}, p99={b.get('p99')}, mean={b.get('mean')} (n={b.get('n')})\n")
    f.write(f"- **agg_pp_tps**: {lat_s.get('agg_pp_tps')}\n")
    f.write(f"- **agg_tg_tps**: {lat_s.get('agg_tg_tps')}\n\n")
    f.write("## Optimization candidates (next steps)\n\n")
    for c in (opt.get("candidates") or [])[:5]:
        f.write(f"- **{c['name']}** (+{c['expected_uplift_pct']}%, {c['blast_radius']}): {c['rationale']}\n")
    f.write("\n## Diagnostics\n\n")
    diag = opt.get("diagnostics", {})
    for k, v in diag.items():
        f.write(f"- {k}: {v}\n")
print(f"wrote {results_path}")
PY

log "DONE. results=$RESULTS run_dir=$OUT_BASE"
echo "$RESULTS"
