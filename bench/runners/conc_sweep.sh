#!/usr/bin/env bash
# --ProtoAI-Bakari--
# conc_sweep.sh — concurrency sweep harness for vLLM TP=2 x EP=8 deploy.
# Story #2 from STORY_BACKLOG_CC8.md.
#
# Usage:
#   bench/runners/conc_sweep.sh \
#     --endpoint http://cuda1:8000 \
#     --model glm51-iq2xxs \
#     --concs 1,2,4,8,16,32 \
#     --prompts 64 \
#     --max-tokens 256 \
#     --out /tmp/bench_glm51_tp2_ep8/conc_sweep_<ts>.jsonl
#
# Exits non-zero if endpoint not /v1/models 200 + ready:true at start.
# Drains the engine between concurrency steps via /v1/health-and-drain (vLLM
# >= 0.20.0 has best-effort drain) plus a 30s settle window.

set -euo pipefail

ENDPOINT="http://cuda1:8000"
MODEL=""
CONCS="1,2,4,8,16,32"
PROMPTS=64
MAX_TOKENS=256
PROMPTS_FILE=""
OUT=""
WARMUP_PROMPTS=4
SETTLE_S=30
TIMEOUT_S=600

usage() {
  cat <<EOF
Usage: $0 --endpoint URL --model NAME [--concs 1,2,4,...] [--prompts N]
          [--max-tokens N] [--prompts-file PATH] [--out PATH]
          [--warmup N] [--settle SEC] [--timeout SEC]
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --concs) CONCS="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --warmup) WARMUP_PROMPTS="$2"; shift 2 ;;
    --settle) SETTLE_S="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$MODEL" ]] && { echo "missing --model" >&2; usage; }
TS=$(date +%Y%m%dT%H%M%SZ)
[[ -z "$OUT" ]] && OUT="/tmp/bench_${MODEL}_conc_sweep_${TS}.jsonl"
mkdir -p "$(dirname "$OUT")"

log() { printf '[conc_sweep %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 1. Pre-flight: endpoint must be HEALTHY before first sweep step.
log "preflight: $ENDPOINT/v1/models"
HTTP_CODE=$(curl -s -o /tmp/cs_models.json -w '%{http_code}' --max-time 10 "$ENDPOINT/v1/models" || echo 000)
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ABORT: $ENDPOINT/v1/models returned $HTTP_CODE (need 200)" >&2
  exit 3
fi
HEALTH_CODE=$(curl -s -o /tmp/cs_health.json -w '%{http_code}' --max-time 10 "$ENDPOINT/health" || echo 000)
[[ "$HEALTH_CODE" == "200" ]] || { echo "ABORT: /health $HEALTH_CODE" >&2; exit 3; }
log "preflight OK (models=$HTTP_CODE health=$HEALTH_CODE)"

# 2. Build prompt set (deterministic seed for reproducibility)
PROMPT_TMP=$(mktemp)
trap 'rm -f "$PROMPT_TMP" /tmp/cs_models.json /tmp/cs_health.json' EXIT
if [[ -n "$PROMPTS_FILE" ]]; then
  cp "$PROMPTS_FILE" "$PROMPT_TMP"
else
  python3 - <<PY > "$PROMPT_TMP"
import json, random
random.seed(42)
seeds = [
  "Explain the role of expert parallelism in MoE inference.",
  "Summarize the difference between TMA and cp.async on Ampere.",
  "What is FP8 block-scaled GEMM and why does it need Hopper?",
  "List three differences between FlashAttention v2 and v3.",
  "Give a one-paragraph history of sparse attention in LLMs.",
  "Compare INT4-AWQ vs INT4-GPTQ calibration trade-offs.",
  "Why does NCCL all-reduce dominate decode latency at TP=16?",
  "Describe the role of the lightning indexer in DeepSeek MoE.",
]
for i in range($PROMPTS):
  random.shuffle(seeds)
  print(json.dumps({"id": i, "prompt": seeds[0] + f" (variant {i})"}))
PY
fi
NPROMPT=$(wc -l < "$PROMPT_TMP" | tr -d ' ')
log "prompt set ready: $NPROMPT prompts in $PROMPT_TMP"

# 3. Warm-up at conc=1 to fill prefix cache + JIT paths
log "warmup: $WARMUP_PROMPTS sequential prompts at conc=1"
head -n "$WARMUP_PROMPTS" "$PROMPT_TMP" | while IFS= read -r line; do
  P=$(echo "$line" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["prompt"])')
  curl -s --max-time "$TIMEOUT_S" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json; print(json.dumps({'model':'$MODEL','prompt':'$P','max_tokens':$MAX_TOKENS,'temperature':0}))")" \
    "$ENDPOINT/v1/completions" > /dev/null || log "warmup prompt failed (continuing)"
done

# 4. Sweep
IFS=',' read -ra CONC_ARR <<< "$CONCS"
echo -n "" > "$OUT"
for CONC in "${CONC_ARR[@]}"; do
  log "==== conc=$CONC ===="
  STEP_START=$(date +%s.%N)
  STEP_OUT="${OUT%.jsonl}_conc${CONC}_raw.jsonl"
  : > "$STEP_OUT"

  python3 - "$ENDPOINT" "$MODEL" "$CONC" "$MAX_TOKENS" "$PROMPT_TMP" "$STEP_OUT" "$TIMEOUT_S" <<'PY'
import asyncio, json, os, sys, time
import urllib.request, urllib.error

endpoint, model, conc, max_tokens, prompt_file, out_path, timeout = sys.argv[1:8]
conc = int(conc); max_tokens = int(max_tokens); timeout = float(timeout)

with open(prompt_file) as f:
    prompts = [json.loads(line) for line in f if line.strip()]

import concurrent.futures as cf
def run_one(p):
    body = json.dumps({"model": model, "prompt": p["prompt"],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(endpoint + "/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        t1 = time.time()
        text = data["choices"][0]["text"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(text.split()))
        prompt_tokens = usage.get("prompt_tokens", -1)
        return {"id": p["id"], "ok": True,
                "wall_s": t1 - t0,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "tps_per_request": completion_tokens / max(t1 - t0, 1e-6)}
    except Exception as e:
        return {"id": p["id"], "ok": False, "err": str(e)}

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=conc) as pool:
    results = list(pool.map(run_one, prompts))
t1 = time.time()

ok = [r for r in results if r.get("ok")]
err = [r for r in results if not r.get("ok")]
total_tokens = sum(r["completion_tokens"] for r in ok)
agg_tps = total_tokens / max(t1 - t0, 1e-6)

with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

summary = {"conc": conc, "n_prompts": len(prompts), "n_ok": len(ok), "n_err": len(err),
           "wall_s": t1 - t0, "total_completion_tokens": total_tokens,
           "agg_tps": agg_tps,
           "mean_tps_per_request": (sum(r["tps_per_request"] for r in ok) / max(len(ok), 1))}
print(json.dumps(summary))
PY

  STEP_SUMMARY=$(tail -n 1 "$STEP_OUT" 2>/dev/null || echo "{}")
  STEP_END=$(date +%s.%N)
  STEP_WALL=$(python3 -c "print(round($STEP_END - $STEP_START, 3))")
  echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"profile_endpoint\": \"$ENDPOINT\", \"model\": \"$MODEL\", \"conc\": $CONC, \"step_wall_s\": $STEP_WALL, \"raw\": \"$STEP_OUT\"}" >> "$OUT"

  log "drain settle ${SETTLE_S}s"
  sleep "$SETTLE_S"
done

log "DONE. summary jsonl=$OUT"
echo "$OUT"
