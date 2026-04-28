#!/bin/bash
# --ProtoAI-Bakari--
# LEAD dispatch (3): single-command post-cmake smoke gate. Zero-wall execution
# from cmake_done milestone — every step automated. Expected wall ≤5 min.
#
# Pipeline:
#   1. (skipped — CC9 owns drain) cluster_drain.sh / sky_drain.sh — assume CC9 already drained
#   2. endpoint health-wait — poll target until /v1/models 200 (max 5 min)
#   3. POST 5 cc6_fixtures.json prompts in sequence — capture into cycle5_capture_<integ>.jsonl
#   4. grade vs baseline.jsonl — emit cosine + top1 + max_abs + jaccard
#   5. bridge l4_capture_ready (announce capture exists) AND l4_result (verdict)
#
# Run:
#   ./cycle5_smoke.sh --integ <label> [--endpoint <url>] [--model <id>] [--no-bridge]
#
# Defaults from cc6_fixtures.json: endpoint=http://10.255.255.4:8000  model=GLM-5.1-MLX
# Exit: 0 gate PASS / 1 gate FAIL / 2 input error / 3 endpoint never came up.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
FIXTURES="${HOME}/AGENT/tools/cc6_fixtures.json"
BASELINE="${REPO}/tests/correctness/baseline.jsonl"

INTEG=""
ENDPOINT=""
MODEL=""
NO_BRIDGE=0
HEALTHWAIT=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --health-wait) HEALTHWAIT="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    *) echo "[cycle5_smoke] unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$INTEG" ]] && { echo "[cycle5_smoke] --integ required" >&2; exit 2; }
[[ ! -f "$FIXTURES" ]] && { echo "[cycle5_smoke] fixtures missing: $FIXTURES" >&2; exit 2; }
[[ ! -f "$BASELINE" ]] && { echo "[cycle5_smoke] baseline missing: $BASELINE — run baseline_capture first" >&2; exit 2; }

# Pull defaults from fixtures
[[ -z "$ENDPOINT" ]] && ENDPOINT=$(python3 -c "import json; print(json.load(open('$FIXTURES')).get('endpoint_default','http://10.255.255.4:8000'))")
[[ -z "$MODEL" ]]    && MODEL=$(python3 -c "import json; print(json.load(open('$FIXTURES')).get('model_default','/Users/z/models/GLM-5.1-RAM-270GB-MLX'))")

CAPTURE_OUT="${HOME}/AGENT/comms/captures/cycle5_${INTEG}.jsonl"
mkdir -p "$(dirname "$CAPTURE_OUT")"
: > "$CAPTURE_OUT"

post_bridge () {
  local topic="$1" body="$2"
  [[ "$NO_BRIDGE" == "1" ]] && return 0
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic "$topic" --body "$body" || true
}

echo "[cycle5_smoke] $(date) integ=$INTEG endpoint=$ENDPOINT"
post_bridge "milestone" "cycle5_smoke start integ=${INTEG} endpoint=${ENDPOINT}"

# ---------- step 2: endpoint health-wait ----------
T0=$(date +%s)
HEALTHY=0
while :; do
  if /usr/bin/curl -fsS --max-time 5 -o /dev/null "${ENDPOINT}/v1/models" 2>/dev/null; then
    HEALTHY=1
    break
  fi
  if (( $(date +%s) - T0 > HEALTHWAIT )); then
    echo "[cycle5_smoke] endpoint $ENDPOINT did not come up in ${HEALTHWAIT}s" >&2
    post_bridge "blocker" "cycle5_smoke endpoint_not_ready ${ENDPOINT} after ${HEALTHWAIT}s integ=${INTEG}"
    exit 3
  fi
  sleep 5    # SLOW_OP polling
done
echo "[cycle5_smoke] endpoint healthy after $(( $(date +%s) - T0 ))s"

# ---------- step 3: POST 5 fixtures ----------
echo "[cycle5_smoke] capturing 5 fixtures → $CAPTURE_OUT"
python3 - "$FIXTURES" "$CAPTURE_OUT" "$ENDPOINT" "$MODEL" <<'PY'
import json, sys, time, urllib.request, urllib.error
fx_path, out_path, endpoint, model = sys.argv[1:5]
fixtures = json.load(open(fx_path))["fixtures"]
with open(out_path, "a") as out_f:
    for fx in fixtures:
        body = {
            "model": model,
            "prompt": fx["prompt"],
            "max_tokens": fx["max_tokens"],
            "temperature": fx.get("temperature", 0),
        }
        if fx.get("stop"):
            body["stop"] = fx["stop"]
        t0 = time.time()
        rec = {"id": fx["id"], "category": fx.get("category", "unknown"),
               "endpoint": endpoint, "model": model, "ts": int(t0)}
        try:
            req = urllib.request.Request(endpoint + "/v1/completions",
                                          data=json.dumps(body).encode(),
                                          headers={"Content-Type": "application/json"},
                                          method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            ch = (resp.get("choices") or [{}])[0]
            rec["text"] = ch.get("text") or (ch.get("message") or {}).get("content") or ""
            rec["final_payload"] = resp
        except urllib.error.HTTPError as e:
            rec["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["wall_s"] = round(time.time() - t0, 3)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  [{fx['id']}] wall={rec['wall_s']}s  text={(rec.get('text','') or rec.get('error',''))[:80]!r}", file=sys.stderr)
PY

# ---------- step 5a: announce l4_capture_ready ----------
post_bridge "l4_capture_ready" "integ=${INTEG} under_test=${CAPTURE_OUT} baseline=${BASELINE}"

# ---------- step 4 + 5b: grade + l4_result ----------
set +e
bash "$HERE/../tests/correctness/cycle5_grade.sh" \
  --integ "$INTEG" \
  --capture "$CAPTURE_OUT" \
  --baseline "$BASELINE" \
  ${NO_BRIDGE:+--no-bridge}
EC=$?
set -e

echo "[cycle5_smoke] $(date) verdict_ec=$EC integ=$INTEG"
exit "$EC"
