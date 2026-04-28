#!/bin/bash
# --ProtoAI-Bakari--
# CC6 daemon (W3 NEW): poll the configured PP baseline endpoint every N seconds.
# Posts bridge.db topic=endpoint_health when state changes (UP→DOWN or DOWN→UP),
# so Story 3 baseline.jsonl population auto-fires the moment the endpoint comes
# back. Without this loop, baseline capture is gated on a human noticing the
# endpoint is alive again.
#
# Run:
#   ./endpoint_health_daemon.sh                                # default mac4:8000, 60s
#   POLL=120 BASELINE_ENDPOINT=http://10.255.255.4:8000 ./endpoint_health_daemon.sh
#   ./endpoint_health_daemon.sh --once                         # single check, exit
#
# Side effects:
#   - bridge.db topic=endpoint_health body="<UP|DOWN> endpoint=<url> model=<id>"
#   - state file: ~/AGENT/comms/cc6_endpoint_state.json (last status, last change ts)
#   - if state flips DOWN→UP and AUTO_FIRE_BASELINE=1 → fires baseline_capture.sh
#     in background.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL="${POLL:-60}"
ENDPOINT="${BASELINE_ENDPOINT:-http://10.255.255.4:8000}"
STATE_FILE="${STATE_FILE:-${HOME}/AGENT/comms/cc6_endpoint_state.json}"
AUTO_FIRE="${AUTO_FIRE_BASELINE:-0}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

mkdir -p "$(dirname "$STATE_FILE")"

probe() {
  local url="$1/v1/models"
  if /usr/bin/curl -fsS --max-time 5 -H 'Accept: application/json' "$url" 2>/dev/null > /tmp/cc6_endpoint_probe.json; then
    local mid
    mid=$(python3 -c 'import json;
try:
    d=json.load(open("/tmp/cc6_endpoint_probe.json"));
    print(d["data"][0]["id"])
except Exception:
    print("")')
    echo "UP $mid"
  else
    echo "DOWN -"
  fi
}

read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo UNKNOWN
  else
    echo UNKNOWN
  fi
}

write_state() {
  local status="$1" model="$2"
  python3 -c "
import json, time, os
d = {'status': '$status', 'model': '$model', 'endpoint': '$ENDPOINT', 'ts': int(time.time())}
open('$STATE_FILE', 'w').write(json.dumps(d, indent=2))
"
}

post_change() {
  local new="$1" model="$2"
  python3 "${HOME}/AGENT/comms/bridge.py" post \
    --from claude-cc6 \
    --topic endpoint_health \
    --body "${new} endpoint=${ENDPOINT} model=${model}" || true
}

while :; do
  prev=$(read_state)
  out=$(probe "$ENDPOINT")
  status=$(echo "$out" | awk '{print $1}')
  model=$(echo "$out" | awk '{$1=""; sub(/^ /,""); print}')
  if [[ "$status" != "$prev" ]]; then
    echo "[endpoint_health] $(date -u +%FT%TZ) state $prev → $status (model=$model)"
    post_change "$status" "${model:--}"
    write_state "$status" "${model:--}"
    if [[ "$prev" == "DOWN" && "$status" == "UP" && "$AUTO_FIRE" == "1" ]]; then
      echo "[endpoint_health] auto-firing baseline_capture.sh in background"
      ( BASELINE_ENDPOINT="$ENDPOINT" \
        BASELINE_OUT="$HERE/baseline.jsonl" \
        BASELINE_CONC="${BASELINE_CONC:-5}" \
        nohup bash "$HERE/baseline_capture.sh" >> "$HERE/baseline_capture.log" 2>&1 & ) || true
    fi
  fi
  [[ "$ONCE" == "1" ]] && break
  sleep "$POLL"   # SLOW_OP polling
done
