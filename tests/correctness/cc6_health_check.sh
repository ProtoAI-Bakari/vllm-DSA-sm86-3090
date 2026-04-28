#!/bin/bash
# --ProtoAI-Bakari--
# Daily health check for the CC6 gate suite. Run via cron / launchd.
# Composes:
#   - test_full_pipeline.sh         (gate-suite bitrot detection)
#   - aggregate_l4_history.py        (verdict trend digest)
#   - cc6_daily_status.py            (bridge.db 24h digest)
#   - endpoint_health_daemon.sh --once  (snapshot endpoint state)
# Posts a single bridge daily_health milestone with overall status.
#
# Run:
#   ./cc6_health_check.sh                        # once, foreground
#   PIPELINE_EC_TOLERATE=1 ./cc6_health_check.sh # treat pipeline FAIL as warn
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS=$(date -u +%FT%TZ)
LOG="${HOME}/AGENT/comms/CC6_HEALTH_CHECK_$(date -u +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "[cc6_health] $TS start" | tee -a "$LOG"

set +e
echo "[cc6_health] stage: test_full_pipeline.sh" | tee -a "$LOG"
bash "$HERE/test_full_pipeline.sh" >> "$LOG" 2>&1
PIPELINE_EC=$?
echo "[cc6_health] stage: aggregate_l4_history.py" | tee -a "$LOG"
python3 "$HERE/aggregate_l4_history.py" >> "$LOG" 2>&1
HISTORY_EC=$?
echo "[cc6_health] stage: cc6_daily_status.py" | tee -a "$LOG"
python3 "$HERE/cc6_daily_status.py" --hours 24 --no-bridge >> "$LOG" 2>&1
DAILY_EC=$?
echo "[cc6_health] stage: endpoint_health_daemon.sh --once" | tee -a "$LOG"
bash "$HERE/endpoint_health_daemon.sh" --once >> "$LOG" 2>&1
ENDPOINT_EC=$?
set -e

TOLERATE="${PIPELINE_EC_TOLERATE:-0}"
if [[ "$PIPELINE_EC" == "0" && "$HISTORY_EC" == "0" && "$DAILY_EC" == "0" && "$ENDPOINT_EC" == "0" ]]; then
  STATUS="GREEN"
elif [[ "$PIPELINE_EC" != "0" && "$TOLERATE" == "0" ]]; then
  STATUS="RED"
else
  STATUS="AMBER"
fi
echo "[cc6_health] $STATUS pipeline=$PIPELINE_EC history=$HISTORY_EC daily=$DAILY_EC endpoint=$ENDPOINT_EC log=$LOG" | tee -a "$LOG"

python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic daily_health \
  --body "${STATUS} ts=${TS} pipeline=${PIPELINE_EC} history=${HISTORY_EC} daily=${DAILY_EC} endpoint=${ENDPOINT_EC} log=${LOG}" || true

[[ "$STATUS" == "GREEN" ]] && exit 0 || exit 1
