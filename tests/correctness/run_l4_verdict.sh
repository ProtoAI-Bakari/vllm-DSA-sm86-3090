#!/bin/bash
# --ProtoAI-Bakari--
# Story 9: L4 integration verdict runner. Given a CC9-shipped capture JSONL,
# runs the gate, posts bridge l4_result, returns verify_gate exit code.
#
# Usage:
#   ./run_l4_verdict.sh --integ <label> --under-test <path> [--baseline <path>]
# Defaults:
#   --baseline ./baseline.jsonl
#   --integ required (e.g. cc3-rev2, cc4-rev1, cc5-int4-rev1)
#
# Side effects:
#   - emits ~/AGENT/comms/CC6_GATE_RESULTS_<integ>.md
#   - posts bridge.db topic=l4_result with PASS/FAIL + report path
#   - exit 0 PASS / 1 FAIL / 2 input error
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${HERE}/baseline.jsonl"
UNDER=""
INTEG=""
GRILL=0
NO_BRIDGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ)       INTEG="$2"; shift 2 ;;
    --under-test)  UNDER="$2"; shift 2 ;;
    --baseline)    BASELINE="$2"; shift 2 ;;
    --grill)       GRILL=1; shift ;;
    --no-bridge)   NO_BRIDGE=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[run_l4_verdict] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INTEG" || -z "$UNDER" ]]; then
  echo "[run_l4_verdict] --integ and --under-test required" >&2
  exit 2
fi
if [[ ! -f "$BASELINE" ]]; then
  echo "[run_l4_verdict] baseline missing: $BASELINE" >&2
  exit 2
fi
if [[ ! -f "$UNDER" ]]; then
  echo "[run_l4_verdict] under-test missing: $UNDER" >&2
  exit 2
fi

REPORT="${HOME}/AGENT/comms/CC6_GATE_RESULTS_${INTEG}.md"
JSON_OUT="${HOME}/AGENT/comms/CC6_GATE_RESULTS_${INTEG}.json"

echo "[run_l4_verdict] integ=$INTEG"
echo "[run_l4_verdict] baseline=$BASELINE"
echo "[run_l4_verdict] under-test=$UNDER"
echo "[run_l4_verdict] report=$REPORT"

set +e
time python3 "$HERE/verify_gate.py" \
  --baseline "$BASELINE" \
  --under-test "$UNDER" \
  --integ "$INTEG" \
  --report "$REPORT" \
  --json-out "$JSON_OUT"
EC=$?
set -e

if [[ "$GRILL" == "1" && -x "$HERE/grill_33.sh" ]]; then
  echo "[run_l4_verdict] running grill_33 adversarial set"
  bash "$HERE/grill_33.sh" --integ "$INTEG" --under-test "$UNDER" || true
fi

if [[ "$EC" == "0" ]]; then
  VERDICT="PASS"
elif [[ "$EC" == "1" ]]; then
  VERDICT="FAIL"
else
  VERDICT="ERROR-EC${EC}"
fi

if [[ "$NO_BRIDGE" == "0" ]]; then
  python3 "${HOME}/AGENT/comms/bridge.py" post \
    --from claude-cc6 \
    --topic l4_result \
    --body "${VERDICT} integ=${INTEG} report=${REPORT}" \
    || echo "[run_l4_verdict] bridge post failed (non-fatal)" >&2
fi

echo "[run_l4_verdict] verdict=$VERDICT"
exit "$EC"
