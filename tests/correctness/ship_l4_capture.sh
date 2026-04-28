#!/bin/bash
# --ProtoAI-Bakari--
# Story W3-NEW: single CC9-callable entrypoint for L4 grading.
#
#   ship_l4_capture.sh --integ <label> --capture <profile.json|baseline.jsonl>
#                      [--baseline <jsonl>]
#                      [--grill] [--longctx] [--no-bridge]
#
# Auto-detects capture format:
#   *.jsonl                      → grade directly
#   *.json (CC9 profile dump)    → run profile_to_jsonl.py first, then grade
# Forwards to run_l4_verdict.sh, optionally chains grill_33 + long_ctx after.
#
# Exit: 0 PASS / 1 FAIL / 2 input error.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTEG=""
CAPTURE=""
BASELINE="${HERE}/baseline.jsonl"
DO_GRILL=0
DO_LONGCTX=0
NO_BRIDGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ)     INTEG="$2"; shift 2 ;;
    --capture)   CAPTURE="$2"; shift 2 ;;
    --baseline)  BASELINE="$2"; shift 2 ;;
    --grill)     DO_GRILL=1; shift ;;
    --longctx)   DO_LONGCTX=1; shift ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    -h|--help)   sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[ship_l4] unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$INTEG" || -z "$CAPTURE" ]] && { echo "[ship_l4] --integ and --capture required" >&2; exit 2; }
[[ ! -f "$CAPTURE" ]] && { echo "[ship_l4] capture missing: $CAPTURE" >&2; exit 2; }

# Detect format
case "$CAPTURE" in
  *.jsonl)
    UNDER="$CAPTURE"
    ;;
  *.json|*.JSON)
    UNDER="${HOME}/AGENT/comms/captures/${INTEG}_converted.jsonl"
    mkdir -p "$(dirname "$UNDER")"
    echo "[ship_l4] converting profile JSON → JSONL: $CAPTURE → $UNDER"
    python3 "$HERE/profile_to_jsonl.py" --in "$CAPTURE" --out "$UNDER" --integ "$INTEG"
    ;;
  *)
    echo "[ship_l4] unknown extension on $CAPTURE — expect .json or .jsonl" >&2
    exit 2
    ;;
esac

EXTRA=()
[[ "$NO_BRIDGE" == "1" ]] && EXTRA+=(--no-bridge)

# Primary: numerics gate
set +e
time bash "$HERE/run_l4_verdict.sh" --integ "$INTEG" --under-test "$UNDER" --baseline "$BASELINE" "${EXTRA[@]}"
EC_GATE=$?
set -e

# Optional: grill_33 (additional adversarial check)
EC_GRILL=0
if [[ "$DO_GRILL" == "1" ]]; then
  set +e
  time bash "$HERE/grill_33.sh" --integ "$INTEG" --under-test "$UNDER" --baseline "$BASELINE" "${EXTRA[@]}"
  EC_GRILL=$?
  set -e
fi

# Optional: long-ctx regression (only if endpoint live + UNDER includes longctx prompts)
EC_LONGCTX=0
if [[ "$DO_LONGCTX" == "1" ]]; then
  set +e
  time bash "$HERE/long_ctx_regression.sh" --integ "$INTEG" "${EXTRA[@]}"
  EC_LONGCTX=$?
  set -e
fi

if [[ "$EC_GATE" == "0" && "$EC_GRILL" == "0" && "$EC_LONGCTX" == "0" ]]; then
  VERDICT="PASS"
else
  VERDICT="FAIL"
fi

if [[ "$NO_BRIDGE" == "0" ]]; then
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic l4_shipped \
    --body "${VERDICT} integ=${INTEG} gate=${EC_GATE} grill=${EC_GRILL} longctx=${EC_LONGCTX} capture=${CAPTURE}" || true
fi

echo "[ship_l4] verdict=$VERDICT  gate=$EC_GATE grill=$EC_GRILL longctx=$EC_LONGCTX"
[[ "$VERDICT" == "PASS" ]] && exit 0 || exit 1
