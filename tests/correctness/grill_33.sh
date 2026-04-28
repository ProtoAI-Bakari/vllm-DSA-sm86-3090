#!/bin/bash
# --ProtoAI-Bakari--
# Story 16 (W3): grill_33 — adversarial 33-prompt set fired immediately after a
# merge to catch regressions that hide in averages but explode on edge cases.
#
# Hooks into run_l4_verdict.sh via --grill flag. Standalone usage:
#     ./grill_33.sh --integ <label> --under-test <dir-or-jsonl> [--baseline <jsonl>]
#
# What it does:
#   1. Generates the fixed 33-prompt grill set (deterministic) into grill_33_prompts.jsonl.
#   2. If --under-test is a directory, runs capture_one.py for each prompt against
#      the endpoint $BASELINE_ENDPOINT, writing under_test_<integ>.jsonl in that dir.
#      If --under-test is a JSONL, treats it as already-captured.
#   3. Runs verify_gate.py with category-restricted thresholds (more permissive
#      on creative/repetition; strict on factual/code/edge).
#   4. Emits ~/AGENT/comms/CC6_GRILL33_<integ>.md and posts bridge topic=grill_result.
#
# Exit: 0 ok / 1 grill FAIL / 2 input error.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTEG=""
UNDER=""
BASELINE="${HERE}/baseline.jsonl"
NO_BRIDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --under-test) UNDER="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[grill_33] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INTEG" || -z "$UNDER" ]]; then
  echo "[grill_33] --integ and --under-test required" >&2
  exit 2
fi

GRILL_PROMPTS="${HERE}/grill_33_prompts.jsonl"
python3 "$HERE/grill_33_gen.py" --out "$GRILL_PROMPTS" >/dev/null

if [[ -d "$UNDER" ]]; then
  OUT="${UNDER%/}/grill33_under_${INTEG}.jsonl"
  : > "$OUT"
  echo "[grill_33] capturing 33 prompts → $OUT (endpoint=${BASELINE_ENDPOINT:-default})"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    BASELINE_OUT="$OUT" echo "$line" | python3 "$HERE/capture_one.py" || true   # SLOW_OP per-prompt
  done < "$GRILL_PROMPTS"
  UNDER="$OUT"
fi

REPORT="${HOME}/AGENT/comms/CC6_GRILL33_${INTEG}.md"
JSON_OUT="${HOME}/AGENT/comms/CC6_GRILL33_${INTEG}.json"

set +e
time python3 "$HERE/verify_gate.py" \
  --baseline "$BASELINE" \
  --under-test "$UNDER" \
  --integ "grill33-${INTEG}" \
  --report "$REPORT" \
  --json-out "$JSON_OUT"
EC=$?
set -e

if [[ "$EC" == "0" ]]; then VERDICT="PASS"; elif [[ "$EC" == "1" ]]; then VERDICT="FAIL"; else VERDICT="ERROR-EC${EC}"; fi

if [[ "$NO_BRIDGE" == "0" ]]; then
  python3 "${HOME}/AGENT/comms/bridge.py" post \
    --from claude-cc6 \
    --topic grill_result \
    --body "${VERDICT} integ=${INTEG} report=${REPORT}" || true
fi

echo "[grill_33] verdict=$VERDICT"
exit "$EC"
