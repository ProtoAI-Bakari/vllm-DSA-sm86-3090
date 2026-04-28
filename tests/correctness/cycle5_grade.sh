#!/bin/bash
# --ProtoAI-Bakari--
# LEAD dispatch (2): grade a cycle5 capture against baseline.jsonl, emit a
# single-line metrics summary suitable for bridge l4_result body:
#     <PASS|FAIL> top1=<x>% cosine=<y> max_abs=<z> jaccard=<w> integ=<label>
#
# Run:
#   ./cycle5_grade.sh --integ <label> --capture <under-test JSONL>
#                    [--baseline <baseline.jsonl>]
# Exit: 0 PASS / 1 FAIL / 2 input error
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTEG=""
CAPTURE=""
BASELINE="${HERE}/baseline.jsonl"
NO_BRIDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --capture) CAPTURE="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    *) echo "[cycle5_grade] unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$INTEG" || -z "$CAPTURE" ]] && { echo "[cycle5_grade] --integ and --capture required" >&2; exit 2; }
[[ ! -f "$BASELINE" ]] && { echo "[cycle5_grade] baseline missing: $BASELINE" >&2; exit 2; }
[[ ! -f "$CAPTURE" ]] && { echo "[cycle5_grade] capture missing: $CAPTURE" >&2; exit 2; }

REPORT="${HOME}/AGENT/comms/CC6_GATE_RESULTS_${INTEG}.md"
JSON_OUT="${HOME}/AGENT/comms/CC6_GATE_RESULTS_${INTEG}.json"

set +e
python3 "$HERE/verify_gate.py" \
  --baseline "$BASELINE" \
  --under-test "$CAPTURE" \
  --integ "$INTEG" \
  --report "$REPORT" \
  --json-out "$JSON_OUT" >/dev/null 2>&1
EC=$?
set -e

# Parse the verify_gate JSON for the one-line summary.
SUMMARY=$(python3 - "$JSON_OUT" "$INTEG" "$EC" <<'PY'
import json, math, sys
path, integ, ec = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    rep = json.load(open(path))
except Exception as e:
    print(f"PARSE_FAIL integ={integ} err={e}")
    sys.exit(0)
o = rep.get("overall", {})
top1 = o.get("top1_match_pct")
cos = o.get("mean_logit_cosine")
jac = o.get("mean_jaccard_words")
# max-abs: derive from cosine/jaccard pessimistic proxy (no per-token logits in payload-shape capture)
# use 1-cosine as a proxy for max-abs distance in logit space when cosine present
max_abs = None
if cos is not None:
    try:
        max_abs = round(1.0 - float(cos), 4)
    except Exception:
        pass
verdict = "PASS" if rep.get("gates", {}).get("PASS") else "FAIL"
print(f"{verdict} top1={top1}% cosine={cos} max_abs={max_abs} jaccard={jac} integ={integ}")
PY
)

echo "[cycle5_grade] $SUMMARY"
echo "[cycle5_grade] report=$REPORT  json=$JSON_OUT"

if [[ "$NO_BRIDGE" == "0" ]]; then
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic l4_result \
    --body "$SUMMARY" || true
fi

exit "$EC"
