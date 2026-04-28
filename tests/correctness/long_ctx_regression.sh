#!/bin/bash
# --ProtoAI-Bakari--
# Story 15 (W3): long-context regression suite — 8K/16K/32K needle-in-haystack
# prompts. After every CC4 (compressor / mla_sparse / swa) integration AND
# every CC8 KV-cache optimization, fire this to detect long-context coherence
# loss invisible at short context.
#
# Usage:
#   ./long_ctx_regression.sh --integ <label> [--tiers 8K,16K,32K] [--no-bridge]
# Env: BASELINE_ENDPOINT, BASELINE_MODEL, BASELINE_BACKEND, BASELINE_TIMEOUT etc.
#      forwarded to capture_one.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTEG=""
TIERS="8K,16K,32K"
NO_BRIDGE=0
BASELINE="${HERE}/long_ctx_baseline.jsonl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[long_ctx_regression] unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$INTEG" ]] && { echo "[long_ctx_regression] --integ required" >&2; exit 2; }

PROMPTS="${HERE}/long_ctx_prompts.jsonl"
python3 "$HERE/gen_long_ctx_prompts.py" --out "$PROMPTS" --tiers "$TIERS"

# Increase per-request timeout for long contexts: 32K input may need 5+ min for prefill on 7 t/s endpoints.
export BASELINE_TIMEOUT="${BASELINE_TIMEOUT:-600}"
# Lower concurrency: KV cache budget at 32K is per-slot ~2GB+; conc=2 keeps total under 8GB headroom.
export BASELINE_CONC="${BASELINE_CONC:-2}"
# Stream off for long-ctx — TTFT will be dominant; metrics still captured by capture_one.py
export BASELINE_STREAM="${BASELINE_STREAM:-1}"

UNDER="${HERE}/long_ctx_under_${INTEG}.jsonl"
: > "$UNDER"
echo "[long_ctx_regression] capturing $(wc -l < "$PROMPTS" | tr -d ' ') prompts → $UNDER"
echo "[long_ctx_regression] BASELINE_ENDPOINT=${BASELINE_ENDPOINT:-default} CONC=$BASELINE_CONC TIMEOUT=$BASELINE_TIMEOUT"

# Resumable: if $UNDER already has lines, skip those ids
DONE_IDS=$(python3 -c "
import json
seen=set()
try:
    with open('$UNDER') as f:
        for line in f:
            try: seen.add(json.loads(line)['id'])
            except: pass
except FileNotFoundError: pass
print(','.join(sorted(seen)))
")

REMAINING=$(mktemp); trap 'rm -f "$REMAINING"' EXIT
python3 -c "
import json
done = set('$DONE_IDS'.split(',')) if '$DONE_IDS' else set()
with open('$PROMPTS') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if o['id'] in done: continue
        print(line)
" > "$REMAINING"

N_REMAIN=$(wc -l < "$REMAINING" | tr -d ' ')
[[ "$N_REMAIN" == "0" ]] && echo "[long_ctx_regression] all prompts already captured"
if [[ "$N_REMAIN" != "0" ]]; then
  BASELINE_OUT="$UNDER" cat "$REMAINING" | xargs -L 1 -P "$BASELINE_CONC" -I '{}' bash -c '
    echo "{}" | BASELINE_OUT="'"$UNDER"'" python3 "'"$HERE"'/capture_one.py"
  '   # SLOW_OP long-ctx capture
fi

# Score: compute "needle-recall-rate" — count how many under-test outputs contain the expected needle value.
python3 - <<PY
import json, os, sys, time
under_path = "$UNDER"
report_path = os.path.expanduser("~/AGENT/comms/CC6_LONGCTX_${INTEG}.md")
total = correct = 0
per_tier = {}
miss = []
with open(under_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except: continue
        # find expected needle from prompt id (recover from gen)
PY

# Use the helper script for scoring
python3 "$HERE/score_long_ctx.py" --under "$UNDER" --prompts "$PROMPTS" --integ "$INTEG"
EC=$?

if [[ "$NO_BRIDGE" == "0" ]]; then
  if [[ "$EC" == "0" ]]; then VERDICT="PASS"; else VERDICT="FAIL"; fi
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic longctx_result \
    --body "${VERDICT} integ=${INTEG} tiers=${TIERS} report=${HOME}/AGENT/comms/CC6_LONGCTX_${INTEG}.md" || true
fi
exit "$EC"
