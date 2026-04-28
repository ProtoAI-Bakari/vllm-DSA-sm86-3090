#!/bin/bash
# --ProtoAI-Bakari--
# Capture baseline outputs for all prompts in prompts.jsonl, conc=4-5 micro-batches.
# Resumable: skips ids already present in baseline.jsonl.
#
# Env (defaults shown):
#   BASELINE_ENDPOINT  http://10.255.255.11:8000   (cuda1 vLLM PP — fall back to sys4 GLM-5.1 MLX with --endpoint)
#   BASELINE_MODEL     (auto-detected from /v1/models if empty)
#   BASELINE_OUT       ./baseline.jsonl
#   BASELINE_CONC      5
#   BASELINE_LOGPROBS  20  (set 0 for MLX servers that reject the param)
#   BASELINE_BACKEND   completions  (set 'chat' for chat-only servers)
#   BASELINE_TIMEOUT   120
#
# Usage:
#   ./baseline_capture.sh                      # default cuda1:8000
#   BASELINE_ENDPOINT=http://10.255.255.4:8000 BASELINE_LOGPROBS=0 BASELINE_BACKEND=chat ./baseline_capture.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS="${BASELINE_PROMPTS:-$HERE/prompts.jsonl}"
OUT="${BASELINE_OUT:-$HERE/baseline.jsonl}"
CONC="${BASELINE_CONC:-5}"
ENDPOINT="${BASELINE_ENDPOINT:-http://10.255.255.11:8000}"

if [[ ! -f "$PROMPTS" ]]; then
  echo "[baseline_capture] generating prompts → $PROMPTS"
  python3 "$HERE/gen_prompts.py" --out "$PROMPTS"
fi

touch "$OUT"
COMPLETED="$(python3 -c "
import json, sys
seen=set()
try:
    with open('$OUT') as f:
        for line in f:
            try:
                seen.add(json.loads(line)['id'])
            except Exception:
                pass
except FileNotFoundError:
    pass
print(len(seen))
print(','.join(sorted(seen)) if seen else '')
")"
N_DONE=$(echo "$COMPLETED" | sed -n '1p')
DONE_IDS=$(echo "$COMPLETED" | sed -n '2p')
N_TOTAL=$(wc -l < "$PROMPTS" | awk '{print $1}')
echo "[baseline_capture] endpoint=$ENDPOINT  prompts=$N_TOTAL  already_done=$N_DONE  conc=$CONC  out=$OUT"

REMAINING="$(mktemp)"
trap 'rm -f "$REMAINING"' EXIT
python3 -c "
import json, sys
done = set('$DONE_IDS'.split(',')) if '$DONE_IDS' else set()
with open('$PROMPTS') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try:
            o=json.loads(line)
        except Exception:
            continue
        if o['id'] in done: continue
        sys.stdout.write(line+'\n')
" > "$REMAINING"
N_REMAIN=$(wc -l < "$REMAINING" | awk '{print $1}')
echo "[baseline_capture] remaining=$N_REMAIN"
[[ "$N_REMAIN" == "0" ]] && { echo "[baseline_capture] DONE — all $N_TOTAL captured."; exit 0; }

T0=$(date +%s)
echo "[baseline_capture] starting at $(date)"
# xargs -P CONC: each child reads its single line on stdin
cat "$REMAINING" | xargs -L 1 -P "$CONC" -I '{}' bash -c '
  echo "{}" | python3 "'"$HERE"'/capture_one.py"
'

T1=$(date +%s)
ELAPSED=$((T1-T0))
echo "[baseline_capture] FINISHED at $(date) — elapsed ${ELAPSED}s"
N_FINAL=$(wc -l < "$OUT" | awk '{print $1}')
N_ERR=$(grep -c '"error"' "$OUT" 2>/dev/null || echo 0)
echo "[baseline_capture] baseline.jsonl lines=$N_FINAL  errors=$N_ERR"
