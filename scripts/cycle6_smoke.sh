#!/bin/bash
# --ProtoAI-Bakari--
# CC0 cycle6 dispatch: DSV4-Flash-FP8 variant of cycle5_smoke.sh.
# Same pipeline (drain→health-wait→POST 5 fixtures→grade→bridge), with:
#   - default endpoint http://10.255.255.11:8000  (cluster vLLM-DSA-sm86 head)
#   - default model deepseek-ai/DeepSeek-V4-Flash-FP8
#   - sets VLLM_TRITON_MLA_SPARSE=1 in env hint (server-side flag, here for record)
#   - bridge topic = l4_verdict (CC0 directive)
#   - longer health-wait (DSV4 weight load ≥2 min)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEG=""
ENDPOINT="${CYCLE6_ENDPOINT:-http://10.255.255.11:8000}"
MODEL="${CYCLE6_MODEL:-deepseek-ai/DeepSeek-V4-Flash-FP8}"
HEALTHWAIT="${CYCLE6_HEALTHWAIT:-600}"
NO_BRIDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --health-wait) HEALTHWAIT="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    *) echo "[cycle6_smoke] unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$INTEG" ]] && { echo "[cycle6_smoke] --integ required (e.g. cc9-cycle6-rev1)" >&2; exit 2; }

EXTRA=()
[[ "$NO_BRIDGE" == "1" ]] && EXTRA+=(--no-bridge)

echo "[cycle6_smoke] DSV4 path: VLLM_TRITON_MLA_SPARSE=1 expected at server side"
echo "[cycle6_smoke] integ=$INTEG endpoint=$ENDPOINT model=$MODEL health_wait=$HEALTHWAIT"

# Reuse cycle5_smoke under the hood — single source of truth for the pipeline,
# only the defaults differ.
exec bash "$HERE/cycle5_smoke.sh" \
  --integ "$INTEG" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --health-wait "$HEALTHWAIT" \
  "${EXTRA[@]}"
