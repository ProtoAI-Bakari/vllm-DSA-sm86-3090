#!/bin/bash
# --ProtoAI-Bakari--
# CC0 23:18Z dispatch (4): post-Bronze INT4 fallback variant of cycle6_smoke.
# Same pipeline; presets for INT4-AWQ-Marlin sm_86 endpoint:
#   - VLLM_USE_AWQ_MARLIN=1 expected at server side
#   - default model deepseek-ai/DeepSeek-V4-Flash-INT4-AWQ
#   - longer health-wait (INT4 quant load + scales upload)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEG=""
ENDPOINT="${CYCLE6_INT4_ENDPOINT:-http://10.255.255.11:8000}"
MODEL="${CYCLE6_INT4_MODEL:-deepseek-ai/DeepSeek-V4-Flash-INT4-AWQ}"
HEALTHWAIT="${CYCLE6_INT4_HEALTHWAIT:-900}"
NO_BRIDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --integ) INTEG="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --health-wait) HEALTHWAIT="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    *) echo "[cycle6_int4] unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$INTEG" ]] && { echo "[cycle6_int4] --integ required" >&2; exit 2; }
EXTRA=()
[[ "$NO_BRIDGE" == "1" ]] && EXTRA+=(--no-bridge)
echo "[cycle6_int4] post-Bronze INT4 fallback path: VLLM_USE_AWQ_MARLIN=1 expected"
echo "[cycle6_int4] integ=$INTEG endpoint=$ENDPOINT model=$MODEL health_wait=$HEALTHWAIT"
exec bash "$HERE/cycle5_smoke.sh" \
  --integ "$INTEG" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --health-wait "$HEALTHWAIT" \
  "${EXTRA[@]}"
