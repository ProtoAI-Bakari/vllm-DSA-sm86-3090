#!/bin/bash
# --ProtoAI-Bakari--
# CC0 directive (d): Cosmic-tier strict gate. Wraps tier_classify.py to
# specifically verify the Cosmic aggregate ≥ 1000 t/s threshold (LEAD §8.1
# stretch goal — only realistically reachable post-INT4 + EPLB + MTP +
# spec-decode + P2P-driver-patch + fabric tuning, per AO1 audit).
#
# Run:
#   ./cosmic_gate.sh --bench <CC8 bench JSON> --integ <label>
#
# Side effects:
#   - reuses tier_classify.py for parsing + verdict
#   - emits ~/AGENT/comms/CC6_COSMIC_<integ>.md
#   - bridge cosmic_result PASS|FAIL with measured aggregate t/s
#
# Exit: 0 cosmic ≥1000 t/s, 1 below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH=""
INTEG=""
NO_BRIDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2 ;;
    --integ) INTEG="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "[cosmic_gate] unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$BENCH" || -z "$INTEG" ]] && { echo "[cosmic_gate] --bench and --integ required" >&2; exit 2; }
[[ ! -f "$BENCH" ]] && { echo "[cosmic_gate] bench file missing: $BENCH" >&2; exit 2; }

REPORT="${HOME}/AGENT/comms/CC6_COSMIC_${INTEG}.md"
JSON_OUT="${HOME}/AGENT/comms/CC6_COSMIC_${INTEG}.json"

# Use python directly so we can parse the agg out and apply Cosmic-strict gate.
python3 - "$BENCH" "$INTEG" "$REPORT" <<'PY'
import json, os, sys, time

bench_path, integ, report_path = sys.argv[1:4]
with open(bench_path) as f:
    blob = json.load(f)

results = blob.get("results") if isinstance(blob, dict) else (blob if isinstance(blob, list) else [])
agg_max = None
for r in results:
    if isinstance(r, dict):
        v = r.get("tps_aggregate")
        if isinstance(v, (int, float)):
            agg_max = max(agg_max or 0.0, float(v))

COSMIC_THRESHOLD = 1000.0
ok = agg_max is not None and agg_max >= COSMIC_THRESHOLD

L = []
verdict = "✓ COSMIC" if ok else "✗ NOT-COSMIC"
L.append(f"# CC6 Cosmic Gate — {integ} — {verdict}")
L.append("")
L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC6]**")
L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
L.append("")
L.append(f"- Aggregate t/s peak: **{agg_max if agg_max is not None else '—'}**")
L.append(f"- Cosmic threshold:    **{COSMIC_THRESHOLD}**")
L.append(f"- Margin: **{(agg_max - COSMIC_THRESHOLD) if agg_max is not None else 'NA'}** t/s")
L.append("")
L.append("Note: Cosmic is the LEAD §8.1 aspirational stretch tier — requires the")
L.append("full optimization stack (INT4-AWQ-Marlin + EPLB + MTP + spec-decode +")
L.append("P2P driver patch + NCCL fabric tuning + fused-MoE) per AO1 audit. Reaching")
L.append("Cosmic is the public-OSS-launch milestone for vllm-DSA-sm86-3090.")
with open(report_path, "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
sys.exit(0 if ok else 1)
PY
EC=$?

if [[ "$NO_BRIDGE" == "0" ]]; then
  if [[ "$EC" == "0" ]]; then
    BODY="COSMIC integ=${INTEG} report=${REPORT}"
  else
    BODY="NOT-COSMIC integ=${INTEG} report=${REPORT}"
  fi
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic cosmic_result --body "$BODY" || true
fi

exit "$EC"
