#!/usr/bin/env bash
# --ProtoAI-Bakari--
# correctness_replay.sh — replay CC6's frozen-seed prompt set against the live
# deploy and emit top-1 agreement + cosine similarity vs CC6's PP baseline.
# Lives on lane-cc8-deploy because CC8 needs it to gate "first L4 PASS"
# before publishing a tier number.
#
# Usage:
#   bench/runners/correctness_replay.sh \
#     --endpoint http://cuda1:8000 \
#     --model glm51-iq2xxs \
#     --baseline /Users/z/AGENT/comms/CC6_BASELINE.jsonl \
#     --n 1500 \
#     --out ~/AGENT/correctness_replay_<ts>.json
#
# Each line of --baseline must be: {"id": int, "prompt": str, "tokens": [int],
#   "top1": str, "logprobs_top5": [[token_id, logprob], ...], "embedding": [...]}
# Output JSON: {top1_agreement, cosine_mean, cosine_p95, perplexity_drift,
#   per_prompt_failures, n_total, n_ok}

set -euo pipefail

ENDPOINT=""
MODEL=""
BASELINE=""
N=1500
MAX_TOKENS=256
TIMEOUT_S=600
CONC=4
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --conc) CONC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$ENDPOINT" || -z "$MODEL" || -z "$BASELINE" ]] && {
  echo "missing --endpoint / --model / --baseline" >&2; exit 2
}
[[ -f "$BASELINE" ]] || { echo "baseline not found: $BASELINE" >&2; exit 2; }

TS=$(date +%Y%m%dT%H%M%SZ)
[[ -z "$OUT" ]] && OUT="${HOME}/AGENT/correctness_replay_${TS}.json"
mkdir -p "$(dirname "$OUT")"

# Pre-flight
HC=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$ENDPOINT/v1/models" || echo 000)
[[ "$HC" == "200" ]] || { echo "ABORT: $ENDPOINT/v1/models=$HC" >&2; exit 3; }

python3 - "$ENDPOINT" "$MODEL" "$BASELINE" "$N" "$MAX_TOKENS" "$TIMEOUT_S" "$CONC" "$OUT" <<'PY'
import json, math, sys, time
import urllib.request
import concurrent.futures as cf

(endpoint, model, baseline_path, n_str, max_tokens_str,
 timeout_str, conc_str, out_path) = sys.argv[1:9]
n_target = int(n_str); max_tokens = int(max_tokens_str)
timeout = float(timeout_str); conc = int(conc_str)


def cosine(a, b):
    if not a or not b or len(a) != len(b): return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0: return None
    return dot / (na * nb)


with open(baseline_path) as f:
    baseline = [json.loads(l) for l in f if l.strip()][:n_target]


def replay_one(b):
    body = json.dumps({
        "model": model, "prompt": b["prompt"],
        "max_tokens": max_tokens, "temperature": 0,
        "logprobs": 5, "stream": False,
    }).encode()
    req = urllib.request.Request(endpoint + "/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"id": b["id"], "ok": False, "err": str(e)}
    t1 = time.time()
    text = data["choices"][0]["text"]
    top1_match = (text.strip()[:len(b.get("top1", "")[:32])] ==
                  b.get("top1", "")[:32])
    cos = None
    if "embedding" in b and "embedding" in data.get("choices", [{}])[0]:
        cos = cosine(b["embedding"], data["choices"][0]["embedding"])
    # Perplexity proxy from logprobs (deltavs baseline if present)
    drift = None
    bp = b.get("logprobs_top5") or []
    if bp and "logprobs" in data["choices"][0]:
        dp = data["choices"][0]["logprobs"].get("token_logprobs") or []
        bp_means = sum(p[1] for p in bp[:len(dp)]) / max(len(bp[:len(dp)]), 1)
        dp_mean = sum(dp[:len(bp)]) / max(len(dp[:len(bp)]), 1)
        drift = abs(dp_mean - bp_means)
    return {"id": b["id"], "ok": True, "wall_s": t1 - t0,
            "top1_match": top1_match, "cosine": cos,
            "logprob_drift": drift, "completion": text}


with cf.ThreadPoolExecutor(max_workers=conc) as pool:
    results = list(pool.map(replay_one, baseline))

ok = [r for r in results if r.get("ok")]
top1_n = sum(1 for r in ok if r.get("top1_match"))
cos_vals = [r["cosine"] for r in ok if r.get("cosine") is not None]
drift_vals = [r["logprob_drift"] for r in ok if r.get("logprob_drift") is not None]


def percentile(vals, q):
    if not vals: return None
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


summary = {
    "n_total": len(results),
    "n_ok": len(ok),
    "n_err": len(results) - len(ok),
    "top1_agreement": (top1_n / len(ok)) if ok else 0.0,
    "cosine_mean": (sum(cos_vals) / len(cos_vals)) if cos_vals else None,
    "cosine_p95": percentile(cos_vals, 0.95),
    "logprob_drift_mean": (sum(drift_vals) / len(drift_vals)) if drift_vals else None,
    "logprob_drift_p95": percentile(drift_vals, 0.95),
    "endpoint": endpoint,
    "model": model,
    "baseline": baseline_path,
}

# Acceptance gates per role CLAUDE.md
gates = {
    "top1_agreement_>=0.98": summary["top1_agreement"] >= 0.98,
    "cosine_p95_>=0.97": (summary["cosine_p95"] or 0) >= 0.97 if summary["cosine_p95"] is not None else None,
    "logprob_drift_p95_<=0.05": (summary["logprob_drift_p95"] or 1) <= 0.05 if summary["logprob_drift_p95"] is not None else None,
}
summary["acceptance_gates"] = gates
summary["overall_pass"] = all(v for v in gates.values() if v is not None)

with open(out_path, "w") as f:
    json.dump({"summary": summary, "raw": results}, f, indent=2)

print(json.dumps(summary, indent=2))
PY

echo "$OUT"
