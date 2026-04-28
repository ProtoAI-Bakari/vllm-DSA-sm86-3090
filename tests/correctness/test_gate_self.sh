#!/bin/bash
# --ProtoAI-Bakari--
# Story W3-NEW: gate self-tests. Synthetic positive + negative fixtures that
# prove the gate runner correctly distinguishes PASS vs FAIL across every axis
# (top-1, Jaccard, cosine, perplexity, edge errors).
#
# Without this, the gate could silently degrade — eg if verify_gate.py is
# refactored and accidentally accepts everything, no merge would notice until
# a real regression slipped through. This test asserts the assertions.
#
# Run:
#   ./test_gate_self.sh                    # all 5 self-tests
#   ./test_gate_self.sh --keep-tmp         # leave /tmp fixtures for inspection
#
# Exit: 0 all self-tests pass, non-zero on first mismatch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP=$(mktemp -d -t cc6_gate_self_XXXX)
KEEP=0
[[ "${1:-}" == "--keep-tmp" ]] && KEEP=1
trap '[[ "$KEEP" == "0" ]] && rm -rf "$TMP" || echo "tmp=$TMP"' EXIT

# ---------- baseline fixture (5 records, all categories represented) ----------
cat > "$TMP/baseline.jsonl" <<'EOF'
{"id":"factual_0000","category":"factual","text":"Au.","final_payload":{"choices":[{"text":"Au.","logprobs":{"top_logprobs":[{"Au":-0.1,"gold":-2.3,"Pt":-5.1},{".":-0.05,"?":-3.0}]}}]}}
{"id":"factual_0001","category":"factual","text":"Jupiter","final_payload":{"choices":[{"text":"Jupiter","logprobs":{"top_logprobs":[{"Jupiter":-0.02,"Saturn":-4.5}]}}]}}
{"id":"mmlu_0000","category":"mmlu_real","subject":"physics","expected_letter":"B","text":"B","final_payload":{"choices":[{"text":"B","logprobs":{"top_logprobs":[{"B":-0.1,"A":-2.0,"C":-2.5,"D":-3.0}]}}]}}
{"id":"code_0000","category":"code","text":"def f(n):\n    return n*2","final_payload":{"choices":[{"text":"def f(n):\n    return n*2"}]}}
{"id":"edge_0000_empty","category":"edge","text":"","final_payload":{"choices":[{"text":""}]}}
EOF

# ---------- positive (identical) ----------
cp "$TMP/baseline.jsonl" "$TMP/under_pass.jsonl"

# ---------- negative: every record diverges ----------
cat > "$TMP/under_fail_total.jsonl" <<'EOF'
{"id":"factual_0000","category":"factual","text":"Gold (Au).","final_payload":{"choices":[{"text":"Gold (Au).","logprobs":{"top_logprobs":[{"Gold":-0.1,"gold":-2.5}]}}]}}
{"id":"factual_0001","category":"factual","text":"Saturn","final_payload":{"choices":[{"text":"Saturn"}]}}
{"id":"mmlu_0000","category":"mmlu_real","subject":"physics","expected_letter":"B","text":"A","final_payload":{"choices":[{"text":"A","logprobs":{"top_logprobs":[{"A":-0.1,"B":-1.5}]}}]}}
{"id":"code_0000","category":"code","text":"def g(x):\n    return None","final_payload":{"choices":[{"text":"def g(x):\n    return None"}]}}
{"id":"edge_0000_empty","category":"edge","text":"x","final_payload":{"choices":[{"text":"x"}]}}
EOF

# ---------- subtle: 1 record diverges (98% match — boundary) ----------
# 5 records → 4/5 = 80% would FAIL; need >49 records to land at ~98%. Build it.
python3 - <<PY
import json, random
random.seed(0xC0FFEE)
base=[]
for i in range(50):
    base.append({"id":f"k_{i:04d}","category":"factual","text":f"answer_{i}","final_payload":{"choices":[{"text":f"answer_{i}"}]}})
with open("$TMP/baseline_50.jsonl","w") as f:
    for o in base: f.write(json.dumps(o)+"\n")
under=[dict(o) for o in base]
# corrupt 1 → 49/50 = 98.0% top-1 (boundary PASS)
under[7]={"id":"k_0007","category":"factual","text":"WRONG","final_payload":{"choices":[{"text":"WRONG"}]}}
with open("$TMP/under_boundary_pass.jsonl","w") as f:
    for o in under: f.write(json.dumps(o)+"\n")
# corrupt 2 → 48/50 = 96.0% (FAIL)
under2=[dict(o) for o in base]
under2[7]={"id":"k_0007","category":"factual","text":"WRONG1","final_payload":{"choices":[{"text":"WRONG1"}]}}
under2[13]={"id":"k_0013","category":"factual","text":"WRONG2","final_payload":{"choices":[{"text":"WRONG2"}]}}
with open("$TMP/under_boundary_fail.jsonl","w") as f:
    for o in under2: f.write(json.dumps(o)+"\n")
PY

# ---------- ID disjoint: under-test missing some ids ----------
head -3 "$TMP/baseline.jsonl" > "$TMP/under_partial.jsonl"

PASS_COUNT=0
FAIL_COUNT=0
declare -a FAILURES=()

run_case() {
  local label="$1" baseline="$2" under="$3" expect_ec="$4"
  set +e
  python3 "$HERE/verify_gate.py" --baseline "$baseline" --under-test "$under" \
    --integ "self-${label}" --report "${TMP}/report_${label}.md" >/dev/null 2>&1
  local got_ec=$?
  set -e
  if [[ "$got_ec" == "$expect_ec" ]]; then
    echo "  [PASS] $label exit=$got_ec (expected=$expect_ec)"
    PASS_COUNT=$((PASS_COUNT+1))
  else
    echo "  [FAIL] $label exit=$got_ec (expected=$expect_ec)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILURES+=("$label")
  fi
}

echo "[test_gate_self] running 5 self-tests..."
run_case "identical_pass"         "$TMP/baseline.jsonl"     "$TMP/under_pass.jsonl"          0
run_case "total_divergence_fail"  "$TMP/baseline.jsonl"     "$TMP/under_fail_total.jsonl"   1
run_case "boundary_98pct_pass"    "$TMP/baseline_50.jsonl"  "$TMP/under_boundary_pass.jsonl" 0
run_case "boundary_96pct_fail"    "$TMP/baseline_50.jsonl"  "$TMP/under_boundary_fail.jsonl" 1
run_case "partial_capture_only_3" "$TMP/baseline.jsonl"     "$TMP/under_partial.jsonl"       0

echo ""
echo "[test_gate_self] PASS=$PASS_COUNT FAIL=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "[test_gate_self] failures: ${FAILURES[*]}"
  exit 1
fi
echo "[test_gate_self] ALL SELF-TESTS GREEN"
exit 0
