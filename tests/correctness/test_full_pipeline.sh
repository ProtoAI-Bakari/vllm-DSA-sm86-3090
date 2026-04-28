#!/bin/bash
# --ProtoAI-Bakari--
# End-to-end CI smoke for the entire CC6 gate suite. Wires every component
# together against synthetic fixtures so a refactor that breaks any stage
# fails fast — not on the next CC9 capture in production.
#
# Stages (each may skip cleanly):
#   1. validate_jsonl on prompts.jsonl + the synthetic baseline fixture
#   2. test_gate_self (5 positive/negative gate sanity tests)
#   3. verify_gate.py against synthetic PASS + synthetic FAIL fixtures
#   4. tier_classify.py against a synthetic Silver bench JSON
#   5. cosmic_gate.sh against synthetic Cosmic + non-Cosmic bench JSONs
#   6. coverage_matrix.py
#
# Skipped on failure of any prerequisite (e.g. CUDA/vllm not installed); each
# stage reports its own status. Exit 0 only if every active stage passed.
#
# Run:
#   ./test_full_pipeline.sh
#   VERBOSE=1 ./test_full_pipeline.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP=$(mktemp -d -t cc6_full_pipeline_XXXX)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
SKIP=0
declare -a FAILED=()

stage() {
  local name="$1"
  shift
  echo
  echo "=== [$name] ==="
  set +e
  if [[ -n "${VERBOSE:-}" ]]; then
    "$@"
  else
    "$@" >/dev/null 2>&1
  fi
  local ec=$?
  set -e
  if [[ "$ec" == "0" ]]; then
    echo "  [PASS] $name"
    PASS=$((PASS+1))
  elif [[ "$ec" == "77" || "$ec" == "5" ]]; then
    echo "  [SKIP] $name (ec=$ec)"
    SKIP=$((SKIP+1))
  else
    echo "  [FAIL] $name (ec=$ec)"
    FAIL=$((FAIL+1))
    FAILED+=("$name")
  fi
}

# Synthetic fixtures (same as test_gate_self.sh)
cat > "$TMP/baseline.jsonl" <<'EOF'
{"id":"a","category":"factual","text":"Paris","final_payload":{"choices":[{"text":"Paris","logprobs":{"top_logprobs":[{"Paris":-0.1,"Lyon":-2.3}]}}]}}
{"id":"b","category":"mmlu_real","subject":"physics","expected_letter":"B","text":"B","final_payload":{"choices":[{"text":"B","logprobs":{"top_logprobs":[{"B":-0.1,"A":-2.0}]}}]}}
EOF
cp "$TMP/baseline.jsonl" "$TMP/under_pass.jsonl"
cat > "$TMP/under_fail.jsonl" <<'EOF'
{"id":"a","category":"factual","text":"Lyon","final_payload":{"choices":[{"text":"Lyon"}]}}
{"id":"b","category":"mmlu_real","subject":"physics","expected_letter":"B","text":"A","final_payload":{"choices":[{"text":"A"}]}}
EOF
cat > "$TMP/bench_silver.json" <<'EOF'
{"model":"smoke","results":[{"conc":4,"tps_per_req":80,"tps_aggregate":320}]}
EOF
cat > "$TMP/bench_cosmic.json" <<'EOF'
{"model":"smoke","results":[{"conc":4,"tps_per_req":300,"tps_aggregate":1200}]}
EOF

stage "validate_jsonl prompts" python3 "$HERE/validate_jsonl.py" prompts "$HERE/prompts.jsonl"
stage "validate_jsonl capture" python3 "$HERE/validate_jsonl.py" capture "$TMP/baseline.jsonl"
stage "test_gate_self.sh"     bash "$HERE/test_gate_self.sh"
stage "verify_gate PASS"      python3 "$HERE/verify_gate.py" --baseline "$TMP/baseline.jsonl" --under-test "$TMP/under_pass.jsonl" --integ pipeline-pass --report "$TMP/r1.md"
stage "verify_gate FAIL→1"    bash -c "python3 '$HERE/verify_gate.py' --baseline '$TMP/baseline.jsonl' --under-test '$TMP/under_fail.jsonl' --integ pipeline-fail --report '$TMP/r2.md'; [[ \$? == 1 ]]"
stage "tier_classify Silver"  python3 "$HERE/tier_classify.py" --in "$TMP/bench_silver.json" --integ pipeline-silver --report "$TMP/tier1.md" --no-bridge
stage "cosmic_gate COSMIC"    bash "$HERE/cosmic_gate.sh" --bench "$TMP/bench_cosmic.json" --integ pipeline-cosmic --no-bridge
stage "cosmic_gate NOT→1"     bash -c "bash '$HERE/cosmic_gate.sh' --bench '$TMP/bench_silver.json' --integ pipeline-not-cosmic --no-bridge; [[ \$? == 1 ]]"
stage "coverage_matrix"       python3 "$HERE/coverage_matrix.py" --report "$TMP/cov.md"
stage "summarize_captures"    python3 "$HERE/summarize_captures.py" "$TMP/baseline.jsonl"

echo
echo "================================================================"
echo "FULL PIPELINE: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
[[ "$FAIL" -gt 0 ]] && { echo "FAILED: ${FAILED[*]}"; exit 1; }
echo "ALL STAGES GREEN"
exit 0
