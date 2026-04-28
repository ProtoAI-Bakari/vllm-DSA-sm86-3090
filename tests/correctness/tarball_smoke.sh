#!/usr/bin/env bash
# tarball_smoke.sh — install a vllm-dsa-sm86 tarball into a fresh venv-3.12
# and confirm it can serve a tiny non-DSA baseline before any cluster fanout.
#
# Goal: gate that any tarball entering install_cluster.sh actually produces a
# bootable vLLM. We deliberately serve a NON-DSA model (e.g. facebook/opt-125m)
# because the DSA path is the very thing CC3/CC4/CC5 are still porting — failing
# the dense path means the rebuild broke vLLM as a whole, not just DSA.
#
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]
# --ProtoAI-Bakari--

set -euo pipefail

TARBALL="${1:-}"
SMOKE_MODEL="${SMOKE_MODEL:-facebook/opt-125m}"
SMOKE_PORT="${SMOKE_PORT:-8765}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"
WORKDIR="${WORKDIR:-$(mktemp -d -t vllm-dsa-smoke-XXXXXX)}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

usage() {
  cat <<EOF
Usage: $0 PATH/TO/vllm-dsa-sm86_v0.20.0+cu128_amd64.tar.gz

  SMOKE_MODEL    HF id of dense model used for boot test (default: facebook/opt-125m)
  SMOKE_PORT     Loopback port used by the test server (default: 8765)
  SMOKE_TIMEOUT  Seconds to wait for /v1/models 200 (default: 180)
  WORKDIR        Override scratch dir (default: mktemp)
  KEEP_WORKDIR=1 Skip cleanup on exit (debug)

Exit codes:
   0 PASS
  30 tarball missing or non-readable
  31 BUILD_INFO assertion failed
  32 INSTALL.sh failed
  33 import vllm failed (post-install)
  34 server boot timeout
  35 /v1/models 200 but completion path errored
  36 server shutdown failed
EOF
}

[[ -z "$TARBALL" || "$1" == "-h" || "$1" == "--help" ]] && { usage; exit 0; }
[[ -r "$TARBALL" ]] || { echo "FATAL: cannot read $TARBALL" >&2; exit 30; }

cleanup() {
  local rc=$?
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$KEEP_WORKDIR" != "1" ]]; then
    rm -rf "$WORKDIR"
  else
    echo "[smoke] kept workdir: $WORKDIR" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT

log() { printf '[smoke %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# 1. Extract the tarball
log "extracting → $WORKDIR"
mkdir -p "$WORKDIR"
tar -xzf "$TARBALL" -C "$WORKDIR"
[[ -d "$WORKDIR/vllm-dsa-sm86" ]] || { log "tarball did not contain vllm-dsa-sm86/"; exit 31; }

cd "$WORKDIR/vllm-dsa-sm86"

# 2. BUILD_INFO assertions (gate must be PASS)
[[ -f BUILD_INFO.json ]] || { log "BUILD_INFO.json missing"; exit 31; }
GATE=$(python3 -c "import json,sys;print(json.load(open('BUILD_INFO.json'))['correctness_gate']['passed'])")
[[ "$GATE" == "True" || "$GATE" == "true" ]] || { log "gate.passed=$GATE"; exit 31; }
ARCHES=$(python3 -c "import json,sys;print(','.join(json.load(open('BUILD_INFO.json'))['cuda_arch_list']))")
[[ "$ARCHES" == *"8.6"* ]] || { log "BUILD_INFO.cuda_arch_list=$ARCHES (need 8.6)"; exit 31; }

# 3. Install into a throwaway venv
TARGET="$WORKDIR/.venv-smoke"
log "INSTALL.sh → $TARGET"
bash INSTALL.sh --target "$TARGET" --cluster-vetted >"$WORKDIR/install.log" 2>&1 || {
  log "INSTALL.sh failed (see $WORKDIR/install.log)"; exit 32;
}

# 4. import vllm smoke
log "import vllm"
"$TARGET/bin/python3" -c "import vllm; print(vllm.__version__)" \
  || { log "import vllm failed"; exit 33; }

# 5. boot a tiny dense model
log "booting $SMOKE_MODEL on :$SMOKE_PORT"
"$TARGET/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model "$SMOKE_MODEL" \
    --port "$SMOKE_PORT" \
    --host 127.0.0.1 \
    --max-model-len 256 \
    --gpu-memory-utilization 0.20 \
    --enforce-eager \
    >"$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

# 6. wait for /v1/models 200
deadline=$(( $(date +%s) + SMOKE_TIMEOUT ))
while (( $(date +%s) < deadline )); do
  if curl -fsS "http://127.0.0.1:$SMOKE_PORT/v1/models" >/dev/null 2>&1; then
    log "/v1/models 200"
    break
  fi
  sleep 2
done
if ! curl -fsS "http://127.0.0.1:$SMOKE_PORT/v1/models" >/dev/null 2>&1; then
  log "server boot timeout (see $WORKDIR/server.log)"
  exit 34
fi

# 7. completion path
log "POST /v1/completions"
RESPONSE=$(curl -fsS -X POST "http://127.0.0.1:$SMOKE_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$SMOKE_MODEL\", \"prompt\": \"hello\", \"max_tokens\": 4, \"temperature\": 0}" 2>&1) \
  || { log "completion failed: $RESPONSE"; exit 35; }

echo "$RESPONSE" | python3 -c \
  "import json,sys;d=json.load(sys.stdin);assert d['choices'][0]['text'],'empty completion';print('OK:',d['choices'][0]['text'][:40])" \
  || { log "completion JSON malformed: $RESPONSE"; exit 35; }

log "SMOKE PASS — tarball $TARBALL serves $SMOKE_MODEL"
exit 0
